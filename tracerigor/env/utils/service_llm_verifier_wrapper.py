# service_llm_verifier_wrapper.py (minimal patched version)
from typing import Dict, Any, List, Tuple
import asyncio, re
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from tracerigor.verifier.verifier.openai_verifier import run_openai_verifier
from tracerigor.utils.response_utils import replace_reasoning_block

def _extract_tag(text: str, tag: str) -> str | None:
    if not text:
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else None

def _replace_think_block(original: str, new_think: str) -> str:
    if not original:
        return original
    # Replace existing <think>…</think> if present
    m = re.search(r"(.*?)(<think>.*?</think>)(.*)", original, flags=re.S | re.I)
    if m:
        return m.group(1) + f"<think>{new_think}</think>" + m.group(3)
    # Otherwise, inject <think>…</think> just before the <answer>…</answer> (if any)
    m2 = re.search(r"(.*?)(<answer>.*?</answer>)(.*)", original, flags=re.S | re.I)
    if m2:
        return m2.group(1) + f"<think>{new_think}</think>" + m2.group(2) + m2.group(3)
    # Fallback: append at end
    return original + f"\n<think>{new_think}</think>"

# --- add near the other helpers at top ---
def _scale_piecewise(t: int, start_step: int, end_step: int, hi: float, lo: float) -> float:
    """
    Returns a scale in [lo, hi].
    - t <= start_step  -> hi
    - t >= end_step    -> lo
    - linear in between
    If end_step <= start_step, returns lo (degenerate "already decayed").
    """
    if end_step <= start_step:
        return lo
    if t <= start_step:
        return hi
    if t >= end_step:
        return lo
    alpha = (t - float(start_step)) / float(end_step - start_step)
    return hi + (lo - hi) * alpha

# --- annealing helper: w_t = w_final + (w_init - w_final) * max(0, 1 - t/T) ---
def _anneal_weight(final_w: float, t: int, warmup_iters: int, init_scale: float) -> float:
    """
    Linearly anneal from init_scale*final_w down to final_w over warmup_iters.
    If warmup_iters <= 0, returns final_w (no anneal).
    """
    if warmup_iters <= 0 or final_w == 0.0:
        return final_w
    init_w = final_w * float(init_scale)
    frac = max(0.0, 1.0 - (float(t) / float(warmup_iters)))
    return final_w + (init_w - final_w) * frac

def _extract_think(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else None

# TODO: refactor to consider subtags (e.g., observation, prediction, reasoning) feedback separately
def _inject_after_valid_action(obs_str: str, feedback: str) -> str:
    """Insert feedback right after the 'After your answer...' line."""
    if not feedback or not isinstance(obs_str, str):
        return obs_str
    anchor = "After your answer, the extracted valid action is"
    try:
        i = obs_str.index(anchor)
        # find the first newline after that sentence (or fall back to end)
        j = obs_str.find("\n", i)
        if j == -1:
            j = len(obs_str)
        block = "\n[Optional feedback on your last action & reasoning]" + feedback
        return obs_str[:j] + block + obs_str[j:]
    except ValueError:
        # anchor not found: append at the end
        return obs_str + "\n[Optional feedback on your last action & reasoning]" + feedback

def _advice_text(out: dict) -> str:
    """
    Return the *full* rationale from <think>..</think> ONLY when verdict is NO.
    Hide everything for YES to avoid biasing the policy while it's already aligned.
    'prompt': None, 'response': "<think>From above-right, a Down move doesn't push the box toward the target; the predicted after-state is incorrect.</think><answer>NO</answer>", 'query_success': True, 'score': 0.0, 'parse_success': True, 'error': None, 'model': 'gpt-5-nano-2025-08-07', 'rubric': 'self_consistency'
    """
    # TODO: remove rewrite tag in response if existing
    verdict = (out.get("verdict") or _extract_tag(out.get("response", ""), "answer") or "").upper()
    if verdict == "YES":
        return ""
    return _extract_think(out.get("response", "")) or ""

def _run_coro_blocking(coro):
    """Run an async coroutine from sync context, even if a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

def service_llm_verifier_wrapper(step_batch_func):
    """
    After step_batch, gather queued verifier requests, invoke LLM verifiers,
    write scores back to env infos, and clear requests.
    - Single global verifier config from self.config.verifier
    - Group by rubric (one call per rubric)
    - Merge back by sample id (robust)
    - Scalars -> metrics.turn_metrics[<rubric>_verifier_score]
    - Optional raw -> info['verifier_raw'] (if keep_raw=True)
    """

    def wrapped_step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results = step_batch_func(self, ids2actions)

        ## Nothing to do if verifier is globally disabled for this service
        ## (We keep this guard cheap; real enable/disable sits in env_cfg.verifier.enabled)
        # any_enabled = False
        # for env_id, (_, _, _, info) in results.items():
        #     if info.get("verifier_requests"):
        #         any_enabled = True
        #         break
        # if not any_enabled:
        #     return results

        # 1) Collect pending requests (flat) and remember which env each sample id came from
        #    Also track which envs *had no requests at all* this turn.
        pending: List[Dict[str, Any]] = []
        id2env: Dict[str, Any] = {}
        envs_with_requests = set()

        ## origin code:
        # for env_id, tup in results.items():
        #     info = tup[3]
        #     for req in (info.get("verifier_requests") or []):
        #         # guard here as well (back-compat: missing flag -> treat as eligible)
        #         if req.get("eligible") is False:
        #             continue
        #         sid = req.get("id")
        #         if not sid:
        #             continue  # skip malformed
        #         pending.append(req)
        #         id2env[sid] = env_id

        # if not pending:
        #     return results
        ## end origin code

        for env_id, tup in results.items():
            info = tup[3]
            vreqs = info.get("verifier_requests") or []
            if vreqs:
                envs_with_requests.add(env_id)
            for req in vreqs:
                sid = req.get("id")
                if not sid:
                    continue  # skip malformed
                pending.append(req)
                id2env[sid] = env_id

        # envs that took a step but produced *no* verifier_requests this turn
        env_ids_all = set(results.keys())
        envs_without_requests = env_ids_all - envs_with_requests

        # 2) Single global verifier config for the whole service/run
        #    but prefer the env-level VerifierConfig if present.
        #    We take the first env in this batch to read its cfg.
        first_env_id = next(iter(results.keys()))
        env_level_cfg = None
        if hasattr(self, "env_configs"):
            env_cfg = self.env_configs.get(first_env_id)
            if env_cfg is not None and getattr(env_cfg, "verifier", None) is not None:
                env_level_cfg = asdict(env_cfg.verifier)

        service_level_cfg = {}
        if hasattr(self, "config") and getattr(self.config, "verifier", None):
            # allow a service-level override if you actually set it
            try:
                service_level_cfg = dict(self.config.verifier)
            except Exception:
                # self.config.verifier might be a dataclass or OmegaConf
                service_level_cfg = asdict(self.config.verifier) if hasattr(self.config.verifier, "__dict__") else {}

        # precedence: env-level > service-level > hard defaults
        vcfg = {**({"use_images": True, "use_observation_text": False}), **service_level_cfg, **(env_level_cfg or {})}

        use_images = vcfg.get("use_images", True)  # default True
        use_obs_txt = vcfg.get("use_observation_text", False)
        model_name   = vcfg.get("model_name", "gpt-5-nano-2025-08-07")
        model_params = vcfg.get("model_params", {"temperature": 1.0})
        weights      = vcfg.get("reward_weights", {})
        keep_raw     = vcfg.get("keep_raw", False)

        # --- anneal config (supports both new piecewise and legacy warmup/init_scale) ---
        anneal_cfg   = vcfg.get("anneal", {}) or {}
        decay_start  = int(anneal_cfg.get("decay_start_step", 35))
        decay_end    = int(anneal_cfg.get("decay_end_step", 100))
        hi_scale     = float(anneal_cfg.get("hi_scale", 1.0))
        lo_scale     = float(anneal_cfg.get("lo_scale", 0.5))
        anneal_enable = anneal_cfg.get("enabled", False)
        # shaping mode
        symmetric_mode = bool(vcfg.get("symmetric_shaping", False))
        gate_mul = bool(vcfg.get("gate_reward_multiplicative", True))

        gate_cfg    = vcfg.get("gating", {}) or {}
        gate_start  = int(gate_cfg.get("start_step", 25))
        gate_harden = int(gate_cfg.get("harden_step", 80))
        big_thr     = float(gate_cfg.get("big_threshold", 1.0))
        alpha_big   = float(gate_cfg.get("floor_big", 0.1))     # tiny floor for big (terminal) rewards when gated
        alpha_small = float(gate_cfg.get("floor_small", 0.2))   # tiny floor for small (per-step) rewards when gated
        beta_small  = float(gate_cfg.get("small_additive", 0.2))  # additive bonus on small rewards
        gate_small_after_harden = bool(gate_cfg.get("gate_small_after_harden", False))

        # --- new enable schedule knobs ---
        enable_after_step   = int(vcfg.get("enable_after_step", 120))  # -1 => enabled from the start
        post_enable_window  = int(vcfg.get("post_enable_window", 50))  # “first 50 steps after enabling”
        gate_small_after_enable = bool(gate_cfg.get("gate_small_after_enable", True))  # whether to gate small rewards multiplicatively after window

        # read-only: current trainer-ish step maintained by the service on reset_batch
        t = getattr(self, "_verifier_train_step", 0)

        # --- hard disable after a given step (skip verifier entirely) ---
        disable_after_step = int(vcfg.get("disable_after_step", -1))  # -1 => never disable
        if disable_after_step >= 0 and t >= disable_after_step:
            # count & drop any queued requests so they don't linger
            dropped = 0
            for env_id, tup in results.items():
                lst = tup[3].pop("verifier_requests", None)
                if lst:
                    dropped += len(lst)
            print(f"[VERIF-SVC] verifier DISABLED at t={t} (threshold={disable_after_step}); skipped {dropped} pending req(s).")
            return results

        # --- new: do not run verifier before enable_after_step ---
        if enable_after_step >= 0 and t < enable_after_step:
            dropped = 0
            for env_id, tup in results.items():
                lst = tup[3].pop("verifier_requests", None)
                if lst:
                    dropped += len(lst)
            if dropped or (t % 50 == 0):  # light debug; prints every ~50 steps even if 0
                print(f"[VERIF-SVC] verifier NOT YET ENABLED (t={t} < {enable_after_step}); "
                    f"skipped {dropped} pending req(s).")
            return results

        # --- verifier is ENABLED for this step ---

        # derive the active rubrics for this service/env, and normalize names
        # so they match the metric keys used when we *do* get a score.
        raw_rubrics = vcfg.get("rubrics") or ["self_consistency"]

        # match the name normalization used elsewhere (history -> history_consistency)
        rubric_names: List[str] = []
        for r in raw_rubrics:
            rubric_names.append("history_consistency" if r == "history" else r)

        # ============================================================
        # NEW: periodic gate (run verifier only every K trainer steps)
        # ============================================================
        run_every_k = int(vcfg.get("run_every_k_steps", 5))
        if run_every_k > 1:
            # Align counting to enable_after_step if you want "every K after enabling"
            base = enable_after_step if enable_after_step >= 0 else 0
            run_this_step = ((t - base) % run_every_k) == 0
        else:
            run_this_step = True

        if not run_this_step:
            # IMPORTANT: we are deliberately skipping verifier even though requests exist.
            # We must keep metrics step-aligned and MUST NOT shape rewards.

            for env_id, tup in results.items():
                info = tup[3]
                tm = info.setdefault("metrics", {}).setdefault("turn_metrics", {})

                tm["verifier_train_step"] = t

                for rub in rubric_names:
                    tm.setdefault(f"{rub}_verifier_score", 0.0)
                    tm.setdefault(f"{rub}_query_success", False)
                    tm.setdefault(f"{rub}_parse_success", False)

                    if rub in ("self_consistency", "self-consistency"):
                        for sub_name in ("aggregate", "observation", "reasoning", "prediction"):
                            tm.setdefault(f"{rub}_{sub_name}_verifier_score", 0.0)

            # clear any queued requests so they don't linger
            for env_id, tup in results.items():
                tup[3].pop("verifier_requests", None)

            # light debug (optional)
            if t % 50 == 0:
                print(f"[VERIF-SVC] periodic skip at t={t} (run_every_k_steps={run_every_k})")

            return results
        # ============================================================
        # END NEW periodic gate
        # ============================================================

        # For envs that had *no* verifier_requests this turn, treat that as
        # "invalid reasoning format" (or invalid action) and set their
        # <rubric>_verifier_score to 0.0 (and subtag scores to 0.0 where applicable).
        for env_id in envs_without_requests:
            obs, reward, done, info = results[env_id]
            tm = info.setdefault("metrics", {}).setdefault("turn_metrics", {})
            tm["verifier_train_step"] = t  # NEW: step-aligned even when no request
            for rub in rubric_names:
                tm.setdefault(f"{rub}_verifier_score", 0.0)

                # NEW: ensure per-subtag self-consistency scores stay step-aligned
                # even when we don't send a verifier request (e.g., invalid action).
                if rub in ("self_consistency", "self-consistency"):
                    for sub_name in ("aggregate", "observation", "prediction", "reasoning"):
                        key = f"{rub}_{sub_name}_verifier_score"
                        tm.setdefault(key, 0.0)

        # If there are no actual requests to send (all invalid-format), we’re done:
        # we just wrote 0.0 for those steps; nothing to query remotely.
        if not pending:
            for env_id, tup in results.items():
                tup[3].pop("verifier_requests", None)
            return results

        # 3) Group by rubric to reduce calls
        rubric2items: Dict[str, List[Dict[str, Any]]] = {}
        for req in pending:
            # rubric2items.setdefault(req["rubric"], []).append({
            #     "id": req["id"],
            #     "reasoning_tokens": req.get("reasoning_tokens"),
            #     "action_tokens": req.get("action_tokens"),
            #     "admissible_actions": req.get("admissible_actions"),
            #     "current_step": req.get("current_step"),
            #     "history": req.get("history"),
            #     "current_observation_text": req.get("current_observation_text"),
            #     "current_observation_image": req.get("current_observation_image"),
            # })
            # make a shallow copy so we don't mutate env info
            item = {
                "id": req["id"],
                "reasoning_tokens": req.get("reasoning_tokens"),
                "action_tokens": req.get("action_tokens"),
                "admissible_actions": req.get("admissible_actions"),
                "current_step": req.get("current_step"),
                "history": req.get("history"),
                "current_observation_text": req.get("current_observation_text"),
                "current_observation_image": req.get("current_observation_image"),
            }

            # strip by config
            if not use_images:
                item["current_observation_image"] = None
            if not use_obs_txt:
                item["current_observation_text"] = ""

            rubric2items.setdefault(req["rubric"], []).append(item)

        # 4) Run verifier per rubric; merge outputs by id (robust to reordering)
        id2out: Dict[str, Dict[str, Any]] = {}
        for rubric, items in rubric2items.items():
            outs = _run_coro_blocking(run_openai_verifier(
                input_data=items,
                rubric=rubric,
                model_name=model_name,
                model_params=model_params,
            ))
            for out in outs:
                sid = out.get("id")
                if sid:
                    out.setdefault("rubric", rubric)
                    id2out[sid] = out

        # 5) Write back scores (and optional reward shaping); clear requests
        new_results = {env_id: list(tup) for env_id, tup in results.items()}
        # optional name normalization for weights lookup
        name_norm = {"self_consistency": "self_consistency",
                    "history": "history_consistency",
                    "grounding": "grounding"}

        for req in pending:
            sid    = req["id"]
            rubric = req["rubric"]
            out    = id2out.get(sid)
            if not out:
                continue
            env_id = id2env.get(sid)
            if env_id is None:
                continue

            obs, reward, done, info = new_results[env_id]
            info.setdefault("metrics", {}).setdefault("turn_metrics", {})
            # record the (global) verifier step for traceability
            info["metrics"]["turn_metrics"]["verifier_train_step"] = t

            score = float(out.get("score", 0.0))
            info["metrics"]["turn_metrics"][f"{rubric}_verifier_score"] = score

            info["metrics"]["turn_metrics"][f"{rubric}_query_success"] = bool(out.get("query_success", False))
            info["metrics"]["turn_metrics"][f"{rubric}_parse_success"] = bool(out.get("parse_success", False))

            # --- NEW: ensure per-subtag keys exist even if out has no sub-scores (parse fail / truncation) ---
            if rubric in ("self_consistency", "self-consistency"):
                for sub_name in ("aggregate", "observation", "reasoning", "prediction"):
                    info["metrics"]["turn_metrics"].setdefault(f"{rubric}_{sub_name}_verifier_score", 0.0)
            # --- END NEW ---

            # --- NEW: record per-subtag scores if provided ---
            # Prefer post-hoc extraction from flat fields added by the verifier
            # (sub_observation_score, sub_reasoning_score, sub_prediction_score, sub_aggregate_score).
            sub_scores = {}
            for sub_name in ("observation", "reasoning", "prediction", "aggregate"):
                k = f"sub_{sub_name}_score"
                if k in out and out[k] is not None:
                    sub_scores[sub_name] = out[k]

            # Fallback: legacy dict-style sub_scores/subtag_scores if ever present
            if not sub_scores:
                sub_scores = out.get("sub_scores") or out.get("subtag_scores") or {}

            # Expected shape: {"observation": 0/1/None, "reasoning": 0/1/None, "prediction": 0/1/None, ...}
            for sub_name, sub_val in sub_scores.items():
                try:
                    sub_f = float(sub_val)
                except (TypeError, ValueError):
                    continue
                # e.g. "self_consistency_observation_verifier_score"
                key = f"{rubric}_{sub_name}_verifier_score"
                info["metrics"]["turn_metrics"][key] = sub_f
            # --- END NEW ---

            apply_rewrite = bool(vcfg.get("apply_rewrite_to_response", True))
            if rubric in ("self_consistency", "self-consistency") and apply_rewrite:
                resp_text = out.get("response") or ""
                verdict = (_extract_tag(resp_text, "answer") or "").strip().upper()
                rewrite = _extract_tag(resp_text, "rewrite")
                if verdict == "NO" and rewrite:
                    # Replace only the reasoning block of the agent's raw response.
                    orig_resp = info.get("llm_raw_response", "")
                    corrected = replace_reasoning_block(orig_resp, rewrite)
                    if corrected and corrected != orig_resp:
                        # Non-destructive: keep both, but prefer the corrected one later
                        info["llm_rewritten_response"] = corrected
                        info["verifier_rewrite"] = rewrite
                        info["verifier_rewrite_applied"] = True

            ## --- attach verifier advice into the observation (optional) ---
            # advice_into_obs      = vcfg.get("advice_into_obs", False)
            # advice_append_to_str = vcfg.get("advice_append_to_obs_str", False)
            # advice_max_chars     = int(vcfg.get("advice_max_chars", 240))

            # if advice_into_obs:
            #     obs_dict = new_results[env_id][0]  # observation dict
            #     if isinstance(obs_dict, dict):
            #         adv_bucket = obs_dict.setdefault("verifier_advice", {})
            #         verdict, rationale = _advice_bits(out, advice_max_chars)  # << mask on YES
            #         adv_bucket[rubric] = {"verdict": verdict, "score": score, "reason": rationale}
            #         if advice_append_to_str and "obs_str" in obs_dict:
            #             one_line = f"\n[Verifier Note] (optional) {rubric}: {verdict}"
            #             if rationale:
            #                 one_line += f" — {rationale}"
            #             obs_dict["obs_str"] = (obs_dict.get("obs_str") or "") + one_line

            # --- optional: attach verifier advice into the observation ---
            # only for eligible reasoning-action pairs: is_format_rewarded and action_is_effective
            advice_into_obs      = vcfg.get("advice_into_obs", True)
            advice_append_to_str = vcfg.get("advice_append_to_obs_str", True)
            if advice_into_obs:
                obs_dict = new_results[env_id][0]  # current observation dict
                if isinstance(obs_dict, dict):
                    rationale = _advice_text(out)  # full <think> text only for NO
                    if rationale:  # only inject when there's actual critique
                        # 1) machine-readable bucket (verdict hidden from the agent)
                        # adv_bucket = obs_dict.setdefault("verifier_advice", {})
                        # adv_bucket[rubric] = {
                        #     "reason": rationale,      # expose ONLY the critique
                        #     # keep verdict/score internal if you like, but don't add to obs:
                        #     # "_verdict": out.get("verdict"),
                        #     # "_score": score,
                        # }
                        # 2) optional one-liner added to obs_str (no verdict, just neutral label)
                        # if advice_append_to_str and "obs_str" in obs_dict:
                        #     obs_dict["obs_str"] += (
                        #         "\n[Optional feedback on your last action & reasoning]"
                        #         f"\n{rationale}"
                        #     )
                        if advice_append_to_str and "obs_str" in obs_dict and not obs_dict.get("_advice_injected", False):
                            obs_dict["obs_str"] = _inject_after_valid_action(obs_dict["obs_str"], rationale)
                            obs_dict["_advice_injected"] = True  # avoid duplicate inserts if multiple rubrics fire

            if keep_raw:
                # store raw outside turn_metrics to avoid cluttering scalar namespace
                info.setdefault("verifier_raw", []).append(out)
                # or trimmed
                # info.setdefault("verifier_raw", []).append({
                #     "id": out["id"],
                #     "response": out.get("response"),
                #     "score": out.get("score"),
                #     "parse_success": out.get("parse_success"),
                #     "query_success": out.get("query_success"),
                #     "error": out.get("error"),
                #     # include token stats when present (chat-completions path)
                #     **({k: out.get(k) for k in ("p_yes", "p_no", "binary_entropy", "topk_entropy_at_label", "verdict_logprob", "verdict_prob_full")
                #         if k in out},),
                # })

            w = float(weights.get(name_norm.get(rubric, rubric), 1.0))
            if w:
                # compute time-varying weight
                if anneal_enable and decay_end > decay_start:
                    w_t = w * _scale_piecewise(t, decay_start, decay_end, hi_scale, lo_scale)
                else:
                    w_t = w

                ## option A: simple gating
                # if gate_mul:
                #     # Loosened gate:
                #     # - For big rewards (>1): trust verifier (multiplicative).
                #     # - For small positive rewards (0,1): let env reward pass unchanged
                #     # - For non-positive rewards: leave as-is (keep penalties / zeros).
                #     if reward > 1.0:
                #         new_results[env_id][1] = reward * (w_t * score)
                #     elif 0.0 < reward < 1.0:
                #         new_results[env_id][1] = reward # + (w_t * score)
                #     else:
                #         new_results[env_id][1] = reward
                # option B: more complex gating
                if gate_mul:
                    if enable_after_step >= 0:
                        # --- new: two-phase gating relative to the enable step ---
                        since_enable = max(0, t - enable_after_step)
                        if since_enable < post_enable_window:
                            # Phase A (first ~50 steps after enabling):
                            # big -> multiplicative with floor; small -> additive; neg/zero -> passthrough
                            if reward > big_thr:
                                mult = alpha_big + (1.0 - alpha_big) * score  # tiny floor
                                new_results[env_id][1] = reward * (w_t * mult)
                            elif reward > 0.0:
                                new_results[env_id][1] = reward + (beta_small * w_t * score)
                            else:
                                new_results[env_id][1] = reward
                        else:
                            # Phase B (after window):
                            # big -> fully multiplicative; small -> either multiplicative-with-floor or small additive
                            if reward > big_thr:
                                new_results[env_id][1] = reward * (w_t * score)
                            elif reward > 0.0:
                                if gate_small_after_enable:
                                    mult = alpha_small + (1.0 - alpha_small) * score
                                    new_results[env_id][1] = reward * (w_t * mult)
                                else:
                                    new_results[env_id][1] = reward + (beta_small/2.0 * w_t * score)
                            else:
                                new_results[env_id][1] = reward
                    else:
                        # Before gate_start: pass-through (optionally keep a tiny beta_small, set beta_small=0 if you want pure pass-through)
                        if t < gate_start:
                            if reward > big_thr:
                                mult = alpha_big * 0.5 + (1.0 - alpha_big * 0.5) * score
                                new_results[env_id][1] = reward * (w_t * mult)
                            elif reward > 0.0:
                                new_results[env_id][1] = reward  # + (beta_small/2 * w_t * score)  # set beta_small=0.0 to disable
                            else:
                                new_results[env_id][1] = reward
                        # Between start and harden: big rewards multiplicative with floor; small rewards additive
                        elif t < gate_harden:
                            if reward > big_thr:
                                mult = alpha_big + (1.0 - alpha_big) * score  # tiny floor
                                new_results[env_id][1] = reward * (w_t * mult)
                            elif reward > 0.0:
                                new_results[env_id][1] = reward + (beta_small * w_t * score)
                            else:
                                new_results[env_id][1] = reward
                        # After harden: optionally gate small rewards too, still with floor (or keep additive)
                        else:
                            if reward > big_thr:
                                new_results[env_id][1] = reward * (w_t * score)
                            elif reward > 0.0:
                                if gate_small_after_harden:
                                    mult = alpha_small + (1.0 - alpha_small) * score  # reuse same alpha; or add alpha_small if you want
                                    new_results[env_id][1] = reward * (w_t * mult)
                                else:
                                    new_results[env_id][1] = reward + (beta_small/2 * w_t * score)
                            else:
                                new_results[env_id][1] = reward
                else:
                    # existing shaping paths
                    if symmetric_mode:
                        # symmetric: delta in [-w_t, +w_t] via transform score∈{0,1} -> sym∈{-1,+1}
                        sym  = 2.0 * score - 1.0
                        delta = w_t * sym
                        new_results[env_id][1] = reward + delta
                    else:
                        # asymmetric: only bonus
                        bonus = max(w_t * score, 0.0)
                        new_results[env_id][1] = reward + bonus

        # Clear consumed requests to avoid re-use next step
        for env_id, tup in new_results.items():
            tup[3].pop("verifier_requests", None)
            # drop the one-shot injection flag (keeps advice text, removes sentinel)
            obs = tup[0]
            if isinstance(obs, dict):
                obs.pop("_advice_injected", None)

        return {env_id: tuple(tup) for env_id, tup in new_results.items()}

    # def wrapped_step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
    #     results = step_batch_func(self, ids2actions)

    #     # Collect all pending requests (flat shape)
    #     pending: List[Tuple[str, Dict[str, Any]]] = []  # (rubric, item)
    #     id2env: Dict[str, Any] = {}
    #     for env_id, tup in results.items():
    #         info = tup[3]
    #         for req in (info.get("verifier_requests") or []):
    #             rub = req["rubric"]
    #             item = {
    #                 "id": req.get("id"),
    #                 "reasoning_tokens": req.get("reasoning_tokens"),
    #                 "action_tokens": req.get("action_tokens"),
    #                 "admissible_actions": req.get("admissible_actions"),
    #                 "current_step": req.get("current_step"),
    #                 "history": req.get("history"),
    #                 "current_observation_text": req.get("current_observation_text"),
    #                 "current_observation_image": req.get("current_observation_image"),
    #             }
    #             if item["id"] is None:
    #                 # Skip malformed entries defensively
    #                 continue
    #             pending.append((rub, item))
    #             id2env[item["id"]] = env_id

    #     if not pending:
    #         return results

    #     # Global verifier config for this service/run
    #     vcfg = getattr(self, "config", None)
    #     vcfg = getattr(vcfg, "verifier", None) or {}
    #     model_name   = vcfg.get("model_name", "gpt-5-nano-2025-08-07")
    #     model_params = vcfg.get("model_params", {"temperature": 0.1})
    #     weights      = vcfg.get("reward_weights", {})
    #     keep_raw     = vcfg.get("keep_raw", False)

    #     # Group by rubric to reduce calls
    #     rubric2items: Dict[str, List[Dict[str, Any]]] = {}
    #     for rub, item in pending:
    #         rubric2items.setdefault(rub, []).append(item)

    #     # Call verifier per rubric and merge back by id
    #     id2out: Dict[str, Dict[str, Any]] = {}
    #     for rub, items in rubric2items.items():
    #         outs = _run_coro_blocking(run_openai_verifier(
    #             input_data=items,
    #             rubric=rub,
    #             model_name=model_name,
    #             model_params=model_params,
    #         ))
    #         for out in outs:
    #             sid = out.get("id")
    #             if sid is not None:
    #                 # annotate for downstream
    #                 out.setdefault("rubric", rub)
    #                 id2out[sid] = out

    #     # Merge back into results; add scalar + (optional) shaped reward
    #     new_results = {env_id: list(tup) for env_id, tup in results.items()}
    #     name_norm = {"self_consistency":"self_consistency",
    #                  "history":"history_consistency",
    #                  "grounding":"grounding"}

    #     for rub, item in pending:
    #         sid = item["id"]
    #         out = id2out.get(sid)
    #         if not out:
    #             continue
    #         env_id = id2env.get(sid)
    #         if env_id is None:
    #             continue
    #         obs, reward, done, info = new_results[env_id]

    #         score = float(out.get("score", 0.0))
    #         info.setdefault("metrics", {}).setdefault("turn_metrics", {})[
    #             f"{rub}_verifier_score"
    #         ] = score

    #         if keep_raw:
    #             info.setdefault("verifier_raw", []).append(out)

    #         w = float(weights.get(name_norm.get(rub, rub), 0.0))
    #         if w:
    #             new_results[env_id][1] = reward + w * score

    #     # Clear consumed requests
    #     for env_id, tup in new_results.items():
    #         tup[3].pop("verifier_requests", None)

    #     return {env_id: tuple(t) for env_id, t in new_results.items()}

    return wrapped_step_batch