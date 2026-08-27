# service_llm_verifier_wrapper.py  (patched)
from typing import Dict, Any, List, Tuple
import asyncio, json
from concurrent.futures import ThreadPoolExecutor

from tracerigor.verifier.verifier.openai_verifier import run_openai_verifier  # async
# If you already added a sync shim, you can import that instead.

def _run_coro_blocking(coro):
    """Run an async coroutine from sync context, even if a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

def _extract_item(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward/forward-compatible extraction of the verifier input item.
    Supports either:
      - new shape: {"payload": {...}, "rubric": "...", "id": "..."}
      - old shape: {"id","rubric","reasoning_tokens","action_tokens",...}
    """
    if "payload" in req and isinstance(req["payload"], dict):
        return req["payload"]
    # old flat shape → construct the expected item
    fields = ("id","reasoning_tokens","action_tokens","admissible_actions",
              "current_step","current_observation_text","current_observation_image","history")
    return {k: req.get(k) for k in fields if k in req}

def service_llm_verifier_wrapper(step_batch_func):
    """
    After step_batch, gather queued verifier requests, invoke LLM verifiers,
    and write scores back to env infos. Also clears requests after use and
    supports per-env verifier config & reward shaping.
    """
    def wrapped_step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results = step_batch_func(self, ids2actions)

        # Collect requests across envs
        pending: List[Dict[str, Any]] = []  # each: {"rubric":..., "item":..., "env_id":..., "vcfg":...}
        for env_id, tup in results.items():
            info = tup[3]
            for req in (info.get("verifier_requests") or []):
                rubric = req["rubric"]
                item   = _extract_item(req)
                # per-env verifier config if available, else fallback to service-level
                vcfg = None
                try:
                    # dataclass VerifierConfig on env_configs[env_id], if present
                    vcfg = getattr(self.env_configs[env_id], "verifier", None)
                except Exception:
                    vcfg = None
                if vcfg is None:
                    vcfg = (getattr(self, "config", None) or {}).get("verifier", {})
                pending.append({"rubric": rubric, "item": item, "env_id": env_id, "vcfg": vcfg})

        if not pending:
            return results

        # Group by (rubric, model_name, model_params) so heterogeneous envs work
        buckets: Dict[tuple, List[Dict[str, Any]]] = {}
        for p in pending:
            v = p["vcfg"] or {}
            model_name   = getattr(v, "model_name", None) or v.get("model_name", "gpt-5-nano-2025-08-07")
            model_params = getattr(v, "model_params", None) or v.get("model_params", {})
            key = (p["rubric"], model_name, json.dumps(model_params, sort_keys=True))
            buckets.setdefault(key, []).append(p)

        # Stable id→env map (for merge-back), and run each bucket
        # We rely on the verifier returning an "id" field per item (your pipeline already sets this).
        for (rubric, model_name, params_json), group in buckets.items():
            model_params = json.loads(params_json) if params_json else {}
            items = [g["item"] for g in group]
            id2env = {g["item"]["id"]: g["env_id"] for g in group}

            outs = _run_coro_blocking(run_openai_verifier(
                input_data=items,
                rubric=rubric,
                model_name=model_name,
                model_params=model_params,
            ))

            # Merge back by item id (order-safe)
            for out in outs:
                sample_id = out.get("id")
                if sample_id is None:
                    continue
                env_id = id2env.get(sample_id)
                if env_id is None:
                    continue
                obs, reward, done, info = results[env_id]

                # Store scores in a consistent place
                info.setdefault("verifier", {}).setdefault("scores", {})[rubric] = float(out.get("score", 0.0))
                info["verifier"].setdefault("raw", {}).setdefault(rubric, []).append({
                    "id": out.get("id"),
                    "response": out.get("response"),
                    "score": out.get("score"),
                    "parse_success": out.get("parse_success"),
                    "query_success": out.get("query_success"),
                    "error": out.get("error"),
                    # token stats (when available):
                    **{k: out.get(k) for k in (
                        "verdict","p_yes","p_no","binary_entropy",
                        "topk_entropy_at_label","verdict_logprob","verdict_prob_full"
                    ) if k in out},
                })

                # Optional: also mirror into turn_metrics for your dashboards
                info.setdefault("metrics", {}).setdefault("turn_metrics", {})[
                    f"{rubric}_verifier_score"
                ] = float(out.get("score", 0.0))

                # Reward shaping (per-env)
                vcfg = group[0]["vcfg"] or {}
                reward_weights = getattr(vcfg, "reward_weights", None) or vcfg.get("reward_weights", {}) or {}
                # name normalization
                weight_key = {
                    "self_consistency": "self_consistency",
                    "history": "history_consistency",
                    "grounding": "grounding",
                }.get(rubric, rubric)
                w = float(reward_weights.get(weight_key, 0.0))
                if w:
                    shaped = w * float(out.get("score", 0.0))
                    results[env_id] = (obs, reward + shaped, done, info)

        # Clear consumed requests to avoid duplicate re-use next step
        for env_id, tup in results.items():
            tup[3].pop("verifier_requests", None)

        return results

    return wrapped_step_batch