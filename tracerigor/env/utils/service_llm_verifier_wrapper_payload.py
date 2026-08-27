# verifier/common/service_llm_verifier_wrapper.py
from typing import Dict, Any, Tuple, List
from collections import defaultdict
from dataclasses import asdict
from tracerigor.verifier.verifier.openai_verifier_sync import run_openai_verifier_sync

def service_llm_verifier_wrapper(step_batch_func):
    """
    Batch-level wrapper:
    - Reads info['verifier_requests'] that env.step populated (fast; no LLM call there).
    - Groups by rubric and calls the Verifier in batches.
    - Merges scores back into each env's info and applies optional reward shaping.
    Expect each info to contain:
        info['verifier_requests'] = [
            {
              "id": <env_id>,
              "rubric": "self_consistency" | "history" | "grounding",
              "payload": {  # the item dict the verifier expects (messages are assembled inside)
                  "id": <sample_id>,
                  "reasoning_tokens": "...",
                  "action_tokens": "...",
                  "admissible_actions": [...],
                  "current_step": int,
                  "current_observation_text": str | None,
                  "current_observation_image": Sequence[path|bytes|... ] | None,
                  "history": [...],  # short history items (optional)
                  # anything else your template needs
              }
            }, ...
        ]
    """
    def _wrapped(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results = step_batch_func(self, ids2actions)

        # Nothing to do if verifier is globally disabled for this service
        # (We keep this guard cheap; real enable/disable sits in env_cfg.verifier.enabled)
        any_enabled = False
        for env_id, (_, _, _, info) in results.items():
            if info.get("verifier_requests"):
                any_enabled = True
                break
        if not any_enabled:
            return results

        # Group all pending requests by rubric
        by_rubric: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        id_to_env = {}
        for env_id, tup in results.items():
            obs, reward, done, info = tup
            for req in info.get("verifier_requests", []):
                r = req.get("rubric")
                item = req.get("payload", {})
                by_rubric[r].append(item)
                id_to_env[item["id"]] = env_id  # map sample id -> env_id for merge-back

        # For each rubric, batch-call the verifier
        for rubric, input_data in by_rubric.items():
            # note: pull per-env cfg just once (we assume same model per service;
            # if heterogeneous, you can split by model_name here)
            # Take any env_id that had requests for this rubric:
            env_id = id_to_env[input_data[0]["id"]]
            vcfg = self.env_configs[env_id].verifier
            model_name = vcfg.model_name
            model_params = vcfg.model_params or {}

            # Run the verifier synchronously (inside Flask/server process)
            ver_results = run_openai_verifier_sync(
                input_data=input_data,
                rubric=rubric,
                model_name=model_name,
                model_params=model_params,
            )

            # Merge results back into per-env infos and (optional) reward shaping
            for r in ver_results:
                # r carries: id, response, score, parse_success, query_success, error, model, etc.
                env_id = id_to_env.get(r["id"])
                if env_id is None:
                    continue
                obs, reward, done, info = results[env_id]

                info.setdefault("verifier", {}).setdefault("scores", {})[rubric] = r["score"]
                info["verifier"].setdefault("raw", {}).setdefault(rubric, []).append({
                    "id": r["id"],
                    "response": r.get("response"),
                    "score": r.get("score"),
                    "parse_success": r.get("parse_success"),
                    "query_success": r.get("query_success"),
                    "error": r.get("error"),
                    # include token stats when present (chat-completions path)
                    **({k: r.get(k) for k in ("p_yes","p_no","binary_entropy","topk_entropy_at_label","verdict_logprob","verdict_prob_full")
                        if k in r},),
                })

                # Optional: reward shaping
                w = (vcfg.reward_weights or {}).get({
                    "self_consistency": "self_consistency",
                    "history": "history_consistency",
                    "grounding": "grounding",
                }.get(rubric, rubric), 0.0)
                if w:
                    shaped = float(r["score"]) * float(w)
                    results[env_id] = (obs, reward + shaped, done, info)

        # Clear consumed requests to avoid re-sending on next step
        for env_id in results:
            results[env_id][3].pop("verifier_requests", None)

        return results
    return _wrapped