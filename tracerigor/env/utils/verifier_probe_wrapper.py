# Process Reward for Evolving Interpretable Reasoning Traces.
# Focus: encourage self-consistent, factually grounded reasoning, etc.
#
# These rewards shape reasoning quality independent of task success, pushing the agent toward:
# - Faithful state attribution (no hallucination)
# - Internal consistency across steps
# - Transparent, auditable intermediate traces
#
# Requirements for an environment to enable these rewards:
# - Must implement get_env_state() -> str
#   (Returns a canonical textual snapshot used as ground-truth for judging grounding/prediction)
# - Must populate info dict with:
#     is_format_rewarded flag (to trigger state-based judging)
#
# The wrappers in this file:
# - Capture pre/post step states
# - Package query tuples
# - Call an LLM verifier service to obtain normalized scores
# - Inject per-turn process rewards into info["metrics"]["turn_metrics"]
#
# Design notes:
# - Rewards are additive (+1 or -1) and weighted via env_config:
# - If format reward is absent, process reward paths short-circuit to zero
# - Multiple versions (v1/v2/v3) exist to support evolving prompt / parsing strategies



# verifier glue for Sokoban rubrics
from typing import Dict, Any, List, Optional

import time
from tracerigor.env.sokoban.utils import sokoban_state_to_sentences
from tracerigor.server.llm_as_judge import run_llm_judge
from tracerigor.server.llm_as_judge_sokoban_frozenlake import run_llm_judge as run_llm_judge_new

import asyncio
from tracerigor.verifier.verifier.openai_verifier import run_openai_verifier  # your coroutine entrypoint
# If you also want an Ollama path, import your ollama runner and branch on config.
from tracerigor.verifier.verifier.common.obs_utils import extract_text_from_obs, extract_images_from_obs

def llm_verifier_probe_wrapper(step_func):
    """
    Collects verifier inputs at each step and stashes them into info['verifier_requests'].
    Does not call any LLM (fast). Batch service wrapper will consume them.
    """
    """
    - Grabs the *pre-step* observation (text+images) from env._last_obs
    - Calls the original step (so parse results are available)
    - Appends a verifier request for each rubric into info['verifier_requests']
    - Updates per-episode memory with (pre-step obs + tokens) for history use
    """
    def wrapped_step(self, action_str: str):
        # 1) capture *pre-step* observation (text + images)
        pre_obs = getattr(self, "_last_obs", None)
        pre_text = extract_text_from_obs(pre_obs) if pre_obs else ""
        pre_imgs = extract_images_from_obs(pre_obs, placeholder=self.config.get("image_placeholder", "<image>")) if pre_obs else []

        # 2) run original step
        # run env step first (so we know success, post obs, etc.)
        obs, reward, done, info = step_func(self, action_str)

        # ensure containers
        info.setdefault("metrics", {}).setdefault("turn_metrics", {})
        info.setdefault("verifier_requests", [])

        # 3) fetch tokens parsed during step
        # --- extract agent reasoning/actions from your parse result (already in info from your step)
        # You already put parse results into info; if not, adapt these two lines to your plumbing:
        reasoning_tokens: str = f"<think>{info.get('think_content', '')}</think>"
        action_tokens: str    = f"<action>{info.get('action_content', ','.join(getattr(self, 'valid_actions', [])))}</action>"

        # 4) add a memory record (pre-step obs + tokens)
        if hasattr(self, "_verifier_mem"):
            self._verifier_mem.add_step(
                observation_text=pre_text,
                observation_images=pre_imgs,
                reasoning_tokens=reasoning_tokens,
                action_tokens=action_tokens,
            )
            # short history for the verifier (no raw images in history entries)
            raw_hist = self._verifier_mem.recent(k=self.config.get("verifier_history_k", 3))
            # filter out “empty-token” entries when building history
            short_hist = [h for h in raw_hist if (h.get("reasoning_tokens") or h.get("action_tokens"))]
        else:
            short_hist = []

        # 5) build requests for the rubrics you want this turn
        # what rubrics to run this turn?
        # e.g. set in cfg: self.config.verifier = {"enabled": True, "rubrics": ["self_consistency", "history", "grounding"]}
        verifier_cfg = self.config.get("verifier", None)
        rubrics = (
            getattr(self, "verifier_rubrics", None)
            or getattr(verifier_cfg, "rubrics", None)
            or (verifier_cfg.get("rubrics") if isinstance(verifier_cfg, dict) else None)
            or ["self_consistency", "history", "grounding"]
        )

        name_map = {
            "self_consistency": "self_consistency",
            "history": "history_consistency",
            "grounding": "grounding",
        }
        # admissible actions (fallback to Sokoban defaults if not exposed by env)
        # TODO: make it generic
        admissible = getattr(self, "admissible_actions", ["up", "down", "left", "right"])
        # keep/fill current step index for history templates
        current_step = getattr(self, "_verifier_step_index", 0)

        # 5a) eligibility gate – only enqueue requests when format was rewarded
        tm = info.get("metrics", {}).get("turn_metrics", {})
        eligible = info.get("is_format_rewarded", False) # and tm.get("action_is_effective", False)

        # (optional) current observation shown to the verifier only for contradiction checks
        # If you already have a textual observation, use it; otherwise leave empty.
        # current_observation_text = getattr(self, "last_observation_text", "")

        # maintain short history for history-consistency rubric
        # each item: { observation_text?, observation_image?, reasoning_tokens?, action_tokens? }
        # hist = getattr(self, "_verifier_history", [])
        # info["verifier_history_len"] = len(hist)

        # Try to get the pre-step state id if available
        # If your env sets _last_obs["state_id"] before step, use that
        pre_state_id = None
        if pre_obs and isinstance(pre_obs, dict):
            pre_state_id = pre_obs.get("state_id", None)
        # Fallback to episode_id + current_step (before increment)
        fallback_id = f"{getattr(self, 'episode_id', 'ep')}_{current_step}"

        if eligible:
            for r in rubrics:
                rr = name_map.get(r, r)
                req = {
                "id": pre_state_id or fallback_id,
                "reasoning_tokens": reasoning_tokens,
                "action_tokens": action_tokens,
                "admissible_actions": [str(a) for a in admissible],
                "current_step": current_step,
                "history": short_hist,
                "current_observation_text": pre_text,       # text obs for contradiction checks
                "current_observation_image": pre_imgs,      # image obs (list of bytes/PIL/paths/urls)
                "obs": None,  # leave None; we pass images explicitly above
                "rubric": rr,
                "eligible": True,
                }
                info["verifier_requests"].append(req)
        ## build a common payload; the verifier templates will consume these fields
        # base_payload = {
        #     "id": info.get("state_id") or f"{getattr(self, 'episode_id', 'ep')}_{current_step}",
        #     "reasoning_tokens": reasoning_tokens,
        #     "action_tokens": action_tokens,
        #     "admissible_actions": [str(a) for a in admissible],
        #     "current_step": current_step,
        #     "history": hist[-3:],  # last few is enough
        #     "current_observation_text": current_observation_text,
        #     # if you have images, add: "current_observation_image": <list|bytes|url>
        # }

        # for r in rubrics:
        #     rr = name_map.get(r, r)
        #     req = {"rubric": rr, **base_payload}
        #     info["verifier_requests"].append(req)

        # bump the step counter (reset sets it implicitly by clearing memory)
        self._verifier_step_index = current_step + 1

        return obs, reward, done, info
    return wrapped_step

## REMOVED
def service_llm_verifier_wrapper(step_batch_func):
    """
    After step_batch, gather all queued verifier requests and score them via your LLM verifier.
    Writes scores into info['metrics']['turn_metrics'][f'{rubric}_verifier_score'].
    Optional: shape reward by a weight per rubric.
    """
    def wrapped_step_batch(self, ids2actions):
        step_batch_results = step_batch_func(self, ids2actions)

        # collect all requests
        requests: List[Dict[str, Any]] = []
        idx_lookup: List[tuple] = []  # (id, index_in_results) to write back
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            reqs = info.get("verifier_requests", [])
            if not reqs:
                continue
            for req in reqs:
                requests.append(req)
                idx_lookup.append((id, len(requests)-1))

        if not requests:
            return step_batch_results

        # group by rubric (so we can call model once per rubric)
        rubric2items: Dict[str, List[Dict[str, Any]]] = {}
        for req in requests:
            rubric2items.setdefault(req["rubric"], []).append(req)

        # choose model + params from config
        model_name    = self.config.get("verifier", {}).get("model_name", "gpt-5-nano-2025-08-07")
        model_params  = self.config.get("verifier", {}).get("model_params", {"temperature": 0.1})
        # If you want to switch to Responses API for gpt-5, your run_gpt_request already handles it via config.api.use_responses_api.

        # call the verifier per rubric
        rubric2results: Dict[str, List[Dict[str, Any]]] = {}
        for rubric, items in rubric2items.items():
            # run_openai_verifier is async → block here safely
            # (Runs in trainer process, not uvicorn. If you *are* under an event loop, use asyncio.run_in_thread.)
            results = asyncio.run(run_openai_verifier(
                input_data=items,
                rubric=rubric,
                model_name=model_name,
                model_params=model_params,
            ))
            rubric2results[rubric] = results

        # write scores back into info + optional reward shaping
        weights = self.config.get("verifier", {}).get("reward_weights", {
            "self_consistency": 0.0,
            "history_consistency": 0.0,
            "grounding": 0.0,
        })

        # Make a mutable copy we can update
        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}

        # Build a flat list back in the same order as `requests`
        flat_results: List[Dict[str, Any]] = []
        for req in requests:
            flat_results.append(
                rubric2results[req["rubric"]].pop(0)  # consume in order sent
            )

        for (id, _req_idx), res in zip(idx_lookup, flat_results):
            score = res.get("score", 0.0)
            rubric = res.get("rubric") or res.get("verifier_rubric")  # depending on what your runner returns
            if rubric is None:
                # if your run_openai_verifier doesn't echo rubric, infer from request list:
                # (ids are aligned; you can also add 'rubric' into res at the source)
                pass
            # write metrics
            info = new_step_batch_results[id][3]
            info.setdefault("metrics", {}).setdefault("turn_metrics", {})
            key = f"{res.get('rubric', 'rubric')}_verifier_score"
            info["metrics"]["turn_metrics"][key] = float(score)

            # optional reward shaping
            w = weights.get(res.get("rubric", ""), 0.0)
            if w:
                new_step_batch_results[id][1] += w * float(score)

            # if you want the raw response saved for audit:
            info["metrics"]["turn_metrics"].setdefault("verifier_raw", []).append({
                "rubric": res.get("rubric"),
                "response": res.get("response"),
                "query_success": res.get("query_success"),
                "parse_success": res.get("parse_success"),
                "verdict": res.get("verdict"),
                "score": score,
            })

        return {id: tuple(result) for id, result in new_step_batch_results.items()}

    return wrapped_step_batch


def env_state_reward_wrapper(step_func):
    def wrapped_step(self, action_str):
        if hasattr(self, 'config') and self.config.get("use_state_reward", False):

            prompt_format = self.config.get("prompt_format", None)
            if prompt_format is None:
                raise ValueError("Prompt format is not specified in the config.")
            assert ("grounding" in prompt_format or "worldmodeling" in prompt_format)

            pre_state = self.get_env_state()
            obs, reward, done, info = step_func(self, action_str)
            post_state = self.get_env_state()

            if "metrics" not in info:
                info["metrics"] = {"turn_metrics": {}, "traj_metrics": {}}
            if "turn_metrics" not in info["metrics"]:
                info["metrics"]["turn_metrics"] = {}

            if info.get("is_format_rewarded", False): # if no format reward, no need to calculate state reward, skipping
                info["use_state_reward"] = True
                if "observation_content" in info and info["observation_content"]:
                    info["observation_state"] = pre_state
                if "prediction_content" in info and info["prediction_content"]:
                    info["prediction_state"] = post_state
            else:
                info["use_state_reward"] = False
                if "observation_content" in info and info["observation_content"]:
                    info["metrics"]["turn_metrics"]["grounding_reward"] = 0.0
                if "prediction_content" in info and info["prediction_content"]:
                    info["metrics"]["turn_metrics"]["worldmodeling_reward"] = 0.0
            return obs, reward, done, info
        else:
            return step_func(self, action_str)
    return wrapped_step

def service_state_reward_wrapper(step_batch_func):
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        if not self.config.get("use_state_reward", False):
            print("[DEBUG] verifier_rubrics wrapper closed")
            return step_batch_results
        print("[DEBUG] verifier_rubrics wrapper enabled")
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            env_name = self.env_configs[id].get("env_name", "default_env")
            if info.get("use_verifier_rubrics", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "grounding",
                        "env_name": env_name,
                    })
                if info.get("think_content", None) and info.get("prediction_state", None):
                    input_to_llm.append({
                        "id": id,
                        "content": info["think_content"],
                        "state": info["prediction_state"],
                        "type": "self_consistency",
                        "env_name": env_name,
                    })

        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            results = run_llm_judge(input_to_llm)
        else:
            return step_batch_results

        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}

        for item, result in zip(input_to_llm, results):
            id = item["id"]
            env_config = self.env_configs[id]
            score= result["score"]
            if item["type"] == "grounding":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "worldmodeling":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)

        return {id: tuple(result) for id, result in new_step_batch_results.items()}

    return wrapped_step_batch


def service_state_reward_wrapper_v2(step_batch_func):
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        if not self.config.get("use_state_reward", False):
            print("[DEUBG] State reward wrapper closed")
            return step_batch_results
        print("[DEUBG] State reward wrapper enabled")
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            env_name = self.env_configs[id].get("env_name", "default_env")
            if info.get("use_state_reward", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["observation_content"],state=info["observation_state"],type="grounding",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "grounding",
                        "env_name": env_name,
                        "prompt":prompt
                    })
                if info.get("prediction_content", None) and info.get("prediction_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["prediction_content"],state=info["prediction_state"],type="worldmodeling",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["prediction_content"],
                        "state": info["prediction_state"],
                        "type": "worldmodeling",
                        "env_name": env_name,
                        "prompt":prompt
                    })

        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            results = run_llm_judge_new(input_to_llm) # a dict containing a set of metrics
        else:
            return step_batch_results

        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}

        for item, result in zip(input_to_llm, results):
            id = item["id"]
            state=item["state"]
            env_config = self.env_configs[id]
            response= result["parsed_response"]
            kwargs={
                "response": response,
                "state": state,
            }
            score=self.calculate_visual_reasoning_reward(**kwargs)
            if item["type"] == "grounding":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "worldmodeling":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)

        return {id: tuple(result) for id, result in new_step_batch_results.items()}

    return wrapped_step_batch


def service_state_reward_wrapper_v3(step_batch_func):
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        if not self.config.get("use_state_reward", False):
            print("[DEUBG] State reward wrapper closed")
            return step_batch_results
        print("[DEUBG] State reward wrapper enabled")
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            env_name = self.env_configs[id].get("env_name", "default_env")
            if info.get("use_state_reward", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["observation_content"],state=info["observation_state"],type="grounding",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "grounding",
                        "env_name": env_name,
                        "prompt":prompt
                    })
                if info.get("prediction_content", None) and info.get("prediction_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["prediction_content"],state=info["prediction_state"],type="worldmodeling",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["prediction_content"],
                        "state": info["prediction_state"],
                        "type": "worldmodeling",
                        "env_name": env_name,
                        "prompt":prompt
                    })

        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            results = run_llm_judge_new(input_to_llm) # a dict containing a set of metrics
        else:
            return step_batch_results

        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}
        grounding_contents= []
        worldmodeling_contents = []
        for item in input_to_llm:
            if item["type"] == "grounding":
                grounding_contents.append(item["content"])
            elif item["type"] == "worldmodeling":
                worldmodeling_contents.append(item["content"])
        self.top_strings_tracker_grounding.add_strings(grounding_contents)
        self.top_strings_tracker_worldmodeling.add_strings(worldmodeling_contents)
        self.top_strings_tracker_grounding.trim_to_m()
        self.top_strings_tracker_worldmodeling.trim_to_m()
        for item, result in zip(input_to_llm, results):
            id = item["id"]
            state=item["state"]
            content= item["content"]
            r_type=item["type"]
            env_name = item["env_name"]
            prompt=item["prompt"]
            env_config = self.env_configs[id]
            response= result["parsed_response"]
            kwargs={
                "response": response,
                "state": state,
                "content": content,
                "r_type": r_type,
                "env_name": env_name,
                "prompt": prompt
            }
            score=self.calculate_visual_reasoning_reward(**kwargs)
            if item["type"] == "grounding":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "worldmodeling":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)

        return {id: tuple(result) for id, result in new_step_batch_results.items()}

    return wrapped_step_batch