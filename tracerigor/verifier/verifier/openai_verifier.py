# verify_api.py
import os
import time
import threading
import uuid
from contextlib import contextmanager
import pathlib, math
from typing import Any, Dict, List, Optional, Sequence, Union, Type


import hydra
import wandb
import asyncio
from fastapi import FastAPI, HTTPException
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict

# from tracerigor.server.gpt_batch_request import run_gpt_request
from tracerigor.server.gpt_batch_request_fast import run_gpt_request_async
from tracerigor.verifier.prompt.sokoban import *
from tracerigor.verifier.prompt.sciworld import (
    SciWorldUniversalTemplate,
    SciWorldGroundingTemplate,
    SciWorldActionCoherenceTemplate,
    SciWorldTemporalConsistencyTemplate,
    SCIWORLD_TEMPLATES,
)
from tracerigor.verifier.prompt.navigation import (
    NavigationUniversalTemplateV2,
    NavigationGroundingTemplate,
    NavigationActionCoherenceTemplate,
    NavigationTemporalConsistencyTemplate,
    NAVIGATION_TEMPLATES,
)
from tracerigor.verifier.prompt.alfworld import (
    ALFWorldUniversalTemplate,
    ALFWorldGroundingTemplate,
    ALFWorldActionCoherenceTemplate,
    ALFWorldTemporalConsistencyTemplate,
    ALFWORLD_TEMPLATES,
)
from tracerigor.verifier.utils.parsers import _universal_score, _yesno_score, _selfconsistency_structured_score, _sciworld_universal_score
from tracerigor.verifier.utils.wandb_helper import _get_table_freq, _get_table_k, log_verifier_batch_to_wandb, log_verifier_examples_steprow
from tracerigor.verifier.verifier.verifier_base import BaseVerifier
from tracerigor.verifier.utils.openai_mm_utils import observation_to_openai_inputs, normalize_image_input, label_placeholders, prepend_or_append_header


import logging
log = logging.getLogger("x_reasoner.openai_verifier_api")
log.setLevel(logging.DEBUG)


def _coerce_image_sequence(
    value: Optional[Union[str, pathlib.Path, Dict[str, Any], bytes, Sequence[Any]]]
) -> Optional[List[Any]]:
    """Normalize a single image or a sequence of images into a list.

    Strings and paths are treated as scalar image inputs rather than iterated as
    character sequences.
    """
    if value is None:
        return None
    if isinstance(value, (str, pathlib.Path, bytes, bytearray, dict)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]

# -----------------------------------------------------------------------------
# Per-process globals for WandB + Hydra
# -----------------------------------------------------------------------------
_WANDB_INITIALIZED: Dict[int, bool] = {}
_GLOBAL_STEPS: Dict[int, int] = {}
_PROCESS_LOCKS: Dict[int, threading.Semaphore] = {}
_HYDRA_LOCKS: Dict[int, threading.Lock] = {}
_HYDRA_INITIALIZED: Dict[int, bool] = {}
_PID_CONFIG: Dict[int, DictConfig] = {}
_WANDB_TABLES: Dict[int, Dict[str, wandb.Table]] = {}

@contextmanager
def wandb_run_context():
    """Ensure WandB run is finished and flag reset per request."""
    try:
        yield
    finally:
        if wandb.run is not None:
            pid = os.getpid()
            _WANDB_TABLES.pop(pid, None)
            wandb.finish()
            # mark that next time we need to init again
            _WANDB_INITIALIZED[pid] = False

def _get_hydra_config(pid: int) -> DictConfig:
    # thread/process-safe Hydra initialization
    if pid not in _HYDRA_LOCKS:
        _HYDRA_LOCKS[pid] = threading.Lock()
    with _HYDRA_LOCKS[pid]:
        if not _HYDRA_INITIALIZED.get(pid, False):
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            two_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_dir = os.path.join(two_level_up, "config")
            # use initialize_config_dir for absolute path
            with hydra.initialize_config_dir(config_dir=config_dir, job_name="openai_verifier", version_base=None):
                cfg = hydra.compose(config_name="verifier_rubrics")
            _HYDRA_INITIALIZED[pid] = True
        if pid not in _PID_CONFIG:
            _PID_CONFIG[pid] = cfg
        return _PID_CONFIG[pid]

# -----------------------------------------------------------------------------
# OpenAI verifier class and specific rubrics
# -----------------------------------------------------------------------------
class OpenAIBaseVerifier(BaseVerifier):
    """
    Base class for verifiers that use OpenAI API.
    """
    def __init__(self, config: DictConfig, model_name: str, client: Optional[Any] = None, model_params: Dict[str, Any] = None):
        super().__init__(config, model_name, client, model_params)


    def assemble_messages(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Assemble provider-specific multimodal messages if present.

        Rationale:
        - Keeps VerifierTemplate provider-agnostic (just returns plain system/user text).
        - Centralizes OpenAI-specific content formatting (text+image parts) here.
        - Allows future override (e.g., subclass for another provider) without changing templates.
        """
        rubric = getattr(self, "rubric", None)
        templates = getattr(self, "_templates", None)
        if not rubric or not templates or rubric not in templates:
            raise ValueError(f"Unknown or unset rubric '{rubric}'. Subclass must define 'rubric' and '_templates'.")
        tmpl_cls = templates[rubric]
        tmpl_obj: VerifierTemplate = tmpl_cls()

        messages = tmpl_obj.build_messages(dict(item))  # returns [{'role':'system','content':...}, {'role':'user','content':...}]
        assert len(messages) >= 2 and messages[0]['role'] == 'system' and messages[1]['role'] == 'user', "Template must return system+user messages"
        system_text = messages[0]['content']
        user_text   = messages[1]['content']

        images = None
        # NEW: prefer explicit images if provided
        if item.get("current_observation_image"):
            images = _coerce_image_sequence(item["current_observation_image"])
        # OLD path: legacy 'obs' structure (kept for back-compat)
        elif item.get("obs") is not None:
            o = observation_to_openai_inputs(item["obs"], item_idx=0)
            images = _coerce_image_sequence(o.get("images"))

        # If no images: call standard chat.completions with text-only messages
        if not images:
            return messages

        # Optional: inspect tokens before sending
        # print(self.estimate_tokens(messages))

        return self.assemble_mm_messages(
            system_text=system_text,
            user_text=user_text,
            images=images,
        )

    # # If you want to replace <image> → [Image k]
    # def label_placeholders(user_text, num_images):
    #     k = 1
    #     while "<image>" in user_text and k <= num_images:
    #         user_text = user_text.replace("<image>", f"[Image {k}]", 1)
    #         k += 1
    #     return user_text

    @staticmethod
    def assemble_mm_messages(
        *,
        system_text: str,
        user_text: str,
        images: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any], bytes]]] = None,
        attach_image_refs_header: bool = True,
        # new flexible controls
        placeholder_strategy: str = "label",   # 'strip' | 'label' | 'keep'
        image_layout: str = "append",          # 'append' | 'interleave'
        header_position: str = "top",          # 'top' | 'bottom'
        label_format: str = "[Image {k}]",
        inline_label_in_interleave: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build OpenAI chat messages (text + optional images).
        If images are provided, they are appended to the user content as additional parts.

        placeholder_strategy:
        - 'strip': remove '<image>' tokens (default legacy behavior)
        - 'label': replace '<image>' -> '[Image k]' (numbered)
        - 'keep' : keep literal '<image>' in text
        image_layout:
        - 'append': single text part + all images appended as image parts
        - 'interleave': split on '<image>', emit text part, then image part, repeat
                        (optionally insert inline labels at each split)
        """
        images = list(images) if images else []
        n_img = len(images)

        # 1) Apply placeholder strategy to user_text
        utxt = user_text
        if placeholder_strategy == "strip":
            utxt = utxt.replace("<image>", "").strip()
        elif placeholder_strategy == "label":
            utxt = label_placeholders(utxt, n_img, label_format=label_format)
        elif placeholder_strategy == "keep":
            pass
        else:
            raise ValueError("placeholder_strategy must be one of: 'strip','label','keep'")

        # 2) Attach optional header
        if n_img and attach_image_refs_header:
            utxt = prepend_or_append_header(utxt, n_img, position=header_position)

        # 3) Build content with chosen layout (chat-style parts first)
        content: List[Dict[str, Any]] = []

        if n_img == 0 or image_layout == "append":
            # Single text part + all image parts
            content.append({"type": "text", "text": utxt})
            for img in images:
                content.append(normalize_image_input(img))

        elif image_layout == "interleave":
            # Split user_text on '<image>' occurrences. If none present, fall back to append.
            if "<image>" not in user_text:
                content.append({"type": "text", "text": utxt})
                for img in images:
                    content.append(normalize_image_input(img))
            else:
                chunks = user_text.split("<image>")
                img_idx = 0
                for i, chunk in enumerate(chunks):
                    # Base text for this chunk
                    text_chunk = chunk
                    # Optionally add inline label where the placeholder was
                    if i < len(chunks) - 1 and inline_label_in_interleave:
                        label = label_format.format(k=img_idx + 1) if img_idx < n_img else "[Image ?]"
                        text_chunk = (text_chunk + ("\n" if text_chunk and not text_chunk.endswith("\\n") else "") + label)
                    # Apply header to the *first* chunk only (to avoid repetition)
                    if i == 0 and n_img and attach_image_refs_header:
                        text_chunk = prepend_or_append_header(text_chunk, n_img, position=header_position)
                    content.append({"type": "text", "text": text_chunk})
                    # Insert the image part after this split if available
                    if i < len(chunks) - 1 and img_idx < n_img:
                        content.append(normalize_image_input(images[img_idx]))
                        img_idx += 1
                # If extra images remain, append them (rare)
                while img_idx < n_img:
                    content.append(normalize_image_input(images[img_idx]))
                    img_idx += 1
        else:
            raise ValueError("image_layout must be one of: 'append','interleave'")
        # TODO: adds a for_responses_api toggle
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def messages_to_responses_input(messages, guard_sentence: bool = False):
        """Map Chat-style messages (from assemble_mm_messages) to Responses input."""
        sys_txt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        user    = next((m for m in messages if m.get("role") == "user"), {"content": ""})
        ucont   = user.get("content", "")

        parts = []
        if isinstance(ucont, list):
            for p in ucont:
                ptype = p.get("type")
                if ptype == "text":
                    txt = p.get("text") or ""
                    if txt:
                        parts.append({"type": "input_text", "text": txt})
                elif ptype == "image_url":
                    iu = p.get("image_url")
                    if isinstance(iu, dict):  # handle {"image_url":{"url": ...}}
                        iu = iu.get("url")
                    if iu:
                        parts.append({"type": "input_image", "image_url": iu})
                # MAYBE WRONG
                elif ptype == "image" and "data" in p:
                    parts.append({"type": "input_image", "image_data": p["data"]})
        else:
            parts.append({"type": "input_text", "text": str(ucont)})

        instr = sys_txt
        if guard_sentence:
            instr = (instr + " Do not output any text before <think> or after </answer>.").strip()

        input_payload = [{"role": "user", "content": parts}]
        return instr, input_payload

    async def evaluate_batch(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompts = [self.get_template(item) for item in input_data]
        metadata = input_data

        # llm_responses = run_gpt_request(prompts, self.config.api)
        ## used when FLASK or other sync code calls async verifier
        # llm_responses = await asyncio.to_thread(run_gpt_request, prompts, self.config.api)
        llm_responses = await run_gpt_request_async(prompts, self.config.api)

        results = []
        for md, resp in zip(metadata, llm_responses):
            text = resp["response"] or ""
            query_success = resp["success"]
            if not query_success or not text.strip():
                score, parse_success, extra = 0.0, False, None
                error = resp.get("error", "empty response")
            else:
                # choose parser
                # parsed = (
                #     self.parse_response(text, metadata_item=md)
                #     if self.rubric == "simulatability"
                #     else self.parse_response(text)
                # )
                # score, parse_success, *maybe_ratio = parsed
                # extra = maybe_ratio[0] if maybe_ratio else None
                # error = None
                if self.rubric == "simulatability":
                    # (score, parse_success, ratio)
                    score, parse_success, extra = self.parse_response(text, metadata_item=md)
                else:
                    parsed = self.parse_response(text)
                    # allow either (score, parse_success) or (score, parse_success, extra)
                    if isinstance(parsed, tuple) and len(parsed) == 3:
                        score, parse_success, extra = parsed
                    else:
                        score, parse_success = parsed
                        extra = None
                error = None

            # Keep only lightweight metadata; drop bulky/non-serializable
            # fields (PIL Images in current_observation_image, history, etc.)
            _BULKY_KEYS = {
                "current_observation_image", "history", "obs",
            }
            entry = {
                **md,  # considering omitting bulky metadata by # **{k: v for k, v in md.items() if k not in _BULKY_KEYS},
                "prompt":            None,       # omit bulky prompt
                "response":          text,
                "query_success":     query_success,
                "score":             score,
                "parse_success":     parse_success,
                "error":             error,
            }
            # if extra is not None:
            #     entry['ratio'] = extra
            # handle extra payloads in a rubric-specific way
            if extra is not None:
                if self.rubric == "simulatability":
                    # keep existing behavior: store as 'ratio'
                    entry["ratio"] = extra
                elif self.rubric == "self_consistency":
                    # extra is the structured JSON dict from SelfConsistencySubtagParser
                    if isinstance(extra, dict):
                        scalars = extra.get("scalar_scores") or extra.get("score") or {}
                        # flatten useful sub-scores into top-level so W&B code can see them
                        for k in ("observation", "reasoning", "prediction", "aggregate"):
                            if k in scalars:
                                entry[f"sub_{k}_score"] = scalars[k]
                else:
                    # fallback: stash as generic 'extra'
                    entry["extra"] = extra

            for k in ("verdict", "p_yes", "p_no", "binary_entropy", "topk_entropy_at_label", "verdict_logprob", "verdict_prob_full", "logprobs"):
                if k in resp:
                    entry[k] = resp[k]

            results.append(entry)

        return results

# class FaithfulnessVerifier(BaseVerifier):
#     system_prompt = "You are an expert evaluator. Think step by step, then <think>…</think><answer>YES/NO</answer>."
#     rubric = "faithfulness"
#     def get_template(self, item):
#         tmpl = self.config.prompt_templates.default_env.faithfulness
#         return tmpl.format(**item)

# class SimulatabilityVerifier(BaseVerifier):
#     system_prompt = "You are an expert evaluator. Provide <predicted_action>…</predicted_action><implicit>YES/NO</implicit>."
#     rubric = "simulatability"
#     def get_template(self, item):
#         tmpl = self.config.prompt_templates.default_env.simulatability
#         return tmpl.format(**item)
#     def parse_response(self, text, metadata_item):
#         # Print out the raw text and ground truth so we can see failures
#         log.debug(f"\n[SimulatabilityVerifier] raw response text:\n{text}\n")
#         log.debug(f"[SimulatabilityVerifier] ground-truth action_tokens: {metadata_item.get('action_tokens')}\n")

#         # Extract predicted actions and implicit flag
#         m_pred = re.search(r'<predicted_action>(.*?)</predicted_action>', text, re.IGNORECASE)
#         if not m_pred:
#             log.debug(f"[SimulatabilityVerifier] missing <predicted_action> in text:\n{text}")

#         m_imp  = re.search(r'<implicit>(YES|NO)</implicit>', text, re.IGNORECASE)
#         if not m_imp:
#             log.debug(f"[SimulatabilityVerifier] missing <implicit> in text:\n{text}")
#         predicted = [a.strip() for a in m_pred.group(1).split(',')] if m_pred else []
#         implic   = m_imp.group(1).upper() if m_imp else "NO"

#         log.debug(f"[SimulatabilityVerifier] parsed predicted={predicted}, implicit={implic}")

#         # Ground truth actions from metadata_item["action_tokens"]
#         gt = re.search(r'<answer>(.*?)</answer>', metadata_item.get("action_tokens",""), re.IGNORECASE)
#         if not gt:
#             log.debug(f"[SimulatabilityVerifier] missing ground-truth <answer> in action_tokens:\n{metadata_item.get('action_tokens')}")
#         actual = [a.strip() for a in gt.group(1).split(',')] if gt else []

#         # Compute ordered match ratio (prefix-wise):
#         match_count = sum(1 for (p, a) in zip(predicted, actual) if p == a)
#         ratio = match_count / len(actual) if actual else 0.0

#         log.debug(f"[SimulatabilityVerifier] match_count={match_count}, ratio={ratio}\n")

#         # Score based on implicit yes/no
#         score = 1.0 if implic == "YES" else 0.0
#         parse_success = bool(m_pred and m_imp and gt)
#         return score, parse_success, ratio

# class LogicalnessVerifier(BaseVerifier):
#     system_prompt = "You are an expert evaluator. Think step by step, then <think>…</think><answer>YES/NO</answer>."
#     rubric = "logicalness"
#     def get_template(self, item):
#         tmpl = self.config.prompt_templates.default_env.logicalness
#         return tmpl.format(**item)


def build_sokoban_verifier(rubric: str) -> Type[OpenAIBaseVerifier]:
    """
    Minimal redesign:
    - Keep template classes (backward compatible).
    - Move the orchestration of message construction into the Verifier so we can
      enrich / restructure prompts (e.g. attach images or future multimodal parts).
    - If later an item includes e.g. item["images"] / item["image_placeholders"],
      we can splice them in here without touching every template class.
    - For now, if the template already has build_messages(), we delegate to it.
      Otherwise we try to build from conventional attributes (system_prompt, user_template).
    - get_template() still returns what run_gpt_request expects (a messages list),
      so no wider code changes required.
    """
    class SokobanVerifier(OpenAIBaseVerifier):
        def __init__(self, config, model_name, model_params):
            super().__init__(config, model_name, client=None, model_params=model_params)
            self.rubric = rubric
            self._templates = {
                "universal": SokobanUniversalTemplateV2,     # NEW: aligned 3-rubric schema
                "universal_v2": SokobanUniversalTemplateV2,
                "universal_v2_temporalfix": SokobanUniversalTemplateV2TemporalFix,  # temporal loophole closed
                "universal_legacy": SokobanUniversalTemplate,
                "grounding": SokobanGroundingTemplate,
                "self_consistency": SokobanActionReasoningConsistencyTemplate,
                "history_consistency": SokobanHistoryConsistencyTemplate,
            }

        def get_template(self, item: Dict[str, Any]) -> str:
            # Delegates to the base-class assemble_messages()
            return self.assemble_messages(item)

        def parse_response(self, text: str):
            if self.rubric in ("universal", "universal_v2", "universal_v2_temporalfix"):
                # V2 (and the temporal-fix variant) use the same 3-rubric JSON schema as SciWorld
                return _sciworld_universal_score(text)
            if self.rubric == "universal_legacy":
                return _universal_score(text)
            if self.rubric == "self_consistency":
                # returns (score, parse_success, extra_dict)
                return _selfconsistency_structured_score(text)
            return _yesno_score(text)

    return SokobanVerifier


def build_sciworld_verifier(rubric: str) -> Type[OpenAIBaseVerifier]:
    """
    Build a SciWorld verifier class for the given rubric.

    SciWorld rubrics:
    - universal: All 3 metrics (grounding, action coherence, temporal consistency)
    - grounding: Binary observation grounding check
    - action_coherence: Binary action-reflection consistency
    - temporal_consistency: Binary history consistency
    """
    class SciWorldVerifier(OpenAIBaseVerifier):
        def __init__(self, config, model_name, model_params):
            super().__init__(config, model_name, client=None, model_params=model_params)
            self.rubric = rubric
            self._templates = {
                "universal": SciWorldUniversalTemplate,
                "grounding": SciWorldGroundingTemplate,
                "action_coherence": SciWorldActionCoherenceTemplate,
                "temporal_consistency": SciWorldTemporalConsistencyTemplate,
            }

        def get_template(self, item: Dict[str, Any]) -> str:
            # Delegates to the base-class assemble_messages()
            return self.assemble_messages(item)

        def parse_response(self, text: str):
            if self.rubric == "universal":
                return _sciworld_universal_score(text)
            return _yesno_score(text)

    return SciWorldVerifier


def build_navigation_verifier(rubric: str) -> Type[OpenAIBaseVerifier]:
    """Build a Navigation verifier class for the given rubric.

    Navigation rubrics (see tracerigor/verifier/prompt/navigation.py):
    - universal / universal_v2: All 3 rubrics in one call (3-rubric JSON schema,
      parsed by `_sciworld_universal_score`).
    - grounding              : Binary observation grounding (image-first).
    - action_coherence       : Binary action-reflection coherence.
    - temporal_consistency   : Binary history consistency.
    - self_consistency       : Alias for action_coherence (back-compat).
    - history_consistency    : Alias for temporal_consistency (back-compat).
    """
    class NavigationVerifier(OpenAIBaseVerifier):
        def __init__(self, config, model_name, model_params):
            super().__init__(config, model_name, client=None, model_params=model_params)
            self.rubric = rubric
            self._templates = {
                "universal": NavigationUniversalTemplateV2,
                "universal_v2": NavigationUniversalTemplateV2,
                "grounding": NavigationGroundingTemplate,
                "action_coherence": NavigationActionCoherenceTemplate,
                "temporal_consistency": NavigationTemporalConsistencyTemplate,
                # Back-compat aliases (mirror what get_navigation_verifier_templates exposes).
                "self_consistency": NavigationActionCoherenceTemplate,
                "history_consistency": NavigationTemporalConsistencyTemplate,
            }

        def get_template(self, item: Dict[str, Any]) -> str:
            return self.assemble_messages(item)

        def parse_response(self, text: str):
            if self.rubric in ("universal", "universal_v2"):
                return _sciworld_universal_score(text)
            return _yesno_score(text)

    return NavigationVerifier


def build_alfworld_verifier(rubric: str) -> Type[OpenAIBaseVerifier]:
    """Build an ALFWorld verifier class for the given rubric.

    ALFWorld rubrics (see tracerigor/verifier/prompt/alfworld.py):
    - universal              : All 3 rubrics in one call (3-rubric JSON schema,
                               parsed by `_sciworld_universal_score`).
    - grounding              : Binary observation grounding (text-only).
    - action_coherence       : Binary action-reflection coherence.
    - temporal_consistency   : Binary history consistency.
    """
    class ALFWorldVerifier(OpenAIBaseVerifier):
        def __init__(self, config, model_name, model_params):
            super().__init__(config, model_name, client=None, model_params=model_params)
            self.rubric = rubric
            self._templates = {
                "universal": ALFWorldUniversalTemplate,
                "grounding": ALFWorldGroundingTemplate,
                "action_coherence": ALFWorldActionCoherenceTemplate,
                "temporal_consistency": ALFWorldTemporalConsistencyTemplate,
            }

        def get_template(self, item: Dict[str, Any]) -> str:
            return self.assemble_messages(item)

        def parse_response(self, text: str):
            if self.rubric == "universal":
                return _sciworld_universal_score(text)
            return _yesno_score(text)

    return ALFWorldVerifier


def build_verifier(env: str, rubric: str) -> Type[OpenAIBaseVerifier]:
    """
    Factory function to build environment-specific verifier.

    Args:
        env: Environment name ('sokoban', 'sciworld', 'navigation', 'alfworld', etc.)
        rubric: Rubric name for evaluation

    Returns:
        Verifier class for the given environment and rubric
    """
    builders = {
        "sokoban": build_sokoban_verifier,
        "sciworld": build_sciworld_verifier,
        "navigation": build_navigation_verifier,
        "alfworld": build_alfworld_verifier,
    }
    if env not in builders:
        raise ValueError(f"Unknown environment: {env}. Available: {list(builders.keys())}")
    return builders[env](rubric)


# -----------------------------------------------------------------------------
# Async run & WandB logging
# -----------------------------------------------------------------------------
async def run_openai_verifier(
    input_data: List[Dict[str, Any]],
    rubric: str,
    model_name: str,
    model_params: Dict[str, Any],
    env: str = "sokoban",  # Default to sokoban for backward compatibility
) -> List[Dict[str, Any]]:
    pid = os.getpid()
    with _PROCESS_LOCKS.setdefault(pid, threading.Semaphore(1)):
        # step & config
        step = _GLOBAL_STEPS.setdefault(pid, -1) + 1
        _GLOBAL_STEPS[pid] = step
        config = _get_hydra_config(pid)
        config.api.name = model_name
        # init WandB
        if not _WANDB_INITIALIZED.get(pid, False):
            run_id = uuid.uuid4().hex[:8]
            wandb.init(
                project=config.wandb.project,
                name=f"{config.wandb.run_name}_{env}_{rubric}_{run_id}",
                config=OmegaConf.to_container(config, resolve=True)
            )
            _WANDB_INITIALIZED[pid] = True

        # pick verifier based on environment
        verifier_cls = build_verifier(env, rubric)
        verifier: OpenAIBaseVerifier = verifier_cls(config, model_name, model_params)
        start = time.time()
        results = await verifier.evaluate_batch(input_data)
        for r in results:
            r["model"] = model_name
        duration = time.time() - start

        # # log scalars
        # num_queried = len(results)
        # num_query_success = sum(r["query_success"] for r in results)
        # query_succ  = num_query_success / num_queried if num_queried else 0.0
        # avg_verifier_score = sum(r["score"] for r in results) / num_queried if num_queried else 0.0
        # parse_succ = sum(r["parse_success"] for r in results) / num_queried if num_queried else 0.0
        # parse_succ_rate_on_query_successes = sum(r["parse_success"] for r in results) / num_query_success if num_query_success else 0.0
        # if rubric == "simulatability":
        #     # only consider entries where ratio is not None
        #     simula_ratios = [r["ratio"] for r in results if r.get("ratio") is not None]
        #     avg_simula_ratio = sum(simula_ratios) / len(simula_ratios) if simula_ratios else 0.0
        # wandb.log({
        #     "step": step,
        #     "duration": duration,
        #     "num_queried": num_queried,
        #     "query_success_rate": query_succ,
        #     "avg_verifier_score": avg_verifier_score,
        #     "parse_succ_rate_all": parse_succ,
        #     "parse_succ_rate_on_successes": parse_succ_rate_on_query_successes,
        #     **({"avg_simula_ratio": avg_simula_ratio} if rubric == "simulatability" else {}),
        # }, step=step)

        # cols = [
        #     'step', 'sample_id', 'reasoning_trace', 'chosen_actions', 'raw_response', 'verifier_score'
        # ]
        # if rubric == "simulatability":
        #     # insert 'simula_ratio' right after 'verifier_score'
        #     cols.append('simula_ratio')
        # cols += [
        #     'parse_ok', 'model_name'
        # ]
        # table = wandb.Table(columns=cols)

        # for r in results:
        #     row = [step, r['id'], r['reasoning_tokens'], r['action_tokens'],
        #         r['response'], r['score']
        #     ]
        #     if rubric == "simulatability":
        #         row.append(r.get('ratio', None))
        #     row += [
        #         r['parse_success'], r['model']
        #     ]
        #     table.add_data(*row)

        # wandb.log({'responses': table}, step=step)

        # wandb already initialized above; function can import wandb internally
        # existing scalar + per-sample table
        log_verifier_batch_to_wandb(
            results=results,
            rubric=rubric,
            step=step,
            duration=duration,
        )

        # step-row tables every N steps (default 10), with K samples per bucket (default 8)
        try:
            freq = _get_table_freq(config, default=10)
            k    = _get_table_k(config, default=8)
            if step % freq == 0:
                log_verifier_examples_steprow(
                    results=results,
                    rubric=rubric,
                    step=step,
                    model_name=model_name,
                    samples_per_bucket=k,
                )
        except Exception as e:
            # don't let logging issues break training
            print(f"[warn] verifier step-row logging skipped: {e}")

        return results


# -----------------------------------------------------------------------------
# FastAPI endpoint
# -----------------------------------------------------------------------------

# Minimal history unit used by universal/history verifiers
class HistoryItem(BaseModel):
    observation_text: Optional[str] = None
    # If this history step was image-based, set True; upstream can render "<image>"
    observation_image: Optional[bool] = False
    images: Optional[List[Any]] = Field(default=None, exclude=True)
    reasoning_tokens: Optional[str] = None
    action_tokens: Optional[str] = None

    # model_config = ConfigDict(arbitrary_types_allowed=True)

class BatchItem(BaseModel):
    id: str
    reasoning_tokens: str
    action_tokens: str

    # Keep but normalize to lowercase for consistency with prompts
    admissible_actions: List[str] = Field(
        default_factory=lambda: ["up", "down", "left", "right"]
    )

    # Step index used by universal/history prompts.
    current_step: Optional[int] = Field(default=None, description="Preferred.")

    # Preferred structured current observation
    current_observation_text: Optional[str] = None
    current_observation_image: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any], bytes]]] = None

    # Back-compat: if provided, overrides the computed value used in the prompt
    _current_observation_text_or_image: Optional[str] = ""

    # Optional short history (last 2–3 usually enough)
    history: List[HistoryItem] = Field(default_factory=list)

    # @field_validator("current_observation_text", mode="before")
    # @classmethod
    # def _clean_text(cls, v):
    #     if v is None: return None
    #     if isinstance(v, float) and math.isnan(v): return None
    #     s = str(v).strip()
    #     return s or None

    @field_validator("current_observation_image", mode="before")
    @classmethod
    def _coerce_images(cls, value):
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (str, pathlib.Path, bytes, bytearray, dict)):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        raise TypeError(
            "current_observation_image must be a path/url/bytes/dict or list thereof"
        )

class BatchRequest(BaseModel):
    items: List[BatchItem]
    rubric: str  # 'faithfulness'|'simulatability'|'logicalness'
    models: List[str] = ["gpt-5-nano-2025-08-07"]  # gpt-5-nano-2025-08-07, gpt-4.1-nano-2025-04-14
    model_params: Optional[Dict[str, Any]] = {}  # e.g. {"temperature":0.2, "top_p":0.8}

app = FastAPI(title="LLM-OpenAI-Verifier (multi-rubrics)",
            #   description="Batch verification of LLM reasoning traces using multiple rubrics.",
            #   version="0.1.0",
            #   contact={
            #       "name": "",
            #       "email": ""
            #   }
              )

@app.post("/batch_verify")
async def batch_verify(req: BatchRequest):
    try:
        data = [i.model_dump() for i in req.items]

        # launch all model‐runs in parallel
        async def single_model_run(model: str):
            with wandb_run_context():
                res = await run_openai_verifier(data, req.rubric, model, req.model_params or {})
            return res

        batches = await asyncio.gather(
            *[single_model_run(m) for m in req.models],
            return_exceptions=False
        )
        # flatten list of lists
        results = [r for batch in batches for r in batch]
        return {"results": results}
    except Exception as e:
        # print full traceback to the server console
        import traceback; traceback.print_exc()
        # return the exception type and message in the 500 response
        detail = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=detail)