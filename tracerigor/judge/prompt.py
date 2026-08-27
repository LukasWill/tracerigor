"""
Judge prompt builder.

Responsibilities:
  1. Maintain a registry of env-specific prompt templates
  2. Build OpenAI-compatible message lists from TurnJudgePackets
  3. Support both "universal" (all rubrics in one call) and "per_rubric" modes
  4. Handle multimodal (text + image) message assembly

Env-specific templates are registered via register_template().
The existing templates from verifier/prompt/ can be adapted to this
registry; this module provides a clean env-agnostic interface on top.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from tracerigor.judge.schema import TurnJudgePacket, JUDGE_OUTPUT_JSON_SCHEMA

# --- Reuse robust image utilities from the verifier pipeline ---
try:
    from tracerigor.verifier.utils.openai_mm_utils import (
        pil_to_data_url_cached,
        normalize_image_input,
        label_placeholders,
        prepend_or_append_header,
    )
    _HAS_VERIFIER_IMG_UTILS = True
except ImportError:
    _HAS_VERIFIER_IMG_UTILS = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# {task_name: {rubric_or_"universal": {"system": str, "user": str}}}
_TEMPLATE_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {}


def register_template(
    task_name: str,
    rubric: str,
    system_prompt: str,
    user_prompt: str,
) -> None:
    """Register a prompt template for (task_name, rubric)."""
    _TEMPLATE_REGISTRY.setdefault(task_name, {})[rubric] = {
        "system": system_prompt,
        "user": user_prompt,
    }


def get_template(task_name: str, rubric: str) -> Optional[Dict[str, str]]:
    """Look up a registered template."""
    return _TEMPLATE_REGISTRY.get(task_name, {}).get(rubric)


def available_templates() -> Dict[str, List[str]]:
    """Return {task_name: [rubric, …]} for all registered templates."""
    return {k: list(v.keys()) for k, v in _TEMPLATE_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Default (env-agnostic) universal prompt
# ---------------------------------------------------------------------------

_DEFAULT_UNIVERSAL_SYSTEM = """You are a strict trace verifier for a multi-turn agent. Judge only the CURRENT step, but use history to catch contradictions.

Your tasks:
1) **Observation Grounding** — Does the reasoning correctly reflect the CURRENT observation?
   Answer YES only if all observation-dependent claims are supported.

2) **Action Coherence** — Does the chosen action logically follow from the reasoning?
   Penalize only clear contradictions between stated intent and action taken.

3) **Temporal Consistency** — Does the reasoning update beliefs appropriately given recent history?
   Check for: beliefs not updated after feedback, stuck behavior, false claims about past actions.

Return a strict JSON object matching the schema below. Be concise and specific.
{json_schema_hint}
"""

_DEFAULT_UNIVERSAL_USER = """Step: {episode_step}

Available actions: {admissible_actions}

History (most recent first):
{history_str}

Current Observation:
{observation_content}

Reasoning (current step):
{reasoning_tokens}

Action taken (current step):
{action_tokens}
"""


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(
    packet: TurnJudgePacket,
    rubric: str = "universal",
    use_images: bool = True,
    json_schema_hint: str = "",
) -> List[Dict[str, Any]]:
    """
    Build an OpenAI-compatible message list from a TurnJudgePacket.

    If a task-specific template is registered for (packet.task_name, rubric),
    it is used. Otherwise falls back to the default universal template.

    Args:
        packet:            The turn-level evidence packet.
        rubric:            "universal" or a specific rubric name.
        use_images:        Whether to include images in the messages.
        json_schema_hint:  Schema snippet to embed in the system prompt.

    Returns:
        [{"role": "system", …}, {"role": "user", …}]
    """
    template = get_template(packet.task_name, rubric)

    if template:
        sys_text = template["system"]
        usr_text = template["user"]
    else:
        sys_text = _DEFAULT_UNIVERSAL_SYSTEM
        usr_text = _DEFAULT_UNIVERSAL_USER

    # Build render context
    history_str = _format_history(packet.history)
    observation_content = packet.current_observation_text or "<image>"

    # Build ground-truth section if available
    from tracerigor.judge.env_templates import build_ground_truth_section
    ground_truth_section = build_ground_truth_section(
        packet.task_name, packet.ground_truth_state
    )

    render_ctx = {
        "episode_step": packet.episode_step,
        "admissible_actions": ", ".join(packet.admissible_actions) if packet.admissible_actions else "N/A",
        "history_str": history_str,
        "observation_content": observation_content,
        "reasoning_tokens": packet.reasoning_tokens or "(none)",
        "action_tokens": packet.action_tokens or "(none)",
        "current_step": packet.episode_step,
        "json_schema_hint": json_schema_hint,
        # New fields from schema (filled by packet_builder per env)
        "task_description": packet.task_description or "N/A",
        "valid_actions": packet.valid_actions or "N/A",
        "ground_truth_section": ground_truth_section,
        # SciWorld-compatible aliases
        "reflection_tokens": packet.reasoning_tokens or "(none)",
        # Backward compat keys for existing verifier templates
        "_current_observation_text_or_image": observation_content,
        "_history_str": history_str,
        "current_observation_text": packet.current_observation_text,
    }

    sys_rendered = _safe_format(sys_text, render_ctx)
    usr_rendered = _safe_format(usr_text, render_ctx)

    # Assemble messages
    system_msg = {"role": "system", "content": sys_rendered}

    # User message: may contain images
    images = packet.current_observation_images if use_images else []
    if images:
        user_content = _build_multimodal_content(usr_rendered, images)
    else:
        user_content = usr_rendered

    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(history: List[Dict[str, Any]]) -> str:
    """Pretty-print a history list for inclusion in the prompt."""
    if not history:
        return "(no history)"
    lines = []
    for i, h in enumerate(history):
        obs = h.get("observation_text") or h.get("obs") or ""
        action = h.get("action_tokens") or h.get("reasoning_action_text") or ""
        reward = h.get("reward", "")
        lines.append(f"[t-{i+1}] obs: {obs[:200]}… | action: {action[:100]} | reward: {reward}")
    return "\n".join(lines)


def _safe_format(template: str, ctx: Dict[str, Any]) -> str:
    """Format a template string, ignoring missing keys."""
    try:
        return template.format(**ctx)
    except KeyError:
        # Partial format: only fill keys that exist
        import string
        formatter = string.Formatter()
        result = []
        for literal, field_name, fmt_spec, conversion in formatter.parse(template):
            result.append(literal)
            if field_name is not None:
                val = ctx.get(field_name)
                if val is not None:
                    result.append(str(val))
                else:
                    result.append("{" + field_name + "}")
        return "".join(result)


def _build_multimodal_content(
    text: str, images: List[Any]
) -> List[Dict[str, Any]]:
    """Build OpenAI vision-style multimodal content parts.

    Uses the cached image conversion from verifier when available
    (LRU cache, numpy support, multiple input formats).  Falls back to a
    minimal PIL-only path when the verifier package is not installed.
    """
    parts: List[Dict[str, Any]] = []

    # Label <image> placeholders: <image> → [Image 1], [Image 2], …
    if _HAS_VERIFIER_IMG_UTILS:
        text = label_placeholders(text, len(images))

    # Split text on <image> or [Image k] placeholders
    segments = re.split(r"<image>|\[Image \d+\]", text)
    for i, seg in enumerate(segments):
        if seg.strip():
            parts.append({"type": "text", "text": seg})
        if i < len(images):
            url = _image_to_data_url(images[i])
            parts.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "auto"},
            })

    # Append any extra images beyond placeholders
    for img in images[len(segments) - 1:]:
        url = _image_to_data_url(img)
        parts.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "auto"},
        })

    return parts if parts else [{"type": "text", "text": text}]


def _image_to_data_url(img) -> str:
    """Convert an image (PIL, numpy, path, bytes, dict) to a data URL string.

    Delegates to the verifier's robust normalize_image_input → pil_to_data_url_cached
    pipeline when available.  Falls back to minimal PIL-only conversion.
    """
    if _HAS_VERIFIER_IMG_UTILS:
        try:
            # normalize_image_input returns {"type": "image_url", "image_url": {"url": ...}}
            part = normalize_image_input(img)
            return part["image_url"]["url"]
        except Exception:
            pass

    # Minimal fallback: PIL images only
    return _pil_to_base64_url(img)


def _pil_to_base64_url(img) -> str:
    """Convert a PIL Image to a data:image/png;base64,… URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
