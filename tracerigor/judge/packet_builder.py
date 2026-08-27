"""
Packet builder — bridges rollout manager data structures to TurnJudgePackets.

This module knows how to extract the right fields from the TraceRigor rollout
recorder and env_states to build judge packets. It is the only place that
couples the judge stack to TraceRigor's specific data layout.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tracerigor.judge.schema import TurnJudgePacket


def build_turn_packet(
    env_id: str,
    task_name: str,
    episode_step: int,
    obs_text: str,
    obs_images: List[Any],
    history: List[Dict[str, Any]],
    raw_trace: str,
    info: Dict[str, Any],
    admissible_actions: Optional[List[str]] = None,
) -> TurnJudgePacket:
    """
    Build a TurnJudgePacket from rollout manager data.

    Args:
        env_id:             Environment identifier.
        task_name:          Task name (e.g. "sokoban", "sciworld").
        episode_step:       Current episode step (1-indexed).
        obs_text:           Observation text from recorder entry.
        obs_images:         Observation images from recorder entry.
        history:            Recent history entries from recorder.
        raw_trace:          Full decoded LLM response string.
        info:               Step info dict from env.
        admissible_actions: List of valid actions for this env.

    Returns:
        A populated TurnJudgePacket.
    """
    # Extract reasoning and action tokens from the raw trace
    # Format-aware extraction: different envs use different tag schemes.
    #   - Sokoban: <think>...</think><action>...</action>
    #   - SciWorld (ReflAct): <reflection>...</reflection><action>...</action>
    reasoning_tokens = (
        _extract_tag(raw_trace, "think")
        or _extract_tag(raw_trace, "reflection")
        or ""
    )
    action_tokens = (
        _extract_tag(raw_trace, "action")
        or _extract_tag(raw_trace, "answer")  # legacy fallback
        or ""
    )

    # Wrap back in canonical <action> tags for the judge
    if reasoning_tokens:
        if _extract_tag(raw_trace, "think") is not None:
            reasoning_tokens = f"<think>{reasoning_tokens}</think>"
        elif _extract_tag(raw_trace, "reflection") is not None:
            reasoning_tokens = f"<reflection>{reasoning_tokens}</reflection>"
    if action_tokens:
        action_tokens = f"<action>{action_tokens}</action>"

    chosen_action = info.get("action_content") or _parse_action(action_tokens)

    # Determine modality
    agent_modality = "vision" if obs_images else "text"

    # Format history for the judge
    formatted_history = _format_recorder_history(history)

    return TurnJudgePacket(
        env_id=env_id,
        episode_step=episode_step,
        task_name=task_name,
        agent_modality=agent_modality,
        current_observation_text=obs_text,
        current_observation_images=obs_images or [],
        history=formatted_history,
        raw_trace=raw_trace,
        reasoning_tokens=reasoning_tokens,
        action_tokens=action_tokens,
        chosen_action=chosen_action,
        available_actions=info.get("available_actions", []),
        admissible_actions=admissible_actions or [],
        # New fields populated from env info
        format_correct=info.get("format_correct", True),
        action_feedback=info.get("action_feedback", ""),
        post_action_observation=info.get("post_action_observation", ""),
        ground_truth_state=info.get("ground_truth_state", {}),
        task_description=info.get("task_description", ""),
        valid_actions=info.get("valid_actions", ""),
    )


def build_packets_from_step_results(
    step_results: Dict[str, Any],
    responses_str: Dict[str, str],
    recorder: Dict[str, List[Dict[str, Any]]],
    env_states: Dict[str, Dict[str, Any]],
    env_configs: Dict[str, Any],
    history_k: int = 3,
    default_admissible: Optional[List[str]] = None,
) -> List[TurnJudgePacket]:
    """
    Build judge packets for a full batch of step results.

    This is the high-level helper called from the rollout manager after
    env_client.step_batch() returns.

    Args:
        step_results:      {env_id: (obs, reward, done, info)}.
        responses_str:     {env_id: decoded_response_string}.
        recorder:          {env_id: [step_record, ...]}.
        env_states:        {env_id: {"step": int, ...}}.
        env_configs:       {env_id: config_obj} for extracting task_name.
        history_k:         Number of recent history entries to include.
        default_admissible: Fallback admissible actions.

    Returns:
        List of TurnJudgePackets (one per env_id).
    """
    packets = []

    for env_id, (obs, reward, done, info) in step_results.items():
        raw_trace = responses_str.get(env_id, "")

        # Get recorder entries: current turn is the last entry (pre-action obs)
        records = recorder.get(env_id, [])

        # The judge for turn t should see the observation available *before* the action.
        # After step_batch, recorder[-1] is the *post-step* recording (with the new obs).
        # So we want recorder[-2] for the pre-action observation.
        if len(records) >= 2:
            pre_action_record = records[-2]
        elif len(records) >= 1:
            pre_action_record = records[-1]
        else:
            pre_action_record = {}

        obs_text = pre_action_record.get("obs_str", "")
        obs_images = pre_action_record.get("image_data", [])

        # History: entries before the current one
        history_start = max(0, len(records) - 1 - history_k)
        history_end = max(0, len(records) - 1)
        history_entries = records[history_start:history_end]

        # Task name from config
        config = env_configs.get(env_id)
        task_name = _get_task_name(config)

        # Admissible actions
        admissible = info.get("available_actions") or default_admissible or []

        episode_step = env_states.get(env_id, {}).get("step", 1)

        packet = build_turn_packet(
            env_id=env_id,
            task_name=task_name,
            episode_step=episode_step,
            obs_text=obs_text,
            obs_images=obs_images,
            history=history_entries,
            raw_trace=raw_trace,
            info=info,
            admissible_actions=admissible,
        )
        packets.append(packet)

    return packets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from <tag>…</tag>."""
    if not text:
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    return m.group(1).strip() if m else None


def _parse_action(action_str: str) -> str:
    """Clean up an action string (strip tags, whitespace, etc.)."""
    if not action_str:
        return ""
    # Remove any XML tags
    clean = re.sub(r"<[^>]+>", "", action_str).strip()
    # Take first line / first comma-separated value
    parts = [p.strip() for p in clean.split(",")]
    return parts[0] if parts else clean


def _format_recorder_history(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format recorder entries into the judge history format."""
    formatted = []
    for r in records:
        entry = {
            "obs": r.get("obs_str", ""),
            "reasoning_action_text": r.get("info", {}).get("llm_raw_response", ""),
            "reward": r.get("reward", 0.0),
            "done": r.get("done", False),
        }
        # Also include parsed tokens if available
        info = r.get("info", {})
        if info.get("think_content"):
            entry["action_tokens"] = f"<action>{info.get('action_content', '')}</action>"
        formatted.append(entry)
    return formatted


def _get_task_name(config) -> str:
    """Extract task name from an env config object."""
    if config is None:
        return "unknown"
    # Try common attribute patterns used in TraceRigor
    for attr in ("task_name", "env_name", "name"):
        val = getattr(config, attr, None)
        if val:
            return str(val)
    # Try to infer from class name
    cls_name = type(config).__name__.lower()
    for task in ("sokoban", "sciworld", "alfworld", "navigation", "frozenlake", "babyai"):
        if task in cls_name:
            return task
    return "unknown"
