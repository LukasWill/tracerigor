"""
Layer-0 deterministic heuristic checks.

Aligned with env-level ViolationTracker patterns (sciworld.ViolationType, sokoban).
Each check mirrors a known agent failure mode:

  FORMAT     — Response cannot be parsed (missing tags, empty action)
  INVALID    — Action not in the admissible/valid set, or env rejected it
  REPETITION — Same action + unchanged observation (stuck loop)

These run before any LLM judge call and can short-circuit obvious failures,
saving judge latency and cost.

Design notes:
  - These checks fire per-turn (unlike ViolationTracker which has consecutive counters).
  - The ViolationTracker in each env handles multi-turn thresholds for early termination.
  - Heuristics here serve a different role: pre-filtering individual turns for the
    LLM judge to avoid wasting judge calls on mechanically-detectable failures.
  - Additionally, heuristic results are injected as ground-truth metadata into the
    judge prompt to reduce ambiguity for the LLM judge.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from tracerigor.judge.config import HeuristicConfig
from tracerigor.judge.schema import TurnJudgePacket


def run_heuristics(
    packet: TurnJudgePacket,
    cfg: HeuristicConfig,
) -> Dict[str, Any]:
    """
    Run all Layer-0 checks on a single turn packet.

    Returns:
        {
            "should_short_circuit": bool,
            "reason": str,
            "checks": {check_name: {"fired": bool, "reason": str}, ...}
        }
    """
    checks = {}

    checks["format_violation"] = _check_format_violation(packet)
    checks["invalid_action"] = _check_invalid_action(packet)
    checks["repetition"] = _check_repetition(packet)
    checks["empty_trace"] = _check_empty_trace(packet, cfg)

    # Short-circuit if any critical check fires
    # FORMAT and INVALID are critical; REPETITION is informational (non-blocking)
    critical = ["format_violation", "invalid_action"]
    for name in critical:
        if checks[name]["fired"]:
            return {
                "should_short_circuit": True,
                "reason": checks[name]["reason"],
                "checks": checks,
            }

    return {
        "should_short_circuit": False,
        "reason": "",
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Individual checks — aligned with env ViolationType patterns
# ---------------------------------------------------------------------------

def _check_format_violation(packet: TurnJudgePacket) -> Dict[str, Any]:
    """
    Check if the response format is broken — verifies BOTH reasoning and action
    tags, not just the action tag.

    Aligned with ViolationType.FORMAT:
      - SciWorld: missing <reflection> or <action> tags
      - Sokoban: missing <think>/<action> tags or unparseable actions

    We rely on the env's own parse result (packet.format_correct) when available,
    then additionally verify that reasoning content exists (unless the format is
    explicitly no-think).
    """
    # If the env parser already flagged format failure, trust it
    if not packet.format_correct:
        return {"fired": True, "reason": "Response format is incorrect (env parser failed)"}

    # Check: no action content at all
    if not packet.chosen_action and not packet.action_tokens:
        return {"fired": True, "reason": "No action parsed from response"}

    # Check: reasoning tags present in raw trace (skip for no-think formats)
    raw = packet.raw_trace or ""
    has_think = bool(re.search(r"<(?:think|reflection)>", raw, re.I))
    has_action = bool(re.search(r"<(?:action|answer)>", raw, re.I))

    if raw and not has_action:
        return {"fired": True, "reason": "Missing <action> tag in raw trace"}

    # Reasoning is expected for most formats; only skip check if trace is very short
    # (no-think format produces no reasoning tags intentionally)
    if raw and not has_think and len(raw.split()) > 15:
        return {"fired": True, "reason": "Missing <think>/<reflection> tag in raw trace"}

    return {"fired": False, "reason": ""}


def _check_invalid_action(packet: TurnJudgePacket) -> Dict[str, Any]:
    """
    Check if the chosen action is invalid.

    Aligned with ViolationType.INVALID_ACTION:
      - SciWorld: "No known action matches" in env feedback
      - Sokoban: action not in ACTION_LOOKUP ("Up","Down","Left","Right")
                 OR action caused no movement (no-op into wall)

    Supports multiple detection strategies:
      1. Env-provided feedback string (e.g. "No known action matches")
      2. Explicit admissible actions list
      3. Available actions list (same concept, different naming)
    """
    # Strategy 1: Env feedback indicates rejection
    feedback = packet.action_feedback
    if feedback and "No known action" in feedback:
        return {
            "fired": True,
            "reason": f"Action rejected by env: {feedback[:100]}",
        }

    # Strategy 2/3: Check against admissible/available action set
    action_set = packet.admissible_actions or packet.available_actions
    if not action_set:
        return {"fired": False, "reason": ""}  # no action set provided

    action = (packet.chosen_action or "").strip().lower()
    admissible = {a.strip().lower() for a in action_set}

    if action and action not in admissible:
        return {
            "fired": True,
            "reason": f"Action '{packet.chosen_action}' not in valid action set",
        }
    return {"fired": False, "reason": ""}


def _check_repetition(packet: TurnJudgePacket) -> Dict[str, Any]:
    """
    Check if the same action was taken with unchanged observation (stuck loop).

    Aligned with ViolationType.REPETITION in SciWorld ViolationTracker:
    the check requires BOTH same action AND similar observation, not just same action.

    This is an informational check (not short-circuit by default) because
    the env-level ViolationTracker handles consecutive thresholds for termination.
    """
    if not packet.history:
        return {"fired": False, "reason": ""}

    current_action = (packet.chosen_action or "").strip().lower()
    if not current_action:
        return {"fired": False, "reason": ""}

    # Compare with most recent history entry
    prev = packet.history[0] if packet.history else {}
    prev_action = (
        prev.get("chosen_action") or prev.get("action_tokens") or prev.get("action") or ""
    )
    # Extract clean action from tagged formats
    m = re.search(r"<(?:answer|action)>(.*?)</(?:answer|action)>", prev_action, re.S | re.I)
    prev_action_clean = m.group(1).strip().lower() if m else prev_action.strip().lower()

    if current_action != prev_action_clean:
        return {"fired": False, "reason": ""}

    # Same action — now check if observation is also unchanged
    curr_obs = _normalize_obs(packet.post_action_observation or packet.current_observation_text or "")
    prev_obs = _normalize_obs(prev.get("observation_text") or prev.get("post_action_observation") or "")

    if curr_obs and prev_obs and curr_obs == prev_obs:
        return {
            "fired": True,
            "reason": f"Repeated action '{current_action}' with unchanged observation",
        }
    return {"fired": False, "reason": ""}


def _check_empty_trace(
    packet: TurnJudgePacket, cfg: HeuristicConfig
) -> Dict[str, Any]:
    """
    Check if the reasoning trace is empty or trivially short.

    This catches cases where the agent produces valid-looking format wrapping
    but with negligible reasoning content. Useful even when format_correct=True,
    as some formats (no_think) intentionally skip reasoning.
    """
    trace = packet.reasoning_tokens or ""
    # Strip all XML tags
    clean = re.sub(r"</?(?:think|observation|reasoning|prediction|reflection|planning|explore|monitor)>", "", trace).strip()
    tokens_approx = len(clean.split())

    if tokens_approx < cfg.min_trace_tokens:
        return {
            "fired": True,
            "reason": f"Trace too short ({tokens_approx} tokens < {cfg.min_trace_tokens})",
        }
    return {"fired": False, "reason": ""}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_obs(obs: str) -> str:
    """Normalize whitespace for observation comparison."""
    return " ".join(obs.split())
