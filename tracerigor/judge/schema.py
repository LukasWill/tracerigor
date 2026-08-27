"""
Judge stack schema definitions.

Defines the data contracts between the rollout manager, judge router,
LLM client, and reward mapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Turn-level judge input packet
# ---------------------------------------------------------------------------

@dataclass
class TurnJudgePacket:
    """Everything the judge needs to evaluate one agent turn."""

    # Identifiers
    env_id: str
    episode_step: int
    task_name: str                            # e.g. "sokoban", "sciworld", "alfworld"
    agent_modality: str = "text"              # "text" | "vision" | "mixed"

    # Current-turn observation (pre-action)
    current_observation_text: str = ""
    current_observation_images: List[Any] = field(default_factory=list)  # PIL Images

    # Post-action observation (for repetition detection)
    post_action_observation: str = ""

    # Recent trajectory window (most-recent first)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Actor output for this turn
    raw_trace: str = ""                       # full decoded response (with <think>…</think><action>…</action>)
    reasoning_tokens: str = ""                # extracted reasoning (e.g. <think>…</think> or <reflection>…</reflection>)
    action_tokens: str = ""                   # extracted action block (e.g. <action>…</action>)
    chosen_action: str = ""                   # parsed clean action string
    format_correct: bool = True               # whether the LLM response was parseable

    # Env metadata
    available_actions: List[str] = field(default_factory=list)
    admissible_actions: List[str] = field(default_factory=list)

    # Action feedback from env (e.g. "No known action matches" in sciworld)
    action_feedback: str = ""

    # Ground truth from env (optional, for judge prompt enrichment)
    # Sokoban: {"state_sentences": [...]}  SciWorld: {"ground_truth_location": "...", "ground_truth_inventory": [...]}
    ground_truth_state: Dict[str, Any] = field(default_factory=dict)

    # Env-specific context for the judge prompt
    task_description: str = ""                # e.g. SciWorld task string
    valid_actions: str = ""                   # formatted list of valid actions for this step

    # Layer-0 heuristic results (filled by heuristics module before LLM call)
    heuristic_flags: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-rubric result from the LLM judge
# ---------------------------------------------------------------------------

@dataclass
class RubricResult:
    """Score for one evaluation dimension."""

    label: str = "uncertain"                  # "pass" | "fail" | "uncertain"
    score: float = 0.5                        # 0.0 | 0.5 | 1.0
    confidence: float = 0.5                   # [0, 1]
    evidence: List[str] = field(default_factory=list)  # ≤3 short bullets


# ---------------------------------------------------------------------------
# Full judge response for one turn
# ---------------------------------------------------------------------------

@dataclass
class JudgeResponse:
    """Aggregated judge output for one turn."""

    env_id: str = ""
    episode_step: int = 0

    # Layer-0 result
    short_circuited: bool = False             # True if heuristics bypassed LLM
    short_circuit_reason: str = ""

    # Layer-1 rubric scores
    rubrics: Dict[str, RubricResult] = field(default_factory=dict)

    # Aggregate
    overall_confidence: float = 0.5
    insufficient_evidence: bool = False

    # Raw LLM output (for audit / debugging)
    raw_llm_response: str = ""
    query_success: bool = False
    parse_success: bool = False
    model_used: str = ""

    # Computed process reward (filled by reward module)
    process_reward: float = 0.0


# ---------------------------------------------------------------------------
# JSON Schema for structured LLM output (used with vLLM guided decoding)
# ---------------------------------------------------------------------------

JUDGE_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observation_grounding": {"$ref": "#/$defs/rubric_yn"},
        "action_coherence": {"$ref": "#/$defs/rubric_yn"},
        "temporal_consistency": {"$ref": "#/$defs/rubric_yn"},
    },
    "required": ["observation_grounding", "action_coherence", "temporal_consistency"],
    "$defs": {
        "rubric_yn": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "yes_no": {"type": "string", "enum": ["YES", "NO"]},
                "evidence": {"type": "string", "maxLength": 200},
            },
            "required": ["yes_no", "evidence"],
        }
    },
}


# ---------------------------------------------------------------------------
# Binary (YES/NO) schema for single-rubric mode (backward compat)
# ---------------------------------------------------------------------------

JUDGE_BINARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "maxLength": 120},
        },
    },
    "required": ["verdict", "confidence", "evidence"],
}
