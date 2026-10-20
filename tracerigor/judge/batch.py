"""
Offline batch judging of recorded agent turns.

Reads turn records (JSON/JSONL), builds ``TurnJudgePacket`` objects, and either

  * runs only the deterministic Layer-0 heuristics and prompt rendering
    (``dry_run=True``; no network, no optional dependencies), or
  * routes the turns through ``JudgeRouter`` against any OpenAI-compatible
    endpoint (requires the ``judge`` extra).

Output records keep the input's pass-through fields (e.g. ``training_step``)
and place the judge result under ``response``, which is the nested schema that
``tracerigor analyze`` reads.

Input record fields (only ``response`` is required):

    id                  unique turn identifier (default: line index)
    task_name           selects env-specific prompt templates (default "generic")
    episode_step        1-indexed step within the episode (default 1)
    observation         observation text shown to the agent before acting
    response            the agent's raw output, e.g. "<think>..</think><answer>..</answer>"
    admissible_actions  optional list of valid actions
    history             optional list of {"observation_text", "action_tokens", "reward"},
                        most recent first
    ground_truth_state  optional env state, e.g. {"state_sentences": [...]} for Sokoban
    action_feedback     optional env feedback for the action
    task_description    optional task string
    format_correct      optional env parser verdict (default True)
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence

from tracerigor.judge.config import JudgeConfig
from tracerigor.judge.env_templates import register_all_templates
from tracerigor.judge.heuristics import run_heuristics
from tracerigor.judge.packet_builder import build_turn_packet
from tracerigor.judge.prompt import build_messages
from tracerigor.judge.schema import TurnJudgePacket

# Fields consumed to build the packet; everything else is passed through.
_PACKET_FIELDS = {
    "id",
    "task_name",
    "episode_step",
    "observation",
    "response",
    "admissible_actions",
    "history",
    "ground_truth_state",
    "action_feedback",
    "task_description",
    "format_correct",
}


def turn_to_packet(turn: Mapping[str, Any], index: int) -> TurnJudgePacket:
    """Convert one recorded turn into a judge packet."""
    if not isinstance(turn.get("response"), str):
        raise ValueError(f"turn {index} needs a string 'response' field")
    info = {
        key: turn[key]
        for key in ("action_feedback", "ground_truth_state", "task_description", "format_correct")
        if key in turn
    }
    packet = build_turn_packet(
        env_id=str(turn.get("id", index)),
        task_name=str(turn.get("task_name", "generic")),
        episode_step=int(turn.get("episode_step", 1)),
        obs_text=str(turn.get("observation", "")),
        obs_images=[],
        history=[],
        raw_trace=turn["response"],
        info=info,
        admissible_actions=list(turn.get("admissible_actions") or []),
    )
    # History is already in the judge's format; build_turn_packet expects
    # rollout-recorder entries instead.
    packet.history = list(turn.get("history") or [])
    return packet


def judge_turns(
    turns: Sequence[Mapping[str, Any]],
    cfg: JudgeConfig,
    *,
    dry_run: bool = False,
    include_prompt: bool = False,
) -> List[Dict[str, Any]]:
    """Judge recorded turns and return one output record per input turn."""
    register_all_templates()
    packets = [turn_to_packet(turn, i) for i, turn in enumerate(turns)]
    ids = [packet.env_id for packet in packets]
    if len(set(ids)) != len(ids):
        raise ValueError("turn ids must be unique")

    if dry_run:
        results = [_dry_run_result(packet, cfg, include_prompt) for packet in packets]
    else:
        from tracerigor.judge.router import JudgeRouter

        router = JudgeRouter(cfg)
        try:
            scored = router.score_batch(packets)
        finally:
            router.close()
        results = []
        for packet in packets:
            response = asdict(scored[packet.env_id])
            response["heuristics"] = packet.heuristic_flags
            record: Dict[str, Any] = {"response": response}
            if include_prompt:
                record["prompt"] = _messages(packet, cfg)
            results.append(record)

    output = []
    for turn, packet, result in zip(turns, packets, results):
        record = {key: value for key, value in turn.items() if key not in _PACKET_FIELDS}
        record.update(
            id=packet.env_id,
            task_name=packet.task_name,
            episode_step=packet.episode_step,
            chosen_action=packet.chosen_action,
        )
        record.update(result)
        output.append(record)
    return output


def _dry_run_result(
    packet: TurnJudgePacket, cfg: JudgeConfig, include_prompt: bool
) -> Dict[str, Any]:
    heuristics = run_heuristics(packet, cfg.heuristics)
    result: Dict[str, Any] = {
        "route": "heuristic_short_circuit" if heuristics["should_short_circuit"] else "llm_judge",
        "short_circuit_reason": heuristics["reason"],
        "heuristics": heuristics["checks"],
    }
    if include_prompt and not heuristics["should_short_circuit"]:
        result["prompt"] = _messages(packet, cfg)
    return result


def _messages(packet: TurnJudgePacket, cfg: JudgeConfig) -> List[Dict[str, Any]]:
    from tracerigor.judge.router import _schema_hint

    use_schema = cfg.provider.use_structured_output
    rubric = "universal" if cfg.mode == "universal" else cfg.rubrics[0]
    return build_messages(
        packet,
        rubric=rubric,
        use_images=False,
        json_schema_hint="" if use_schema else _schema_hint(),
    )
