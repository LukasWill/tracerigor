"""Pinned judge semantics: short-circuit penalty, default shaping, history order."""

import pytest

from tracerigor.judge.config import GatingConfig, HeuristicConfig, JudgeConfig, RewardConfig
from tracerigor.judge.heuristics import run_heuristics
from tracerigor.judge.packet_builder import build_packets_from_step_results
from tracerigor.judge.prompt import _format_history
from tracerigor.judge.reward import compute_process_reward
from tracerigor.judge.router import JudgeRouter
from tracerigor.judge.schema import JudgeResponse, RubricResult, TurnJudgePacket

RUBRICS = ("observation_grounding", "action_coherence", "temporal_consistency")


def _response(*scores: float, insufficient: bool = False) -> JudgeResponse:
    return JudgeResponse(
        rubrics={name: RubricResult(score=score) for name, score in zip(RUBRICS, scores)},
        insufficient_evidence=insufficient,
    )


# ---------------------------------------------------------------------------
# Short-circuit penalty
# ---------------------------------------------------------------------------

def test_short_circuit_uses_heuristic_penalty_by_default() -> None:
    reward = compute_process_reward(JudgeResponse(short_circuited=True), RewardConfig())
    assert reward == pytest.approx(-0.20)


def test_short_circuit_penalty_is_clamped_to_reward_range() -> None:
    response = JudgeResponse(short_circuited=True)
    assert compute_process_reward(response, RewardConfig(), short_circuit_penalty=-0.10) == pytest.approx(-0.10)
    assert compute_process_reward(response, RewardConfig(), short_circuit_penalty=-0.90) == pytest.approx(-0.25)


def test_router_passes_configured_short_circuit_penalty(requires_judge_client) -> None:
    cfg = JudgeConfig.from_dict(
        {
            "provider": {"base_url": "http://127.0.0.1:9/v1", "model": "stub"},
            "heuristics": {"short_circuit_penalty": -0.05},
        }
    )
    cfg.audit.enabled = False
    packet = TurnJudgePacket(env_id="e0", episode_step=1, task_name="sokoban", format_correct=False)
    router = JudgeRouter(cfg)
    try:
        verdict = router.score_batch([packet])["e0"]
    finally:
        router.close()
    assert verdict.short_circuited
    assert verdict.process_reward == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# Shaping modes
# ---------------------------------------------------------------------------

# Default weights 0.35 / 0.40 / 0.25 over (grounding, coherence, temporal).
@pytest.mark.parametrize(
    "scores, asymmetric, symmetric",
    [
        ((1.0, 1.0, 1.0), 0.25, 0.25),    # r_raw 1.00 → 1.0, clamped
        ((1.0, 0.0, 0.0), 0.0, -0.25),    # r_raw 0.35 → -0.30, clamped
        ((0.5, 0.5, 0.5), 0.0, 0.0),      # r_raw 0.50 → 0
        ((0.0, 1.0, 0.0), 0.0, -0.20),    # r_raw 0.40 → -0.20
    ],
)
def test_default_shaping_is_asymmetric(scores, asymmetric, symmetric) -> None:
    response = _response(*scores)
    assert compute_process_reward(response, RewardConfig()) == pytest.approx(asymmetric)
    assert compute_process_reward(response, RewardConfig(symmetric_shaping=True)) == pytest.approx(symmetric)


def test_asymmetric_bonus_below_clamp_and_annealing() -> None:
    # r_raw = 0.35 + 0.25 = 0.60 → 2·0.60 − 1 = 0.20 in both modes.
    response = _response(1.0, 0.0, 1.0)
    assert compute_process_reward(response, RewardConfig()) == pytest.approx(0.20)
    assert compute_process_reward(response, RewardConfig(symmetric_shaping=True)) == pytest.approx(0.20)

    # Annealing halfway between hi=1.0 and lo=0.5 → w_t = 0.75.
    gating = GatingConfig(anneal_enabled=True, anneal_start_step=0, anneal_end_step=100)
    assert compute_process_reward(response, RewardConfig(), gating, train_step=50) == pytest.approx(0.15)


def test_insufficient_evidence_only_reduces_the_asymmetric_bonus() -> None:
    # r_raw = 1.0 − 0.15 = 0.85 → 0.70, clamped to 0.25.
    assert compute_process_reward(_response(1.0, 1.0, 1.0, insufficient=True), RewardConfig()) == pytest.approx(0.25)
    # r_raw = 0.60 − 0.15 = 0.45 → no bonus by default, −0.10 when symmetric.
    response = _response(1.0, 0.0, 1.0, insufficient=True)
    assert compute_process_reward(response, RewardConfig()) == pytest.approx(0.0)
    assert compute_process_reward(response, RewardConfig(symmetric_shaping=True)) == pytest.approx(-0.10)


def test_alpha_small_does_not_affect_reward() -> None:
    response = _response(0.0, 1.0, 0.0)
    for cfg in (RewardConfig(), RewardConfig(gate_mul=True)):
        baseline = compute_process_reward(response, cfg)
        cfg.alpha_small = 0.9
        assert compute_process_reward(response, cfg) == pytest.approx(baseline)


# ---------------------------------------------------------------------------
# Rollout history order
# ---------------------------------------------------------------------------

def _recorder(n: int):
    return [
        {
            "obs_str": f"obs{i}",
            "info": {"llm_raw_response": f"resp{i}", "think_content": "t", "action_content": f"a{i}"},
            "reward": float(i),
        }
        for i in range(n)
    ]


def _build(records, history_k: int):
    step_results = {"e0": ({}, 0.0, False, {"action_content": "Right"})}
    (packet,) = build_packets_from_step_results(
        step_results=step_results,
        responses_str={"e0": "<think>go right</think><action>Right</action>"},
        recorder={"e0": records},
        env_states={"e0": {"step": len(records) - 1}},
        env_configs={},
        history_k=history_k,
    )
    return packet


def test_recorder_history_is_most_recent_first() -> None:
    # records[-1] is the post-step record and is excluded; the window is the
    # history_k records before it, newest first.
    packet = _build(_recorder(6), history_k=3)
    assert [h["obs"] for h in packet.history] == ["obs4", "obs3", "obs2"]
    assert packet.current_observation_text == "obs4"

    lines = _format_history(packet.history).splitlines()
    assert lines[0].startswith("[t-1] obs: obs4")
    assert lines[-1].startswith("[t-3] obs: obs2")


def test_recorder_history_short_episode() -> None:
    assert [h["obs"] for h in _build(_recorder(3), history_k=5).history] == ["obs1", "obs0"]
    assert _build(_recorder(1), history_k=3).history == []


def test_repetition_check_compares_against_history_zero() -> None:
    packet = TurnJudgePacket(
        env_id="e0",
        episode_step=2,
        task_name="sokoban",
        chosen_action="Right",
        current_observation_text="wall ahead",
        history=[
            {"action_tokens": "<action>Right</action>", "observation_text": "wall ahead"},
            {"action_tokens": "<action>Up</action>", "observation_text": "open floor"},
        ],
    )
    assert run_heuristics(packet, HeuristicConfig())["checks"]["repetition"]["fired"]
    packet.history.reverse()
    assert not run_heuristics(packet, HeuristicConfig())["checks"]["repetition"]["fired"]
