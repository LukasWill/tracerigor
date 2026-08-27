"""CPU-only smoke tests for the judge integration layer."""

from tracerigor.judge import JudgeConfig, JudgeIntegration, register_all_templates
from tracerigor.judge.config import HeuristicConfig, RewardConfig
from tracerigor.judge.heuristics import run_heuristics
from tracerigor.judge.prompt import available_templates, build_messages
from tracerigor.judge.reward import compute_process_reward
from tracerigor.judge.schema import JudgeResponse, RubricResult, TurnJudgePacket


def test_judge_configuration_and_templates() -> None:
    register_all_templates()
    assert available_templates()

    config = JudgeConfig.from_dict(
        {
            "enabled": True,
            "provider": {
                "base_url": "http://localhost:8001/v1",
                "model": "local-test-model",
            },
        }
    )
    assert config.provider.model == "local-test-model"

    packet = TurnJudgePacket(env_id="test_0", episode_step=1, task_name="sokoban")
    packet.reasoning_tokens = "The box is right of the player, so move right."
    packet.action_tokens = "<answer>Right</answer>"
    packet.chosen_action = "Right"
    packet.admissible_actions = ["Up", "Down", "Left", "Right"]
    messages = build_messages(packet, rubric="universal", use_images=False)
    assert [message["role"] for message in messages] == ["system", "user"]

    result = run_heuristics(packet, HeuristicConfig())
    assert "should_short_circuit" in result


def test_reward_and_disabled_integration() -> None:
    response = JudgeResponse(env_id="test_0")
    response.rubrics = {
        "grounding": RubricResult(label="pass", score=1.0, confidence=0.9),
        "action_coherence": RubricResult(label="pass", score=1.0, confidence=0.8),
        "temporal_consistency": RubricResult(label="fail", score=0.0, confidence=0.8),
    }
    reward = compute_process_reward(response, RewardConfig())
    assert 0.0 <= reward <= 1.0

    integration = JudgeIntegration(JudgeConfig(enabled=False))
    results = {
        "env0": (
            {"obs_str": "hello"},
            1.0,
            False,
            {"metrics": {"turn_metrics": {}}},
        )
    }
    assert integration.process_step_batch(results, {"env0": "test"}, {}, {}) == results
