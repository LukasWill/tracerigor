import pytest

def _skip_if_missing():
    # Ensure external deps are present, otherwise skip gracefully
    pytest.importorskip("minigrid", reason="MiniGrid/BabyAI not installed")

def test_babyai_service_batch_smoke():
    _skip_if_missing()

    from tracerigor.env.babyai_text.service_config import BabyAITextServiceConfig
    from tracerigor.env.babyai_text.service import BabyAITextService

    # 1) Construct the service
    svc_cfg = BabyAITextServiceConfig(
        format_type="grounding_worldmodeling",
        max_actions_per_step=1,
        add_example=True,
        image_tag="<image>",
    )
    service = BabyAITextService(svc_cfg)

    # 2) Create two environments with potentially different subtasks
    ids2configs = {
        "env_a": {
            "env_config": {
                "env_id": "BabyAI-MixedTrainLocal-v0",
                "subtask": "goto",
                "format_penalty": 0.0,
                "binary_reward": False,
                "babyai_kwargs": {},
            }
        },
        "env_b": {
            "env_config": {
                "env_id": "BabyAI-MixedTrainLocal-v0",
                "subtask": "pickup",
                "format_penalty": 0.0,
                "binary_reward": False,
                "babyai_kwargs": {},
            }
        },
    }
    service.create_environments_batch(ids2configs)

    # 3) Reset both
    reset_out = service.reset_batch({"env_a": 0, "env_b": 1})
    assert "env_a" in reset_out and "env_b" in reset_out
    obs_a, info_a = reset_out["env_a"]
    obs_b, info_b = reset_out["env_b"]
    assert isinstance(obs_a, dict) and "obs_str" in obs_a
    assert isinstance(obs_b, dict) and "multi_modal_data" in obs_b

    # 4) Step both with plausible BabyAI-formatted replies
    ids2actions = {
        "env_a": "<think>Start moving.</think><answer>go forward</answer>",
        "env_b": "<think>Face the object.</think><answer>turn left</answer>",
    }
    step_out = service.step_batch(ids2actions)
    assert "env_a" in step_out and "env_b" in step_out
    s_obs_a, r_a, done_a, info_a = step_out["env_a"]
    s_obs_b, r_b, done_b, info_b = step_out["env_b"]
    assert isinstance(s_obs_a, dict) and isinstance(s_obs_b, dict)
    assert isinstance(r_a, float) and isinstance(done_a, bool)

    # 5) System prompts for both envs
    sys_prompts = service.get_system_prompts_batch(["env_a", "env_b"])
    assert isinstance(sys_prompts["env_a"], str) and "PLAY!" in sys_prompts["env_a"]

    # 6) Episode-end reward (not meaningful after one step, but should run)
    final_rewards = service.compute_reward_batch(["env_a", "env_b"])
    assert "env_a" in final_rewards and isinstance(final_rewards["env_a"], float)

    # 7) Close both
    service.close_batch(["env_a", "env_b"])


if __name__ == "__main__":
    test_babyai_service_batch_smoke()
