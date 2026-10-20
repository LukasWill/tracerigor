"""Environment registry behaviour that depends on optional extras."""

import importlib.util

import pytest

from tracerigor.env import environment_status, get_environment


def test_lazy_simulators_report_missing_runtime_packages() -> None:
    status = environment_status()
    if importlib.util.find_spec("scienceworld") is None:
        assert status["sciworld"]["status"] == "unavailable"
        assert "scienceworld" in status["sciworld"]["detail"]


def test_sokoban_episode_with_envs_extra() -> None:
    """Sokoban must import without the legacy judge-server stack (Hydra, W&B)."""
    # Check the real dependency: other tests may stub gym_sokoban when it is absent.
    if importlib.util.find_spec("gym") is None:
        pytest.skip("requires the envs extra")
    spec = get_environment("sokoban")
    env = spec["env_cls"](spec["config_cls"](render_mode="text"))
    try:
        obs, _ = env.reset(seed=7)
        assert " P " in obs["obs_str"]
        _, _, _, info = env.step("<think>Push the box right.</think><action>Right</action>")
        assert info["metrics"]["turn_metrics"]["action_is_valid"]
    finally:
        env.close()
