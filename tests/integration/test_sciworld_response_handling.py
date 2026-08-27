import sys
import types


def _install_gym_sokoban_stub() -> None:
    if "gym_sokoban.envs.sokoban_env" in sys.modules:
        return

    gym_sokoban_module = types.ModuleType("gym_sokoban")
    gym_sokoban_envs_module = types.ModuleType("gym_sokoban.envs")
    gym_sokoban_env_module = types.ModuleType("gym_sokoban.envs.sokoban_env")

    class DummySokobanEnv:
        pass

    gym_sokoban_env_module.SokobanEnv = DummySokobanEnv
    gym_sokoban_envs_module.sokoban_env = gym_sokoban_env_module
    gym_sokoban_module.envs = gym_sokoban_envs_module

    sys.modules["gym_sokoban"] = gym_sokoban_module
    sys.modules["gym_sokoban.envs"] = gym_sokoban_envs_module
    sys.modules["gym_sokoban.envs.sokoban_env"] = gym_sokoban_env_module


def _install_together_stub() -> None:
    if "together" in sys.modules:
        return

    together_module = types.ModuleType("together")

    class DummyAsyncTogether:
        pass

    together_module.AsyncTogether = DummyAsyncTogether
    sys.modules["together"] = together_module


_install_gym_sokoban_stub()
_install_together_stub()

from tracerigor.env.sciworld.env import SciWorldEnv
from tracerigor.env.sciworld.env_config import SciWorldEnvConfig
from tracerigor.utils.response_utils import extract_reasoning_content, replace_reasoning_block


def test_sciworld_reflact_parser_accepts_single_pair():
    env = SciWorldEnv(SciWorldEnvConfig(prompt_format="reflact"))

    parsed = env.parse_func(
        "<reflection>I should inspect the room first.</reflection><action>look around</action>"
    )

    assert parsed["format_correct"] is True
    assert parsed["reflection_content"] == "I should inspect the room first."
    assert parsed["think_content"] == "I should inspect the room first."
    assert parsed["action_content"] == "look around"
    assert parsed["actions"] == ["look around"]


def test_sciworld_reflact_parser_rejects_multiple_pairs():
    env = SciWorldEnv(SciWorldEnvConfig(prompt_format="reflact"))

    parsed = env.parse_func(
        "<reflection>First idea.</reflection><action>look around</action>"
        "<reflection>Second idea.</reflection><action>open door</action>"
    )

    assert parsed["format_correct"] is False
    assert parsed["reflection_content"] == ""
    assert parsed["action_content"] == ""
    assert parsed["actions"] == []


def test_extract_reasoning_content_supports_reflection():
    response = "<reflection>Track the current room before acting.</reflection><action>look around</action>"

    assert extract_reasoning_content(response) == "Track the current room before acting."


def test_replace_reasoning_block_preserves_reflact_format():
    original = "<reflection>Old reflection.</reflection><action>look around</action>"

    rewritten = replace_reasoning_block(original, "New reflection.")

    assert rewritten == "<reflection>New reflection.</reflection><action>look around</action>"