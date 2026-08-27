import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def _install_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    openai_module = types.ModuleType("openai")

    class DummyAsyncOpenAI:
        pass

    openai_module.AsyncOpenAI = DummyAsyncOpenAI
    sys.modules["openai"] = openai_module


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_install_gym_sokoban_stub()
_install_together_stub()
_install_openai_stub()


def test_navigation_reflact_prompt_uses_grounding_icl_in_reflact_format():
    prompt = _load_module("navigation_prompt_for_test", "tracerigor/env/navigation/prompt.py")

    assert "reflact" in prompt.FORMAT_CONFIGS
    assert "reflact_diverse" in prompt.FORMAT_CONFIGS

    format_prompt = prompt.format_prompt["reflact"](
        max_actions_per_step=5,
        action_sep=",",
        add_example=True,
    )
    assert "<reflection>...</reflection><action>...</action>" in format_prompt
    assert "<reflection>I am in a living room" in format_prompt
    assert "<action>moveahead,moveahead,rotateright,moveahead,moveahead</action>" in format_prompt

    diverse_prompt = prompt.format_prompt["reflact_diverse"](
        max_actions_per_step=5,
        action_sep=",",
        add_example=True,
    )
    assert "There is a couch to my left" in diverse_prompt
    assert "I can see the kitchen doorway to my right" in diverse_prompt
    assert "I am at the entrance of a bedroom" in diverse_prompt

    system_prompt = prompt.system_prompt(format="reflact")
    assert "<reflection>I am in a living room" in system_prompt
    assert "<action>moveahead, moveahead, rotateright, moveahead, moveahead</action>" in system_prompt

    diverse_system_prompt = prompt.system_prompt(format="reflact_diverse")
    assert "I can see the kitchen doorway to my right" in diverse_system_prompt
    assert "I am at the entrance of a bedroom" in diverse_system_prompt


def test_navigation_reflact_parser_accepts_action_tags():
    parse_utils = _load_module("parse_utils_for_navigation_test", "tracerigor/env/utils/parse_utils.py")

    parsed = parse_utils.PARSE_FUNC_MAP["reflact"](
        "<reflection>The target is ahead-left, and I should move closer.</reflection>"
        "<action>moveahead,moveleft</action>",
        action_sep=",",
        max_actions=5,
    )

    assert parsed["format_correct"] is True
    assert parsed["reflection_content"] == "The target is ahead-left, and I should move closer."
    assert parsed["think_content"] == parsed["reflection_content"]
    assert parsed["actions"] == ["moveahead", "moveleft"]


def test_navigation_config_id_includes_prompt_format_for_env_reuse():
    source = (ROOT / "tracerigor/env/navigation/env_config.py").read_text()

    assert '"prompt_format"' in source
    assert 'id_fields = ["eval_set", "render_mode", "max_actions_per_step", "prompt_format"]' in source
