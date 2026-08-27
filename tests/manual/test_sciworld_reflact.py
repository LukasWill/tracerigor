"""
Test script to verify SciWorld environment works correctly with:
1. SciWorldEnvConfig seeds generation (simulating create_dataset flow)
2. reflact and react prompt formats
3. LLM response parsing and environment stepping

Note: This test imports SciWorld components directly (not through tracerigor.env)
to avoid gym_sokoban dependency in environments where Sokoban is not installed.

Run: python test_sokoban/test_sciworld_reflact.py
"""
import sys
import os

# Add the workspace root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base config first without triggering tracerigor.env.__init__
# This avoids importing Sokoban which requires gym_sokoban
from tracerigor.env.base.base_env_config import BaseEnvConfig


def test_sciworld_config_seeds_generation():
    """Test 1: Verify SciWorldEnvConfig can generate seeds (simulating create_dataset)"""
    print("=" * 60)
    print("TEST 1: Verify SciWorldEnvConfig seeds generation")
    print("=" * 60)

    # Import directly from sciworld module
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    # Test config matching the structure of env_config.yaml
    config = SciWorldEnvConfig(
        generalization_level=0,
        split="train",
        env_step_limit=100,
        simplifications_preset="easy",
        render_mode="text",
        prompt_format="free_think",
        use_history=True,
        history_length=2,
    )

    # Verify generate_seeds works (this is what create_dataset uses)
    train_seeds = config.generate_seeds(5, seed=42)
    test_seeds = config.generate_seeds(3, seed=42)

    assert len(train_seeds) == 5, f"Expected 5 train seeds, got {len(train_seeds)}"
    assert len(test_seeds) == 3, f"Expected 3 test seeds, got {len(test_seeds)}"
    assert all(isinstance(s, int) for s in train_seeds), "Seeds should be integers"

    # Verify config_id works
    config_id = config.config_id()
    assert "sciworld" in config_id.lower() or "L0" in config_id, f"config_id should identify the env: {config_id}"

    print(f"✅ Successfully generated seeds: {len(train_seeds)} train, {len(test_seeds)} test")
    print(f"   Sample seeds: {train_seeds[:3]}...")
    print(f"   config_id: {config_id}")


def test_sciworld_reflact_format():
    """Test 2: Verify SciWorld env with reflact prompt format"""
    print("\n" + "=" * 60)
    print("TEST 2: Verify SciWorld env with reflact prompt format")
    print("=" * 60)

    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(
        prompt_format="reflact",
        generalization_level=0,
        split="train",
        simplifications_preset="easy"
    )
    env = SciWorldEnv(config)

    # Reset the environment
    obs, info = env.reset(seed=42)
    assert obs is not None, "Observation should not be None"
    assert 'task_description' in info, "Info should contain task_description"
    print(f"✅ Environment reset successfully!")
    print(f"   Task: {info['task_description'][:60]}...")

    # Test parsing with reflact format
    test_response = "<reflection>Location: kitchen. Inventory: empty. Task goal: complete experiment.</reflection><action>look around</action>"
    parsed = env.parse_func(test_response)

    assert parsed['format_correct'] == True, "Format should be correct"
    assert 'Location: kitchen' in parsed['reflection_content'], "Should parse reflection content"
    assert parsed['actions'] == ['look around'], f"Should parse action, got {parsed['actions']}"
    print(f"✅ Parsed reflact response correctly")
    print(f"   reflection_content: {parsed['reflection_content'][:50]}...")
    print(f"   actions: {parsed['actions']}")

    # Step the environment
    obs2, reward, done, info2 = env.step(test_response)
    assert obs2 is not None, "Observation after step should not be None"
    print(f"✅ Step executed: reward={reward}, done={done}")


def test_sciworld_react_format():
    """Test 3: Verify SciWorld env with react prompt format"""
    print("\n" + "=" * 60)
    print("TEST 3: Verify SciWorld env with react prompt format")
    print("=" * 60)

    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(
        prompt_format="react",
        generalization_level=0,
        split="train",
        simplifications_preset="easy"
    )
    env = SciWorldEnv(config)

    obs, info = env.reset(seed=123)
    print(f"✅ Environment reset with react format")

    # Test parsing with react format
    test_response = "<think>I need to look around first to see what's in the room.</think><action>look around</action>"
    parsed = env.parse_func(test_response)

    assert parsed['format_correct'] == True, "Format should be correct"
    assert 'look around' in parsed['think_content'], "Should parse think content"
    assert parsed['actions'] == ['look around'], f"Should parse action, got {parsed['actions']}"
    print(f"✅ Parsed react response correctly")
    print(f"   think_content: {parsed['think_content'][:50]}...")
    print(f"   actions: {parsed['actions']}")

    # Step the environment
    obs2, reward, done, info2 = env.step(test_response)
    assert obs2 is not None
    print(f"✅ Step executed: reward={reward}, done={done}")


def test_format_prompts():
    """Test 4: Verify format_prompt generates correct prompts for react/reflact"""
    print("\n" + "=" * 60)
    print("TEST 4: Verify format_prompt for react/reflact")
    print("=" * 60)

    from tracerigor.env.sciworld.prompt import format_prompt, FORMAT_CONFIGS

    # Verify react and reflact are in FORMAT_CONFIGS
    assert "react" in FORMAT_CONFIGS, "react should be in FORMAT_CONFIGS"
    assert "reflact" in FORMAT_CONFIGS, "reflact should be in FORMAT_CONFIGS"

    # Test react prompt
    react_prompt = format_prompt["react"](max_actions_per_step=1, action_sep=",", add_example=True)
    assert "<think>" in react_prompt, "React prompt should mention <think> tag"
    assert "<action>" in react_prompt, "React prompt should mention <action> tag"
    assert "current condition" in react_prompt.lower() or "plan" in react_prompt.lower(), \
        "React prompt should mention thinking about condition/planning"
    print(f"✅ React format prompt generated correctly")

    # Test reflact prompt
    reflact_prompt = format_prompt["reflact"](max_actions_per_step=1, action_sep=",", add_example=True)
    assert "<reflection>" in reflact_prompt, "Reflact prompt should mention <reflection> tag"
    assert "<action>" in reflact_prompt, "Reflact prompt should mention <action> tag"
    assert "reflect" in reflact_prompt.lower(), "Reflact prompt should mention reflection"
    print(f"✅ Reflact format prompt generated correctly")

    print(f"\nReact prompt:\n{react_prompt[:200]}...")
    print(f"\nReflact prompt:\n{reflact_prompt[:200]}...")


def test_special_tokens():
    """Test 5: Verify reflection tokens are in special_token_list"""
    print("\n" + "=" * 60)
    print("TEST 5: Verify reflection tokens in special_token_list")
    print("=" * 60)

    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    # Use SciWorldEnvConfig instead of abstract BaseEnvConfig
    config = SciWorldEnvConfig()
    assert "<reflection>" in config.special_token_list, "<reflection> should be in special_token_list"
    assert "</reflection>" in config.special_token_list, "</reflection> should be in special_token_list"
    print(f"✅ Reflection tokens present in special_token_list")
    print(f"   special_token_list: {config.special_token_list}")


def test_sciworld_config_with_reflact():
    """Test 6: Verify SciWorldEnvConfig works with reflact format"""
    print("\n" + "=" * 60)
    print("TEST 6: Test SciWorldEnvConfig with reflact format")
    print("=" * 60)

    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    # Config with reflact format
    config = SciWorldEnvConfig(
        generalization_level=0,
        split="train",
        simplifications_preset="easy",
        prompt_format="reflact",  # Using reflact!
    )

    # Verify config is created correctly
    assert config.prompt_format == "reflact"

    # Verify generate_seeds still works
    seeds = config.generate_seeds(3, seed=42)
    assert len(seeds) == 3

    print(f"✅ SciWorldEnvConfig created with reflact format")
    print(f"   prompt_format: {config.prompt_format}")
    print(f"   generated seeds: {seeds}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SCIWORLD REACT/REFLACT INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_sciworld_config_seeds_generation()
        test_sciworld_reflact_format()
        test_sciworld_react_format()
        test_format_prompts()
        test_special_tokens()
        test_sciworld_config_with_reflact()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
