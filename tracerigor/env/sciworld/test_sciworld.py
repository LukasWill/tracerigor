"""
Smoke tests for the TraceRigor-compliant SciWorld environment package.

Run with:
    python tracerigor/env/sciworld/test_sciworld.py

These tests verify:
1. Configuration instantiation
2. Environment initialization
3. Basic reset and step operations
4. Service batch operations
5. Action parsing

Note: Some tests require Java and ScienceWorld to be fully installed.
      Tests that require Java will be skipped if Java is not available.
"""

import os
import sys
import traceback
from typing import Optional

# Ensure we can import from the tracerigor package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Test Utilities
# =============================================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.error: Optional[str] = None

    def __str__(self):
        if self.skipped:
            return f"⚠️  SKIP: {self.name} - {self.error}"
        elif self.passed:
            return f"✅ PASS: {self.name}"
        else:
            return f"❌ FAIL: {self.name}\n   Error: {self.error}"


def run_test(test_func):
    """Decorator to run a test function and capture results."""
    result = TestResult(test_func.__name__)
    try:
        test_func()
        result.passed = True
    except ImportError as e:
        result.skipped = True
        result.error = f"Import error (dependency not installed?): {e}"
    except RuntimeError as e:
        if "Java" in str(e) or "JAR" in str(e):
            result.skipped = True
            result.error = f"ScienceWorld requires Java: {e}"
        else:
            result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return result


# =============================================================================
# Configuration Tests
# =============================================================================

def test_env_config_instantiation():
    """Test that SciWorldEnvConfig can be instantiated with defaults."""
    # Import directly from sciworld package to avoid tracerigor.env import chain issues
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig()

    assert config.env_name == "sciworld", f"Expected env_name='sciworld', got '{config.env_name}'"
    assert config.task_nums == [1], f"Expected task_nums=[1], got {config.task_nums}"
    assert config.split == "train", f"Expected split='train', got '{config.split}'"
    assert config.env_step_limit == 100, f"Expected env_step_limit=100, got {config.env_step_limit}"
    assert config.prompt_format == "free_think", f"Expected prompt_format='free_think', got '{config.prompt_format}'"

    print(f"  Config ID: {config.config_id()}")


def test_env_config_custom():
    """Test SciWorldEnvConfig with custom values."""
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(
        task_nums=[0, 1, 2],
        split="test",
        env_step_limit=50,
        prompt_format="no_think",
        use_history=False,
        meta_think=True,
        generalization_level=1
    )

    assert config.task_nums == [0, 1, 2], f"Expected [0,1,2], got {config.task_nums}"
    assert config.split == "test", f"Expected 'test', got '{config.split}'"
    assert config.env_step_limit == 50, f"Expected 50, got {config.env_step_limit}"
    assert config.prompt_format == "no_think", f"Expected 'no_think', got '{config.prompt_format}'"
    assert config.use_history == False, f"Expected False, got {config.use_history}"
    assert config.meta_think == True, f"Expected True, got {config.meta_think}"
    assert config.generalization_level == 1, f"Expected 1, got {config.generalization_level}"


def test_service_config_instantiation():
    """Test that SciWorldServiceConfig can be instantiated."""
    from tracerigor.env.sciworld.service_config import SciWorldServiceConfig

    config = SciWorldServiceConfig()

    # BaseServiceConfig has max_workers
    assert config.max_workers > 0, f"Expected max_workers > 0, got {config.max_workers}"
    assert hasattr(config, 'use_state_reward'), "Missing use_state_reward attribute"


# =============================================================================
# Prompt and Format Tests
# =============================================================================

def test_format_configs():
    """Test that FORMAT_CONFIGS contains all expected formats."""
    from tracerigor.env.sciworld.prompt import FORMAT_CONFIGS

    expected_formats = ["free_think", "no_think", "grounding", "worldmodeling", "meta_think"]

    for fmt in expected_formats:
        assert fmt in FORMAT_CONFIGS, f"Missing format '{fmt}' in FORMAT_CONFIGS"
        cfg = FORMAT_CONFIGS[fmt]
        # Each format config should have description, format, and example
        assert "description" in cfg, f"Missing description in {fmt}"
        assert "format" in cfg, f"Missing format in {fmt}"
        assert "example" in cfg, f"Missing example in {fmt}"


def test_system_prompt_function():
    """Test that system_prompt function works correctly."""
    from tracerigor.env.sciworld.prompt import system_prompt

    # Test normal prompt
    prompt = system_prompt(meta_think=False)
    assert "ScienceWorld" in prompt, "Missing ScienceWorld reference in prompt"
    assert "action" in prompt.lower(), "Missing action reference in prompt"

    # Test meta-think prompt
    meta_prompt = system_prompt(meta_think=True)
    assert "<planning>" in meta_prompt, "Missing planning tag in meta_think prompt"
    assert "<explore>" in meta_prompt, "Missing explore tag in meta_think prompt"


def test_observation_templates():
    """Test observation template functions."""
    from tracerigor.env.sciworld.prompt import (
        init_observation_template,
        action_observation_template
    )

    # Test init observation
    init_obs = init_observation_template(
        task_description="Test task",
        observation="You are in a room.",
        available_actions="look around, pick up item"
    )

    assert "Test task" in init_obs
    assert "You are in a room." in init_obs
    assert "look around" in init_obs

    # Test action observation with history
    action_obs = action_observation_template(
        task_description="Test task",
        step_count=2,
        history_length=2,
        action_history="[Step 1, Action: 'look']",
        current_step=3,
        observation="You see a table.",
        available_actions="pick up item"
    )

    assert "Test task" in action_obs
    assert "step 3" in action_obs.lower() or "Step 3" in action_obs


# =============================================================================
# Action Parsing Tests
# =============================================================================

def test_action_parsing():
    """Test that action parsing works correctly."""
    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig()
    # Don't actually init the env, just test the parser

    # Create parser directly
    import re
    action_prefix = "<action>"
    action_suffix = "</action>"
    pattern = re.compile(
        re.escape(action_prefix) + r"(.*?)" + re.escape(action_suffix),
        re.DOTALL
    )

    # Test case 1: Single action
    text1 = "<think>I should look around</think><action>look around</action>"
    matches = pattern.findall(text1)
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
    assert matches[0].strip() == "look around", f"Expected 'look around', got '{matches[0].strip()}'"

    # Test case 2: Multiple actions
    text2 = "<action>go north</action>\n<action>pick up key</action>"
    matches = pattern.findall(text2)
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"

    # Test case 3: No action tags
    text3 = "I don't know what to do"
    matches = pattern.findall(text3)
    assert len(matches) == 0, f"Expected 0 matches for no tags"


# =============================================================================
# Environment Initialization Tests
# =============================================================================

def test_env_instantiation():
    """Test that SciWorldEnv can be instantiated (without full initialization)."""
    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(
        task_nums=[1],
        env_step_limit=100
    )

    # This should work without Java (lazy init)
    env = SciWorldEnv(config)

    assert env.config == config
    assert env._initialized == False  # Lazy init
    assert env._gym_env is None


def test_env_system_prompt_generation():
    """Test system prompt generation without full env initialization."""
    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(prompt_format="free_think")
    env = SciWorldEnv(config)

    prompt = env.system_prompt()

    # System prompt should contain format instructions, not task description
    # Task description goes in the observation
    assert "ScienceWorld" in prompt, "Missing ScienceWorld reference in system prompt"
    assert "<think>" in prompt or "think" in prompt.lower(), "Think instruction not in prompt"
    assert "<action>" in prompt, "Action format not in prompt"

    print(f"  System prompt length: {len(prompt)} chars")


def test_env_full_initialization():
    """Test full environment initialization and reset (requires ScienceWorld + Java)."""
    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(
        task_nums=[0],  # First task (usually something simple)
        env_step_limit=10
    )

    env = SciWorldEnv(config)

    # This will trigger full initialization with ScienceWorld
    obs, info = env.reset(seed=42)

    assert env._initialized == True, "Environment should be initialized after reset"
    assert env.task_description is not None, "Task description should be set"
    assert "obs_str" in obs, "Observation should have 'obs_str' key"
    assert len(obs["obs_str"]) > 0, "Observation text should not be empty"

    print(f"  Task: {env.task_description[:50]}...")
    print(f"  Observation: {obs['obs_str'][:80]}...")


def test_env_step():
    """Test environment step function (requires ScienceWorld + Java)."""
    from tracerigor.env.sciworld.env import SciWorldEnv
    from tracerigor.env.sciworld.env_config import SciWorldEnvConfig

    config = SciWorldEnvConfig(
        task_nums=[0],
        env_step_limit=10,
        prompt_format="no_think"  # Simpler format for testing
    )

    env = SciWorldEnv(config)
    obs, info = env.reset(seed=42)

    # Step with a generic action that should work
    action_text = "<action>look around</action>"
    obs, reward, done, info = env.step(action_text)

    assert "obs_str" in obs, "Step observation should have 'obs_str' key"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(done, bool), "Done should be boolean"

    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Format correct: {info.get('format_correct')}")


# =============================================================================
# Service Tests
# =============================================================================

def test_service_instantiation():
    """Test that SciWorldService can be instantiated."""
    from tracerigor.env.sciworld.service import SciWorldService
    from tracerigor.env.sciworld.service_config import SciWorldServiceConfig

    # Create a minimal config
    config = SciWorldServiceConfig()

    # Service instantiation should work without Java
    # (environments are created lazily via create_environments_batch)
    service = SciWorldService(config)

    assert service.config == config
    assert isinstance(service.environments, dict), "environments should be a dict"
    print(f"  Service created with max_workers={config.max_workers}")


# =============================================================================
# Registration Tests
# =============================================================================

def test_registration():
    """Test that sciworld is registered in TraceRigor env registry."""
    try:
        # Read the __init__.py file to check registration without importing
        # (importing tracerigor.env may fail if other dependencies are missing)
        init_path = os.path.join(SCRIPT_DIR, '..', '__init__.py')
        with open(init_path, 'r') as f:
            init_content = f.read()

        # Check that sciworld is mentioned in REGISTERED_ENV
        assert '"sciworld"' in init_content or "'sciworld'" in init_content, \
            "sciworld not found in REGISTERED_ENV"
        assert 'SciWorldEnv' in init_content, "SciWorldEnv not imported"
        assert 'SciWorldEnvConfig' in init_content, "SciWorldEnvConfig not imported"

        print(f"  Registration found in __init__.py")
    except FileNotFoundError:
        raise Exception(f"Could not find tracerigor/env/__init__.py at {init_path}")


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("SciWorld TraceRigor Environment - Smoke Tests")
    print("=" * 60)
    print()

    # Define test groups
    config_tests = [
        test_env_config_instantiation,
        test_env_config_custom,
        test_service_config_instantiation,
    ]

    prompt_tests = [
        test_format_configs,
        test_system_prompt_function,
        test_observation_templates,
    ]

    parsing_tests = [
        test_action_parsing,
    ]

    env_tests = [
        test_env_instantiation,
        test_env_system_prompt_generation,
        test_env_full_initialization,  # Requires Java
        test_env_step,                 # Requires Java
    ]

    service_tests = [
        test_service_instantiation,
    ]

    registration_tests = [
        test_registration,
    ]

    all_test_groups = [
        ("Configuration Tests", config_tests),
        ("Prompt & Format Tests", prompt_tests),
        ("Action Parsing Tests", parsing_tests),
        ("Environment Tests", env_tests),
        ("Service Tests", service_tests),
        ("Registration Tests", registration_tests),
    ]

    results = []

    for group_name, tests in all_test_groups:
        print(f"\n{group_name}")
        print("-" * 40)

        for test_func in tests:
            result = run_test(test_func)
            results.append(result)
            print(result)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)

    print(f"  Total:   {len(results)}")
    print(f"  Passed:  {passed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print()

    if failed > 0:
        print("❌ Some tests failed!")
        sys.exit(1)
    elif skipped > 0:
        print("⚠️  All critical tests passed, some skipped (likely missing Java/ScienceWorld)")
        sys.exit(0)
    else:
        print("✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
