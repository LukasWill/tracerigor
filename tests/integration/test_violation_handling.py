"""
Test script for violation handling mechanism in SciWorld environment.

Tests three types of violations:
1. Format violations (missing <reflection>/<action> tags)
2. Invalid actions ("No known action matches")
3. Repetition (same action with unchanged observation)
"""

from tracerigor.env.sciworld.env import SciWorldEnv, ViolationTracker, ViolationType
from tracerigor.env.sciworld.env_config import SciWorldEnvConfig


def test_violation_tracker_unit():
    """Unit test for ViolationTracker class."""
    print("=" * 60)
    print("Testing ViolationTracker unit tests")
    print("=" * 60)

    tracker = ViolationTracker(
        format_threshold=3,
        invalid_action_threshold=5,
        repetition_threshold=3,
    )

    # Test 1: Format violations
    print("\n1. Testing format violations (threshold=3)...")
    for i in range(3):
        should_term, reason = tracker.record_step(
            format_correct=False,
            action="",
            observation="Some observation"
        )
        print(f"   Step {i+1}: should_terminate={should_term}, reason={reason}")
        if i < 2:
            assert not should_term, f"Should not terminate after {i+1} format violations"
        else:
            assert should_term, "Should terminate after 3 format violations"
            assert reason == ViolationType.FORMAT

    # Reset and test invalid actions
    tracker.reset()
    print("\n2. Testing invalid actions (threshold=5)...")
    for i in range(5):
        should_term, reason = tracker.record_step(
            format_correct=True,
            action=f"invalid_action_{i}",  # Different actions to avoid repetition
            observation=f"No known action matches that input. (attempt {i})",  # Different obs
            action_feedback=f"No known action matches that input. (attempt {i})"
        )
        print(f"   Step {i+1}: should_terminate={should_term}, reason={reason}")
        if i < 4:
            assert not should_term, f"Should not terminate after {i+1} invalid actions"
        else:
            assert should_term, "Should terminate after 5 invalid actions"
            assert reason == ViolationType.INVALID_ACTION

    # Reset and test repetition
    tracker.reset()
    print("\n3. Testing repetition (threshold=3)...")
    for i in range(4):  # Need 4 steps: 1 initial + 3 repetitions
        should_term, reason = tracker.record_step(
            format_correct=True,
            action="wait1",
            observation="You wait for 1 iteration. Nothing happens."
        )
        print(f"   Step {i+1}: should_terminate={should_term}, reason={reason}")
        if i < 3:
            # First step is not repetition (no prior), steps 2-3 are repetitions but below threshold
            pass
        else:
            assert should_term, "Should terminate after 3 consecutive repetitions"
            assert reason == ViolationType.REPETITION

    # Test that different action resets counter
    tracker.reset()
    print("\n4. Testing counter reset when action changes...")
    # Two repetitions
    tracker.record_step(True, "wait1", "Same observation")
    tracker.record_step(True, "wait1", "Same observation")
    # Different action - should reset
    should_term, _ = tracker.record_step(True, "look around", "You look around...")
    assert not should_term
    print(f"   After different action: consecutive_repetitions={tracker.consecutive_repetitions}")
    assert tracker.consecutive_repetitions == 0, "Counter should reset after different action"

    # Test concurrent format violation + repetition
    tracker.reset()
    print("\n5. Testing concurrent format violation + repetition...")
    # Format violation that repeats (same action, same obs)
    tracker.record_step(False, "wait1", "You wait.")  # Format violation, step 1
    tracker.record_step(False, "wait1", "You wait.")  # Format + rep 1
    should_term, reason = tracker.record_step(False, "wait1", "You wait.")  # Format + rep 2
    print(f"   After 3 format violations + 2 repetitions: F={tracker.consecutive_format_violations}, R={tracker.consecutive_repetitions}")
    assert tracker.consecutive_format_violations == 3, "Should have 3 format violations"
    assert tracker.consecutive_repetitions == 2, "Should have 2 repetitions"
    assert should_term, "Should terminate (format threshold reached)"
    assert reason == ViolationType.FORMAT, "Format takes priority over repetition"

    print("\n✅ All unit tests passed!")


def test_env_integration():
    """Integration test with actual SciWorld environment."""
    print("\n" + "=" * 60)
    print("Testing SciWorld environment integration")
    print("=" * 60)

    # Create config with violation handling enabled
    config = SciWorldEnvConfig(
        task_nums=[1],
        env_step_limit=20,
        prompt_format="reflact",
        enable_violation_termination=True,
        format_violation_threshold=2,  # Lower for faster testing
        invalid_action_threshold=3,
        repetition_threshold=2,
        violation_penalty=-1.0,
    )

    print(f"\nConfig: violation thresholds = format:{config.format_violation_threshold}, "
          f"invalid:{config.invalid_action_threshold}, repetition:{config.repetition_threshold}")

    try:
        env = SciWorldEnv(config)
        obs, info = env.reset(seed=0)
        print(f"\n✅ Environment initialized successfully")
        print(f"   Task: {info['task_description'][:100]}...")

        # Test format violation termination
        print("\n5. Testing format violation termination in env...")
        obs, info = env.reset(seed=0)

        for i in range(3):
            # Send malformed response (no tags)
            response = "I should do something"
            obs, reward, done, info = env.step(response)
            print(f"   Step {i+1}: done={done}, reward={reward:.2f}, "
                  f"violation_terminated={info.get('violation_terminated', False)}")

            if done and info.get('violation_terminated'):
                print(f"   ✅ Terminated due to: {info.get('termination_reason')}")
                break

        # Test invalid action termination
        print("\n6. Testing invalid action termination in env...")
        obs, info = env.reset(seed=0)

        for i in range(5):
            # Send valid format but invalid action
            response = "<reflection>Testing invalid action</reflection><action>fly to moon</action>"
            obs, reward, done, info = env.step(response)
            print(f"   Step {i+1}: done={done}, reward={reward:.2f}, "
                  f"violation_terminated={info.get('violation_terminated', False)}")

            if done and info.get('violation_terminated'):
                print(f"   ✅ Terminated due to: {info.get('termination_reason')}")
                break

        env.close()
        print("\n✅ Integration tests completed!")

    except ImportError as e:
        print(f"\n⚠️ ScienceWorld not installed, skipping integration test: {e}")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise


def test_config():
    """Test configuration options."""
    print("\n" + "=" * 60)
    print("Testing configuration options")
    print("=" * 60)

    # Test default config
    config_default = SciWorldEnvConfig()
    print(f"\nDefault config:")
    print(f"  enable_violation_termination: {config_default.enable_violation_termination}")
    print(f"  format_violation_threshold: {config_default.format_violation_threshold}")
    print(f"  invalid_action_threshold: {config_default.invalid_action_threshold}")
    print(f"  repetition_threshold: {config_default.repetition_threshold}")
    print(f"  violation_penalty: {config_default.violation_penalty}")

    # Test custom config
    config_custom = SciWorldEnvConfig(
        enable_violation_termination=False,
        format_violation_threshold=5,
        invalid_action_threshold=10,
        repetition_threshold=5,
        violation_penalty=-2.0,
    )
    print(f"\nCustom config:")
    print(f"  enable_violation_termination: {config_custom.enable_violation_termination}")
    print(f"  format_violation_threshold: {config_custom.format_violation_threshold}")
    print(f"  invalid_action_threshold: {config_custom.invalid_action_threshold}")
    print(f"  repetition_threshold: {config_custom.repetition_threshold}")
    print(f"  violation_penalty: {config_custom.violation_penalty}")

    print("\n✅ Config tests passed!")


if __name__ == "__main__":
    test_config()
    test_violation_tracker_unit()
    test_env_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
