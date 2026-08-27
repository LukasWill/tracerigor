#!/usr/bin/env python3
"""
Test script for SciWorld LLM Judge prompts.

This script demonstrates the prompt generation without requiring API access.
Run this to see what prompts would be sent to the LLM judge.

Usage:
    python test_sciworld_prompts.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tracerigor.verifier.prompt.sciworld import (
    SciWorldUniversalTemplate,
    SciWorldGroundingTemplate,
    SciWorldActionCoherenceTemplate,
    SciWorldTemporalConsistencyTemplate,
    get_sciworld_template,
)


def test_universal_template():
    """Test the universal (combined) template."""
    print("=" * 60)
    print("Testing SciWorld Universal Template")
    print("=" * 60)

    template = SciWorldUniversalTemplate()

    # Sample data from a SciWorld trajectory
    data = {
        "task_description": "Your task is to determine which of the two inclined planes (plastic, bronze) has the most friction.",
        "current_step": 5,
        "current_observation_text": """This room is called the foundry. In it, you see:
    the agent
    a substance called air
    a blast furnace, which is turned off. The blast furnace door is closed.
    a sink, which is turned off. In the sink is: nothing.
    a table. On the table is: nothing.
You also see:
    A door to the outside (that is open)""",
        "reflection_tokens": "Location: foundry. Inventory: empty. Task goal: determine which inclined plane has more friction. I need to find the inclined planes to conduct the experiment. The foundry has a blast furnace, sink, and table. I should look for the inclined planes elsewhere.",
        "action_tokens": "go to door to outside",
        "valid_actions": "['activate OBJ', 'close OBJ', 'go OBJ', 'look around', 'look at OBJ', 'look in OBJ']",
        "history": [
            {
                "step": 3,
                "observation_text": "This outside location is called the outside. Here you see: the agent, air, axe, fire pit...",
                "action": "go to door to foundry",
            },
            {
                "step": 4,
                "observation_text": "You move through the door to the foundry.",
                "action": "look around",
            },
        ],
    }

    messages = template.build_messages(data)

    print("\n[SYSTEM PROMPT]:")
    print("-" * 40)
    print(messages[0]["content"][:1000])
    print("..." if len(messages[0]["content"]) > 1000 else "")

    print("\n[USER PROMPT]:")
    print("-" * 40)
    print(messages[1]["content"])

    return True


def test_grounding_template():
    """Test the observation grounding template."""
    print("\n" + "=" * 60)
    print("Testing SciWorld Grounding Template")
    print("=" * 60)

    template = SciWorldGroundingTemplate()

    data = {
        "current_observation_text": """This room is called the kitchen. In it, you see:
    the agent
    a chair. On the chair is: nothing.
    a counter. On the counter is: a bowl (containing a banana, an orange), a drawer.
    a sink, which is turned off. In the sink is: nothing.
    a stove, which is turned off.
You also see:
    A door to the hallway (that is open)""",
        "reflection_tokens": "Location: kitchen. Inventory: empty. I see a bowl with fruits on the counter, and a closed drawer. The stove is off. I should check the drawer for the recipe.",
    }

    messages = template.build_messages(data)

    print("\n[SYSTEM PROMPT]:")
    print("-" * 40)
    print(messages[0]["content"])

    print("\n[USER PROMPT]:")
    print("-" * 40)
    print(messages[1]["content"])

    return True


def test_action_coherence_template():
    """Test the action coherence template."""
    print("\n" + "=" * 60)
    print("Testing SciWorld Action Coherence Template")
    print("=" * 60)

    template = SciWorldActionCoherenceTemplate()

    data = {
        "reflection_tokens": "Location: kitchen. Inventory: flour. Task goal: make dough. I have the flour and need water. The sink can provide water. I should activate the sink to get water.",
        "action_tokens": "activate sink",
        "valid_actions": "['activate OBJ', 'pick up OBJ', 'open OBJ', 'go OBJ']",
    }

    messages = template.build_messages(data)

    print("\n[SYSTEM PROMPT]:")
    print("-" * 40)
    print(messages[0]["content"])

    print("\n[USER PROMPT]:")
    print("-" * 40)
    print(messages[1]["content"])

    # Test contradictory case
    print("\n[Testing Contradictory Case]:")
    print("-" * 40)

    data_bad = {
        "reflection_tokens": "Location: kitchen. Task goal: make dough. I should go to the pantry to find flour.",
        "action_tokens": "look around",  # Contradicts the stated intent to go to pantry
        "valid_actions": "['go OBJ', 'look around', 'pick up OBJ']",
    }

    messages_bad = template.build_messages(data_bad)
    print("Reflection says: 'go to pantry'")
    print("Action taken: 'look around'")
    print("Expected verdict: NO (contradiction)")

    return True


def test_temporal_consistency_template():
    """Test the temporal consistency template."""
    print("\n" + "=" * 60)
    print("Testing SciWorld Temporal Consistency Template")
    print("=" * 60)

    template = SciWorldTemporalConsistencyTemplate()

    data = {
        "current_step": 8,
        "reflection_tokens": "Location: greenhouse. Inventory: empty. Task goal: find inclined planes. I have already checked the foundry and outside areas. The greenhouse is a new location I haven't explored yet.",
        "history": [
            {
                "step": 5,
                "observation_text": "This outside location is called the outside. Here you see: the agent, air, axe...",
                "action": "go to door to foundry",
            },
            {
                "step": 6,
                "observation_text": "You move through the door to the foundry.",
                "action": "look around",
            },
            {
                "step": 7,
                "observation_text": "This room is called the foundry. In it, you see: blast furnace, sink, table...",
                "action": "go to door to greenhouse",
            },
        ],
    }

    messages = template.build_messages(data)

    print("\n[SYSTEM PROMPT]:")
    print("-" * 40)
    print(messages[0]["content"])

    print("\n[USER PROMPT]:")
    print("-" * 40)
    print(messages[1]["content"])

    # Test inconsistent case
    print("\n[Testing Inconsistent Case]:")
    print("-" * 40)

    data_bad = {
        "current_step": 8,
        "reflection_tokens": "Location: kitchen. I have been exploring the kitchen area. The foundry was my starting point.",  # Wrong - history shows we're in greenhouse
        "history": data["history"],
    }

    print("History shows: outside -> foundry -> greenhouse")
    print("Reflection claims: in kitchen, started at foundry")
    print("Expected verdict: NO (location contradiction)")

    return True


def test_template_registry():
    """Test the template registry function."""
    print("\n" + "=" * 60)
    print("Testing Template Registry")
    print("=" * 60)

    rubrics = ["universal", "grounding", "action_coherence", "temporal_consistency"]

    for rubric in rubrics:
        template = get_sciworld_template(rubric)
        print(f"  ✓ {rubric}: {template.template_id}")

    # Test invalid rubric
    try:
        get_sciworld_template("invalid_rubric")
        print("  ✗ Should have raised ValueError for invalid rubric")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError for invalid rubric: {e}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SciWorld LLM Judge Prompt Tests")
    print("=" * 60)

    tests = [
        ("Universal Template", test_universal_template),
        ("Grounding Template", test_grounding_template),
        ("Action Coherence Template", test_action_coherence_template),
        ("Temporal Consistency Template", test_temporal_consistency_template),
        ("Template Registry", test_template_registry),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
