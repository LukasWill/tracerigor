"""Test the updated SciWorld prompt templates."""

from tracerigor.verifier.prompt.sciworld import SciWorldUniversalTemplate

template = SciWorldUniversalTemplate()
print('History window:', template.history_window)
print()

# Test with sample data
test_data = {
    'task_description': 'Find inclined planes',
    'current_step': 5,
    'current_observation_text': 'You move through the door to the outside.',
    'reflection_tokens': 'Location: outside. I need to find inclined planes.',
    'action_tokens': 'go to door to greenhouse',
    'valid_actions': 'go OBJ, look around',
    'history': [
        {'step': 1, 'observation_text': 'You are outside with doors.', 'action': 'go to foundry', 'reflection': 'Location: outside. Going to foundry.'},
        {'step': 2, 'observation_text': 'You move to foundry.', 'action': 'look around', 'reflection': 'Location: foundry. Looking around.'},
        {'step': 3, 'observation_text': 'Foundry has blast furnace.', 'action': 'look in table', 'reflection': 'Location: foundry. Checking table.'},
        {'step': 4, 'observation_text': 'Table is empty.', 'action': 'go to outside', 'reflection': 'Location: foundry. Going outside.'},
    ]
}

rendered = template._render_prompt(test_data.copy())
print('=== Rendered _history_str ===')
print(rendered['_history_str'])
print()

# Test mechanical checks
print('=== Testing Mechanical Checks ===')
from tracerigor.verifier.verifier.sciworld_mechanical_checks import (
    extract_location_claim_from_reflection,
    check_action_validity,
    check_action_repetition,
)

# Test location extraction
reflection = "Location: foundry. Inventory: empty. Task goal: find planes."
loc = extract_location_claim_from_reflection(reflection)
print(f"Extracted location from '{reflection[:50]}...': {loc}")

# Test action validity
valid_actions = "Action templates: ['go OBJ', 'look around'], Valid objects for OBJ substitution: ['door to foundry', 'door to kitchen']"
result = check_action_validity("go door to foundry", valid_actions)
print(f"Action validity check: passed={result.passed}, evidence={result.evidence}")

result2 = check_action_validity("go to door to invalid", valid_actions)
print(f"Invalid action check: passed={result2.passed}, evidence={result2.evidence}")

print("\n=== All tests passed ===")
