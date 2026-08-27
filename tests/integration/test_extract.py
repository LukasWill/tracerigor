#!/usr/bin/env python3
"""Test extract_valid_actions function."""
import re

def extract_valid_actions(turn_text: str) -> str:
    """Extract valid actions from a turn.

    Format: Valid_actions: ['action1 OBJ', ...], OBJ needs to be replaced with one of the following objects: ['obj1', ...]
    Returns the full string including both the action templates and the object substitution list.
    """
    # Match the full valid_actions string including the OBJ substitution list
    full_pattern = r"Valid_actions:\s*(\[.*?\]),\s*OBJ needs to be replaced with one of the following objects:\s*(\[.*?\])"
    match = re.search(full_pattern, turn_text, re.DOTALL)
    if match:
        action_templates = match.group(1)
        object_list = match.group(2)
        return f"Action templates: {action_templates}, Valid objects for OBJ substitution: {object_list}"

    # Fallback: try to match just the action list
    simple_match = re.search(r"Valid_actions:\s*(\[.*?\])", turn_text, re.DOTALL)
    if simple_match:
        return f"Action templates: {simple_match.group(1)}"
    return "N/A"

# Test with sample text from actual data
test_text = """Valid_actions: ['activate OBJ', 'close OBJ', 'deactivate OBJ', 'dunk OBJ in OBJ', 'eat OBJ', 'flush OBJ', 'focus on OBJ', 'go OBJ', 'inventory', 'look around', 'look at OBJ', 'look in OBJ', 'mix OBJ', 'move OBJ to OBJ', 'open OBJ', 'pick up OBJ', 'pour OBJ in OBJ', 'put down OBJ', 'read OBJ', 'reset task', 'task', 'teleport OBJ', 'use OBJ on OBJ', 'wait', 'wait1'], OBJ needs to be replaced with one of the following objects: ['agent', 'air', 'art studio', 'art studio door', 'bedroom', 'bedroom door', 'door to greenhouse', 'door to kitchen', 'door to living room', 'door to workshop', 'greenhouse', 'hallway', 'kitchen', 'living room', 'picture', 'workshop']
 example: <action>focus on door</action>"""

result = extract_valid_actions(test_text)
print("Result:")
print(result)
print()
print("Length:", len(result))
