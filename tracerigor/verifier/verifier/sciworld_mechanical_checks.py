"""
Mechanical (non-LLM) verification checks for SciWorld.

These checks provide ground-truth comparisons that don't require LLM inference,
useful for:
1. Pre-filtering steps with obvious failures before LLM evaluation
2. Providing ground-truth labels for LLM judge validation
3. Saving tokens by avoiding LLM calls for mechanically-detectable issues
"""

import re
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MechanicalCheckResult:
    """Result of a mechanical verification check."""
    passed: bool
    evidence: str
    ground_truth: Optional[Any] = None


def get_agent_location_from_object_tree(object_tree: Dict) -> Optional[str]:
    """
    Extract the agent's current location from ScienceWorld's object tree.

    The object tree is a nested dictionary where objects are contained within
    locations/containers. The agent's location is the parent container of the agent.

    Args:
        object_tree: Result from env.getObjectTree()

    Returns:
        Location name (e.g., "kitchen", "outside") or None if not found
    """
    def find_agent(obj: Dict, parent_name: str = "root") -> Optional[str]:
        if not isinstance(obj, dict):
            return None

        # Check if this object is the agent
        if obj.get("name") == "agent":
            return parent_name

        # Recursively search in contents
        contents = obj.get("contents", {})
        if isinstance(contents, dict):
            for child_key, child_obj in contents.items():
                result = find_agent(child_obj, obj.get("name", parent_name))
                if result:
                    return result
        return None

    return find_agent(object_tree)


def get_agent_inventory_from_object_tree(object_tree: Dict) -> list:
    """
    Extract the agent's inventory from ScienceWorld's object tree.

    The agent's inventory is the contents of the agent object.

    Args:
        object_tree: Result from env.getObjectTree()

    Returns:
        List of item names in inventory
    """
    def find_agent_contents(obj: Dict) -> Optional[list]:
        if not isinstance(obj, dict):
            return None

        if obj.get("name") == "agent":
            contents = obj.get("contents", {})
            if isinstance(contents, dict):
                # Filter out meta-objects like "terminal" and "inventory"
                return [
                    item.get("name", key)
                    for key, item in contents.items()
                    if item.get("name") not in ("terminal 1", "terminal 2", "inventory")
                ]
            return []

        contents = obj.get("contents", {})
        if isinstance(contents, dict):
            for child_obj in contents.values():
                result = find_agent_contents(child_obj)
                if result is not None:
                    return result
        return None

    return find_agent_contents(object_tree) or []


def extract_location_claim_from_reflection(reflection: str) -> Optional[str]:
    """
    Extract location claim from ReflAct-style reflection.

    Looks for patterns like:
    - "Location: kitchen"
    - "Location: outside"
    - "I am in the kitchen"

    Args:
        reflection: The reflection text

    Returns:
        Claimed location or None if no clear claim found
    """
    if not reflection:
        return None

    # Pattern 1: "Location: X" (most common in ReflAct)
    match = re.search(r'Location:\s*([a-zA-Z\s]+?)(?:\.|,|$)', reflection, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()

    # Pattern 2: "I am in (the) X"
    match = re.search(r'I am (?:in|at) (?:the\s+)?([a-zA-Z\s]+?)(?:\.|,|$)', reflection, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()

    return None


def check_location_grounding(
    reflection: str,
    object_tree: Dict,
) -> MechanicalCheckResult:
    """
    Mechanically verify if the reflection's location claim matches ground truth.

    Args:
        reflection: The agent's reflection text
        object_tree: Result from env.getObjectTree()

    Returns:
        MechanicalCheckResult with pass/fail and evidence
    """
    claimed_location = extract_location_claim_from_reflection(reflection)
    if claimed_location is None:
        return MechanicalCheckResult(
            passed=True,  # Can't fail if no claim made
            evidence="No explicit location claim found in reflection",
            ground_truth=None
        )

    actual_location = get_agent_location_from_object_tree(object_tree)
    if actual_location is None:
        return MechanicalCheckResult(
            passed=True,  # Can't verify without ground truth
            evidence="Could not determine ground truth location from object tree",
            ground_truth=None
        )

    # Normalize for comparison
    claimed_norm = claimed_location.lower().strip()
    actual_norm = actual_location.lower().strip()

    # Check if they match (allowing for minor variations)
    if claimed_norm == actual_norm or claimed_norm in actual_norm or actual_norm in claimed_norm:
        return MechanicalCheckResult(
            passed=True,
            evidence=f"Location claim '{claimed_location}' matches ground truth '{actual_location}'",
            ground_truth=actual_location
        )
    else:
        return MechanicalCheckResult(
            passed=False,
            evidence=f"Location mismatch: claimed '{claimed_location}', actual '{actual_location}'",
            ground_truth=actual_location
        )


def check_action_validity(
    action: str,
    valid_actions_str: str,
    observation_feedback: str = "",
) -> MechanicalCheckResult:
    """
    Mechanically verify if the action was valid/accepted by the environment.

    IMPORTANT: SciWorld has an internal syntax interpreter that accepts many actions
    not in the explicit valid_actions list. Therefore, we use two approaches:

    1. **Primary (if feedback available)**: Check if observation contains
       "No known action matches" - this is the definitive signal of invalid action.
    2. **Fallback (no feedback)**: Check against valid_actions list, but mark as
       "uncertain" since the action might still be accepted by SciWorld's interpreter.

    Args:
        action: The action taken
        valid_actions_str: String containing valid actions and objects
        observation_feedback: The observation/feedback AFTER the action was taken.
                             If this contains "No known action matches", the action was invalid.

    Returns:
        MechanicalCheckResult with pass/fail and evidence
    """
    if not action:
        return MechanicalCheckResult(
            passed=True,
            evidence="No action to validate",
            ground_truth=None
        )

    # PRIMARY CHECK: If we have observation feedback, check for explicit rejection
    if observation_feedback:
        if "no known action matches" in observation_feedback.lower():
            return MechanicalCheckResult(
                passed=False,
                evidence=f"Action '{action}' rejected by environment: 'No known action matches'",
                ground_truth="env_rejected"
            )
        else:
            # Env accepted the action (even if not in explicit valid_actions list)
            return MechanicalCheckResult(
                passed=True,
                evidence=f"Action '{action}' was accepted by environment (no rejection feedback)",
                ground_truth="env_accepted"
            )

    # FALLBACK: No feedback available - check against valid_actions list
    # Mark result as "uncertain" since SciWorld may accept actions not in list
    if not valid_actions_str:
        return MechanicalCheckResult(
            passed=True,
            evidence="Cannot verify action validity without valid_actions or observation feedback",
            ground_truth=None
        )

    # Parse valid objects from the string
    obj_match = re.search(r"(?:OBJ needs to be replaced with|Valid objects for OBJ substitution).*?\[(.*?)\]",
                          valid_actions_str, re.IGNORECASE | re.DOTALL)
    if obj_match:
        objects_str = obj_match.group(1)
        valid_objects = [o.strip().strip("'\"") for o in objects_str.split(",")]
    else:
        valid_objects = []

    # Extract action templates
    template_match = re.search(r"(?:Valid_actions|Action templates):\s*\[(.*?)\]",
                               valid_actions_str, re.IGNORECASE | re.DOTALL)
    if template_match:
        templates_str = template_match.group(1)
        templates = [t.strip().strip("'\"") for t in templates_str.split(",")]
    else:
        templates = []

    # Check if action matches any template with valid object substitution
    action_clean = action.strip().lower()

    for template in templates:
        template_clean = template.strip().lower()

        if "obj" not in template_clean:
            # Direct action match (no OBJ substitution needed)
            if action_clean == template_clean:
                return MechanicalCheckResult(
                    passed=True,
                    evidence=f"Action '{action}' matches template '{template}'",
                    ground_truth=templates
                )
        else:
            # Check OBJ substitution
            for obj in valid_objects:
                substituted = template_clean.replace("obj", obj.lower())
                if action_clean == substituted:
                    return MechanicalCheckResult(
                        passed=True,
                        evidence=f"Action '{action}' matches '{template}' with OBJ='{obj}'",
                        ground_truth=templates
                    )

    # Action not in explicit list - but may still be accepted by SciWorld
    # Mark as passed=True but with uncertain evidence
    return MechanicalCheckResult(
        passed=True,  # Don't fail without env feedback
        evidence=f"Action '{action}' not in explicit valid_actions list, but may be accepted by SciWorld interpreter (no env feedback to confirm)",
        ground_truth={"templates": templates, "objects": valid_objects, "status": "uncertain"}
    )


def check_action_repetition(
    current_action: str,
    current_observation: str,
    history: list,
) -> MechanicalCheckResult:
    """
    Check if the current action is a repetition of a recently failed action.

    A repetition is: same action taken when observation hasn't meaningfully changed.

    Args:
        current_action: The action taken this step
        current_observation: The observation from this step
        history: List of past steps with 'action' and 'observation_text' keys

    Returns:
        MechanicalCheckResult with pass/fail and evidence
    """
    if not history or len(history) < 2:
        return MechanicalCheckResult(
            passed=True,
            evidence="Not enough history to check for repetition",
            ground_truth=None
        )

    action_clean = current_action.strip().lower() if current_action else ""

    # Count consecutive repetitions of the same action
    consecutive_same = 0
    for h in reversed(history):
        past_action = (h.get("action") or "").strip().lower()
        if past_action == action_clean:
            consecutive_same += 1
        else:
            break

    if consecutive_same >= 2:
        return MechanicalCheckResult(
            passed=False,
            evidence=f"Action '{current_action}' repeated {consecutive_same + 1} times consecutively",
            ground_truth=consecutive_same + 1
        )

    return MechanicalCheckResult(
        passed=True,
        evidence="No problematic repetition detected",
        ground_truth=consecutive_same
    )


# =============================================================================
# Aggregated mechanical check for pre-filtering
# =============================================================================

@dataclass
class MechanicalPrefilterResult:
    """Aggregated result of all mechanical checks for a step."""
    location_check: MechanicalCheckResult
    action_validity_check: MechanicalCheckResult
    action_repetition_check: MechanicalCheckResult
    empty_reflection: bool = False  # True if reflection is empty/missing

    @property
    def any_failure(self) -> bool:
        """True if any mechanical check failed."""
        return (
            not self.location_check.passed or
            not self.action_validity_check.passed or
            not self.action_repetition_check.passed
        )

    @property
    def has_format_or_validity_issue(self) -> bool:
        """True if empty reflection OR invalid action (for v2.2 skip logic)."""
        return self.empty_reflection or not self.action_validity_check.passed

    @property
    def failure_summary(self) -> str:
        """Summary of failures for logging."""
        failures = []
        if self.empty_reflection:
            failures.append("Empty reflection")
        if not self.location_check.passed:
            failures.append(f"Location: {self.location_check.evidence}")
        if not self.action_validity_check.passed:
            failures.append(f"Action validity: {self.action_validity_check.evidence}")
        if not self.action_repetition_check.passed:
            failures.append(f"Repetition: {self.action_repetition_check.evidence}")
        return "; ".join(failures) if failures else "All checks passed"


def run_mechanical_prefilter(
    reflection: str,
    action: str,
    current_observation: str,
    valid_actions_str: str,
    history: list,
    object_tree: Optional[Dict] = None,
    next_observation: str = "",  # Observation AFTER action was taken (for validity check)
) -> MechanicalPrefilterResult:
    """
    Run all mechanical checks on a step.

    This can be used to:
    1. Pre-filter steps before LLM evaluation
    2. Provide ground truth for LLM judge validation
    3. Skip LLM calls for mechanically-detected issues

    Args:
        reflection: The agent's reflection text
        action: The action taken
        current_observation: The observation from this step (before action)
        valid_actions_str: String containing valid actions and objects
        history: List of past steps
        object_tree: Optional object tree for ground truth location (if available)
        next_observation: The observation AFTER the action was taken. Used to check
                         if action was rejected ("No known action matches").

    Returns:
        MechanicalPrefilterResult with all check results
    """
    # Check for empty reflection (format issue)
    empty_reflection = not reflection or not reflection.strip()

    # Location check (requires object_tree for ground truth)
    if object_tree:
        location_check = check_location_grounding(reflection, object_tree)
    else:
        location_check = MechanicalCheckResult(
            passed=True,
            evidence="No object tree provided for ground truth location check",
            ground_truth=None
        )

    # Action validity check (uses next_observation if available for definitive check)
    action_validity_check = check_action_validity(
        action,
        valid_actions_str,
        observation_feedback=next_observation
    )

    # Action repetition check
    action_repetition_check = check_action_repetition(action, current_observation, history)

    return MechanicalPrefilterResult(
        location_check=location_check,
        action_validity_check=action_validity_check,
        action_repetition_check=action_repetition_check,
        empty_reflection=empty_reflection,
    )
