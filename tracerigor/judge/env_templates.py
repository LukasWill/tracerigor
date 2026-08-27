"""
Environment-specific prompt template adapters.

Registers verifier prompt templates into the judge's template registry.
Templates are imported directly from verifier/prompt/ where
possible, ensuring that the proven prompt text is reused verbatim.

Each task registers:
  - "universal": All rubrics in one call

To add a new environment:
  1. Create prompt templates (system + user) for the env
  2. Add a register_<env>() function below
  3. Call it from register_all_templates()
"""
from __future__ import annotations

from tracerigor.judge.prompt import register_template


# ---------------------------------------------------------------------------
# SciWorld — inherited verbatim from verifier/prompt/sciworld.py
# ---------------------------------------------------------------------------

# The full text is kept here (not imported) so the judge package stays
# self-contained — but the wording is *identical* to the verifier originals.

_SCIWORLD_UNIVERSAL_SYSTEM = """You are a strict SciWorld verifier. Judge only the CURRENT step, but use history to catch contradictions.

SciWorld is a text-based virtual environment for elementary science tasks. The agent uses ReflAct-style reasoning:
<reflection>...</reflection><action>...</action>

The reflection should describe:
- Current location
- Inventory state
- Task goal
- Current progress
- Next step reasoning

OBSERVATION TYPES (critical for grounding):
1. **Room description**: "This room is called X. In it, you see: ..." → Location is X
2. **Movement confirmation**: "You move through the door to X." → Location is now X (does NOT list inventory or other objects)
3. **Action result**: "Inside the table is: nothing" or "I'm not sure how to use X." → Location is UNCHANGED from prior step
4. **Error message**: "No known action matches that input." → Location is UNCHANGED (action failed)

Your tasks:
1) **Observation Grounding** — Does the reflection's LOCATION claim match the current or contextually-inferred observation?
   - ONLY verify: current location claim
   - For observation type 2 (movement confirmation): only check if location matches the destination
   - For observation types 3 & 4: location should match the LAST VALID location inferred from movement history
   - Do NOT penalize: claims about past actions, claims about objects/entities visible from prior observations
   - ONLY penalize: clearly WRONG location claims that contradict the observable/inferable location

2) **Action Coherence** — Does the <action> logically follow from the <reflection>?
   - The action should match what the reflection says is the "next step"
   - Or the action directly implements or reasonably interprets the reflection's stated intent
   - Penalize only CLEAR contradictions between reflection intent and action taken

   Do NOT focus on action validity (that is checked mechanically elsewhere). Focus on:
   - Is the action what the reflection says the agent should do next?
   - Is there logical coherence between the reasoning and the chosen action?

3) **Temporal Consistency** — Does the <reflection> update beliefs appropriately given the HISTORY?
   CORE PRINCIPLE: Good reasoning updates beliefs/plans when evidence changes.

   Check for these FAILURES:
   a) **Belief not updated after action feedback**: Does reflection claim something that recent observations (current + history) contradict?
      Check BOTH location claims AND action outcome claims (object acquisition, state changes, contents revealed, etc.):
      - After a SUCCESSFUL action: reflection should acknowledge the actual outcome
        - Movement success ("You move to X") → location updated to X
        - Pick up success ("You move X to inventory") → object now acquired
        - Look/open success → contents revealed should be acknowledged
      - After a FAILED action ("No known action matches" or similar): reflection should NOT claim the intended outcome
        - Movement failed → location remains LAST CONFIRMED, not intended destination
        - Pick up failed → object NOT acquired (don't claim "I have X")
        - Interaction failed → state unchanged (don't claim the intended effect happened)
      - Example FAILURE: Observation shows "You move the flour to inventory", but reflection says "flour is not movable"
      - Example FAILURE: Move to kitchen failed with "No known action matches", but reflection claims "Location: kitchen"
      - Example FAILURE: "pick up flour" failed, but reflection claims "I have acquired the flour"
   b) **Stuck behavior**: Does reflection fail to adapt after repeated negative feedback?
      - Same or similar action failed 2+ times in history but reflection does NOT explicitly acknowledge the failure AND continues the same/similar approach without meaningful adaptation
      - NOTE: Implicit adaptation alone (silently trying something different) is NOT sufficient if the agent also makes incorrect claims about location or action outcomes
   c) **False claims about past actions**:
      - Reflection claims "I have checked/inspected/looked at/in X" but no such action appears in history
      - "Seeing" an object in room description ≠ "checking" it (checking requires explicit look at/in action)
      - Example FAILURE: "blast furnace has been checked" but history only shows "look around"
   d) **Contradiction with prior reasoning**:
      - Current reflection contradicts the agent's own prior reasoning from recent steps
      - Example FAILURE: Prior reflection concluded "object A is not here", current reflection claims "I found object A here" without new evidence

   Answer YES if: Reflection appropriately acknowledges recent feedback (success/failure) and updates beliefs/plans accordingly
   Answer NO if: Any failure (a, b, c, d) is present

Return a strict JSON object with EXACTLY these keys:
{
  "observation_grounding": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"},
  "action_coherence": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"},
  "temporal_consistency": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"}
}

Be concise and specific in evidence. Do not add extra keys.
"""

# Ground truth section template (inserted when ground truth available from replay)
_SCIWORLD_GROUND_TRUTH_SECTION = """
[Ground Truth from Environment — USE THIS for location verification]
Agent's actual location: {ground_truth_location}
Agent's inventory: {ground_truth_inventory_str}
"""

_SCIWORLD_UNIVERSAL_USER = """Task: {task_description}

Step index: {episode_step}

History (most recent steps, if available):
{history_str}

Current Observation:
{observation_content}
{ground_truth_section}
Reflection to verify:
{reasoning_tokens}

Action taken:
{action_tokens}

Valid actions for this step:
{valid_actions}
"""


def register_sciworld_templates() -> None:
    """Register SciWorld judge prompt templates (universal only)."""
    register_template(
        task_name="sciworld",
        rubric="universal",
        system_prompt=_SCIWORLD_UNIVERSAL_SYSTEM,
        user_prompt=_SCIWORLD_UNIVERSAL_USER,
    )


# ---------------------------------------------------------------------------
# Sokoban — refined universal prompt covering all 3 rubrics
# ---------------------------------------------------------------------------

# Import the V2 prompts from the canonical source (verifier/prompt/sokoban.py)
# to keep judge and verifier in sync.
from tracerigor.verifier.prompt.sokoban import (
    SOKOBAN_UNIVERSAL_V2_SYSTEM as _SOKOBAN_UNIVERSAL_SYSTEM,
    SOKOBAN_GROUND_TRUTH_SECTION_V2 as _SOKOBAN_GROUND_TRUTH_SECTION,
    SOKOBAN_UNIVERSAL_V2_USER as _SOKOBAN_UNIVERSAL_USER,
)


def register_sokoban_templates() -> None:
    """Register Sokoban judge prompt templates (universal V2 only)."""
    register_template(
        task_name="sokoban",
        rubric="universal",
        system_prompt=_SOKOBAN_UNIVERSAL_SYSTEM,
        user_prompt=_SOKOBAN_UNIVERSAL_USER,
    )


# ---------------------------------------------------------------------------
# Ground truth builders (env-specific formatters for replay data)
# ---------------------------------------------------------------------------


def build_sciworld_ground_truth(ground_truth_state: dict) -> str:
    """Format SciWorld ground truth state into a prompt section."""
    location = ground_truth_state.get(
        "location", ground_truth_state.get("ground_truth_location", "unknown")
    )
    inventory = ground_truth_state.get(
        "inventory", ground_truth_state.get("ground_truth_inventory", [])
    )
    inv_str = ", ".join(inventory) if inventory else "empty"
    return _SCIWORLD_GROUND_TRUTH_SECTION.format(
        ground_truth_location=location,
        ground_truth_inventory_str=inv_str,
    )


def build_sokoban_ground_truth(ground_truth_state: dict) -> str:
    """Format Sokoban ground truth state into a prompt section."""
    state_text = ground_truth_state.get("state_text", "")
    if not state_text:
        state_text = "\n".join(ground_truth_state.get("state_sentences", []))
    if not state_text:
        return ""
    return _SOKOBAN_GROUND_TRUTH_SECTION.format(ground_truth_state_text=state_text)


# Dispatch table for ground truth builders
_GROUND_TRUTH_BUILDERS = {
    "sciworld": build_sciworld_ground_truth,
    "sokoban": build_sokoban_ground_truth,
}


def build_ground_truth_section(task_name: str, ground_truth_state: dict) -> str:
    """Build the ground truth section for a given env, returns empty string if no builder."""
    builder = _GROUND_TRUTH_BUILDERS.get(task_name)
    if builder and ground_truth_state:
        return builder(ground_truth_state)
    return ""


# ---------------------------------------------------------------------------
# Registration entrypoint
# ---------------------------------------------------------------------------

def register_all_templates() -> None:
    """Register all env-specific templates. Call once at startup."""
    register_sciworld_templates()
    register_sokoban_templates()
