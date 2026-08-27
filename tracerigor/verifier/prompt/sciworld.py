"""
SciWorld LLM Judge Prompt Templates

Evaluates Chain-of-Thought (CoT) quality for text-based SciWorld environment.
Designed for offline validation samples with ReflAct-style reasoning:
    <reflection>...</reflection><action>...</action>

Three evaluation dimensions:
1. Groundedness/Veridicality: Does the reflection accurately describe the current observation?
2. Action Coherence: Does the action logically follow from the stated reflection?
3. Temporal Consistency: Is the reflection consistent with recent history (memory)?
"""

from tracerigor.verifier.prompt.verifier_template_base import VerifierTemplate


# ============================================================================
# 1) Universal SciWorld Verifier (All 3 metrics in one call)
# ============================================================================
SCIWORLD_UNIVERSAL_SYSTEM = """You are a strict SciWorld verifier. Judge only the CURRENT step, but use history to catch contradictions.

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

# User prompt template - supports optional ground truth location from trajectory replay
SCIWORLD_UNIVERSAL_USER = """Task: {task_description}

Step index: {current_step}

History (most recent steps, if available):
{_history_str}

Current Observation:
{current_observation_text}
{_ground_truth_section}
Reflection to verify:
{reflection_tokens}

Action taken:
{action_tokens}

Valid actions for this step:
{valid_actions}
"""

# Ground truth section template (inserted when ground truth available from replay)
GROUND_TRUTH_SECTION_TEMPLATE = """
[Ground Truth from Environment - USE THIS for location verification]
Agent's actual location: {ground_truth_location}
Agent's inventory: {ground_truth_inventory_str}
"""


class SciWorldUniversalTemplate(VerifierTemplate):
    """Universal SciWorld verifier: grounding, action coherence, temporal consistency."""

    # Default history window size to match rollout's window_size
    DEFAULT_HISTORY_WINDOW = 5

    def __init__(self, history_window: int = None):
        super().__init__(
            template_id="sciworld.universal",
            description="Universal SciWorld verifier: grounding, action coherence, temporal consistency.",
            required_keys=("current_observation_text", "reflection_tokens", "action_tokens"),
            system_prompt=SCIWORLD_UNIVERSAL_SYSTEM,
            user_prompt=SCIWORLD_UNIVERSAL_USER,
        )
        self.history_window = history_window or self.DEFAULT_HISTORY_WINDOW

    def _render_prompt(self, data: dict) -> dict:
        """Override to handle SciWorld-specific data transformations."""
        # Default values for optional fields
        data.setdefault("task_description", "N/A")
        data.setdefault("current_step", "N/A")
        data.setdefault("valid_actions", "N/A")

        # Handle ground truth section (from trajectory replay)
        ground_truth_location = data.get("ground_truth_location")
        ground_truth_inventory = data.get("ground_truth_inventory", [])

        if ground_truth_location:
            # Format ground truth section
            inventory_str = ", ".join(ground_truth_inventory) if ground_truth_inventory else "(empty)"
            data["_ground_truth_section"] = GROUND_TRUTH_SECTION_TEMPLATE.format(
                ground_truth_location=ground_truth_location,
                ground_truth_inventory_str=inventory_str,
            )
        else:
            data["_ground_truth_section"] = ""  # No ground truth available

        # Handle history formatting - include reflection/reasoning CoT
        history = data.get("history", [])
        if isinstance(history, list) and history:
            # Format history entries with observation, action, AND reflection
            formatted_history = []
            for i, h in enumerate(history):
                step = h.get("step", i + 1)
                obs = h.get("observation_text", "N/A")
                action = h.get("action", "N/A")
                reflection = h.get("reflection", "")

                # Do NOT truncate: judge must see the same context the agent saw
                # during generation for accurate temporal consistency assessment.
                entry = f"[Step {step}]\nObservation: {obs}\nAction: {action}"
                if reflection:
                    entry += f"\nReasoning: {reflection}"
                formatted_history.append(entry)

            # Use configurable history window (default 5 to match rollout)
            data["_history_str"] = "\n\n".join(formatted_history[-self.history_window:])
        else:
            data["_history_str"] = "(No prior history)"

        return data


# ============================================================================
# 2) Observation Grounding Only (Binary)
# ============================================================================
SCIWORLD_GROUNDING_SYSTEM = """You verify whether the agent's reflection's LOCATION claim is correct given the observation.

OBSERVATION TYPES (learn to recognize them):
1. **Room description**: "This room is called X. In it, you see: ..." → Location is X
2. **Movement confirmation**: "You move through the door to X." → Location is now X (does NOT list inventory or other objects)
3. **Action result**: "Inside the table is: nothing" or "I'm not sure how to use X." → Location is UNCHANGED from context
4. **Error message**: "No known action matches that input." → Location is UNCHANGED (invalid action failed)

For observation types 2, 3, and 4: the observation does NOT explicitly list inventory, doors, or room contents.

RULES:
- ONLY verify: Does the reflection's LOCATION claim match the observation?
- For type 1: Check if location matches the room name in observation
- For type 2: Check if location matches the movement destination
- For types 3 & 4: Verify location claims against history, not current observation
- Do NOT penalize: inventory claims, claims about past actions, claims about doors/objects from prior observations

Answer YES if: Location claim matches the observation or is plausible for types 3/4
Answer NO if: Location claim clearly contradicts the observation (e.g., observation says "kitchen" but reflection says "outside")

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>
"""

SCIWORLD_GROUNDING_USER = """Current Observation:
{current_observation_text}

Reflection to verify:
{reflection_tokens}
"""


class SciWorldGroundingTemplate(VerifierTemplate):
    """Binary observation grounding check for SciWorld."""
    def __init__(self):
        super().__init__(
            template_id="sciworld.grounding",
            description="Binary observation grounding (text-only).",
            required_keys=("current_observation_text", "reflection_tokens"),
            system_prompt=SCIWORLD_GROUNDING_SYSTEM,
            user_prompt=SCIWORLD_GROUNDING_USER,
        )


# ============================================================================
# 3) Action Coherence Only (Binary)
# ============================================================================
SCIWORLD_ACTION_COHERENCE_SYSTEM = """You evaluate whether the agent's action logically follows from its stated reflection.

The agent uses ReflAct format:
<reflection>...</reflection><action>...</action>

FOCUS: Is the action a logical outcome of the agent's stated reasoning?

Check:
1) Does the reflection state a clear "next step" or intent?
2) Does the action implement or reasonably interpret that stated intent?
3) Is there any contradiction between what the reflection says and what action is taken?

Do NOT focus on whether the action is valid/executable (that is checked mechanically elsewhere).
Focus only on logical coherence between the reasoning and the chosen action.

Answer YES if:
- The action directly implements what the reflection states as the next step
- OR the action is a reasonable interpretation of the reflection's intent

Answer NO if:
- The action contradicts the reflection's stated intent
- The action is completely unrelated to the reflection's reasoning
- The reflection says one thing but the action does something else

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>
"""

SCIWORLD_ACTION_COHERENCE_USER = """Reflection:
{reflection_tokens}

Action taken:
{action_tokens}
"""


class SciWorldActionCoherenceTemplate(VerifierTemplate):
    """Binary action-reflection consistency check for SciWorld."""
    def __init__(self):
        super().__init__(
            template_id="sciworld.action_coherence",
            description="Binary action-reflection consistency.",
            required_keys=("reflection_tokens", "action_tokens"),
            system_prompt=SCIWORLD_ACTION_COHERENCE_SYSTEM,
            user_prompt=SCIWORLD_ACTION_COHERENCE_USER,
        )

    def _render_prompt(self, data: dict) -> dict:
        data.setdefault("valid_actions", "N/A")
        return data


# ============================================================================
# 4) Temporal Consistency Only (Binary)
# ============================================================================
SCIWORLD_TEMPORAL_CONSISTENCY_SYSTEM = """You evaluate whether the agent's CURRENT reflection appropriately updates beliefs given the HISTORY.

CORE PRINCIPLE: Good reasoning updates beliefs/plans when evidence changes. Bad reasoning clings to outdated beliefs or makes false claims.

Check for these FAILURES (answer NO if any present):

a) **Belief not updated after movement feedback**:
   - After a successful move ("You move through the door to X"), reflection should update location to X
   - After a failed move ("No known action matches"), location should remain the LAST CONFIRMED location
   - Example FAILURE: Move to kitchen failed, but reflection claims "Location: kitchen"

b) **Stuck behavior without explicit acknowledgment**:
   - Same or similar action failed 2+ times in history
   - Reflection does NOT explicitly mention the failures AND still attempts a similar approach
   - NOTE: Implicit adaptation alone (trying something different) is NOT sufficient if the agent also makes incorrect claims
   - Example FAILURE: 3 consecutive "wait1" actions with identical observations and no plan change

c) **False claims about past actions**:
   - Reflection claims "I have checked/inspected/looked at X" but no such action appears in history
   - Note: "Seeing" an object in a room description is NOT the same as "checking" it
   - Example FAILURE: Reflection says "blast furnace has been checked" but history shows only "look around"

d) **Contradiction with prior reasoning**:
   - Current reflection contradicts the agent's own prior reasoning from recent steps
   - Example FAILURE: Prior reflection concluded "object A is not here", current claims "I found object A here" without new evidence

Answer YES if:
- Reflection acknowledges recent feedback (success/failure) appropriately
- Beliefs are updated to match current evidence
- Plans adapt when prior attempts fail
- OR this is an early step with no contradictory history

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>
"""

SCIWORLD_TEMPORAL_CONSISTENCY_USER = """Recent History (last {history_length} steps):
{_history_str}

Current Reflection to verify:
{reflection_tokens}

Current step index: {current_step}
"""


class SciWorldTemporalConsistencyTemplate(VerifierTemplate):
    """Binary temporal/history consistency check for SciWorld."""
    def __init__(self):
        super().__init__(
            template_id="sciworld.temporal_consistency",
            description="Binary temporal/history consistency.",
            required_keys=("reflection_tokens",),
            system_prompt=SCIWORLD_TEMPORAL_CONSISTENCY_SYSTEM,
            user_prompt=SCIWORLD_TEMPORAL_CONSISTENCY_USER,
        )

    def _render_prompt(self, data: dict) -> dict:
        # Handle history formatting
        history = data.get("history", [])
        data["history_length"] = len(history) if isinstance(history, list) else 0

        if isinstance(history, list) and history:
            formatted_history = []
            for i, h in enumerate(history):
                step = h.get("step", i + 1)
                obs = h.get("observation_text", "N/A")
                action = h.get("action", "N/A")
                # Do NOT truncate: judge must see the same context the agent saw
                formatted_history.append(f"[Step {step}]\nObservation: {obs}\nAction: {action}")
            data["_history_str"] = "\n\n".join(formatted_history)
        else:
            data["_history_str"] = "(No prior history available)"

        data.setdefault("current_step", "N/A")
        return data


# ============================================================================
# Template Registry
# ============================================================================
SCIWORLD_TEMPLATES = {
    "universal": SciWorldUniversalTemplate,
    "grounding": SciWorldGroundingTemplate,
    "action_coherence": SciWorldActionCoherenceTemplate,
    "temporal_consistency": SciWorldTemporalConsistencyTemplate,
}


def get_sciworld_template(rubric: str) -> VerifierTemplate:
    """Get a SciWorld template by rubric name."""
    if rubric not in SCIWORLD_TEMPLATES:
        raise ValueError(f"Unknown SciWorld rubric: {rubric}. Available: {list(SCIWORLD_TEMPLATES.keys())}")
    return SCIWORLD_TEMPLATES[rubric]()
