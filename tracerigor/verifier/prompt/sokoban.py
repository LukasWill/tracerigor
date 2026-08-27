
# from .verifier_template_base import VerifierTemplate
# from ..utils import registry
from tracerigor.verifier.prompt.verifier_template_base import VerifierTemplate
from tracerigor.verifier.utils import registry


# =============================================================================
# Refined Universal Sokoban Verifier (aligned with SciWorld 3-rubric schema)
# =============================================================================
# Uses the same JSON output keys as SciWorld (observation_grounding,
# action_coherence, temporal_consistency) so the _sciworld_universal_score
# parser can be reused without modification.

SOKOBAN_UNIVERSAL_V2_SYSTEM = """You are a strict Sokoban verifier. Judge only the CURRENT step, but use history to catch contradictions.

Sokoban is a spatial puzzle: push boxes onto targets in a grid.
The agent observes the board as an **image** each turn.  Sometimes you are also
given a separate text description of the ground-truth spatial state.

Visual elements:
  Player — small green alien figure.
  Box    — yellow crate marked with an orange "X".
  Target — black tile with a red diamond outline.
Admissible actions: Up, Down, Left, Right.  The agent may issue 1–3 comma-separated actions per turn.

Orientation (all relative terms refer to the image / grid axes):
  "above" = lower row index (top of image)   "below" = higher row index
  "left"  = lower column index               "right" = higher column index

The agent reasons inside `<reflection>...</reflection>` before choosing
`<action>...</action>`.  Common patterns in the reflection:
  - Position claims: "I am to the right of the box, same row."
  - Spatial relations: "The box is to the right of the target, same row."
  - Progress / plan: "I need to push the box left."

Your tasks:

1) **Observation Grounding** — Does the reflection correctly describe the
   CURRENT board state?
   - First inspect the CURRENT observation image.
   - If a ground-truth spatial state is provided below the observation, treat
     it as the authoritative reference for precise position verification.
   - If no ground-truth spatial state is provided, judge directly from the
     image and only penalise CLEAR contradictions.
   - For exact spatial claims such as **same row**, **same column**,
     **directly above/below**, or **left/right of**, answer YES only if the
     image clearly supports all such claims. Do not ignore one contradictory
     claim just because other parts of the reflection are correct.
   - Common errors to catch:
     a) **Incorrect direction**: e.g., "box is above the target" when it is
        actually below.
     b) **Incorrect axis alignment**: e.g., "same row" when they are in the
        same column, or vice versa.
     c) **Swapped reference frames**: describing the box relative to the player
        but getting the direction reversed.
   - Do NOT penalise omissions — only penalise claims that CONTRADICT the
     ground truth.

2) **Action Coherence** — Does the chosen action logically follow from the
   stated reflection?
   - The reflection should mention or clearly imply the direction of intended
     movement or push, and the action must match that stated intent.
   - Judge this against the reflection AS WRITTEN, not against whether the
     reflection is factually correct. If the reflection is grounded incorrectly
     but the action still follows that mistaken plan, Action Coherence should
     be YES and Observation Grounding should carry the factual penalty.
   - Use the reflection's own described geometry as the reference. If the
    reflection's stated positions imply that a different move would be needed,
    or that the proposed move cannot produce the claimed push (i.e., physically
    impossible under Sokoban mechanics), answer NO even if the action repeats
    a direction word from the plan sentence.
   - If the reflection's spatial facts and its explicit plan sentence conflict,
    treat the reflection itself as internally inconsistent and answer NO.
      Example: if the reflection says the player is to the right of the box and
      plans to push the box right, a Right action is incoherent because pushing
      right would require the player to be left of the box.
   - Common errors to catch:
     a) **Direction mismatch**: reflection says "push box left" but action is
        Right.
     b) **Multi-action incoherence**: the sequence of comma-separated actions
        is internally contradictory (e.g., Left,Right cancels out without
        reason).
     c) **Missing justification**: reflection does not mention or imply any
        direction; action appears arbitrary.
   - Do NOT judge whether the action is *optimal* or *effective* — only
     whether it is consistent with the stated reasoning.

3) **Temporal Consistency** — Does the CURRENT reflection UPDATE BELIEFS
   appropriately given the HISTORY (if any)?

   CORE PRINCIPLE: Good reasoning updates beliefs/plans when evidence
   changes — a successful action shifts the position, a blocked action
   demands re-planning, a new observation supersedes the old one.
   Reflections that stay STATIC in the face of new feedback are temporally
   inconsistent even when they make no explicit contradictory claim —
   passivity in the presence of new evidence is itself a failure mode.

   History may include prior reasoning, proposed actions, executed actions,
   and optional board-state text. Treat executed actions as the authoritative
   record of what actually happened in that turn.
   If history lacks board-state text, reason conservatively from executed
   actions, the current image, and explicit prior claims. Do NOT assume hidden
   obstacles or exact intermediate states unless they follow directly.
   If there is no prior history, answer YES unless the current reflection
   invents prior actions, outcomes, or progress that never happened.

   Check for these failures (answer NO if ANY is present):

   a) **Position / action-outcome claims contradict history**:
      - After a SUCCESSFUL move: position should update accordingly.
        e.g., moved Left → player column should decrease by 1.
      - After a PARTIALLY EXECUTED multi-action turn: only the executed
        sub-actions happened. Do NOT assume later proposed sub-actions also
        succeeded.
      - After a FAILED/BLOCKED move (wall, unmovable box blocked by wall): position should
        remain unchanged — do NOT claim the intended destination.
        e.g., action was Left but player hit a wall → reflection should NOT
        claim a new position to the left.
      - Example FAILURE: Agent moved Left last turn, but current reflection
        still describes the same position as before the move.
      - Example FAILURE: Agent proposed Left,Left but only one Left executed,
        while the current reflection assumes a two-cell shift.

   b) **Belief not updated after action feedback (STATIC / STALE reasoning)**:
      - The current reflection's position description, plan, or
        "needs to push X" statement is essentially UNCHANGED from the
        immediately prior reflection, even though the prior action was
        executed (fully or partially), blocked, or produced a new
        observation that should have shifted the description.
      - This is a failure EVEN WHEN the current reflection makes no
        explicit contradictory claim. The failure is PASSIVITY, not
        contradiction: the agent is ignoring that anything happened.
      - For templated reflections ("Position: ... / Task goal: ... /
        Progress: ..."), an unchanged "Position:" line after any executed
        action is a strong STATIC-reasoning signal.
      - Exception: if the prior action was fully invalid (no sub-action
        executed AND no board-state change) AND the reflection explicitly
        acknowledges this (e.g., "my last attempt was blocked"), updating
        the plan while keeping the position is fine — that is NOT passivity.
      - Example FAILURE: Prior reflection "Player is right of box, push
        right"; prior action Right,Right executed=[] (blocked); current
        reflection (near-identical wording) "Player is right of box, push
        right". Judge: NO — the reflection ignored the block.
      - Example FAILURE: Prior action Right executed successfully; current
        "Position:" line is a verbatim copy of the prior reflection's
        "Position:" line with no one-cell shift acknowledged.

   c) **Stuck / oscillating actions without acknowledgment**:
      - Same or alternating actions (Up then Down, or the same action
        twice) occur in history, the reflection does NOT explicitly
        acknowledge the repeated failure AND continues the same or
        near-identical approach without meaningful adaptation.
      - Implicit adaptation alone (silently trying something different) is
        NOT sufficient if the agent also makes incorrect claims about
        position or action outcomes.

   d) **False claims about past actions**: reflection asserts something
      happened in history that did not occur (e.g., "I already moved right
      to push the box onto the target" when history shows no such event).

   e) **Contradiction with prior reasoning**: current claims conflict with
      the agent's own prior reflection from recent steps without new
      evidence to justify the change.

   Answer YES if: the reflection appropriately acknowledges recent feedback
   (success, partial execution, failure, or new observation) and updates
   beliefs/plans accordingly — OR there is genuinely nothing to update
   (e.g., first turn with no history, or no executed action and no state
   change).
   Answer NO if: any failure (a, b, c, d, e) is present.

Return a strict JSON object with EXACTLY these keys:
{{
  "observation_grounding": {{"yes_no": "YES|NO", "evidence": "<=2 short bullets"}},
  "action_coherence": {{"yes_no": "YES|NO", "evidence": "<=2 short bullets"}},
  "temporal_consistency": {{"yes_no": "YES|NO", "evidence": "<=2 short bullets"}}
}}

Be concise and specific in evidence. Do not add extra keys.
"""

# Ground truth section template for Sokoban (uses sokoban_state_to_sentences)
SOKOBAN_GROUND_TRUTH_SECTION_V2 = """
[Ground Truth Spatial State — USE THIS for position verification]
{ground_truth_state_text}
"""

SOKOBAN_UNIVERSAL_V2_USER = """Step index: {episode_step}

History (most recent steps):
{history_str}

Current Observation:
{observation_content}
{ground_truth_section}
Reasoning to verify (current step):
{reasoning_tokens}

Action taken (current step):
{action_tokens}
"""


class SokobanUniversalTemplateV2(VerifierTemplate):
    """Refined universal Sokoban verifier aligned with SciWorld 3-rubric schema.

    Designed for VLM agent traces with:
    - Image observations (converted to text ground truth for the judge)
    - <reflection>...</reflection><action>...</action> format
    - Short episodes (1–5 turns), multi-action per turn (Up,Down,Left,Right)
    - Windowed history matching agent's generation-time context
    """

    DEFAULT_HISTORY_WINDOW = 5

    def __init__(self, history_window: int = None):
        super().__init__(
            template_id="sokoban.universal_v2",
            description="Refined universal Sokoban verifier: grounding, action coherence, temporal consistency.",
            required_keys=("reasoning_tokens", "action_tokens"),
            system_prompt=SOKOBAN_UNIVERSAL_V2_SYSTEM,
            user_prompt=SOKOBAN_UNIVERSAL_V2_USER,
        )
        self.history_window = history_window or self.DEFAULT_HISTORY_WINDOW

    def _render_prompt(self, data: dict) -> dict:
        """Sokoban-specific prompt rendering.

        Key differences from SciWorld:
        - Current observations are multimodal (image-first, optional text fallback).
        - Optional replay ground truth is rendered in a separate section.
        - History entries may include executed-action feedback and optional
          ground-truth board-state text.
        - No valid_actions field (fixed action set).
        """
        data.setdefault("episode_step", data.get("current_step", "N/A"))

        # Ground truth section (from sokoban_state_to_sentences)
        gt_text = data.get("ground_truth_state_text")
        if gt_text:
            data["ground_truth_section"] = SOKOBAN_GROUND_TRUTH_SECTION_V2.format(
                ground_truth_state_text=gt_text,
            )
        else:
            data["ground_truth_section"] = ""

        # Observation content — prefer the current image whenever available.
        if data.get("current_observation_text") and data["current_observation_text"].strip():
            data["observation_content"] = data["current_observation_text"]
        elif data.get("current_observation_image"):
            data["observation_content"] = "<image>"
        else:
            data["observation_content"] = "(no observation)"

        # History formatting — keep it text-only for token efficiency, but expose
        # executed-action feedback and optional board-state text when available.
        history = data.get("history", [])
        if isinstance(history, list) and history:
          formatted = []
          for i, h in enumerate(history):
            step = h.get("step", i + 1)
            action = h.get("action", "N/A")
            reflection = h.get("reflection", "")
            obs_text = h.get(
              "observation_state_text",
              h.get("observation_text", ""),
            )
            executed_text = h.get("executed_actions_text")
            outcome_note = h.get("action_outcome_note")

            entry = f"[Step {step}]"
            if obs_text:
              entry += f"\nBoard state: {obs_text}"
            if reflection:
              entry += f"\nReasoning: {reflection}"
            entry += f"\nProposed action: {action}"
            if executed_text:
              entry += f"\nExecuted actions: {executed_text}"
            if outcome_note:
              entry += f"\nOutcome note: {outcome_note}"
            formatted.append(entry)

          data["history_str"] = "\n\n".join(formatted[-self.history_window:])
        else:
            data["history_str"] = "(No prior history)"

        return data


# =============================================================================
# Temporal-rubric loophole fix (minimal, non-destructive variant)
# =============================================================================
# The V2 temporal rubric's YES clause granted a pass whenever there was "no
# executed action and no state change". This let the judge score stuck,
# non-executing late turns (blocked moves, repeated "Done") as temporally
# CONSISTENT even when the reflection did NOT acknowledge being stuck --
# directly contradicting failures (b) STATIC reasoning and (c) stuck/oscillating
# of the SAME rubric, and inflating late-turn temporal in long/stuck (failing)
# episodes. We close ONLY that clause and leave every other byte of the prompt
# (grounding rubric, action rubric, JSON schema) identical, so any downstream
# score change is attributable to this single edit. The fix is acknowledgment-
# gated and therefore unbiased: a stuck turn is still YES if the reflection
# admits the block and re-plans, and only flips to NO when the reflection
# silently repeats the same reasoning -- it cannot manufacture a decay.

_TEMPORAL_YES_CLAUSE_ORIG = """   Answer YES if: the reflection appropriately acknowledges recent feedback
   (success, partial execution, failure, or new observation) and updates
   beliefs/plans accordingly — OR there is genuinely nothing to update
   (e.g., first turn with no history, or no executed action and no state
   change)."""

_TEMPORAL_YES_CLAUSE_FIXED = """   Answer YES if: the reflection appropriately acknowledges recent feedback
   (success, partial execution, failure, or new observation) and updates
   beliefs/plans accordingly — OR this is genuinely the first turn, with no
   prior history to update against.
   CRITICAL — a blocked or invalid prior action (nothing executed, no
   board-state change) is NOT, by itself, grounds for YES. When nothing
   executed and the board did not change, answer YES ONLY IF the current
   reflection EXPLICITLY acknowledges the blocked/failed attempt and adapts
   its plan. If the reflection instead restates essentially the same
   position/plan as though nothing happened, that is STATIC reasoning —
   answer NO (failures b, c). A frozen board is a reason to re-plan, not a
   licence to repeat the same reasoning."""

SOKOBAN_UNIVERSAL_V2_SYSTEM_TEMPORALFIX = SOKOBAN_UNIVERSAL_V2_SYSTEM.replace(
    _TEMPORAL_YES_CLAUSE_ORIG, _TEMPORAL_YES_CLAUSE_FIXED
)
assert SOKOBAN_UNIVERSAL_V2_SYSTEM_TEMPORALFIX != SOKOBAN_UNIVERSAL_V2_SYSTEM, (
    "Temporal YES-clause not found for replacement; check whitespace / em-dash."
)


class SokobanUniversalTemplateV2TemporalFix(SokobanUniversalTemplateV2):
    """V2 verifier with the temporal-consistency YES-clause loophole closed.

    Byte-identical to SokobanUniversalTemplateV2 except the temporal rubric no
    longer treats "no executed action / no state change" as automatic
    consistency: a stuck/blocked turn is YES only if the reflection acknowledges
    the block and re-plans, NO if it silently repeats the same reasoning.
    Observation-grounding and action-coherence rubrics and the JSON output
    schema are unchanged, so within-episode shape changes isolate temporal.
    """

    def __init__(self, history_window: int = None):
        super().__init__(history_window=history_window)
        self.template_id = "sokoban.universal_v2_temporalfix"
        self.system_prompt = SOKOBAN_UNIVERSAL_V2_SYSTEM_TEMPORALFIX


# =============================================================================
# Legacy universal template (preserved for backward compatibility)
# =============================================================================

SOKOBAN_UNIVERSAL_SYSTEM = """You are a strict Sokoban verifier. Judge only the CURRENT step, but use history to catch contradictions.
Sokoban symbols: walls #, floor _, targets O, boxes X, player P, box-on-target √, player-on-target S.
Admissible actions: {admissible_actions}. One and only one action is taken each step.

Your tasks:
1) Factual Grounding — Does the chain-of-thought reasoning correctly reflect ONLY the CURRENT observation (text grid or image)?
2) Action-Reasoning Self-Consistency — Does the stated action follow from the reasoning provided?
3) History Consistency — Does the CURRENT reasoning contradict either (a) the immediately preceding reasoning, or (b) the state changes implied by the last actions over the last TWO steps?

Return a strict JSON object with EXACTLY these keys:
{
  "factual_grounding": {"yes_no": "YES|NO", "evidence": "<=2 bullets"},
  "action_reasoning_consistency": {"yes_no": "YES|NO", "evidence": "<=2 bullets"},
  "history_consistency": {"yes_no": "YES|NO", "evidence": "<=2 bullets"},
#   "score": {
#     "grounding": 0|1,
#     "behavioral": 0|1,
#     "history": 0|1,
#     "aggregate": 0.0
  }
}
Be concise and specific in evidence. Do not add extra keys.
"""

SOKOBAN_UNIVERSAL_USER = """Step index: {current_step}

History (most recent first, include at least 2 if available):
{_history_str}

Current Observation:
{_current_observation_text_or_image}

Reasoning to verify (current step):
{reasoning_tokens}

Action taken (current step):
{action_tokens}
"""

class SokobanUniversalTemplate(VerifierTemplate):
    def __init__(self):
        super().__init__(
            template_id="sokoban.universal",
            description="Universal Sokoban verifier: grounding, behavioral, history.",
            required_keys=("admissible_actions", "current_step", "history", "reasoning_tokens", "action_tokens"),
            system_prompt=SOKOBAN_UNIVERSAL_SYSTEM,
            user_prompt=SOKOBAN_UNIVERSAL_USER,
        )

# ---------------------------------
# 2) Grounding (image or text), bin
# ---------------------------------
SOKOBAN_GROUNDING_SYSTEM = """You verify whether the chain-of-thought reasoning matches the CURRENT observation exactly.
- If observation is text, check entities/adjacencies in the grid.
- If observation is an image, check visual locations/adjacencies of player, boxes, targets, and walls.
Answer YES only if the reasoning's referenced facts are supported by the observation; otherwise NO.
Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

SOKOBAN_GROUNDING_SYSTEM_C = """You verify whether the chain-of-thought reasoning matches the CURRENT observation exactly.
- If observation is text, check entities/adjacencies in the grid.
- If observation is an image, check visual locations/adjacencies of player, boxes, targets, and walls.
- Prefer <observation> sub-tag inside <think> if present; otherwise check any observation-dependent claims.
Answer YES only if the reasoning's referenced facts are supported by the observation; otherwise NO.
Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""
# + If there are no observation-dependent claims, state "no claims" in <think> and answer YES.

SOKOBAN_GROUNDING_SYSTEM_B = """Verify whether the chain-of-thought matches the CURRENT observation.
If text: check grid entities/adjacencies. If image: locations/adjacencies of player, boxes, targets, walls.
Prefer <observation> sub-tag if present; otherwise check any observation-dependent claims.
YES only if all such claims are supported; otherwise NO. If there are no observation-dependent claims, say "no claims" in <think> and answer YES.
Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

SOKOBAN_GROUNDING_USER = """Current Observation:
{_current_observation_text_or_image}

Reasoning to verify:
{reasoning_tokens}
"""

# Other variations:
"""You are a fact-checker. For each atomic claim in the reasoning about the CURRENT observation, mark it as SUPPORTS/REFUTES/UNCLEAR based on the observation.
Output JSON with a list of checks. Keep each claim short (≤15 words).

{
  "checks": [
    {"claim":"P is left of X","verdict":"SUPPORTS|REFUTES|UNCLEAR","evidence":"quote grid snippet or visual note"},
    {"claim":"Cell right of X is wall","#2 verdict…":"…"}
  ],
  "overall_grounded":"YES|NO"
}"""

# # Merged system + user into a single user prompt (system left empty for compatibility)
# SOKOBAN_GROUNDING_SYSTEM = ""

# SOKOBAN_GROUNDING_USER = """You are a strict Sokoban grounding verifier. Verify whether the chain-of-thought matches the CURRENT observation exactly.
# Guidelines:
# - If observation is text, check entities (P, X, √, S, O, #, _) and their adjacencies / positions.
# - If observation is an image, check visual locations and adjacencies of player, boxes, targets, and walls.
# - Ignore unstated speculation or future planning.
# Answer YES only if every stated factual claim in the reasoning is directly supported by the observation; otherwise NO.

# Current Observation:
# {_current_observation_text_or_image}

# Reasoning:
# {reasoning_tokens}
# Think step by step and return exactly: <think>…</think><answer>YES|NO</answer>.
# Return exactly: <think>…</think><answer>YES|NO</answer>.
# """

class SokobanGroundingTemplate(VerifierTemplate):
    def __init__(self):
        super().__init__(
            template_id="sokoban.grounding",
            description="Binary factual grounding (text or image).",
            required_keys=("reasoning_tokens", "_current_observation_text_or_image"),
            system_prompt=SOKOBAN_GROUNDING_SYSTEM,
            user_prompt=SOKOBAN_GROUNDING_USER,
        )

# ----------------------------------------------
# 3) Action–Reasoning Consistency (behavioral)
# ----------------------------------------------
SOKOBAN_SELF_CONSISTENCY_SYSTEM = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). Do NOT require the reasoning to restate the observation.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

SOKOBAN_SELF_CONSISTENCY_SYSTEM_WITH_PRED = """You are an expert consistency evaluator for Sokoban.
You see:
(1) the environment's current observation (text and/or image), and
(2) the agent's chain-of-thought reasoning of the form:
  <think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>

Evaluate three dimensions:

1) observation_consistency (<observation>):
   - Check whether the described relative positions (above/below/same-row, left/right/same-col) are factually compatible with the current observation.
   - Answer YES if there are no clear factual contradictions.
   - Answer NO if any stated relation is clearly false or impossible.
   - Do NOT require the agent to mention every detail; only penalize incorrect statements.

2) reasoning_consistency (<reasoning>) — this is the PRIMARY dimension:
   YES requires ALL:
   - The reasoning mentions or clearly implies the key determinants for the chosen action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
   - No internal contradictions in the reasoning.
   - The reasoning's stated direction aligns with the action's direction.
   - If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
   - If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). You do NOT need the reasoning to restate the observation.

3) prediction_consistency (<prediction>):
   - You do NOT know the true future state; judge only logical consistency with the current observation, Sokoban rules, and the chosen action(s).
   - Answer YES if the predicted relations are plausible and follow from the moves.
   - Answer NO if they contradict Sokoban mechanics or what would logically happen.
   - If there is no meaningful <prediction> (missing or empty), set yes_no to "N/A".

OUTPUT FORMAT (strict):
Return a single JSON object and nothing else, with this structure:

{
  "observation_consistency": {
    "yes_no": "YES" | "NO",
    "evidence": ["one very short bullet about observation"]
  },
  "reasoning_consistency": {
    "yes_no": "YES" | "NO",
    "evidence": [
      "first very short bullet about reasoning",
      "optional second very short bullet about reasoning"
    ]
  },
  "prediction_consistency": {
    "yes_no": "YES" | "NO" | "N/A",
    "evidence": [] or ["one very short bullet about prediction"]
  }
}

Brevity / cost control:
- Each evidence bullet must be very concise (~15 words max).
- Each evidence entry must be a single-line, JSON-safe string.
- Do NOT include newlines, backslashes, or double quotes inside evidence.
- For observation, use exactly 1 bullet.
- For reasoning, use at most 2 bullets.
- For prediction, use 0-1 bullets; you may use [] when yes_no is "N/A".
- No extra text outside the JSON.
"""


SOKOBAN_SELF_CONSISTENCY_SYSTEM_NO_PRED = """You are an expert evaluator for Sokoban.
You see:
(1) the environment's current observation (text and/or image), and
(2) the agent's chain-of-thought answer of the form:
  <think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>

Your job is to judge, in a fine-grained way, whether the agent's reasoning is self-consistent and grounded.

Evaluate two dimensions:

1) observation_consistency (<observation>):
   - Check whether the described relative positions (above/below/same-row, left/right/same-col) are factually compatible with the current observation.
   - Answer YES if there are no clear factual contradictions.
   - Answer NO if any stated relation is clearly false or impossible.
   - Do NOT require the agent to mention every detail; only penalize incorrect statements.

2) reasoning_consistency (<reasoning>) — this is the PRIMARY dimension:
   YES requires ALL:
   - The reasoning mentions or clearly implies the key determinants for the chosen action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
   - No internal contradictions in the reasoning.
   - The reasoning's stated direction aligns with the action's direction.
   - If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
   - If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). You do NOT need the reasoning to restate the observation.

OUTPUT FORMAT (strict):
Return a single JSON object and nothing else, with this structure:

{
  "observation_consistency": {
    "yes_no": "YES" | "NO",
    "evidence": ["one very short bullet about observation"]
  },
  "reasoning_consistency": {
    "yes_no": "YES" | "NO",
    "evidence": [
      "first very short bullet about reasoning",
      "optional second very short bullet about reasoning"
    ]
  }
}

Brevity / cost control:
- Each evidence bullet must be very concise (~15 words max).
- Each evidence entry must be a single-line, JSON-safe string.
- Do NOT include newlines, backslashes, or double quotes inside evidence.
- For observation, use exactly 1 bullet.
- For reasoning, use at most 2 bullets.
- No extra text outside the JSON.
"""

SOKOBAN_SELF_CONSISTENCY_SYSTEM_SUB = """
Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.

You will see the environment's current observation (text and/or image) and an agent answer of the form:
<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>

Evaluate three aspects:

1) observation_consistency (<observation>):
   - Check whether the described relative positions (above/below/same-row, left/right/same-col) are factually compatible with the current observation.
   - Answer YES if there are no clear factual contradictions.
   - Answer NO if any stated relation is clearly false or impossible.
   - Do NOT require the agent to mention every detail; only penalize incorrect statements.

2) reasoning_consistency (<reasoning>) — this is the PRIMARY dimension:
   YES requires ALL:
   - The reasoning mentions or clearly implies the key determinants for the chosen action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
   - No internal contradictions in the reasoning.
   - The reasoning's stated direction aligns with the action's direction.
   - If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
   - If a current observation is provided, you may use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). You do NOT need the reasoning to restate the observation.

3) prediction_consistency (<prediction>, if present):
   - Treat <prediction> as the expected state AFTER executing the chosen action(s).
   - You do NOT know the true future state; judge only logical consistency with the current observation, Sokoban rules, and the chosen action(s).
   - Answer YES if the predicted relations are plausible and follow from the moves.
   - Answer NO if they contradict Sokoban mechanics or what would logically happen.
   - If there is no meaningful <prediction> (missing or empty), use "N/A".

OUTPUT FORMAT (strict):
Return a single JSON object and nothing else, with this structure:

{
  "observation_consistency": {
    "yes_no": "YES" | "NO",
    "evidence": [
      "short bullet 1 about observation",
      "short bullet 2 about observation"
    ]
  },
  "reasoning_consistency": {
    "yes_no": "YES" | "NO",
    "evidence": [
      "short bullet 1 about reasoning",
      "short bullet 2 about reasoning"
    ]
  },
  "prediction_consistency": {
    "yes_no": "YES" | "NO" | "N/A",
    "evidence": [
      "short bullet 1 about prediction (or why N/A)",
      "short bullet 2 about prediction (or why N/A)"
    ]
  }
}

Brevity / cost control:
- Each evidence list should contain 2-3 very short, concrete bullets.
- No extra text outside the JSON.
"""

# ask llm to rewrite agent's trace when NO
SOKOBAN_SELF_CONSISTENCY_SYSTEM_R = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). Do NOT require the reasoning to restate the observation.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Output format:
- Return exactly: <think>…</think><answer>YES|NO</answer>.
- If the answer is NO, ALSO append a minimal correction in <rewrite>…</rewrite> that fixes ONLY incorrect keywords/phrases about (a) observation grounding (entities, relations, directions), (b) reasoning direction/intent, or (c) predicted outcome. Preserve any existing sub-tags (<observation>, <reasoning>, <prediction>) if present, do not alter the action, and keep ≤ 40 tokens.
"""


SOKOBAN_SELF_CONSISTENCY_SYSTEM_J = """You are a strict Sokoban step-level self-consistency verifier. Verify only the CURRENT step. The chain-of-thought comes from a separate LLM agent under RL finetuning; assess its reasoning as written. Do not coach, rewrite, propose alternatives, or grade optimality.

Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). Do NOT require the reasoning to restate the observation.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

# gpt-4.1-nano is sensitive to the feasibility of actions while gpt-5-nano does not have this issue.
# (feasibility-aware variant)
SOKOBAN_SELF_CONSISTENCY_SYSTEM_F = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes), including **feasibility contradictions** where the reasoning's implied spatial preconditions for the action are not met by the current positions (e.g., moving down is infeasible if the box is above the player). Do NOT require the reasoning to restate the observation.
- If the action cannot be executed from the described/observed position under the reasoning's own assumptions, answer NO.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

# actually we can filter out pairs with any invalid actions before gpt can assess them
SOKOBAN_SELF_CONSISTENCY_SYSTEM_G = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- Before any other judgment, verify that every action is an EXACT member of the provided Admissible actions list (case-insensitive; trim whitespace). If any action is outside the set, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). Do NOT require the reasoning to restate the observation.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

# actually we can filter out pairs with any invalid actions before gpt can assess them; I implemented this filtering
SOKOBAN_SELF_CONSISTENCY_SYSTEM_H = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- Before any other judgment, verify that every action is an EXACT member of the provided Admissible actions list (case-insensitive; trim whitespace). If any action is outside the set, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes), including **feasibility contradictions** where the reasoning's implied spatial preconditions for the action are not met by the current positions (e.g., moving down is infeasible if the box is above the player). Do NOT require the reasoning to restate the observation.
- If the action cannot be executed from the described/observed position under the reasoning's own assumptions, answer NO.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""


SOKOBAN_SELF_CONSISTENCY_SYSTEM_D = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action in Sokoban.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for that action (e.g., moving to a free cell, or pushing a box with a free cell beyond, etc.).
- No internal contradictions in the reasoning.
- The reasoning's stated direction aligns with the action's direction.
- If the reasoning is ambiguous between multiple actions (e.g., "maybe left or up") or lacks an explicit/implicit direction, answer NO.
- If a current observation is provided, use it only to reject contradictions (e.g., the reasoning asserts a fact the observation clearly refutes). Do NOT require the reasoning to restate the observation.
- If sub-tags appear (<observation>, <reasoning>, <prediction>): base on <reasoning>; accept <prediction> only if supported; on conflict, prefer <reasoning>.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

SOKOBAN_SELF_CONSISTENCY_SYSTEM_D_OPTIMIZED = """# Role and Objective
- Assess whether the chain-of-thought reasoning in Sokoban appropriately justifies the selected action.
# Instructions
- Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
- Answer YES only if all the following are met:
  - The reasoning mentions or clearly implies the key determinants for the action (e.g., moving to a free cell, pushing a box with a free cell beyond).
  - There are no internal contradictions in the reasoning.
  - The reasoning's stated direction aligns with the chosen action.
- Answer NO if:
  - The reasoning is ambiguous between actions (e.g., 'maybe left or up'), or lacks explicit/implicit direction.
- If provided, use the current observation only to reject contradictions (i.e., when the reasoning clearly opposes the observation), but do NOT require the reasoning to restate it.
- If there are sub-tags like `<observation>`, `<reasoning>`, `<prediction>`:
  - Use `<reasoning>` as the primary basis.
  - Accept `<prediction>` only if it is supported by `<reasoning>`; if there is a conflict, defer to `<reasoning>`.
- After generating your response, briefly validate in 1-2 lines that your answer matches all criteria and make a next-step decision if any doubt remains.

# Cost Control / Brevity Rules
- The <think> section must be no more than 50 tokens.
- Use at most 2 concise bullets or 1 short sentence in <think>.

# Output Format
- Return exactly:
  - `<think>... </think>`
  - `<answer>YES|NO</answer>`"""

# Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
# Evaluate whether the chain-of-thought ALONE justifies the chosen action for THIS step.
# Evaluate whether the chain-of-thought reasoning justifies ONLY the provided action in the CURRENT step.


SOKOBAN_SELF_CONSISTENCY_SYSTEM_E = """Evaluate whether the chain-of-thought reasoning actually justifies the chosen action (which may be a single move OR a sequence of moves) in Sokoban.
If a sequence (e.g., "left, up, down"), judge the WHOLE ordered sequence.
YES requires ALL:
- The reasoning mentions or clearly implies the key determinants for EACH move in the sequence (e.g., moving to a free cell, or pushing a box with a free cell beyond, etc.).
- No internal contradictions in the reasoning.
- The reasoning's stated / implied directions align exactly and in order with the action sequence (no extra, missing, or alternative moves; no ambiguity like "maybe left or up").
- If a current observation is provided, use it only to reject contradictions (do NOT require restating it).
- If sub-tags appear (<observation>, <reasoning>, <prediction>): base on <reasoning>; accept <prediction> only if supported; on conflict, prefer <reasoning>.
Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

SOKOBAN_SELF_CONSISTENCY_SYSTEM_B = """Evaluate whether the chain-of-thought reasoning justifies ONLY the provided action in the CURRENT step.
Answer YES only if ALL:
- Direction match: The reasoning explicitly or implicitly supports exactly the same direction as the action (no alternative directions mentioned).
- Causal justification: Mentions the minimal Sokoban precondition(s) for that move (e.g., target cell free; or box with free cell beyond for a push).
- No contradictions, no hallucinated board facts, no speculative multi-step planning.
- Not generic: Vague text like "moving to improve position" or "continue" without directional justification => NO.
- If reasoning is missing, empty, or purely meta (e.g., "I will move") => NO.
- If reasoning cites multiple possible actions or is ambiguous => NO.

Use observation (if present) ONLY to reject contradictions—do not restate it.

Cost control / brevity rules:
- Think section max 50 tokens.
- Use at most 2 concise bullets OR 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>.
"""

# TODO: Add more details on what constitutes a valid action and reasoning.
# Think step by step and end with exactly: <think>…</think><answer>YES|NO</answer>.
# Superficial or unrelated reasoning that doesn't bear on the action is insufficient.
# - If an image is attached, refer to it exactly as 'Image 1' (or 'Image k' if multiple). Do NOT introduce new image labels or synonyms.

SOKOBAN_SELF_CONSISTENCY_USER = """Reasoning trace to verify:
{reasoning_tokens}

Action taken:
{action_tokens}

Admissible actions:
{admissible_actions}

(Optional) Current environment observation for contradiction checks only:
{_current_observation_text_or_image}
"""

SOKOBAN_SELF_CONSISTENCY_USER_HEADER_INLINE = """Attached images: [1]. Refer as 'Image 1'.

Reasoning to verify:
<think>now…</think>

Action taken:
left

Admissible actions:
['up', 'down', 'left', 'right']

(Optional) Current Observation for contradiction checks only:
[Image 1]
"""

SOKOBAN_SELF_CONSISTENCY_USER_INTERLEAVED = """Reasoning to verify:
<think>now…</think>

Action taken:
left

Admissible actions:
['up', 'down', 'left', 'right']

(Optional) Current Observation for contradiction checks only (Image 1):
"""



class SokobanActionReasoningConsistencyTemplate(VerifierTemplate):
    def __init__(self):
        super().__init__(
            template_id="sokoban.self_consistency",
            description="Binary action-reasoning self-consistency.",
            # required_keys=("reasoning_tokens", "action_tokens", "admissible_actions", "_current_observation_text_or_image"),
            required_keys=("reasoning_tokens", "action_tokens", "admissible_actions"),
            system_prompt=SOKOBAN_SELF_CONSISTENCY_SYSTEM,
            user_prompt=SOKOBAN_SELF_CONSISTENCY_USER,
        )

# ----------------------------------------------
# 4) History Consistency (>=2 steps), binary
# ----------------------------------------------
SOKOBAN_HISTORY_SYSTEM = """Check whether the CURRENT chain-of-thought reasoning contradicts either:
(a) the immediately preceding reasoning (if present), OR
(b) the implied consequences of the last actions across the last steps (use ≥2 if available).
Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>, where YES = consistent, NO = contradiction/drift.
"""
# immediately preceding state/action consequences

SOKOBAN_HISTORY_SYSTEM_B = """Check whether the CURRENT chain-of-thought reasoning, against the immediately previous step(s) (use ≥2 if available), contradicts either:
(a) the immediately preceding reasoning (if present), OR
(b) the implied consequences of the last actions.
If the previous step included a <prediction> or explicit consequence, the current reasoning/action must align unless the current observation provides new evidence AND the reasoning states the change.
Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>, where YES = consistent, NO = contradiction/drift.
"""
# If the immediately previous chain-of-thought includes a <prediction> (or explicit consequence), check that the current reasoning/action does not contradict it unless the current observation provides new information that justifies a change; if so, the reasoning must say so.

SOKOBAN_HISTORY_SYSTEM_C = """Check whether the CURRENT chain-of-thought reasoning contradicts either:
(a) the immediately preceding reasoning (if present), OR
(b) the implied consequences of the last actions across the last steps (use ≥2 if available), unless current observation provides new evidence AND the reasoning states the change.
Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>…</think><answer>YES|NO</answer>, where YES = consistent, NO = contradiction/drift.
"""

SOKOBAN_HISTORY_USER = """You are now at step {current_step}. Recent History (most recent first; at least one):
{_history_str}

Current Observation:
{_current_observation_text_or_image}

Current Reasoning:
{reasoning_tokens}
"""

# { "observation_text": ..., "observation_image": True/False, "reasoning_tokens": ..., "action_tokens": ... }

class SokobanHistoryConsistencyTemplate(VerifierTemplate):
    def __init__(self):
        super().__init__(
            template_id="sokoban.history_consistency",
            description="Binary history consistency using last two steps' reasoning and implied consequences.",
            required_keys=("current_step", "_current_observation_text_or_image", "reasoning_tokens"),
            system_prompt=SOKOBAN_HISTORY_SYSTEM,
            user_prompt=SOKOBAN_HISTORY_USER,
        )

# ----------------------------------------------
# 5) Anti-Hallucination Grounding, binary
# ----------------------------------------------
SOKOBAN_ANTI_HALLUCINATION_SYSTEM = """Check whether the CURRENT chain-of-thought reasoning contradicts either:
(a) the immediately preceding reasoning, OR
(b) the implied consequences of the last actions across the last TWO steps.
Return exactly: <think>…</think><answer>YES|NO</answer>, where YES = consistent, NO = contradiction/drift.
"""

SOKOBAN_ANTI_HALLUCINATION_USER = """You are now at step {current_step}. Recent History (most recent first; at least two):
{_history_str}

Current Observation:
{_current_observation_text_or_image}

Current Reasoning:
{reasoning_tokens}
"""

# ------------------------------
# Factory / Registry helpers
# ------------------------------

# Template registry for programmatic access
SOKOBAN_TEMPLATES = {
    "universal": SokobanUniversalTemplateV2,       # NEW: aligned with SciWorld 3-rubric schema
    "universal_v2": SokobanUniversalTemplateV2,     # Explicit alias
    "universal_v2_temporalfix": SokobanUniversalTemplateV2TemporalFix,  # temporal YES-clause loophole closed
    "universal_legacy": SokobanUniversalTemplate,   # Old template (for backward compat)
    "grounding": SokobanGroundingTemplate,
    "self_consistency": SokobanActionReasoningConsistencyTemplate,
    "history_consistency": SokobanHistoryConsistencyTemplate,
}


def get_sokoban_template(rubric: str) -> VerifierTemplate:
    """Get a Sokoban template by rubric name."""
    if rubric not in SOKOBAN_TEMPLATES:
        raise ValueError(f"Unknown Sokoban rubric: {rubric}. Available: {list(SOKOBAN_TEMPLATES.keys())}")
    return SOKOBAN_TEMPLATES[rubric]()


def get_sokoban_verifier_templates():
    objs = {
        "sokoban.universal": SokobanUniversalTemplateV2(),
        "sokoban.universal_v2": SokobanUniversalTemplateV2(),
        "sokoban.universal_v2_temporalfix": SokobanUniversalTemplateV2TemporalFix(),
        "sokoban.universal_legacy": SokobanUniversalTemplate(),
        "sokoban.grounding": SokobanGroundingTemplate(),
        "sokoban.self_consistency": SokobanActionReasoningConsistencyTemplate(),
        "sokoban.history_consistency": SokobanHistoryConsistencyTemplate(),
    }
    # optional global registry
    for k, v in objs.items():
        try:
            registry.register(k, v)
        except KeyError:
            pass
    return objs

if __name__ == "__main__":
    items = [{
        "id":"ex-1",
        "current_step":17,
        "history":[
            {"observation_text":"...", "reasoning_tokens":"<think>t-1</think>", "action_tokens":"left"},
            {"observation_text":"...", "reasoning_tokens":"<think>t-2</think>", "action_tokens":"up"}
        ],
        # "current_observation_text":"#####\n#_PXO#\n#####",
        "current_observation_image": "<image>",  # Placeholder for image
        "reasoning_tokens": "<think>now…</think>",
        "action_tokens": "left",
        "admissible_actions": ["up","down","left","right"]
    }]
    batch_items = [
    {
        "id": "1",
        "reasoning_tokens": "<think><observation>The player is in the top-left corner, and there is a box to the right.</observation><reasoning>Since the player is in the top-left corner, the first action should be to move left to get closer to the box. The box is to the right, so the next action should be to move right to get closer to the box.</reasoning><prediction>The player will move left, then right.</prediction></think>",
        "action_tokens": "<answer>Left,Right,Down</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)",
    },
    {
        "id": "2",
        "reasoning_tokens": "<think><observation>The player is to the left of the box and the target is to the right of the box.</observation><reasoning>Since the player is to the left of the box and the target is to the right of the box, the first action should be to move to the right to align with the target. The box is in the way, so the next action should be to move down to push the box to the right.</reasoning><prediction>The player will move right, then down.</prediction></think>",
        "action_tokens": "<answer>Right,Down,Right</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)",
    },
    {
        "id": "3",
        "reasoning_tokens": "<think><observation>The player is in the top right corner, and there is a box to the left.</observation><reasoning>Since the player is in the top right corner, the first action should be to move left to get closer to the box. The box is to the left, so the next action should be to move left.</reasoning><prediction>The player will move left.</prediction></think>",
        "action_tokens": "<answer>Left,Down,Left</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)",
    },
    ]
    verifiers = get_sokoban_verifier_templates()
    self_consistency = verifiers["sokoban.self_consistency"]
    self_consistency_messages = self_consistency.build_messages(batch_items[0])
    print("Self-consistency messages:")
    for msg in self_consistency_messages:
        print(f"{msg['role']}: {msg['content']}")