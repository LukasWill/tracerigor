"""
Navigation (AI2-THOR home-robot) LLM Judge Prompt Templates.

Evaluates Chain-of-Thought (CoT) quality for the VLM agent's interactions with the
visual navigation environment under the ReflAct format:
    <reflection>...</reflection><action>...</action>

Three evaluation rubrics (same schema as SciWorld / Sokoban v2 — so the universal
3-rubric JSON parser `_sciworld_universal_score` can be reused without changes):
  1. observation_grounding  — does the reflection accurately describe the current
                              first-person image?
  2. action_coherence       — does the ordered action sequence logically follow
                              from the reflection's stated intent?
  3. temporal_consistency   — does the reflection update beliefs/plans
                              appropriately given the action / feedback history?

Navigation-specific design notes (vs. Sokoban / SciWorld):
  - Observations are first-person RGB images of a household scene; no grid, no
    discrete cell positions. Spatial words are agent-egocentric.
  - The action set mixes translations (moveahead/back/left/right) with rotations
    (rotateleft/right) and camera tilts (lookup/lookdown). Confusing translation
    with rotation is a navigation-specific action-coherence failure mode.
  - The agent's training prompt embeds a "living room / va  se / kitchen doorway"
    worked example. A small fraction of very-early-training trajectories parrot
    that example verbatim even when the actual image / instruction do not match
    it (empirically: ~1/128 at step_0, ~4/128 at step_10, ~0 from step_20 on,
    in the finegrained-navigation-reflact_2hky0t5o sweep). The judge prompt
    lists this as a niche grounding failure mode, not as a primary one.
  - Multi-action turns (up to 5 sub-actions, comma-separated) need to be judged
    as ordered plans. Env feedback reports only the LAST sub-action's status.
"""

from tracerigor.verifier.prompt.verifier_template_base import VerifierTemplate
from tracerigor.verifier.utils import registry


# =============================================================================
# Universal Navigation Verifier V2 (all 3 rubrics in one call)
# =============================================================================
# JSON output keys match SciWorld / Sokoban v2 so the shared parser works.

NAV_UNIVERSAL_V2_SYSTEM = """You are a strict robot-navigation verifier. Judge only the CURRENT step, but use history to catch contradictions.

The agent is a home robot navigating an indoor scene (AI2-THOR-style). It observes the world as a first-person RGB image each turn — the image is the camera view from the robot's head. There is no map, grid, or absolute coordinate frame; all spatial words refer to the current camera view.

Task: navigate close to a target object specified by the Human Instruction. The instruction may name the target directly ("DeskLamp", "Pot") or describe it indirectly ("a device to set an alarm", "a deep cooking vessel to braise meat"). The reflection should identify the target consistently with the instruction.

Admissible actions (1–5 per turn, comma-separated, executed left-to-right):
  moveahead   — translate forward
  moveback    — translate backward
  moveleft    — STRAFE left (translation; heading and view do NOT change)
  moveright   — STRAFE right (translation; heading and view do NOT change)
  rotateleft  — rotate body 90° to the left (heading and view DO change)
  rotateright — rotate body 90° to the right (heading and view DO change)
  lookup      — tilt camera 30° up (no translation)
  lookdown    — tilt camera 30° down (no translation)

Critical action semantics (navigation-specific):
  • `moveleft` / `moveright` are STRAFES — they do NOT turn the agent. If the reflection plans to "turn right" / "rotate to face X", the correct action is `rotateright`, NOT `moveright`. Conversely, "step / strafe left" corresponds to `moveleft`, NOT `rotateleft`.
  • `lookup` / `lookdown` only re-aim the camera; they do not move the agent. If the reflection plans to "check the floor" or "inspect what's overhead", the corresponding action is `lookdown` / `lookup`.
  • Spatial words in the reflection ("left", "right", "in front of", "behind", "near", "far") are agent-egocentric and refer to the CURRENT camera view. "Behind me" is not observable from the current image unless the agent has rotated to face it.

Environment-feedback semantics:
  • "Last action is executed successfully" / "not executed successfully" reports the status of the LAST sub-action of the prior turn only. Earlier sub-actions of that turn may have partially advanced the agent and changed the image; later sub-actions following a failed one are not executed. The current image is the authoritative state.

The agent reasons inside <reflection>...</reflection> and emits the move sequence inside <action>...</action>. The reflection is expected to cover: the current visual state, the target object (per the instruction), progress so far, and any obstacle/correction needed.

Your tasks:

1) Observation Grounding — Does the reflection correctly describe the CURRENT first-person image?
   Inspect the image and check the reflection's claims about:
     • Room type (kitchen / bedroom / living room / bathroom / office, ...).
     • Visible furniture and major objects ("there is a desk / stove / couch / TV / ...").
     • Target identification — does the object the reflection nominates as the target ("the target is the DeskLamp") (a) actually appear in the image, and (b) satisfy the Human Instruction's semantic description? (E.g., the instruction "a luminous device on my desk to read at night" requires a lamp-like object, NOT a phone or laptop.)
     • Agent-egocentric direction of objects relative to the camera ("to my left", "in front of", "near the doorway").
   Penalise (answer NO):
     a) Hallucinated objects — reflection claims to see X but X is not present in the image (e.g., "I see a pot on the stove" with no pot visible; "There is a couch to my left" while the image shows a kitchen).
     b) Wrong target nomination — the reflection points to a visible object as "the target" that clearly does not satisfy the instruction's semantic description (e.g., naming a phone as the target for "a luminous device to read at night", or a coffee maker for "a vessel to cook rice"). This is a grounding failure because the reflection misreads what role the visible object plays.
     c) Wrong egocentric direction — object claimed left when clearly right, claimed in front when behind, etc.
     d) Wrong room type — claims kitchen when the image is a bedroom, etc.
     e) Verbatim parroting of the prompt's "I am in a living room... a vase... kitchen doorway" worked example while the image and/or instruction clearly do not match it. (Niche failure — see module note.)
   Do NOT penalise:
     - Omissions — the reflection is not required to be exhaustive; only penalise claims that CONTRADICT the image / instruction.
     - Reasonable synonyms (e.g., "kettle" vs "pot", "monitor" vs "TV") so long as the choice does not change the target identity.
     - Soft / hedged language ("appears to be on the desk", "likely on the counter") — judge against the image and allow plausible uncertainty.

   No-claims case: If the reflection makes NO claims about the current scene, room, visible objects, or target identity — e.g., a pure plan adjustment after a prior failure such as "Since moving left was unsuccessful, I should try moving forward instead." — there is nothing to ground-check. Answer YES. Observation Grounding scores whether claims about the visible scene are contradicted; it does NOT require the reflection to describe the scene. (Plan adjustments are scored under Action Coherence / Temporal Consistency.)

   What counts as a contradiction (definitional clarification, applies to bullets a–d above):
     • A claim is contradicted when the image clearly shows a DIFFERENT prominent object in place of the claimed one, OR a structurally different room/layout than asserted, OR an object in a clearly different egocentric direction than asserted.
     • Ambiguity, clutter, partial occlusion, distance, and peripheral framing are NOT by themselves contradictions. Do not penalise a claim merely because the claimed object is not unambiguously visible in the current view — judge the reflection's claims against what IS clearly visible, not against what cannot be ruled out.

2) Action Coherence — Does the ordered action SEQUENCE logically follow from the reflection?
   The reflection should state or clearly imply a movement intent: a direction (forward / left / right / back), a target to approach, a rotation, or a sensing move (lookup / lookdown). The ordered action sequence must match that stated intent.
   - Treat the action list as an ordered plan (e.g., `moveahead, moveahead, rotateright, moveahead` = step forward twice, then turn right, then step forward once).
   - Judge against the reflection AS WRITTEN, not against whether the reflection's facts are correct (factual errors are scored under Observation Grounding). If the reflection is mis-grounded but the action faithfully implements its mistaken plan, Action Coherence is YES.
   Common errors to catch (answer NO if any apply):
     a) Translate ↔ rotate confusion: reflection explicitly plans to "rotate" / "turn to face X" but the action is `moveright`/`moveleft` (a strafe, no rotation); or says "strafe / step left" but action is `rotateleft`. Only flag when the reflection's verb is unambiguous; reflections that say only "move toward X on the right" are ambiguous and are NOT a confusion by themselves.
     b) Direction mismatch: the reflection's stated direction or unambiguous spatial relation is absent from or contradicted by the action sequence (e.g., reflection "I should move right" / "try moving right instead" but the action contains no `moveright` and is dominated by `moveleft` / `moveahead`; or reflection says the target is directly behind and the action only moves ahead, with no turn/backtrack/repositioning). This also covers within-turn plan/action contradictions where the reflection explicitly proposes a correction the action does not implement.
     c) Empty / vague reasoning: reflection is purely meta ("I will move", "I need to navigate", "I should try a different approach") with NO direction, no axis, no rotation/tilt intent, and no reference to the target — and the action sequence cannot be plausibly tied to the reflection.
     d) Sensing-vs-moving mismatch: reflection explicitly plans `lookdown` / `lookup` to inspect floor/ceiling but the action contains only translations (or vice versa).
   Allow:
     - Multi-action sequences whose overall composition matches the stated intent, even if sub-actions repeat (e.g., reflection "move forward several times then turn right" → `moveahead, moveahead, moveahead, rotateright`).
     - Coarse goals ("move toward the desk") paired with any concrete sequence whose component directions do NOT contradict the stated goal. Do not require the sequence to be the most direct or efficient route.
     - Lateral micro-adjustment under a coarse goal: when the reflection states only a coarse "move toward X" goal (no explicit commitment to a single axis or to avoiding any direction), a single off-axis strafe (`moveleft` or `moveright`) interleaved among `moveahead`s counts as a lateral alignment toward X — NOT a direction mismatch. Only flag direction mismatch when the reflection explicitly committed to a direction / axis (or to avoiding one) and the action violates it.
   Do NOT judge whether the action is OPTIMAL or whether it will SUCCEED.

3) Temporal Consistency — Does the CURRENT reflection update beliefs / plans appropriately given the HISTORY?

   CORE PRINCIPLE: Good reasoning updates beliefs and plans when evidence changes — a successful sub-action produces a new viewpoint, a failed sub-action means the agent's view is unchanged, and repeated failures should provoke a different plan, not the same one. Reflections that stay STATIC in the face of new feedback are temporally inconsistent even when they make no explicit contradictory claim — passivity in the presence of new evidence is itself a failure mode.

   What you get in the History (per step, oldest-first): the prior `Reasoning` (reflection text) and the prior `Proposed action` sequence; an `Env feedback` line if available — and this feedback is ALWAYS exactly "Last action is executed successfully." or "Last action is not executed successfully.", i.e., the status of the LAST sub-action of that turn only. There is NO per-sub-action breakdown: when feedback is "not executed successfully", you cannot tell from history alone whether earlier sub-actions also failed or were partially executed. Use the current image (the authoritative state) to disambiguate.

   If there is no prior history, answer YES unless the current reflection invents prior actions, outcomes, or progress that never happened.

   Check for these failures (answer NO if ANY is present):

   a) Static / stale reasoning (passivity):
      The current reflection's description of viewpoint, surroundings, target visibility, or planned direction is essentially UNCHANGED from a recent prior reflection, even though the prior turn executed actions or produced a new image that should have shifted the description. Verbatim or near-verbatim repetition of the immediately preceding reflection is a strong signal.
      Example FAILURE: prior step's reflection is "Looking at the kitchen, I see the deep vessel on the counter. I need to move towards it.", the prior turn's `moveleft` got "Last action is not executed successfully", and the current reflection is the same sentence with no acknowledgment of the failed move.

   b) False progress claims:
      Reflection asserts "Now I see X" / "Now I am near X" / "I have reached X" when (i) recent feedback was "not executed successfully" AND (ii) the current image does not actually support the new claim. "Now" implies a change since the previous turn; if the image shows no relevant change, this is hallucinated progress.

   c) Stuck / oscillating actions without acknowledgment:
      Recent history shows the agent alternating directions (moveleft → moveright → moveleft) or repeating a near-identical action sequence after failure, AND the current reflection does NOT verbally acknowledge the repeated failure AND continues a near-identical approach without meaningful adaptation (e.g., trying a rotate, a lookdown/lookup, or a different macro plan).
      Implicit adaptation alone (silently switching direction without verbal acknowledgment) is NOT sufficient if the agent also makes incorrect claims about progress.

   d) False claims about past actions:
      Reflection asserts something happened in history that did not occur — e.g., "I already rotated to face the kitchen" when no `rotateleft` / `rotateright` appears in the Proposed-action history; or "After looking down, I saw the plate" when no `lookdown` was proposed in any prior turn.

   e) Contradiction with prior reasoning:
      Current claims conflict with the agent's own prior reflection from recent steps without any new evidence (new viewpoint after a successful move, or an explicit re-examination) to justify the flip.
      Example FAILURE: the prior step's reflection named the target as the DeskLamp; the current step (with no movement between them, image unchanged) names the target as the laptop.

   f) Mis-attribution of partial execution:
      Because the env feedback covers only the last sub-action, the reflection should NOT assume the whole sequence succeeded just because the last sub-action did, nor that the whole sequence failed just because the last sub-action did. Cross-check with the current image. Asserting "I moved 5 steps forward" purely on the basis of feedback (without image support) is a NO.

   Answer YES if: the reflection appropriately acknowledges recent feedback and updates beliefs and plans accordingly — OR there is genuinely nothing to update (first turn, or no prior executed action and no new viewpoint).
   Answer NO if: any failure (a–f) is present.

Return a strict JSON object with EXACTLY these keys:
{
  "observation_grounding": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"},
  "action_coherence": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"},
  "temporal_consistency": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"}
}
Be concise and specific in evidence. Do not add extra keys.
"""

# Optional ground-truth section (e.g., from a future scene-state replay).
NAV_GROUND_TRUTH_SECTION_V2 = """
[Ground Truth Spatial State — USE THIS for verification]
{ground_truth_state_text}
"""

NAV_UNIVERSAL_V2_USER = """Human Instruction: {instruction}

Step index: {episode_step}

History (most recent steps):
{history_str}

Current Observation:
{observation_content}
{ground_truth_section}
Reflection to verify (current step):
{reasoning_tokens}

Action taken (current step):
{action_tokens}
"""


class NavigationUniversalTemplateV2(VerifierTemplate):
    """Universal navigation verifier aligned with SciWorld / Sokoban v2 3-rubric schema.

    Designed for the VLM ReflAct agent (`<reflection>...</reflection><action>...</action>`)
    on AI2-THOR-style visual navigation:
      - First-person image observations (passed in via the verifier's image-attach path).
      - Multi-action turns (1–5 sub-actions, comma-separated).
      - Mixed translation / rotation / camera-tilt actions.
      - Windowed history matching the agent's generation-time context.
    """

    DEFAULT_HISTORY_WINDOW = 5

    def __init__(self, history_window: int = None):
        super().__init__(
            template_id="navigation.universal_v2",
            description="Universal navigation verifier: grounding, action coherence, temporal consistency.",
            required_keys=("reasoning_tokens", "action_tokens"),
            system_prompt=NAV_UNIVERSAL_V2_SYSTEM,
            user_prompt=NAV_UNIVERSAL_V2_USER,
        )
        self.history_window = history_window or self.DEFAULT_HISTORY_WINDOW

    def _render_prompt(self, data: dict) -> dict:
        data.setdefault("episode_step", data.get("current_step", "N/A"))
        data.setdefault("instruction", data.get("human_instruction", "N/A"))

        # Optional ground truth section.
        gt_text = data.get("ground_truth_state_text")
        if gt_text:
            data["ground_truth_section"] = NAV_GROUND_TRUTH_SECTION_V2.format(
                ground_truth_state_text=gt_text,
            )
        else:
            data["ground_truth_section"] = ""

        # Observation content — prefer image; fall back to optional text.
        if data.get("current_observation_text") and data["current_observation_text"].strip():
            data["observation_content"] = data["current_observation_text"]
        elif data.get("current_observation_image"):
            data["observation_content"] = "<image>"
        else:
            data["observation_content"] = "(no observation)"

        # History formatting — text-only for token efficiency. The navigation
        # env exposes only a coarse last-sub-action `env_feedback` per turn
        # (no per-sub-action breakdown); `executed_actions_text` is rendered
        # only if the orchestrator opts to provide it. The system prompt
        # describes the history schema honestly.
        history = data.get("history", [])
        if isinstance(history, list) and history:
            formatted = []
            for i, h in enumerate(history):
                step = h.get("step", i + 1)
                action = h.get("action", "N/A")
                reflection = h.get("reflection", "")
                obs_text = h.get("observation_text", "")
                executed_text = h.get("executed_actions_text")
                feedback_text = h.get("env_feedback") or h.get("action_outcome_note")

                entry = f"[Step {step}]"
                if obs_text:
                    entry += f"\nViewpoint: {obs_text}"
                if reflection:
                    entry += f"\nReasoning: {reflection}"
                entry += f"\nProposed action: {action}"
                if executed_text:
                    entry += f"\nExecuted sub-actions: {executed_text}"
                if feedback_text:
                    entry += f"\nEnv feedback: {feedback_text}"
                formatted.append(entry)

            data["history_str"] = "\n\n".join(formatted[-self.history_window:])
        else:
            data["history_str"] = "(No prior history)"

        return data


# =============================================================================
# Binary single-rubric variants (extracted from the universal body)
# =============================================================================
# These mirror SciWorld's per-rubric binaries, for cases where the orchestrator
# wants to score one rubric at a time. The semantic content matches the
# corresponding section of NAV_UNIVERSAL_V2_SYSTEM.

# ---- 1) Observation Grounding (binary) -------------------------------------
NAV_GROUNDING_SYSTEM = """You verify whether the agent's reflection accurately describes the CURRENT first-person navigation image. Spatial words in the reflection are agent-egocentric (relative to the camera).

Check the reflection's claims about:
  - Room type (kitchen / bedroom / living room / bathroom / office, ...).
  - Visible furniture and major objects ("desk / stove / couch / TV / ...").
  - Target nomination — does the object the reflection labels as the target appear in the image, and does it fit the Human Instruction's semantic description?
  - Agent-egocentric directions ("left / right / in front of / behind / near / far").

Penalise (NO):
  - Hallucinated objects (claimed but not in image).
  - Wrong target nomination vs. the instruction's semantic description (e.g., naming a phone as the target for "a luminous device to read at night").
  - Wrong egocentric direction (claimed left when actually right, etc.).
  - Wrong room type.

Do NOT penalise:
  - Omissions; only contradictions count.
  - Reasonable synonyms (e.g., "kettle" vs "pot") unless they change the target identity.
  - Hedged/soft language ("appears to be on the counter") supported by the image.

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>.
"""

NAV_GROUNDING_USER = """Human Instruction: {instruction}

Current Observation:
{_current_observation_text_or_image}

Reflection to verify:
{reasoning_tokens}
"""


class NavigationGroundingTemplate(VerifierTemplate):
    """Binary observation grounding check for navigation."""

    def __init__(self):
        super().__init__(
            template_id="navigation.grounding",
            description="Binary observation grounding (image-first).",
            required_keys=("reasoning_tokens",),
            system_prompt=NAV_GROUNDING_SYSTEM,
            user_prompt=NAV_GROUNDING_USER,
        )

    def _render_prompt(self, data: dict) -> dict:
        data.setdefault("instruction", data.get("human_instruction", "N/A"))
        return super()._render_prompt(data)


# ---- 2) Action Coherence (binary) ------------------------------------------
NAV_ACTION_COHERENCE_SYSTEM = """You evaluate whether the ordered action sequence logically follows from the agent's reflection in a home robot navigation task.

Action set (1–5 per turn, comma-separated, executed left-to-right):
  moveahead / moveback / moveleft / moveright   — translations (strafes; no heading change)
  rotateleft / rotateright                      — 90° rotations (heading and view change)
  lookup / lookdown                             — 30° camera tilts (no translation)

Critical semantics:
  • `moveleft` / `moveright` are STRAFES, NOT turns. "Turn right" → `rotateright`, not `moveright`.
  • `lookup` / `lookdown` only re-aim the camera; they do not move the agent.

Treat the action list as an ordered plan. Judge against the reflection AS WRITTEN (factual errors are scored under Observation Grounding).

Answer NO if:
  a) Translate ↔ rotate confusion: reflection explicitly plans a "rotate" / "turn to face X" but action is `moveright`/`moveleft` (a strafe); or vice versa. Reflections that only say "move toward X on the right" are ambiguous and NOT a confusion by themselves.
  b) Direction mismatch: the reflection's stated direction is absent from or contradicted by the action sequence (e.g., "try moving right instead" but the action is dominated by `moveleft` / `moveahead` with no `moveright`).
  c) Empty / vague reflection (no direction, axis, rotation, or tilt intent; purely meta — "I will move", "I need to navigate") with action not derivable from it.
  d) Sensing-vs-moving mismatch: reflection explicitly plans `lookdown` to inspect the floor but action contains only translations (or vice versa).

Allow:
  - Multi-action sequences whose overall composition matches the stated intent, even with repetition.
  - Coarse goals ("move toward the desk") paired with any concrete sequence whose component directions do NOT contradict the goal. Do not require the route to be direct or efficient.

Do NOT judge whether the action is OPTIMAL or whether it will SUCCEED.

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>.
"""

NAV_ACTION_COHERENCE_USER = """Reflection:
{reasoning_tokens}

Action taken:
{action_tokens}

Admissible actions:
{admissible_actions}

(Optional) Current Observation for contradiction checks only:
{_current_observation_text_or_image}
"""


class NavigationActionCoherenceTemplate(VerifierTemplate):
    """Binary action–reflection coherence check for navigation."""

    def __init__(self):
        super().__init__(
            template_id="navigation.action_coherence",
            description="Binary action-reflection coherence.",
            required_keys=("reasoning_tokens", "action_tokens", "admissible_actions"),
            system_prompt=NAV_ACTION_COHERENCE_SYSTEM,
            user_prompt=NAV_ACTION_COHERENCE_USER,
        )


# ---- 3) Temporal Consistency (binary) --------------------------------------
NAV_TEMPORAL_CONSISTENCY_SYSTEM = """You evaluate whether the CURRENT navigation reflection appropriately updates beliefs given the HISTORY.

CORE PRINCIPLE: Good reasoning updates beliefs/plans when evidence changes. A successful sub-action produces a new viewpoint; a failed sub-action means the view is unchanged; repeated failures should provoke a different plan, not the same one. Passivity in the face of new feedback is itself a failure mode.

Env feedback semantics: "Last action is executed successfully" / "not executed successfully" reports only the LAST sub-action's status. Earlier sub-actions may have partially advanced the agent; later ones following a failed sub-action are not executed.

Answer NO if ANY of these failures is present:
  a) Static / stale reasoning — current reflection's viewpoint / target / plan description is essentially unchanged from a recent prior reflection, despite intervening executed actions or a new image. Verbatim or near-verbatim repetition is a strong signal.
  b) False progress — "Now I see / Now I am near / I have reached X" after recent failed sub-actions, without the image actually changing in a supporting way.
  c) Stuck / oscillating — recent history alternates directions or repeats a near-identical sequence after failure, AND the reflection does NOT verbally acknowledge the repeated failure AND continues a near-identical approach without meaningful adaptation (rotate, look, or a different macro plan).
  d) False claims about past actions — claims of executed rotates / look-downs / movements that do not appear in the action history.
  e) Contradiction with prior reflection — flips on target identity, room, or geometry without new supporting evidence.
  f) Mis-attribution of partial execution — assuming the whole sequence succeeded (or failed) just because the last sub-action did.

Answer YES if: the reflection appropriately acknowledges feedback (success / partial / failure / new viewpoint) and updates beliefs / plans accordingly — OR there is genuinely nothing to update (first turn, or no executed sub-action and no new viewpoint).

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>.
"""

NAV_TEMPORAL_CONSISTENCY_USER = """Step index: {current_step}

Recent History (most recent first; at least one if available):
{_history_str}

Current Observation:
{_current_observation_text_or_image}

Current Reflection to verify:
{reasoning_tokens}
"""


class NavigationTemporalConsistencyTemplate(VerifierTemplate):
    """Binary temporal/history consistency check for navigation."""

    DEFAULT_HISTORY_WINDOW = 5

    def __init__(self, history_window: int = None):
        super().__init__(
            template_id="navigation.temporal_consistency",
            description="Binary temporal/history consistency.",
            required_keys=("reasoning_tokens",),
            system_prompt=NAV_TEMPORAL_CONSISTENCY_SYSTEM,
            user_prompt=NAV_TEMPORAL_CONSISTENCY_USER,
        )
        self.history_window = history_window or self.DEFAULT_HISTORY_WINDOW

    def _render_prompt(self, data: dict) -> dict:
        history = data.get("history", [])
        if isinstance(history, list) and history:
            formatted = []
            for i, h in enumerate(history):
                step = h.get("step", i + 1)
                action = h.get("action", "N/A")
                reflection = h.get("reflection", "")
                executed_text = h.get("executed_actions_text")
                feedback_text = h.get("env_feedback") or h.get("action_outcome_note")

                entry = f"[Step {step}]"
                if reflection:
                    entry += f"\nReasoning: {reflection}"
                entry += f"\nProposed action: {action}"
                if executed_text:
                    entry += f"\nExecuted sub-actions: {executed_text}"
                if feedback_text:
                    entry += f"\nEnv feedback: {feedback_text}"
                formatted.append(entry)
            data["_history_str"] = "\n\n".join(formatted[-self.history_window:])
        else:
            data["_history_str"] = "(No prior history available)"

        data.setdefault("current_step", "N/A")
        return super()._render_prompt(data)


# =============================================================================
# Factory / registry helpers
# =============================================================================

NAVIGATION_TEMPLATES = {
    "universal": NavigationUniversalTemplateV2,
    "universal_v2": NavigationUniversalTemplateV2,
    "grounding": NavigationGroundingTemplate,
    "action_coherence": NavigationActionCoherenceTemplate,
    "temporal_consistency": NavigationTemporalConsistencyTemplate,
    # Back-compat aliases (older codepaths referenced these names).
    "self_consistency": NavigationActionCoherenceTemplate,
    "history_consistency": NavigationTemporalConsistencyTemplate,
}


def get_navigation_template(rubric: str) -> VerifierTemplate:
    """Get a navigation template by rubric name."""
    if rubric not in NAVIGATION_TEMPLATES:
        raise ValueError(
            f"Unknown navigation rubric: {rubric}. "
            f"Available: {list(NAVIGATION_TEMPLATES.keys())}"
        )
    return NAVIGATION_TEMPLATES[rubric]()


def get_navigation_verifier_templates():
    objs = {
        "navigation.universal": NavigationUniversalTemplateV2(),
        "navigation.universal_v2": NavigationUniversalTemplateV2(),
        "navigation.grounding": NavigationGroundingTemplate(),
        "navigation.action_coherence": NavigationActionCoherenceTemplate(),
        "navigation.temporal_consistency": NavigationTemporalConsistencyTemplate(),
        # Back-compat aliases.
        "navigation.self_consistency": NavigationActionCoherenceTemplate(),
        "navigation.history_consistency": NavigationTemporalConsistencyTemplate(),
    }
    for k, v in objs.items():
        try:
            registry.register(k, v)
        except KeyError:
            pass
    return objs


if __name__ == "__main__":
    sample = {
        "id": "ex-1",
        "current_step": 3,
        "instruction": "navigate to the DeskLamp in the room and be as close as possible to it",
        "history": [
            {
                "step": 1,
                "reflection": "I see a desk with a lamp on it. I will move toward the desk.",
                "action": "moveahead, moveahead, moveahead, moveleft, moveahead",
                "executed_actions_text": "moveahead, moveahead, moveahead, moveleft, moveahead",
                "env_feedback": "Last action is executed successfully.",
            },
            {
                "step": 2,
                "reflection": "Since moving left was unsuccessful, I should try moving right instead.",
                "action": "moveright, moveright, moveright, moveahead, moveright",
                "executed_actions_text": "moveright, moveright, moveright, moveahead, moveright",
                "env_feedback": "Last action is not executed successfully.",
            },
        ],
        "current_observation_image": "<image>",
        "reasoning_tokens": "<reflection>Now I see the DeskLamp on the desk. I should move toward it.</reflection>",
        "action_tokens": "<action>moveahead, moveahead, moveleft, moveahead, moveleft</action>",
        "admissible_actions": [
            "moveahead", "moveback", "moveleft", "moveright",
            "rotateleft", "rotateright", "lookup", "lookdown",
        ],
    }
    verifiers = get_navigation_verifier_templates()
    universal = verifiers["navigation.universal"]
    msgs = universal.build_messages(sample)
    print("=== Navigation universal v2 messages ===")
    for m in msgs:
        print(f"\n[{m['role']}]\n{m['content']}")
