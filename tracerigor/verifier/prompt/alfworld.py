"""
ALFWorld LLM Judge Prompt Templates.

Evaluates Chain-of-Thought (CoT) quality for the LLM agent's interactions with
the text-based ALFWorld environment under the ReflAct format:
    <reflection>...</reflection><action>...</action>

Three evaluation rubrics (same schema as SciWorld / Sokoban v2 / Navigation v2,
so the universal 3-rubric JSON parser `_sciworld_universal_score` can be reused
without changes):

  1. observation_grounding  — does the reflection accurately describe what the
                              CURRENT observation (and the explicit
                              admissible-commands list) actually conveys about
                              the agent's location, what is visible there, what
                              the agent is holding, and the state of objects?
  2. action_coherence       — does the chosen single ALFWorld command logically
                              follow from the reflection's stated intent?
  3. temporal_consistency   — does the reflection update beliefs / plans
                              appropriately given the history of observations
                              and actions — particularly after an inadmissible
                              action (the env keeps the state frozen and the
                              prompt appends a "(Note: ... did not advance.)"
                              line), a state-transforming command, or a put?

ALFWorld-specific design notes (vs. SciWorld / Sokoban / Navigation):
  - Text-only env. Observations are short scripted sentences from TextWorld
    (e.g., "You arrive at countertop 1. On the countertop 1, you see ...",
    "You pick up the X from the Y.", "You cool the X using the fridge 1.",
    "You move the X to the Y."). There is no image and no grid.
  - The env exposes an explicit `Admissible commands: [...]` list every turn
    and only commands in that list advance the world. When the agent emits a
    command NOT in the list, the env (a) does NOT advance state, (b) re-emits
    the PRIOR turn's observation verbatim, and (c) the prompt template
    appends "(Note: your previous action was not in admissible_commands; the
    environment state did not advance.)". This is the dominant source of
    early-training temporal-consistency failures: the agent reads the stale
    observation as confirmation that its rejected action succeeded.
  - Tasks are long-horizon embodied recipes (find → take → optional
    transform [cool/heat/clean/slice] → put). State changes are produced
    ONLY by the relevant transformation verb (`cool X with fridge`,
    `heat X with microwave`/`stoveburner`, `clean X with sinkbasin`,
    `use desklamp`). `examine X` and `look` are read-only and do NOT change
    object state, location, or inventory — claims like "the apple is cool
    after examining it" are temporal-consistency failures.
  - History window is small (`history_length=2` in the run we analyse);
    older steps appear as action-only entries. The judge prompt must
    therefore reason from the (compressed) history plus the current
    observation, not from a long unfiltered transcript.
  - Reflection is ONE SENTENCE in the ReflAct style — typically
    "Currently, I am at <loc>, [not] holding <obj>, ..." — so the judge
    should NOT penalise terseness or absence of structured "Location: /
    Inventory: /" fields.
"""

from tracerigor.verifier.prompt.verifier_template_base import VerifierTemplate
from tracerigor.verifier.utils import registry


# =============================================================================
# Universal ALFWorld Verifier (all 3 rubrics in one call)
# =============================================================================
# JSON output keys match SciWorld / Sokoban v2 / Navigation v2 so the shared
# parser works.

ALFWORLD_UNIVERSAL_SYSTEM = """You are a strict ALFWorld verifier. Judge only the CURRENT step, but use history to catch contradictions.

ALFWorld is a text-based embodied household environment (TextWorld / ALFRED). The agent receives a one-sentence task instruction (e.g., "put some candle on toilet", "put a hot egg in fridge", "examine the bowl with the desklamp", "put two toiletpaper in toilet") and a short scripted observation each turn. The agent uses ReflAct-style reasoning:
<reflection>...</reflection><action>...</action>
exactly ONE command per turn.

The reflection is typically a single sentence in the form "Currently, I am at <location>, [not] holding <object>, <progress / next-step intent>." It should anchor the agent in (a) the current location, (b) what (if anything) is in the agent's inventory, (c) the visible objects relevant to the task, (d) the goal, and (e) the next step.

ENVIRONMENT MECHANICS YOU MUST KNOW:

A) Admissible-commands gating. Every user turn includes an explicit
   `Admissible commands: [...]` list. ONLY commands in that list advance
   the world. Common command families:
     • `go to <receptacle>`
     • `open <receptacle>` / `close <receptacle>` (cabinets, drawers,
       microwave, fridge, safe — most start closed)
     • `take <obj> from <receptacle>`
     • `move <obj> to <receptacle>` OR `put <obj> in/on <receptacle>`
       (the env exposes whichever phrasing is admissible for the current
       receptacle/object — the agent often invents one when the other is
       required; this is a leading source of inadmissible actions)
     • State-transformations: `cool <obj> with fridge <n>`,
       `heat <obj> with microwave <n>` (or `stoveburner <n>`),
       `clean <obj> with sinkbasin <n>`, `slice <obj> with <knife>`
     • `use <obj>` (specifically `use desklamp <n>` to turn it on for
       "examine X with desklamp" tasks)
     • Read-only: `examine <obj>`, `look`, `inventory`, `help`

B) Inadmissible-action behaviour. When the agent emits a command that is
   NOT in `Admissible commands`, the env (i) does NOT advance state,
   (ii) re-shows the PRIOR observation, and (iii) the prompt appends:
       "(Note: your previous action was not in admissible_commands;
        the environment state did not advance.)"
   The CURRENT observation in that case is STALE — it describes the
   state BEFORE the rejected action, not after.

C) Observation conventions (text-only; learn to recognise them):
     1. Initial room view: "-= Welcome to TextWorld, ALFRED! =- … you see a
        cabinet 1, a cabinet 2, a coffeemachine 1, …"
     2. Navigation success: "You arrive at X." (optionally followed by
        "On the X, you see ..." or "The X is closed.")
     3. Open: "You open the X. The X is open. In it, you see ..."
     4. Take: "You pick up the X from the Y."
     5. Put/move success: "You move the X to the Y." (also
        "You put the X in/on the Y." for some objects)
     6. State-transformation success:
          • "You cool the X using the fridge n."
          • "You heat the X using the microwave n." (or stoveburner)
          • "You clean the X using the sinkbasin n."
          • "You turn on the desklamp n."  (from `use desklamp n`)
     7. Look at current spot: "You are facing the X. Next to it, you see ..."
     8. Inventory: "You are carrying: ..." or "You are not carrying anything."
     9. Examine: "There's nothing special about X." or
          "This is a hot X." / "This is a cool X." (examine only READS
          state; it does not transform anything).
    10. Help: long generic command-syntax dump beginning with
          "Available commands:" — emitted when the agent issues `help`.
          This is NOT a confirmation of any task progress.

D) State-changing vs read-only verbs (used by Temporal Consistency
   below). `cool` / `heat` / `clean` / `slice` / `use desklamp` are the
   ONLY verbs that change an object's state. `take` / `put` / `move`
   are the only verbs that change inventory or placement. `examine`,
   `look`, `inventory`, `help` are READ-ONLY: "This is a hot X." after
   `examine X` is the env READING the existing state, not creating it.

Your tasks:

1) Observation Grounding — Does the reflection correctly describe what
   the CURRENT observation literally says about the visible scene and
   the agent's relation to the task target?
   Scope (current-observation-only): check the reflection's claims
   about (i) current location, (ii) objects visible AT the current
   location as listed in the current observation, and (iii) which
   object class the reflection nominates as the task target.
   Inventory, object-state, placement, and past-action claims are
   evaluated under Temporal Consistency, NOT here — even when they
   appear in the same sentence.
   Exception: such a claim IS in-scope here when the reflection
   explicitly presents the CURRENT observation itself as evidence
   that inventory / placement / object-state / task-progress has
   changed — e.g., "X is now in Y", "I have just placed X on Y",
   "Now I see X has been placed", "Having moved X to Y, the task is
   complete" — AND the literal current observation directly
   contradicts that change. The most common contradicting current-
   obs shapes are: (i) a stale "You pick up X from Z" carried over
   by an inadmissible action (often with the "(Note: ... did not
   advance.)" line), (ii) an "On Y, you see nothing." or other
   enumeration at the claimed destination that omits the claimed-
   placed object, (iii) the `help` command-syntax dump, (iv) an
   `examine`/`look` reply that reads existing state rather than a
   new change. Such a reflection misreads the CURRENT observation
   and IS a grounding failure here — even though the same failure
   may also be flagged under Temporal Consistency. That overlap is
   intentional, not an error.
   Penalise (NO) if any of these CLEAR contradictions with the current
   observation or the task instruction is present:
     a) Hallucinated visible object — reflection claims an object is
        present at the current location but the current observation
        does not list it. Synonyms / close confusables count when
        they change identity: "cup" ≠ "mug" if the task specifies
        "mug" (the env exposes both as distinct objects).
     b) Wrong current location — reflection names a location that
        the current observation contradicts (e.g., current obs is
        "You arrive at countertop 1." but reflection says "I am at
        the fridge").
     c) Wrong target object — the object the reflection labels as
        the task target is incompatible with the task instruction
        (e.g., task is "put some candle on toilet" but the
        reflection treats a `soapbar` as the target).
     d) Misread of a read-only / no-progress observation as task
        progress — e.g., interpreting the CURRENT `help` command-syntax dump
        ("Available commands: ...") or a bare CURRENT `examine`/`look` reply
        as direct confirmation that something was placed, transformed, or
        completed. The dump and the look/examine reply are
        observations only; they confirm no action outcome.
   Do NOT penalise:
     - Omissions. The reflection is not required to be exhaustive.
     - Minor preposition slip ("I am in the cabinet 1" vs "at
       cabinet 1") so long as it does not change the semantic
       location.
     - Hedged language ("Currently, I have not found a mug yet")
       supported by recent observations.
     - Inventory / object-state / placement / past-action claims —
       scored under Temporal Consistency, UNLESS the reflection
       explicitly treats the CURRENT observation itself as evidence
       for such a change (see the Exception above).
   No-claims case: if the reflection makes essentially no scene-or-
   target claim (e.g., a pure plan adjustment), there is nothing to
   ground-check — answer YES.

2) Action Coherence — Does the chosen single command logically follow
   from the reflection's stated intent?
   The reflection should state or clearly imply what to DO next: find
   somewhere, take a specific object, put / move an object somewhere,
   transform an object (cool/heat/clean/slice/use), open a receptacle,
   or check inventory / look. The action must implement that intent.
   Judge against the reflection AS WRITTEN: factual mis-grounding is
   scored under Observation Grounding, NOT here. If the reflection is
   mis-grounded but the action faithfully implements its mistaken plan,
   Action Coherence is YES.
   Penalise (NO) if:
     a) Verb / intent mismatch — reflection says "I need to take the
        mug" but the action is `go to <somewhere else>`; or "I need
        to place it in the toiletpaperhanger" but the action is
        `take ...`. The action must match the next-step verb the
        reflection commits to.
     b) Object mismatch — reflection commits to operating on object
        X (e.g., "take the toiletpaper 2") but the action targets a
        different object (e.g., `take soapbar 1`).
     c) Destination mismatch — reflection commits to a destination
        receptacle (e.g., "move the candle to the toilet") but the
        action targets a different receptacle (e.g., `move candle 1
        to shelf 1`). Only the DESTINATION must match; the choice of
        verb form ("move" vs "put in/on") does not matter here.
     d) Read-only action under an act-intent reflection — reflection
        states a concrete *committed* physical next step (specifically
        take / put / cool / heat / clean / use / open <named-object>)
        but the action is `help` / `inventory` / `look` / `examine`
        with no stated diagnostic purpose. (If the reflection
        explicitly says "let me check my inventory" / "let me look
        around", `inventory` / `look` IS coherent.)
        Exception: under a SEARCH / FIND intent (e.g., "I need to
        find X", "still searching for Y", "have not found Z yet"),
        BOTH `look` AND any `go to <receptacle>` ARE coherent — they
        are the env's only exploration primitives, and the rubric
        does NOT require the diagnostic verb to be invoked
        explicitly.
     e) Empty / purely meta reflection ("I need to continue", "I
        will take action") whose action cannot be plausibly tied to
        any stated intent.
   Allow:
     - Search / exploration under a coarse "find X" / "search for X"
       reflection: ANY `go to <receptacle>` is coherent — do NOT
       impose household priors about which receptacles are
       "plausible" locations for X (e.g., apples can be on
       stoveburners; watches on sofas; mugs in coffeemachines —
       household-knowledge plausibility is NOT a coherence check).
     - The reflection mentioning a final destination while choosing a
       prerequisite intermediate step (e.g., "I need to put the mug
       in the coffeemachine, so I'll go to the coffeemachine first"
       + action `go to coffeemachine 1`) — multi-step plans are fine
       as long as the chosen action is on the stated path.
   Do NOT judge whether the action is OPTIMAL, whether it is in
   `Admissible commands`, or whether it will SUCCEED — those are
   scored mechanically elsewhere.

3) Temporal Consistency — Does the CURRENT reflection update beliefs /
   plans appropriately given the HISTORY?

   CORE PRINCIPLE: Good reasoning updates beliefs and plans when
   evidence changes — a successful action changes inventory / state /
   location, an inadmissible action changes NOTHING, a state-frozen
   loop demands a different plan rather than the same one.
   Reflections that stay STATIC in the face of new feedback are
   temporally inconsistent even when they make no explicit
   contradictory claim — passivity in the presence of new evidence is
   itself a failure mode.

   If there is no prior history, answer YES unless the current
   reflection invents prior actions, outcomes, or progress that never
   happened.

   Reference rule: a `take` / `put` / `move` success in history is the
   only thing that changes inventory or placement; a `cool` / `heat` /
   `clean` / `slice` / `use desklamp` success is the only thing that
   changes object state. Claims that assert any of these effects
   without the corresponding history evidence fail this rubric.

   Interpretive note on the agent's "I have X" / "I have found X" /
   "I see X" phrasings in the reflection (the dominant alfworld
   surface form). Evaluate these against the evidence the environment
   has actually supplied, regardless of where the location qualifier
   sits in the sentence (glued: "I have X at Y" vs comma-separated:
   "I am at Y, ..., I have X" — score them the same):
     • If X appears in the CURRENT observation at the agent's current
       location, OR was enumerated by a recent `arrive at Y` /
       `open Y` observation at the relevant location, treat the claim
       as a VISIBILITY/recall claim — SUPPORTED, do NOT require a
       `take X` in history. The list-form "I have X, Y, Z" inside a
       multi-clause template means "X, Y, Z are visible/available
       here".
     • If X has appeared in NO past or current observation AND there
       is no `take X` in history, treat as a possession/recall claim
       that fails (this would trip failure (c) below).
     • "I have [VERB-ed] X" (checked / examined / opened / taken /
       moved / put / cooled / heated / cleaned / sliced) is a
       PAST-ACTION claim — see failure (g) for the per-verb
       requirements.
   Apply this disambiguation BEFORE deciding which failure (b–c–g)
   could apply.

   Check for these failures (answer NO if ANY is present):

   a) Belief not updated after an inadmissible action. The prior
      turn's action was inadmissible — the env signals this both by
      re-emitting the previous observation (so the current
      observation looks identical to the one before the rejected
      action) and by the "(Note: ... did not advance.)" line in the
      prompt — yet the reflection claims the intended outcome
      happened. Canonical FAILURE: prior `move candle 1 to toilet 1`
      rejected; current observation is the stale "You pick up the
      candle 1 from the shelf 1."; reflection says "Having moved the
      candle 1 to the toilet 1, I have now completed the task." → NO.

   b) Hallucinated state transformation. Reflection asserts a state
      change ("the egg is hot", "the mug is clean", "the desklamp is
      on") with no corresponding successful transformation verb
      (`heat` / `cool` / `clean` / `slice` / `use desklamp`) in the
      recent history. Note: an `examine X` reply such as "This is a
      hot/cool X." READS pre-existing state and does NOT license a
      newly-asserted transformation (the only legitimate "is hot"
      claim must be backed by either a successful transformation in
      history or such an examine reply).

   c) Hallucinated possession / placement. Reflection asserts "I have
      / am holding X" or "X is on/in Y" with no corresponding
      successful `take` / `put` / `move` for that X in history.
      Canonical FAILURE: reflection "Currently, I have the
      toiletpaper 2 and need to place it" after the agent only did
      `go to toiletpaperhanger 1` — no `take toiletpaper 2 from ...`
      executed yet.

   d) False task-completion claim. Reflection asserts "the task is
      complete", "no further action is required", or "I have
      successfully done X" when at least one required sub-step
      (find / take / transform / put) has no corresponding success
      in history. Help-spam loops ("I have completed the task" →
      `help` → "Available commands:" → "I have completed the
      task" → `help` …) are the strongest signature.

   e) Static / stale reasoning (passivity). Current reflection's
      progress / next-step description is essentially UNCHANGED from
      a recent prior reflection even though the prior turn executed
      an action or produced a new observation that should have
      shifted it. Verbatim or near-verbatim repetition across
      multiple turns is a strong signal — e.g., "Currently, I have
      no <item> and no relevant items." repeated while the agent
      keeps navigating between rooms that already revealed their
      contents.

   f) Stuck / oscillating actions without acknowledgment. Recent
      history shows the agent repeating a near-identical command (or
      oscillating between two) without progress, AND the current
      reflection does NOT verbally acknowledge the repetition AND
      continues the same approach without meaningful adaptation.
      Implicit adaptation alone (silently switching command without
      verbal acknowledgement) is NOT sufficient if the agent ALSO
      makes incorrect claims about inventory, state, or progress.

   g) False claims about past actions. Reflection asserts something
      happened in history that did not. Precise verb definitions:
        • "I have checked X" / "have searched X" / "have looked at X"
          is supported only if the agent has actually OBSERVED X's
          contents.
            - For OPEN receptacles (countertop, diningtable, shelf,
              sinkbasin, sidetable, bathtubbasin, ottoman, sofa,
              dresser, toilet, stoveburner, coffeemachine, ...),
              `go to X` is sufficient — the obs at arrival enumerates
              contents ("On the X, you see ...").
            - For CLOSED receptacles (cabinet, drawer, fridge,
              microwave, safe, garbagecan when shown closed, ...),
              `go to X` alone is NOT sufficient — bare arrival yields
              only "The X is closed." and contents remain unknown.
              The agent must additionally have a successful `open X`
              in history to support "checked X".
        • "I have examined X" refers strictly to an explicit
          `examine X` action in history; `look` and `go to` do NOT
          count.
        • "I have opened X" / "I have closed X" requires the
          corresponding successful `open X` / `close X` action.
        • "I have taken X" / "I am holding X" requires a successful
          `take X from <Y>` action.
        • "I have moved/put X [in/on] Y" requires a successful
          `move X to Y` or `put X in/on Y`.
        • "I have cooled/heated/cleaned/sliced X" requires the
          corresponding successful state-transformation action.
      Other "After [VERB-ing] X ..." claims must be backed by an
      actual VERB action in history.

   h) Goal drift / contradiction with prior reasoning. Current claims
      contradict the agent's own prior reflection without new
      evidence to justify the flip — e.g., prior reflection committed
      to "put on toilet" (matching the task) but the current
      reflection commits to "put on shelf"; or the target object
      class silently switches between turns.

   Answer YES if: the reflection appropriately acknowledges recent
   feedback — success, inadmissible, or no-op — and updates beliefs
   and plans accordingly — OR there is genuinely nothing to update
   (first turn, or no executed action and no state change).
   Answer NO if: any failure (a–h) is present.

Return a strict JSON object with EXACTLY these keys:
{
  "observation_grounding": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"},
  "action_coherence": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"},
  "temporal_consistency": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"}
}
Be concise and specific in evidence. Do not add extra keys.
"""

# Optional ground-truth section (currently unused — ALFWorld emits text
# observations directly; reserved for future trajectory-replay augmentation
# where a structured world-state dump could be injected).
ALFWORLD_GROUND_TRUTH_SECTION = """
[Ground Truth World State — USE THIS for verification]
{ground_truth_state_text}
"""

ALFWORLD_UNIVERSAL_USER = """Task: {task_description}

Step index: {current_step}

History (most recent steps, oldest first):
{history_str}

Current Observation:
{observation_content}
{inadmissible_note_block}{ground_truth_section}
Admissible commands at this step:
{admissible_commands_str}

Reflection to verify (current step):
{reflection_tokens}

Action taken (current step):
{action_tokens}
"""


class ALFWorldUniversalTemplate(VerifierTemplate):
    """Universal ALFWorld verifier: grounding, action coherence, temporal consistency.

    Designed for the LLM ReflAct agent on the text-based ALFWorld env:
      - Text-only observations (no image).
      - One command per turn, drawn from an explicit `admissible_commands`
        list surfaced at every step.
      - Windowed history matching the agent's generation-time context
        (the run we analyse uses `history_length=2`).
      - Inadmissible actions freeze the world and re-show the prior
        observation; the prompt template appends a `(Note: ...)` line in
        that case.
    """

    DEFAULT_HISTORY_WINDOW = 5

    def __init__(self, history_window: int = None):
        super().__init__(
            template_id="alfworld.universal",
            description="Universal ALFWorld verifier: grounding, action coherence, temporal consistency.",
            required_keys=("reflection_tokens", "action_tokens"),
            system_prompt=ALFWORLD_UNIVERSAL_SYSTEM,
            user_prompt=ALFWORLD_UNIVERSAL_USER,
        )
        self.history_window = history_window or self.DEFAULT_HISTORY_WINDOW

    def _render_prompt(self, data: dict) -> dict:
        data.setdefault("task_description", data.get("task", "N/A"))
        data.setdefault("current_step", data.get("step_index", "N/A"))

        # Optional ground-truth section (currently unused).
        gt_text = data.get("ground_truth_state_text")
        if gt_text:
            data["ground_truth_section"] = ALFWORLD_GROUND_TRUTH_SECTION.format(
                ground_truth_state_text=gt_text,
            )
        else:
            data["ground_truth_section"] = ""

        # Current observation — ALFWorld is text-only.
        obs_text = data.get("current_observation_text") or ""
        if obs_text.strip():
            data["observation_content"] = obs_text
        else:
            data["observation_content"] = "(no observation)"

        # Surface the inadmissible-action note if the orchestrator flagged
        # the *previous* action as inadmissible. Two compatible inputs:
        #   - boolean `last_action_inadmissible`
        #   - or pre-formatted `inadmissible_note` string (overrides bool)
        note_str = data.get("inadmissible_note")
        if not note_str and data.get("last_action_inadmissible"):
            note_str = (
                "(Note: your previous action was not in admissible_commands; "
                "the environment state did not advance.)"
            )
        data["inadmissible_note_block"] = (
            f"{note_str}\n" if note_str else ""
        )

        # Admissible commands at the current step.
        admissible = data.get("admissible_commands")
        if isinstance(admissible, (list, tuple)):
            data["admissible_commands_str"] = ", ".join(admissible) or "(none surfaced)"
        elif isinstance(admissible, str) and admissible.strip():
            data["admissible_commands_str"] = admissible
        else:
            data["admissible_commands_str"] = "(not provided)"

        # History formatting — text-only. Per-step entries surface the
        # observation, the reasoning (reflection), the chosen action, and
        # — when the env flagged it — whether that action was inadmissible.
        history = data.get("history", [])
        if isinstance(history, list) and history:
            formatted = []
            for i, h in enumerate(history):
                step = h.get("step", i + 1)
                obs = h.get("observation_text", "")
                reflection = h.get("reflection", "")
                action = h.get("action", "N/A")
                # Inadmissible flag may be either a boolean or an explicit note.
                inadm = h.get("last_action_inadmissible")
                outcome_note = h.get("action_outcome_note")

                entry = f"[Step {step}]"
                if obs:
                    entry += f"\nObservation: {obs}"
                if reflection:
                    entry += f"\nReasoning: {reflection}"
                entry += f"\nAction: {action}"
                if outcome_note:
                    entry += f"\nOutcome note: {outcome_note}"
                elif inadm:
                    entry += (
                        "\nOutcome note: previous action was not in "
                        "admissible_commands; environment state did not advance."
                    )
                formatted.append(entry)

            data["history_str"] = "\n\n".join(formatted[-self.history_window:])
        else:
            data["history_str"] = "(No prior history)"

        return data


# =============================================================================
# Binary single-rubric variants (extracted from the universal body)
# =============================================================================
# These mirror SciWorld's per-rubric binaries, for cases where the orchestrator
# wants to score one rubric at a time. Semantic content matches the
# corresponding section of ALFWORLD_UNIVERSAL_SYSTEM.

# ---- 1) Observation Grounding (binary) -------------------------------------
ALFWORLD_GROUNDING_SYSTEM = """You verify whether the agent's reflection accurately describes the CURRENT ALFWorld observation and the state derivable from it plus recent history.

ALFWorld is a text-based household environment. Observations are short scripted sentences:
  • "You arrive at X."  /  "On the X, you see ..."  /  "The X is closed."
  • "You open the X. The X is open. In it, you see ..."
  • "You pick up the X from the Y."
  • "You move the X to the Y."  /  "You put the X in/on the Y."
  • "You cool/heat/clean the X using the Y."   (state-transforming)
  • "You turn on the desklamp n."              (`use desklamp`)
  • "You are facing the X. Next to it, you see ..."   (`look`)
  • "You are carrying: ..."  /  "You are not carrying anything."   (`inventory`)
  • "There's nothing special about X." / "This is a hot/cool X."   (`examine` — READ-ONLY)
  • "Available commands: ..."                    (`help` — no task progress)

Inadmissible-action behaviour: if the prior action was not in `admissible_commands`, the env keeps state frozen and re-shows the prior observation; the prompt appends "(Note: ... did not advance.)". A STALE current observation is NOT confirmation that the intended action succeeded.

Check the reflection's claims about: current location, visible objects, inventory, object state (hot/cool/clean/on), and target identification (consistent with the Task).

Penalise (NO):
  a) Hallucinated visible object — claims an object is here that the current observation does not list.
  b) Hallucinated inventory / placement — claims to be holding X, or that X has been placed at Y, with no successful `take` / `put` / `move` for X in recent history (especially when the most recent action was inadmissible).
  c) Hallucinated state transformation — claims "X is cool/hot/clean/sliced/on" with no successful `cool` / `heat` / `clean` / `slice` / `use` in history. `examine` does NOT change state.
  d) Wrong target — labels an object as the task target that is incompatible with the task instruction.
  e) Misreading the `help` dump as task completion.

Do NOT penalise:
  - Omissions; only contradictions count.
  - Preposition slips ("in cabinet 1" vs "at cabinet 1") that don't change the semantic location.
  - Hedged claims ("I have not found a mug yet") supported by recent observations.
  - The single-sentence ReflAct style; do NOT require structured fields.

No-claims case: if the reflection makes essentially no scene / inventory / state / target claim (pure plan adjustment), answer YES — nothing to ground-check.

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>.
"""

ALFWORLD_GROUNDING_USER = """Task: {task_description}

Current Observation:
{_current_observation_text_or_image}
{inadmissible_note_block}
Reflection to verify:
{reflection_tokens}
"""


class ALFWorldGroundingTemplate(VerifierTemplate):
    """Binary observation grounding check for ALFWorld."""

    def __init__(self):
        super().__init__(
            template_id="alfworld.grounding",
            description="Binary observation grounding (text-only ALFWorld).",
            required_keys=("current_observation_text", "reflection_tokens"),
            system_prompt=ALFWORLD_GROUNDING_SYSTEM,
            user_prompt=ALFWORLD_GROUNDING_USER,
        )

    def _render_prompt(self, data: dict) -> dict:
        data.setdefault("task_description", data.get("task", "N/A"))
        note_str = data.get("inadmissible_note")
        if not note_str and data.get("last_action_inadmissible"):
            note_str = (
                "(Note: your previous action was not in admissible_commands; "
                "the environment state did not advance.)"
            )
        data["inadmissible_note_block"] = f"{note_str}\n" if note_str else ""
        return super()._render_prompt(data)


# ---- 2) Action Coherence (binary) ------------------------------------------
ALFWORLD_ACTION_COHERENCE_SYSTEM = """You evaluate whether the chosen single ALFWorld command logically follows from the agent's reflection (one command per turn, ReflAct format: <reflection>...</reflection><action>...</action>).

Common ALFWorld command families:
  • Navigate: `go to <receptacle>`
  • Container ops: `open <receptacle>` / `close <receptacle>`
  • Pick / place: `take <obj> from <receptacle>` , `move <obj> to <receptacle>` / `put <obj> in/on <receptacle>`
  • State-transform: `cool <obj> with fridge n` , `heat <obj> with microwave n` (or stoveburner) , `clean <obj> with sinkbasin n` , `slice <obj> with <knife>`
  • Special: `use desklamp n` (turn on for "examine X with desklamp" tasks)
  • Read-only: `examine <obj>` , `look` , `inventory` , `help`

Judge against the reflection AS WRITTEN. Factual mis-grounding is scored under Observation Grounding, not here. If the reflection is mis-grounded but the action faithfully implements its mistaken plan, Action Coherence is YES.

Answer NO if:
  a) Verb / intent mismatch — reflection commits to a next-step verb (take / put / cool / heat / clean / use / open / go) that the action does NOT implement (e.g., reflection "I need to take the toiletpaper" but action is `go to <somewhere else>`).
  b) Object mismatch — reflection commits to acting on object X but the action targets a different object.
  c) Receptacle mismatch — reflection commits to a specific destination receptacle but the action's destination is a different one. NOTE: writing `move` vs `put` (verb-form choice) is an admissibility issue, not a coherence issue; only flag DESTINATION mismatch.
  d) Read-only action under an act-intent reflection — reflection states a concrete physical next step but the action is `help` / `inventory` / `look` / `examine` with no stated diagnostic purpose. (If the reflection explicitly invokes the diagnostic — "let me check my inventory" — the read-only action IS coherent.)
  e) Empty / purely meta reflection ("I need to continue") whose action cannot be plausibly tied to any stated intent.

Allow:
  - Exploration moves under a high-level goal ("I need to find a mug") — any `go to <plausible-receptacle>` is coherent.
  - Multi-step plans where the action is on the stated path ("I need to put the mug in the coffeemachine, so I'll go there first" + `go to coffeemachine 1`).

Do NOT judge whether the action is OPTIMAL, in `Admissible commands`, or whether it will SUCCEED.

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>.
"""

ALFWORLD_ACTION_COHERENCE_USER = """Reflection:
{reflection_tokens}

Action taken:
{action_tokens}

Admissible commands at this step:
{admissible_commands_str}
"""


class ALFWorldActionCoherenceTemplate(VerifierTemplate):
    """Binary action-reflection coherence check for ALFWorld."""

    def __init__(self):
        super().__init__(
            template_id="alfworld.action_coherence",
            description="Binary action-reflection coherence (ALFWorld).",
            required_keys=("reflection_tokens", "action_tokens"),
            system_prompt=ALFWORLD_ACTION_COHERENCE_SYSTEM,
            user_prompt=ALFWORLD_ACTION_COHERENCE_USER,
        )

    def _render_prompt(self, data: dict) -> dict:
        admissible = data.get("admissible_commands")
        if isinstance(admissible, (list, tuple)):
            data["admissible_commands_str"] = ", ".join(admissible) or "(none surfaced)"
        elif isinstance(admissible, str) and admissible.strip():
            data["admissible_commands_str"] = admissible
        else:
            data["admissible_commands_str"] = "(not provided)"
        return super()._render_prompt(data)


# ---- 3) Temporal Consistency (binary) --------------------------------------
ALFWORLD_TEMPORAL_CONSISTENCY_SYSTEM = """You evaluate whether the CURRENT ALFWorld reflection appropriately updates beliefs / plans given the HISTORY.

CORE PRINCIPLE: Good reasoning updates beliefs and plans when evidence changes — a successful action changes inventory / state / location, an inadmissible action changes NOTHING, and a state-frozen loop demands a different plan rather than the same one. Passivity in the face of new feedback is itself a failure mode.

Env feedback semantics:
  • Successful action → new observation describing the change ("You pick up the X from Y.", "You move the X to Y.", "You cool the X using Y.", "You arrive at X.").
  • Inadmissible action → state frozen; current observation is a STALE repeat of the prior one; the prompt template appends "(Note: ... did not advance.)". The intended outcome did NOT happen.
  • `examine X` and `look` are READ-ONLY. They never change inventory, state, or location; "This is a hot/cool X." is a READ of state, not a transformation.
  • `help` returns a static command-syntax dump; it is NOT a confirmation of task progress.

Answer NO if ANY of these failures is present:

  a) Inadmissible-action mis-attribution — prior action was inadmissible (Note line / stale current obs) yet the reflection claims the intended outcome happened. Canonical: prior `move candle 1 to toilet 1` was inadmissible, current obs is the stale "You pick up the candle 1 ..." line, current reflection: "Having moved the candle to the toilet, the task is complete." → NO.
  b) Hallucinated state transformation — claims "X is cool/hot/clean/sliced/on" with no corresponding `cool` / `heat` / `clean` / `slice` / `use` success in history.
  c) Hallucinated possession / placement — claims "I have X" or "X is on/in Y" with no successful `take` / `put` / `move` for X in history.
  d) False progress / false task-completion — "task complete", "no further action required", "I have successfully placed X" when a required sub-step has no successful counterpart in history. Help-spam loops are the strongest signature.
  e) Static / stale reasoning — current reflection is essentially unchanged (often verbatim) from a recent prior one despite intervening actions / observations that should have shifted it.
  f) Stuck / oscillating actions without acknowledgment — repeats / oscillates a command without verbally acknowledging the repetition AND without meaningful adaptation.
  g) False claims about past actions — asserts a step happened in history that did not occur.
  h) Goal drift / contradiction with prior reasoning — current claims contradict the agent's own prior reflection (e.g., target object class or destination receptacle silently switched) without new evidence.

Answer YES if: the reflection appropriately acknowledges recent feedback and updates beliefs / plans accordingly — OR there is genuinely nothing to update (first turn, or no executed action and no state change).

Brevity: <think> <= 50 tokens, <=2 bullets or 1 short sentence.
Return exactly: <think>...</think><answer>YES|NO</answer>.
"""

ALFWORLD_TEMPORAL_CONSISTENCY_USER = """Step index: {current_step}

Recent History (oldest first; up to {history_length} steps):
{_history_str}

Current Observation:
{_current_observation_text_or_image}
{inadmissible_note_block}
Current Reflection to verify:
{reflection_tokens}
"""


class ALFWorldTemporalConsistencyTemplate(VerifierTemplate):
    """Binary temporal/history consistency check for ALFWorld."""

    DEFAULT_HISTORY_WINDOW = 5

    def __init__(self, history_window: int = None):
        super().__init__(
            template_id="alfworld.temporal_consistency",
            description="Binary temporal/history consistency (ALFWorld).",
            required_keys=("reflection_tokens",),
            system_prompt=ALFWORLD_TEMPORAL_CONSISTENCY_SYSTEM,
            user_prompt=ALFWORLD_TEMPORAL_CONSISTENCY_USER,
        )
        self.history_window = history_window or self.DEFAULT_HISTORY_WINDOW

    def _render_prompt(self, data: dict) -> dict:
        # Let the base populate `_current_observation_text_or_image` and its
        # default `_history_str` first, then override `_history_str` /
        # `history_length` with the ALFWorld-specific formatting. Otherwise
        # the base's `str(history)` would clobber our formatted string.
        data = super()._render_prompt(data)

        history = data.get("history", [])
        if isinstance(history, list) and history:
            formatted = []
            for i, h in enumerate(history):
                step = h.get("step", i + 1)
                obs = h.get("observation_text", "")
                reflection = h.get("reflection", "")
                action = h.get("action", "N/A")
                inadm = h.get("last_action_inadmissible")
                outcome_note = h.get("action_outcome_note")

                entry = f"[Step {step}]"
                if obs:
                    entry += f"\nObservation: {obs}"
                if reflection:
                    entry += f"\nReasoning: {reflection}"
                entry += f"\nAction: {action}"
                if outcome_note:
                    entry += f"\nOutcome note: {outcome_note}"
                elif inadm:
                    entry += (
                        "\nOutcome note: previous action was not in "
                        "admissible_commands; environment state did not advance."
                    )
                formatted.append(entry)
            data["_history_str"] = "\n\n".join(formatted[-self.history_window:])
            data["history_length"] = min(len(history), self.history_window)
        else:
            data["_history_str"] = "(No prior history available)"
            data["history_length"] = 0

        data.setdefault("current_step", "N/A")

        note_str = data.get("inadmissible_note")
        if not note_str and data.get("last_action_inadmissible"):
            note_str = (
                "(Note: your previous action was not in admissible_commands; "
                "the environment state did not advance.)"
            )
        data["inadmissible_note_block"] = f"{note_str}\n" if note_str else ""

        return data


# =============================================================================
# Factory / registry helpers
# =============================================================================

ALFWORLD_TEMPLATES = {
    "universal": ALFWorldUniversalTemplate,
    "grounding": ALFWorldGroundingTemplate,
    "action_coherence": ALFWorldActionCoherenceTemplate,
    "temporal_consistency": ALFWorldTemporalConsistencyTemplate,
}


def get_alfworld_template(rubric: str) -> VerifierTemplate:
    """Get an ALFWorld template by rubric name."""
    if rubric not in ALFWORLD_TEMPLATES:
        raise ValueError(
            f"Unknown ALFWorld rubric: {rubric}. "
            f"Available: {list(ALFWORLD_TEMPLATES.keys())}"
        )
    return ALFWORLD_TEMPLATES[rubric]()


def get_alfworld_verifier_templates():
    objs = {
        "alfworld.universal": ALFWorldUniversalTemplate(),
        "alfworld.grounding": ALFWorldGroundingTemplate(),
        "alfworld.action_coherence": ALFWorldActionCoherenceTemplate(),
        "alfworld.temporal_consistency": ALFWorldTemporalConsistencyTemplate(),
    }
    for k, v in objs.items():
        try:
            registry.register(k, v)
        except KeyError:
            pass
    return objs


if __name__ == "__main__":
    sample = {
        "id": "alf-ex-1",
        "task": "put some candle on toilet.",
        "current_step": 4,
        "history": [
            {
                "step": 2,
                "observation_text": (
                    "You arrive at shelf 1. On the shelf 1, you see a candle 1, "
                    "a soapbar 2, and a soapbar 1."
                ),
                "reflection": (
                    "Now I have identified a candle 1 on shelf 1, which is the "
                    "required item to complete the task."
                ),
                "action": "take candle 1 from shelf 1",
                "last_action_inadmissible": False,
            },
            {
                "step": 3,
                "observation_text": "You pick up the candle 1 from the shelf 1.",
                "reflection": (
                    "With the candle picked up from shelf 1, the next step is "
                    "to move it to the toilet."
                ),
                "action": "move candle 1 to toilet 1",
                "last_action_inadmissible": True,
            },
        ],
        "current_observation_text": "You pick up the candle 1 from the shelf 1.",
        "last_action_inadmissible": True,
        "admissible_commands": [
            "go to toilet 1", "go to shelf 2", "put candle 1 in/on toilet 1",
            "examine candle 1", "look", "inventory", "help",
        ],
        "reflection_tokens": (
            "<reflection>Now that I have picked up the candle 1, I should "
            "move it to the toilet 1 to complete the task.</reflection>"
        ),
        "action_tokens": "<action>move candle 1 to toilet 1</action>",
    }
    verifiers = get_alfworld_verifier_templates()
    universal = verifiers["alfworld.universal"]
    msgs = universal.build_messages(sample)
    print("=== ALFWorld universal messages ===")
    for m in msgs:
        print(f"\n[{m['role']}]\n{m['content']}")
