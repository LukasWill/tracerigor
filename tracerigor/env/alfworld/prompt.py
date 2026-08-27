"""System prompt, observation templates, and format prompts for ALFWorld."""
from typing import Callable, Dict, List


def system_prompt(render_mode: str = "text", **kwargs) -> str:
    """System prompt describing the ALFWorld household setting.

    Intentionally minimal: ALFWorld surfaces a per-step ``admissible_commands``
    list that is the single source of truth for what the agent may emit. A
    constant action-type taxonomy in the system prompt tends to invite the
    LLM to synthesise commands that aren't in the current admissible set,
    producing invalid actions. We mirror the verl-agent / GiGPO reference
    here and let admissible_commands carry the action surface.
    """
    base = (
        "You are an expert agent operating in the ALFRED Embodied Environment. "
        "Each turn you are given a task instruction, the current observation, "
        "and a list of admissible actions for the current situation. You must "
        "pick exactly one command from that admissible list and emit it "
        "verbatim — do not invent or rephrase commands that are not listed."
    )

    if render_mode == "vision":
        base += (
            " You will also receive an egocentric image of the current scene; "
            "use it together with the textual observation when choosing your action."
        )

    return base


def init_observation_template(
    task: str,
    observation: str,
    admissible_commands: str,
    **kwargs,
) -> str:
    """Initial-step observation template (no history)."""
    return f"""Your task is: {task}

Initial observation: {observation}

Admissible commands: [{admissible_commands}]

Decide your next action."""


def action_observation_template(
    task: str,
    action_history: str,
    current_step: int,
    observation: str,
    admissible_commands: str,
    last_action_valid: bool,
    **kwargs,  # accepts legacy history_length / step_count / last_action /
               # cumulative_reward kwargs so older call sites stay valid.
) -> str:
    """Subsequent-step observation template (with history).

    Four pieces of per-turn feedback were intentionally dropped from an
    earlier version of this template because they were either misleading
    or redundant against the history block emitted by
    :func:`format_action_history`:

    1. **Cumulative reward.** ALFWorld emits its task reward sparsely and
       only on terminal success (``win_reward × won``). The running total
       in our env additionally accumulates ``format_reward (+0.5/turn)``
       and ``invalid_action_penalty (-0.1/turn)`` noise, so the displayed
       number tracks format compliance × turn count rather than task
       progress. Surfacing it to the agent is actively misleading.
    2. **"Your last attempted action was: 'X'"** — the most recent action
       is already shown in the history block immediately above as
       ``Action: '<X>'``. Echoing it adds tokens and zero information.
    3. **Duplicate step counters.** "You have taken N step(s) so far"
       followed by "You are now at step N+1" said the same thing twice in
       different phrasings; the terse ``Step N.`` form is kept.
    4. **"Most recent {history_length} (observation, action) pair(s)"
       label.** ``format_action_history`` always emits *all* prior steps —
       older ones in a condensed action-only form, the last
       ``history_length`` with full observations — so the count in the
       header is stale once ``len(buffers) > history_length`` (the block
       contains many more lines than the label advertises). Replaced
       with a neutral "Episode history so far:" header.

    When the previous action was *not* in ``admissible_commands`` we still
    surface a one-line note: the ALFWorld env leaves the observation
    static in that case, and a naive reader of the history block might
    otherwise wonder why the environment "did not change" after the
    action was issued.
    """
    inadmissible_note = (
        ""
        if last_action_valid
        else (
            "\n(Note: your previous action was not in admissible_commands; "
            "the environment state did not advance.)"
        )
    )
    return f"""Your task is: {task}

Episode history so far:
{action_history}

Step {current_step}. Current observation: {observation}{inadmissible_note}

Admissible commands: [{admissible_commands}]

Decide your next action."""


# ---------------------------------------------------------------------------
# Format configurations (mirrors the canonical TraceRigor convention)
# ---------------------------------------------------------------------------

FORMAT_CONFIGS: Dict[str, Dict] = {
    "free_think": {
        "description": "First think step-by-step about the situation, then output the action.",
        "format": "<think>...</think><action>...</action>",
        "example": "<think>The task wants a clean mug on the desk. I should first locate a mug.</think><action>go to countertop 1</action>",
    },
    "no_think": {
        "description": "Output only the action, no reasoning.",
        "format": "<action>...</action>",
        "example": "<action>go to countertop 1</action>",
    },
    "grounding": {
        "description": "First describe the current observed state, then your reasoning, then the action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "example": (
            "<think><observation>I am in the middle of the room facing countertop 1. I can see a mug 1 and a knife 1.</observation>"
            "<reasoning>I should pick up mug 1 and then clean it at the sink.</reasoning></think>"
            "<action>take mug 1 from countertop 1</action>"
        ),
    },
    "worldmodeling": {
        "description": "First reason about the situation, predict the next state, then output the action.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": (
            "<think><reasoning>I need a clean mug, so I'll move to the sink.</reasoning>"
            "<prediction>After moving I will be facing sinkbasin 1 with a sink and faucet visible.</prediction></think>"
            "<action>go to sinkbasin 1</action>"
        ),
    },
    "grounding_worldmodeling": {
        "description": "Describe the observation, reason, predict the next state, then output the action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": (
            "<think><observation>I am holding mug 1 at sinkbasin 1.</observation>"
            "<reasoning>Cleaning is done by 'clean OBJ with sinkbasin'.</reasoning>"
            "<prediction>The mug will become clean and I will still hold it.</prediction></think>"
            "<action>clean mug 1 with sinkbasin 1</action>"
        ),
    },

    # ---------------------------------------------------------------------
    # ReflAct (arXiv:2505.15182). Per Fig. 16 / 18 of the paper's appendix
    # K.1.2, ALFWorld reflections in the ReflAct framework are a SINGLE
    # SENTENCE of prose that anchors the agent in its current state
    # ("Currently, I am at <loc>, [not] holding <obj>, …") and connects the
    # last observation to the task goal. They are NOT structured
    # "Location: … / Inventory: …" debug-prints. The XML wrapping is kept
    # because it is what our parser consumes; only the *content style*
    # mirrors the paper.
    # ---------------------------------------------------------------------
    "reflact": {
        "description": (
            "First reflect in ONE sentence on the agent's state in relation "
            "to the task goal — typically your current location, what (if "
            "anything) you are holding, and how the last observation moves "
            "you toward the goal — then output a single action."
        ),
        "format": "<reflection>...</reflection><action>...</action>",
        # Three modes drawn from Fig. 16/18 of the ReflAct paper, restated
        # in our own words on ALFWorld scenarios we synthesised so as not
        # to reproduce the paper's exemplars verbatim:
        #   1) initial planning  — sketches the recipe, picks first search loc
        #   2) state-grounded look-result — "Currently, I am at … not holding …"
        #   3) near-completion progress check — goal object in hand, last step
        "examples": [
            (
                "<reflection>To solve the task, I need to find a mug, clean "
                "it at the sinkbasin, and then place it on the coffeetable; "
                "mugs are most likely on a countertop, in a cabinet, or on "
                "the diningtable, so I will start by going to countertop 1."
                "</reflection><action>go to countertop 1</action>"
            ),
            (
                "<reflection>Currently, I am at countertop 1, not holding "
                "anything, and searching for a mug to clean and place on the "
                "coffeetable, but I only see a knife 1 and a peppershaker 1."
                "</reflection><action>go to countertop 2</action>"
            ),
            (
                "<reflection>Currently, I am at sinkbasin 1, holding a clean "
                "mug 1, and the task is nearly complete with only the "
                "placement on the coffeetable remaining."
                "</reflection><action>go to coffeetable 1</action>"
            ),
        ],
        # The default single example uses the mid-task state-grounded mode,
        # which is the dominant pattern across Fig. 16/18.
        "example": (
            "<reflection>Currently, I am at countertop 1, not holding "
            "anything, and searching for a mug to clean and place on the "
            "coffeetable, but I only see a knife 1 and a peppershaker 1."
            "</reflection><action>go to countertop 2</action>"
        ),
    },

    # Same three modes surfaced together (initial plan / state-grounded look /
    # near-completion). Useful at ICL / inference time when you want the model
    # to see the full range of reflection styles the paper uses.
    "reflact_diverse": {
        "description": (
            "First reflect in ONE sentence on the agent's state in relation "
            "to the task goal — your current location, what (if anything) "
            "you are holding, and how the last observation moves you toward "
            "the goal — then output a single action. Your reflection may "
            "take the form of an initial plan, a state-grounded look at "
            "what you just observed, or a near-completion progress check."
        ),
        "format": "<reflection>...</reflection><action>...</action>",
        "examples": [
            (
                "<reflection>To solve the task, I need to find a mug, clean "
                "it at the sinkbasin, and then place it on the coffeetable; "
                "mugs are most likely on a countertop, in a cabinet, or on "
                "the diningtable, so I will start by going to countertop 1."
                "</reflection><action>go to countertop 1</action>"
            ),
            (
                "<reflection>Currently, I am at countertop 1, not holding "
                "anything, and searching for a mug to clean and place on the "
                "coffeetable, but I only see a knife 1 and a peppershaker 1."
                "</reflection><action>go to countertop 2</action>"
            ),
            (
                "<reflection>Currently, I am at sinkbasin 1, holding a clean "
                "mug 1, and the task is nearly complete with only the "
                "placement on the coffeetable remaining."
                "</reflection><action>go to coffeetable 1</action>"
            ),
        ],
        "example": None,
    },
}


def _format_prompt_factory(format_type: str) -> Callable:
    def prompt_function(**kwargs) -> str:
        import random as _random

        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)
        # For reflact_diverse, surface all examples by default so the model
        # sees the full range of reasoning styles.
        if format_type == "reflact_diverse":
            use_diverse_examples = kwargs.get("use_diverse_examples", True)
            random_example = kwargs.get("random_example", False)
        else:
            use_diverse_examples = kwargs.get("use_diverse_examples", False)
            random_example = kwargs.get("random_example", False)

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown prompt_format: {format_type}")
        cfg = FORMAT_CONFIGS[format_type]

        text = (
            f"You may issue up to {max_actions_per_step} action(s) per turn, "
            f"separated by '{action_sep}'.\n"
            f"{cfg['description']}\n"
            f"Your response must follow this exact structure:\n{cfg['format']}"
        )
        if add_example:
            examples = cfg.get("examples") or []
            single = cfg.get("example")

            if use_diverse_examples and examples:
                text += "\n\nExamples:"
                for i, ex in enumerate(examples, 1):
                    text += f"\n{i}. {ex}"
            elif random_example and examples:
                text += f"\n\ne.g. {_random.choice(examples)}"
            elif single:
                text += f"\n\ne.g. {single}"
        return text

    return prompt_function


format_prompt: Dict[str, Callable] = {
    name: _format_prompt_factory(name) for name in FORMAT_CONFIGS
}


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------

def format_action_history(buffers: List[Dict], history_length: int = 2) -> str:
    """Compact textual rendering of the rolling (obs, action) history."""
    if not buffers:
        return "(no prior steps)"

    recent_start = max(0, len(buffers) - history_length)
    older = buffers[:recent_start]
    recent = buffers[recent_start:]

    out_lines: List[str] = []
    for j, record in enumerate(older):
        out_lines.append(f"  [Step {j + 1}] Action: '{record['action']}'")
    for j, record in enumerate(recent):
        step_no = recent_start + j + 1
        out_lines.append(
            f"  [Step {step_no}] Observation: {record['text_obs']!r}, "
            f"Action: '{record['action']}'"
        )
    return "\n" + "\n".join(out_lines)
