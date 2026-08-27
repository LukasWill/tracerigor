ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object one step in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object one step in front of you",
}


def get_instruction_prompt(env, mission="BabyAI-MixedTrainLocal-v0"):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!
""".strip()

    return instruction_prompt

def system_prompt(mission="BabyAI-MixedTrainLocal-v0", **kwargs):
    """
    Return a concise, reusable system-level prompt for BabyAI navigation/manipulation.
    Draws the action list and tips from the BabyAI module-level constants.
    """
    # Use the shared helper so the action strings stay in sync with BabyAI wrappers
    return get_instruction_prompt(env=None, mission=mission)


def init_observation_template(observation="", image_tag="<image>", **kwargs):
    """
    Initial observation presented to the agent before the first action.
    `observation` is the textual description produced by BabyAITextCleanLangWrapper
    (typically its long_term_context). We also surface the image placeholder.
    """
    return f"""[Initial Observation]
{observation}

Visual snapshot: {image_tag}

Decide your first move to progress toward the mission.
"""


def action_template(valid_action=None, observation="", image_tag="<image>", **kwargs):
    """
    Per-step prompt that reminds the agent of the extracted valid action
    (from the previous turn) and shows the current observation.
    """
    valid_action_str = ", ".join(valid_action) if valid_action else "None"
    return f"""After your answer, the extracted valid action: {valid_action_str}

Current state (textual description):
{observation}

Current visual snapshot: {image_tag}

Choose your next move. Use only one of the valid actions if possible.
"""


# FORMAT_CONFIGS parallel the Blackjack prompt file, adapted to BabyAI semantics
FORMAT_CONFIGS = {
    "free_think": {
        "description": (
            "You should first give your reasoning (plan and legality checks), "
            "and then your answer (one BabyAI action)."
        ),
        "format": "<think>...</think><answer>...</answer>",
        "example": (
            "<think>I need to reach the red door. The path is blocked on the right, "
            "so I should face left first, then approach.</think><answer>turn left</answer>"
        ),
    },

    "no_think": {
        "description": "You should provide only your answer (a single BabyAI action).",
        "format": "<answer>...</answer>",
        "example": "<answer>go forward</answer>",
    },

    "grounding": {
        "description": (
            "You should first describe what you see, then give your reasoning "
            "including legality checks, and finally your answer."
        ),
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": (
            "<think><observation>I face a corridor; a yellow key is two tiles ahead.</observation>"
            "<reasoning>Legal to move forward; approaching the key is required before pickup.</reasoning>"
            "</think><answer>go forward</answer>"
        ),
    },

    "worldmodeling": {
        "description": (
            "You should first give your reasoning, then predict the immediate state after your action, "
            "and finally your answer."
        ),
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": (
            "<think><reasoning>I must face the key to approach it.</reasoning>"
            "<prediction>After turning left, I will face the hallway leading to the key.</prediction>"
            "</think><answer>turn left</answer>"
        ),
    },

    "grounding_worldmodeling": {
        "description": (
            "You should first describe the observation, then your reasoning, then predict the state "
            "after your action, and finally your answer."
        ),
        "format": (
            "<think><observation>...</observation><reasoning>...</reasoning>"
            "<prediction>...</prediction></think><answer>...</answer>"
        ),
        "example": (
            "<think><observation>A blue door is in front; I hold no object.</observation>"
            "<reasoning>I must get adjacent to toggle the door; moving forward is legal.</reasoning>"
            "<prediction>After stepping forward, I will be in front of the blue door and can toggle.</prediction>"
            "</think><answer>go forward</answer>"
        ),
    },
}


def format_prompt_generator(format_type):
    """Generate a format-specific instruction block, identical pattern to Blackjack."""
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]

        # Expose the canonical BabyAI action names in-line (helps general LLMs stay on-rail)
        action_list = ", ".join([f'"{a}"' for a in [
            "turn left", "turn right", "go forward", "pick up", "drop", "toggle"
        ]])

        base_prompt = (
            f"You can take up to {max_actions_per_step} action(s) at a time, "
            f"separated by '{action_sep}'.\n"
            f"{config['description']}\n"
            f"\nAllowed actions: {action_list}\n"
            "Your answer MUST be a single allowed action unless explicitly instructed otherwise."
        )

        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""

        if add_example:
            example = config["example"].format(action_sep=action_sep)
            return base_prompt + "\n" + f"e.g. {example}"

        return base_prompt

    return prompt_function


# Dict of callable prompt builders, same shape as Blackjack
format_prompt = {fmt: format_prompt_generator(fmt) for fmt in FORMAT_CONFIGS}


if __name__ == "__main__":
    # quick manual test
    print("System prompt:")
    print(system_prompt(mission="BabyAI-MixedTrainLocal-v0/goto"))
    print("\n" + "=" * 50 + "\n")
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=1, action_sep=",", add_example=True))
        print("\n" + "=" * 30 + "\n")