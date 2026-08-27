"""
SciWorld Prompt Templates for TraceRigor.

This module defines system prompts, observation templates, and format configurations
for the ScienceWorld environment following TraceRigor conventions.
"""
from typing import Dict, Callable


# =============================================================================
# Action Definitions (constant across all prompts)
# =============================================================================

SCIWORLD_ACTIONS = """[
{"action": "open OBJ", "description": "open a container"},
{"action": "close OBJ", "description": "close a container"},
{"action": "activate OBJ", "description": "activate a device"},
{"action": "deactivate OBJ", "description": "deactivate a device"},
{"action": "connect OBJ to OBJ", "description": "connect electrical components"},
{"action": "disconnect OBJ", "description": "disconnect electrical components"},
{"action": "use OBJ [on OBJ]", "description": "use a device/item"},
{"action": "look around", "description": "describe the current room"},
{"action": "look at OBJ", "description": "describe an object in detail"},
{"action": "look in OBJ", "description": "describe a container's contents"},
{"action": "read OBJ", "description": "read a note or book"},
{"action": "move OBJ to OBJ", "description": "move an object to a container"},
{"action": "pick up OBJ", "description": "move an object to the inventory"},
{"action": "put down OBJ", "description": "drop an inventory item"},
{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"},
{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"},
{"action": "mix OBJ", "description": "chemically mix a container"},
{"action": "go to LOC", "description": "move to a new location"},
{"action": "eat OBJ", "description": "eat a food"},
{"action": "flush OBJ", "description": "flush a toilet"},
{"action": "focus on OBJ", "description": "signal intent on a task object"},
{"action": "wait", "description": "take no action for 10 iterations"},
{"action": "wait1", "description": "take no action for 1 iteration"},
{"action": "task", "description": "describe current task"},
{"action": "inventory", "description": "list your inventory"}
]"""


# =============================================================================
# System Prompt Function
# =============================================================================

def system_prompt(meta_think: bool = False, **kwargs) -> str:
    """
    Generate the system prompt for SciWorld environment.

    Args:
        meta_think: Whether to use meta-thinking multi-phase reasoning prompts
        **kwargs: Additional arguments (for future extensibility)

    Returns:
        System prompt string describing the environment and rules
    """
    base_prompt = f"""You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.

Here are the actions you may take:
{SCIWORLD_ACTIONS}

Note: OBJ and LOC should be replaced with actual objects/locations from your available actions."""

    if meta_think:
        base_prompt += """

When reasoning, you may use ONE of the following reasoning modes:
- <planning>: Plan the full task by breaking it into high-level steps. Use at the beginning or when replanning.
- <explore>: Think creatively about possible locations, items, or actions when facing obstacles.
- <reflection>: Analyze errors and consider alternative approaches when stuck.
- <monitor>: Track progress and ensure alignment with your overall plan during normal execution."""

    return base_prompt


# =============================================================================
# Observation Templates
# =============================================================================

def init_observation_template(
    task_description: str,
    observation: str,
    available_actions: str,
    **kwargs
) -> str:
    """
    Generate initial observation template (no history).

    Args:
        task_description: Description of the current task
        observation: Current environment observation
        available_actions: List of available actions

    Returns:
        Formatted initial observation string
    """
    return f"""Your current task is: {task_description}

Your current observation is: {observation}

Current available actions:
{available_actions}

Now it's your turn to take an action."""


def action_observation_template(
    task_description: str,
    step_count: int,
    history_length: int,
    action_history: str,
    current_step: int,
    observation: str,
    available_actions: str,
    planning: str = None,
    **kwargs
) -> str:
    """
    Generate observation template with history.

    Args:
        task_description: Description of the current task
        step_count: Total steps taken so far
        history_length: Number of historical steps included
        action_history: Formatted string of recent actions and observations
        current_step: Current step number
        observation: Current environment observation
        available_actions: List of available actions
        planning: Previous planning (for meta_think mode)

    Returns:
        Formatted observation string with history
    """
    base_obs = f"""Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step} and your current observation is: {observation}

Current available actions:
{available_actions}"""

    if planning:
        base_obs += f"\n\nYour previous overall plan is: {planning}. Please strictly adhere to your plan."

    base_obs += "\n\nNow it's your turn to take an action."

    return base_obs


# =============================================================================
# Format Configurations (following TraceRigor convention)
# =============================================================================

FORMAT_CONFIGS: Dict[str, Dict] = {
    "free_think": {
        "description": "You should first reason step-by-step about the current situation, then provide your action.",
        "format": "<think>...</think><action>...</action>",
        "example": "<think>I need to find a beaker to mix chemicals. Let me look around the room first.</think><action>look around</action>"
    },

    "no_think": {
        "description": "You should provide only your action.",
        "format": "<action>...</action>",
        "example": "<action>look around</action>"
    },

    "grounding": {
        "description": "You should first describe your observation of the current state, then your reasoning, and finally your action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "example": "<think><observation>I am in the kitchen. I can see a stove, a sink, and several containers on the counter.</observation><reasoning>To complete the task, I first need to find the necessary equipment. Let me examine what's available.</reasoning></think><action>look around</action>"
    },

    "worldmodeling": {
        "description": "You should first reason about the current situation, then predict what state you will be in after your action, and finally provide your action.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": "<think><reasoning>I need to heat the water to continue the experiment.</reasoning><prediction>After activating the stove, the water in the beaker should start heating up.</prediction></think><action>activate stove</action>"
    },

    "grounding_worldmodeling": {
        "description": "You should first describe your observation, then your reasoning, then predict the outcome, and finally provide your action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": "<think><observation>I see a beaker with water on the stove. The stove is currently off.</observation><reasoning>I need to heat this water for the experiment.</reasoning><prediction>Activating the stove will cause the water to heat up gradually.</prediction></think><action>activate stove</action>"
    },

    "meta_think": {
        "description": "You should use ONE reasoning mode (<planning>, <explore>, <reflection>, or <monitor>), then provide your action.",
        "format": "<planning|explore|reflection|monitor>...</planning|explore|reflection|monitor><action>...</action>",
        "example": "<monitor>I'm currently at step 3 of my plan. The next action should be to pick up the thermometer.</monitor><action>pick up thermometer</action>"
    },

    # ==========================================================================
    # ReAct and ReflAct frameworks (from ReflAct paper: arXiv:2505.15182)
    # ==========================================================================

    "react": {
        "description": "You should first think about the current condition and plan for your future actions, and then output your action in this turn.",
        "format": "<think>...</think><action>...</action>",
        "example": "<think>Current condition: I am in the kitchen with a beaker on the counter. The task requires me to heat water. Plan: First, I'll pick up the beaker, then move it to the stove, and finally activate the stove.</think><action>pick up beaker</action>"
    },

    "reflact": {
        "description": "You should first reflect on the agent's state, including your current location, inventory, and focused object, in relation to the task goal. Then, output the action for this turn.",
        "format": "<reflection>...</reflection><action>...</action>",
        # Multiple diverse examples to encourage varied reasoning patterns
        # (only one is randomly selected during format prompt generation)
        "examples": [
            # Example 1: Structured state-tracking style (Location/Inventory/Goal)
            "<reflection>Location: kitchen. Inventory: empty. Task goal: heat water to 100 degrees. Current progress: I have not yet acquired any items. The beaker with water is on the counter. Next step toward goal: pick up the beaker to begin the heating process.</reflection><action>pick up beaker</action>",

            # Example 2: Narrative obstacle-handling style (from ReflAct paper)
            "<reflection>I'm in the bedroom, but there are no useful materials here for creating the paint I need. The art studio would likely have the supplies I'm looking for, so I should navigate there next.</reflection><action>go to art studio</action>",

            # Example 3: Progress-monitoring and adaptation style
            "<reflection>Just moved to the foundry. I haven't found the thermometer yet despite checking the kitchen. Since the task requires measuring temperature, I should systematically search each room. Let me look around here first.</reflection><action>look around</action>",
        ],
        # Keep single example for backward compatibility
        "example": "<reflection>Location: kitchen. Inventory: empty. Task goal: heat water to 100 degrees. Current progress: I have not yet acquired any items. The beaker with water is on the counter. Next step toward goal: pick up the beaker to begin the heating process.</reflection><action>pick up beaker</action>"
    },

    # ReflAct with diverse examples enabled (uses all examples in prompt)
    "reflact_diverse": {
        "description": "You should first reflect on the agent's state in relation to the task goal. Your reflection can take various forms - you may describe your location and inventory, reason about obstacles, monitor your progress, or plan your next steps. Then, output the action for this turn.",
        "format": "<reflection>...</reflection><action>...</action>",
        "examples": [
            "<reflection>Location: kitchen. Inventory: empty. Task goal: heat water to 100 degrees. The beaker with water is on the counter. Next step: pick up the beaker.</reflection><action>pick up beaker</action>",
            "<reflection>I'm in the bedroom with no useful materials for my task. The art studio likely has what I need.</reflection><action>go to art studio</action>",
            "<reflection>Just arrived at the foundry. Haven't found the thermometer yet. Let me search this room before moving on.</reflection><action>look around</action>",
        ],
        "example": None  # Will use examples list instead
    }
}


def format_prompt_generator(format_type: str) -> Callable:
    """
    Generate a prompt function for the specified format type.

    Args:
        format_type: Type of format prompt to generate

    Returns:
        A function that generates the format prompt string
    """
    def prompt_function(**kwargs) -> str:
        import random

        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        # For reflact_diverse, automatically enable diverse examples
        # unless explicitly overridden by the caller
        if format_type == "reflact_diverse":
            use_diverse_examples = kwargs.get("use_diverse_examples", True)  # Default True for reflact_diverse
            random_example = kwargs.get("random_example", False)  # Don't randomly select for diverse
        else:
            use_diverse_examples = kwargs.get("use_diverse_examples", False)
            random_example = kwargs.get("random_example", False)  # Randomly select one example

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")

        config = FORMAT_CONFIGS[format_type]

        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
{config["description"]}"""

        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""

        if add_example:
            # Check if we have multiple examples
            examples = config.get("examples", [])
            single_example = config.get("example")

            if use_diverse_examples and examples:
                # Show all diverse examples (useful for inference/evaluation)
                base_prompt += "\n\nExamples:"
                for i, ex in enumerate(examples, 1):
                    base_prompt += f"\n{i}. {ex}"
            elif examples and random_example:
                # Randomly select one example from the list (default for training)
                selected_example = random.choice(examples)
                base_prompt += f"\n\ne.g. {selected_example}"
            elif single_example:
                # Fallback to single example
                base_prompt += f"\n\ne.g. {single_example}"

        return base_prompt

    return prompt_function


# Generate the format prompt dictionary
format_prompt: Dict[str, Callable] = {
    format_type: format_prompt_generator(format_type)
    for format_type in FORMAT_CONFIGS
}


# =============================================================================
# History Formatting Utilities
# =============================================================================

def format_action_history(
    buffers: list,
    history_length: int = 2,
    include_full_output: bool = False
) -> str:
    """
    Format action history for inclusion in observations.

    Args:
        buffers: List of history records with 'text_obs', 'action', and optionally 'full_output'
        history_length: Number of recent steps to include with full observations
        include_full_output: Whether to include reasoning traces

    Returns:
        Formatted action history string
    """
    if not buffers:
        return ""

    all_actions = [record["action"] for record in buffers]
    recent_history = buffers[-history_length:] if history_length > 0 else []
    recent_start_index = max(0, len(buffers) - history_length)

    action_history = ""

    # Show older actions in condensed form
    for j in range(recent_start_index):
        action = all_actions[j]
        step_number = j + 1
        action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

    # Show recent history with full observations
    for j, record in enumerate(recent_history):
        step_number = recent_start_index + j + 1
        env_obs = record["text_obs"]
        action = record["action"]
        action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

    # Optionally include reasoning traces
    if include_full_output:
        trace_length = min(3, len(buffers))
        start_index = len(buffers) - trace_length
        action_history += "\n- recent reasoning process: \n"
        for j, record in enumerate(buffers[-trace_length:]):
            if 'full_output' in record:
                step_number = start_index + j + 1
                action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"

    return action_history.strip()


if __name__ == "__main__":
    # Test the prompts
    print("System prompt (normal):")
    print(system_prompt(meta_think=False))
    print("\n" + "="*50 + "\n")

    print("System prompt (meta_think):")
    print(system_prompt(meta_think=True))
    print("\n" + "="*50 + "\n")

    print("Init observation template:")
    print(init_observation_template(
        task_description="Heat water to 100 degrees",
        observation="You are in the kitchen. You see a stove and a beaker.",
        available_actions="look around, pick up beaker, activate stove"
    ))
    print("\n" + "="*50 + "\n")

    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=1, action_sep=",", add_example=True))
        print("\n" + "="*30 + "\n")
