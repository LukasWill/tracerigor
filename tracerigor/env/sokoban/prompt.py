def system_prompt(**kwargs):
    return """You are an expert Sokoban solver.
# Goal:
Push all boxes onto targets. (You do NOT need to stand on targets; only boxes on targets count.)

# Rules:
1. You can only push boxes — never pull. Plan ahead to avoid deadlocks.
2. You cannot walk through walls or boxes.
3. Pushing a box into a corner or against a wall where it can never reach a target is an irreversible deadlock — avoid this.

# Visual elements (shown as an image):
- Player: a small green alien-like figure with two antennae and black eyes. This is you.
- Box: a yellow crate marked with an orange "X". You need to push this onto a target.
- Target: a black tile outlined in red with a small red diamond in the center.

# Orientation (all relative terms refer to the image axes):
- "above" = closer to the top of the image (lower row index)
- "below" = closer to the bottom (higher row index)
- "left"  = closer to the left edge (lower column index)
- "right" = closer to the right edge (higher column index)

# Actions (each moves you one square unless blocked):
- Up / Down / Left / Right
If the destination square contains a box and the next square beyond it (in the same direction) is empty or a target, you push the box one square in that direction."""

# - Player orientation: The player sprite always faces upward (toward the top); facing direction never changes and has no gameplay effect.

# Symbols (If image is provided there are no symbols):
# `#` Wall | `_` Floor | `O` Target | `X` Box | `P` You | `√` Box on Target | `S` You on Target

# # Legality check:
# Before any move, explicitly verify the move is legal—i.e. the destination square must be empty or a target, and if it contains a box, the next square in that same direction must also be empty or a target.

# Your admissible actions are ["left", "down", "right", "up"].

# You only move into a box to push it in the same direction you move, and only if the square beyond the box is empty or a target.

# "<think><observation>The box is above and right of the player, and the target is above and right of the player.</observation><reasoning>Grounding routine: (1) align with the box without pushing, (2) verify a free corridor from the box toward the target, (3) execute a single push. Concretely, I move Up to be left of the box (same-row), then Right to push the box right onto the target. Legality checks: Up is legal (cell above the player is free). After Up, I am left of the box; the cell to the right of the box is the target (free), so pushing Right is legal and does not create a corner trap.</reasoning><prediction>After these moves, relative to the player: the box will be same-row and right, and the target will be same-row and right.</prediction></think><action>Up{action_sep}Right</action>"

def init_observation_template(**kwargs):
    observation = kwargs.get("img_str", "The player is near a box")
    state_summary = kwargs.get("state_summary", "")
    extra = f"\n[State Summary] {state_summary}" if state_summary else ""

    if kwargs.get("turn_wise_update", False):
        return f"""Current observation is:
{observation}{extra}
Decide your next action(s)."""

    return f"""[Initial Observation]:
{observation}{extra}
Decide your next action(s)."""

def action_template(**kwargs):
    valid_action = kwargs.get("valid_action", "Down")
    observation = kwargs.get("img_str", "The player pushed the box closer to the target")
    state_summary = kwargs.get("state_summary", "")
    extra = f"\n[State Summary] {state_summary}" if state_summary else ""
    # Only show reflect hint when the verifier is actively providing feedback
    show_reflect_hint = kwargs.get("show_reflect_hint", False)
    reflect_hint = "\n(If optional feedback on your last action & reasoning is shown above, briefly reflect on it and adjust your plan.)" if show_reflect_hint else ""

    if kwargs.get("turn_wise_update", False):
        return f"""Current observation is:
{observation}{extra}
Decide your next action(s)."""

    return f"""After your answer, the extracted valid action is {valid_action}.
After that action, the observation is:
{observation}{extra}
Decide your next action(s).{reflect_hint}"""

# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><action>...</action>",
        "description": "You should first give your reasoning, and then your answer.",
        "example": (
            "Example 1:\n"
            "<think><reasoning>The box is one step below me, and the target is two steps below me, I need to go down then push the box down to the target.</reasoning></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><reasoning>The box is to my right but blocked below; I should step right, then up to get into pushing position.</reasoning></think>"
            "<action>Right{action_sep}Up</action>"
        )
    },

    "no_think": {
        "format": "<action>...</action>",
        "description": "You should provide only your answer.",
        "example": "<action>Down{action_sep}Down</action>"
    },

    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "example": (
            "Example 1:\n"
            "<think><observation>The box is below and in the same column as the player. The target is below and in the same column as the player, further past the box.</observation>"
            "<reasoning>Everything is vertically aligned. I push down to move the box onto the target.</reasoning></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><observation>The box is in the same row and to the left of the player. The target is below and to the left of the player.</observation>"
            "<reasoning>The target is below the box, so I need to push it down. I must get above the box first — detour up, then left to stand above it.</reasoning></think>"
            "<action>Up{action_sep}Left</action>"
        ),
        "example_single": (
            "Example 1:\n"
            "<think><observation>The box is below and in the same column as the player. The target is below and in the same column as the player, further past the box.</observation>"
            "<reasoning>Vertically aligned. I push down.</reasoning></think>"
            "<action>Down</action>\n"
            "Example 2:\n"
            "<think><observation>The box is in the same row and to the left of the player. The target is below and to the left of the player.</observation>"
            "<reasoning>I need to get above the box to push it down toward the target. I start detouring upward.</reasoning></think>"
            "<action>Up</action>"
        ),
        "additional_info": "Inside the `<observation>` tags, describe the position of the `target` (red) and each `box` (yellow) relative to the player. For each object, you must specify both its vertical (`above`, `below`, `same`) and horizontal (`left`, `right`, `same`) direction."
    },

    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "description": "You should first give your reasoning about which moves to make (including legality checks), then predict the next state after executing those moves, and finally output your answer.",
        "example": (
            "Example 1:\n"
            "<think><reasoning>I go right to align with the box's column, then down to push the box onto the target. Legality: Right is free; after Right I am above the box and the cell below the box is the target (free).</reasoning>"
            "<prediction>After those moves, the box will be below and in the same column as the player. The target will be below and in the same column (the box is on the target).</prediction></think>"
            "<action>Right{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><reasoning>I detour up to avoid contacting the box, then left to stand above it. Legality: Up is free; after Up, Left is free.</reasoning>"
            "<prediction>After those moves, the box will be below and in the same column as the player. The target will be below and to the left of the player.</prediction></think>"
            "<action>Up{action_sep}Left</action>"
        ),
        "additional_info": "Inside `<prediction>` tags, describe where each `box` (yellow) and each `target` (red) will be after you have executed your planned actions, specifying each object's both vertical (`above`, `below`, `same`) and horizontal (`left`, `right`, `same`) relation to the player."
    },

    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "description": "From the current observation, describe what you see, then give your reasoning (including legality checks), then predict the future state after your proposed moves, and finally output your answer.",
        "additional_info": "Within `<observation>` tags, describe the current positions of each `box` and each `target` relative to the player. Within `<prediction>` tags, describe their relative positions after your planned moves. Your observation/prediction must include two relations per entity: (1) a vertical relation in {`above`, `below`, `same`} and (2) a horizontal relation in {`left`, `right`, `same`}.",
        "example": (
            "Example 1:\n"
            "<think><observation>The box is below and to the right of the player. The target is below and to the right of the player.</observation>"
            "<reasoning>I move Right to align with the box's column, then Down to push it onto the target. Legality: the cell to the right is free; after Right I am above the box (same column), and the cell below the box is the target (free), so a Down push is legal.</reasoning>"
            "<prediction>After these moves, the box will be below and same column as the player. The target will be below and same column (the box is on the target).</prediction></think>"
            "<action>Right{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><observation>The box is in the same row and to the left of the player. The target is below and to the left of the player.</observation>"
            "<reasoning>The target is below the box, so I push the box down. I must get above the box first. Detour: Up to avoid contacting the box, then Left to stand above it, then Down to push. Legality: Up is free; after Up, Left is free; after Up+Left I am above the box and the cell below it is the target (free).</reasoning>"
            "<prediction>After these moves, the box will be below and same column as the player. The target will be below and same column (the box is on the target).</prediction></think>"
            "<action>Up{action_sep}Left{action_sep}Down</action>"
        )
    },

    "grounding_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": (
            "Example 1:\n"
            "<think><observation>####\n#_P#\n#__#\n#_X#\n#_O#</observation>"
            "<reasoning>I need to go down then push the box down to reach the target</reasoning></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><observation>#####\n#P__#\n#_X_#\n#_O_#\n#####</observation>"
            "<reasoning>Move Right to align with the box, then Down to push toward the target</reasoning></think>"
            "<action>Right{action_sep}Down</action>"
        )
    },

    "worldmodeling_symbolic": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "description": "You should first give your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": (
            "Example 1:\n"
            "<think><reasoning>I need to go down then push the box down to reach the target</reasoning>"
            "<prediction>####\n#__#\n#__#\n#_P#\n#_√#</prediction></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><reasoning>Step Right then Up to set up a vertical push</reasoning>"
            "<prediction>#####\n#_P_#\n#_X_#\n#_O_#\n#####</prediction></think>"
            "<action>Right{action_sep}Up</action>"
        )
    },

    "grounding_worldmodeling_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The observation and state should be represented as grids using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": (
            "Example 1:\n"
            "<think><observation>####\n#_P#\n#__#\n#_X#\n#_O#</observation>"
            "<reasoning>I need to go down then push the box down to reach the target</reasoning>"
            "<prediction>####\n#__#\n#__#\n#_P#\n#_√#</prediction></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><observation>#####\n#P__#\n#_X_#\n#_O_#\n#####</observation>"
            "<reasoning>Right then Down to push the box onto the target</reasoning>"
            "<prediction>#####\n#_P_#\n#___#\n#_√_#\n#####</prediction></think>"
            "<action>Right{action_sep}Down</action>"
        )
    },

    "grounding_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be in the format of {{\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}}",
        "example": (
            "Example 1:\n"
            "<think><observation>{{\"player\":(1,2),\"box\":(3,2),\"target\":(4,2)}}</observation>"
            "<reasoning>I need to go down then push the box down to the target</reasoning></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><observation>{{\"player\":(2,1),\"box\":(2,2),\"target\":(2,4)}}</observation>"
            "<reasoning>Move Right twice to push the box toward the target</reasoning></think>"
            "<action>Right{action_sep}Right</action>"
        )
    },

    "worldmodeling_structured": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be in the format of {{\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}}",
        "example": (
            "Example 1:\n"
            "<think><reasoning>I need to go down then push the box down to the target</reasoning>"
            "<prediction>{{\"player\":(3,2),\"box\":(4,2),\"target\":(4,2)}}</prediction></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><reasoning>Go Right to align, then Up to prepare a vertical push</reasoning>"
            "<prediction>{{\"player\":(1,2),\"box\":(2,2),\"target\":(2,4)}}</prediction></think>"
            "<action>Right{action_sep}Up</action>"
        )
    },

    "grounding_worldmodeling_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and prediction should be in the format of [{\"object_id\":target,\"vertical_relation\":\"xxx\",\"horizontal_relation\":\"xxx\"},{\"object_id\":box,\"vertical_relation\":\"xxx\",\"horizontal_relation\":\"xxx\"}]",
        "example": (
            "Example 1:\n"
            "<think><observation>[{{\"object_id\":\"target\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}},{{\"object_id\":\"box\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}}]</observation>"
            "<reasoning>I need to go down then push the box down to the target</reasoning>"
            "<prediction>[{{\"object_id\":\"target\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}},{{\"object_id\":\"box\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}}]</prediction></think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think><observation>[{{\"object_id\":\"box\",\"vertical_relation\":\"same\",\"horizontal_relation\":\"right\"}},{{\"object_id\":\"target\",\"vertical_relation\":\"same\",\"horizontal_relation\":\"right\"}}]</observation>"
            "<reasoning>Move Right to stand left of the box, then Right to push it onto the target</reasoning>"
            "<prediction>[{{\"object_id\":\"box\",\"vertical_relation\":\"same\",\"horizontal_relation\":\"same\"}},{{\"object_id\":\"target\",\"vertical_relation\":\"same\",\"horizontal_relation\":\"same\"}}]</prediction></think>"
            "<action>Right{action_sep}Right</action>"
        )
    },

    # ==========================================================================
    # ReAct and ReflAct frameworks (from ReflAct paper: arXiv:2505.15182)
    # ==========================================================================

    "react": {
        "format": "<think>...</think><action>...</action>",
        "description": "You should first think about the current condition and plan for your future actions, and then output your action in this turn.",
        "example": (
            "Example 1:\n"
            "<think>Current condition: The box is directly below me, same column. The target is below the box, same column. Plan: I am already above the box and aligned. I push down two times.</think>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<think>Current condition: The box is to my left, same row. The target is below and to the left. Plan: I need to push the box down, which requires standing above it. I detour up first to avoid bumping the box sideways, then move left to get above it.</think>"
            "<action>Up{action_sep}Left</action>"
        ),
        "example_single": (
            "Example 1:\n"
            "<think>Current condition: I am directly above the box. The target is below the box. Plan: Push down.</think>"
            "<action>Down</action>\n"
            "Example 2:\n"
            "<think>Current condition: The box is to my left, same row. The target is below the box. Plan: I need to get above the box to push it down. I start detouring up.</think>"
            "<action>Up</action>"
        )
    },

    "reflact": {
        "format": "<reflection>...</reflection><action>...</action>",
        "description": "You should first reflect on the agent's state, including your current position relative to the box and target, in relation to the task goal. Then, output your action for this turn.",
        "example": (
            "Example 1:\n"
            "<reflection>Position: I am directly above the box, same column. The box is directly above the target, same column. Task goal: push the box onto the target. Progress: The box and target are vertically aligned — two downward pushes solve it.</reflection>"
            "<action>Down{action_sep}Down</action>\n"
            "Example 2:\n"
            "<reflection>Position: The box is to my left, same row. The target is below the box. Task goal: push the box down onto the target. Progress: I need to be above the box to push it down, but I am to its right. I must detour around it — go up first to avoid bumping it, then left to position above it.</reflection>"
            "<action>Up{action_sep}Left</action>"
        ),
        "example_single": (
            "Example 1:\n"
            "<reflection>Position: I am directly above the box, same column. The box is above the target, same column. Task goal: push the box onto the target. Progress: The box and target are aligned vertically — I push down.</reflection>"
            "<action>Down</action>\n"
            "Example 2:\n"
            "<reflection>Position: The box is to my left, same row. The target is below the box. Task goal: push the box down onto the target. Progress: I need to get above the box first, so I start detouring upward.</reflection>"
            "<action>Up</action>"
        )
    },
}

def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified format type.

    Args:
        format_type (str): The format type to generate a prompt function for

    Returns:
        function: A function that generates a prompt for the specified format
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format.

        Args:
            max_actions_per_step (int): Maximum number of actions allowed per step
            action_sep (str): Separator between actions
            add_example (bool): Whether to add an example

        Returns:
            str: The formatted prompt
        """
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", "|")
        add_example = kwargs.get("add_example", True)
        config = FORMAT_CONFIGS[format_type]

        # Build the base prompt text
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}"""

        # Add additional information if available
        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info']}"

        # Add response format instruction
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""

        # Select example variant: use single-action examples when available
        # and max_actions_per_step == 1, otherwise fall back to default.
        if add_example:
            if max_actions_per_step == 1 and "example_single" in config:
                ex = config["example_single"]
            else:
                ex = config["example"]
            # Handle both string and tuple/list examples
            if isinstance(ex, (list, tuple)):
                examples_text = "\n\n".join(
                    e.format(action_sep=action_sep) for e in ex
                )
                return base_prompt + "\n" + f"e.g. {examples_text}"
            else:
                example = ex.format(action_sep=action_sep)
                return base_prompt + '\n' + f"e.g. {example}"

        return base_prompt

    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type)
                for format_type in FORMAT_CONFIGS}


visual_reasoning_reward_prompt="""You are a text parser assistant. Your task is to extract relative spatial relationships between objects and the 'player' from a given text description (either an observation or a prediction) in a Sokoban environment, and output this information in a structured JSON format.

**Input:** You will receive a block of text that describes a state or a predicted state.
**Output:** You must output a JSON array. Each object in the array describes the relationship of one specific non-player object to the player.

**Objects to Look For:**
- target
- **box** (Treat any mention of 'box', 'box0', 'box1', etc., as referring to the general 'box' type.)

**Required JSON Output Format:**
A JSON array [...] where each element is an object {{...}} with the following keys:
- "object_id": string. The identifier of the object. Must be "target", or "box". **For any mention of any box (box0, box1, 'a box', the box'), this MUST be "box".**
- "vertical_relation": string or null. The vertical position relative to the player. Must be one of: "above", "below", "same", or null if not mentioned or unclear.
- "horizontal_relation": string or null. The horizontal position relative to the player. Must be one of: "left", "right", "same", or null if not mentioned or unclear.

**Instructions:**
1.  Read the input text carefully.
2.  Identify descriptions of relative position for any of the "Objects to Look For" with respect to the "player".
3.  **Crucially, for any mention of any box (e.g., "box0", "box1", "the box", "a box"), extract its relationship but set its "object_id" to "box" in the output JSON.** If multiple boxes have distinct relationships, create a separate JSON object entry for each distinct relationship, all with "object_id": "box".
4.  Extract the vertical and horizontal components of the relationship. **Map variations in phrasing (like "on the left side", "to the right of the player", "same position as player", "same spot as target", "same spot as a box") to the specific allowed terms ("above", "below", "same", "left", "right").**
5.  **Handling Absolute Positions (Edge Case):** If the text describes an object's position using *absolute* terms like "top-left corner", "bottom-right corner", and the player's relative position is implied (e.g., player is known to be at the opposite corner), attempt to infer the relative position *relative to the player* using the allowed terms ("above", "below", "left", "right", "same"). For example, if the text says "target is at the top-left corner" and the player is implied to be at the bottom-right, infer "target is above and left of player". *However, prioritize extraction from text that uses direct relative terminology.*
6.  **Handling Partial or Unclear Information:** If a specific object (target, box) is mentioned but *only* its vertical or horizontal relationship is described (or only implicitly), set the other relationship to null. If a relationship is too vague ("near") or doesn't use terms that can be mapped to the allowed set, set both vertical_relation and horizontal_relation to null for that object, or consider omitting the object's entry if the description is entirely unmappable.
7.  **Ignoring Irrelevant Information:** Ignore any part of the text that does not describe the relative position of an object from the "Objects to Look For" list with respect to the player, especially descriptions of actions, reasons, or generic objects without IDs.
8.  If an object from the "Objects to Look For" list is *not* mentioned in the input text with a relevant description, do NOT include it in the output JSON array.
9.  If the input text contains no extractable relationships for the listed objects using the allowed terms and inference rules, output an empty JSON array [].
10. Your output must be *only* the JSON string, and it must be valid JSON.

**Example 1 (Target and Box):**
Input Text:
The target is above and left of the player. A box is right of the player.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "target",
    "vertical_relation": "above",
    "horizontal_relation": "left"
  }},
  {{
    "object_id": "box",
    "vertical_relation": null, // Vertical relation not mentioned
    "horizontal_relation": "right"
  }}
]
```
*(Note: The box's vertical relation was not specified, so it's null.)*

**Example 2 (Multiple Boxes):**
Input Text:
There is a box above me and another box left of me. The target is far above and left.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "box",
    "vertical_relation": "above",
    "horizontal_relation": null // Horizontal relation not mentioned for the first box
  }},
  {{
    "object_id": "box",
    "vertical_relation": null, // Vertical relation not mentioned for the second box
    "horizontal_relation": "left"
  }},
  {{
    "object_id": "target",
    "vertical_relation": "above",
    "horizontal_relation": "left" // "far above and left" maps to "above" and "left"
  }}
]
```
*(Note: Two distinct box relationships are listed, both with object_id "box". Vertical/horizontal relations are null if not explicitly mentioned.)*

**Example 3 (Box on Target / Same Place):**
Input Text:
The box is on the target, and they are both at the same spot as the player.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "box",
    "vertical_relation": "same",
    "horizontal_relation": "same"
  }},
  {{
    "object_id": "target",
    "vertical_relation": "same",
    "horizontal_relation": "same"
  }}
]
```
*(Note: "on the target" combined with "same spot as player" implies the box is also at the same spot as the player.)*

**Example 4 (Absolute Corners - Sokoban Context):**
Input Text:
The player is at the bottom-right corner. The target is at the top-right. A box is at the bottom-left.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "target",
    "vertical_relation": "above", // Target at top-right relative to player at bottom-right
    "horizontal_relation": "same"
  }},
  {{
    "object_id": "box",
    "vertical_relation": "same", // Box at bottom-left relative to player at bottom-right
    "horizontal_relation": "left"
  }}
]
```
*(Note: Inferring relative positions from absolute corner descriptions based on player's known corner.)*

**Example 5 (No Relevant Info):**
Input Text:
I need to find the correct path.

Objects to Look For: target, box

Expected JSON Output:
```json
[]
```

---
Input Text to Parse:
{prediction}

Objects to Look For: target, box # Ensure this list matches the main list above if needed

JSON Output:
"""

if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"

    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep, add_example=True))
        print("\n" + "="*50 + "\n")
    print(visual_reasoning_reward_prompt.format(prediction="123"))