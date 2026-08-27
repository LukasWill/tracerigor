import re
from typing import Dict, List
import json



def parse_freethink(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think>...</think><action>...</action>

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <think> and <action> tags
    - think_content: the content inside <think> tag
    - action_content: the content inside <action> (or legacy <answer>) tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    #Pattern to check for content strictly in the format <think>...</think><action>...</action>
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)


    # Pattern to extract content from think and answer tags
    extraction_pattern = r'<think>(.*?)</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list: # remove all special tokens in responses to forbid confusion in training
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + "<action>" + action_content.strip() + "</action>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_no_think(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <action>...</action>

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <action> tag
    - think_content: empty string (no think content in this format)
    - action_content: the content inside <action> (or legacy <answer>) tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the format <action>...</action>
    strict_pattern = r'^\s*<(?:answer|action)>(.*?)</(?:answer|action)>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from action/answer tag
    extraction_pattern = r'<(?:answer|action)>(.*?)</(?:answer|action)>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    #format_correct = match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        action_content = match.group(1)
        think_content = ""  # No think content in this format
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<action>" + action_content.strip() + "</action>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - observation_content: the content inside <observation> tag
    - think_content: the entire content inside <think> tag
    - reasoning_content: the content inside <reasoning> tag
    - action_content: the content inside <action> (or legacy <answer>) tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>'
    match = re.search(extraction_pattern, response, re.DOTALL)

    if not match:
        observation_content, reasoning_content, action_content, actions = "", "", "", []
        think_content = ""
    else:
        observation_content = match.group(1)
        reasoning_content = match.group(2)
        action_content = match.group(3)
        think_content = "<observation>" + observation_content + "</observation><reasoning>" + reasoning_content + "</reasoning>"

        if special_token_list is not None:
            for special_token in special_token_list:
                observation_content = observation_content.replace(special_token, "").strip()
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<action>" + action_content.strip() + "</action>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - think_content: the entire content inside <think> tag
    - reasoning_content: the content inside <reasoning> tag
    - prediction_content: the content inside <prediction> tag
    - action_content: the content inside <action> (or legacy <answer>) tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>'
    match = re.search(extraction_pattern, response, re.DOTALL)

    if not match:
        reasoning_content, prediction_content, action_content, actions = "", "", "", []
        think_content = ""
    else:
        reasoning_content = match.group(1)
        prediction_content = match.group(2)
        action_content = match.group(3)
        think_content = "<reasoning>" + reasoning_content + "</reasoning><prediction>" + prediction_content + "</prediction>"

        if special_token_list is not None:
            for special_token in special_token_list:
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                prediction_content = prediction_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<action>" + action_content.strip() + "</action>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - observation_content: the content inside <observation> tag
    - reasoning_content: the content inside <reasoning> tag
    - prediction_content: the content inside <prediction> tag
    - think_content: the entire content inside <think> tag
    - action_content: the content inside <action> (or legacy <answer>) tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<(?:answer|action)>(.*?)</(?:answer|action)>'
    match = re.search(extraction_pattern, response, re.DOTALL)

    if not match:
        observation_content, reasoning_content, prediction_content, action_content, actions = "", "", "", "", []
        think_content = ""
    else:
        observation_content = match.group(1)
        reasoning_content = match.group(2)
        prediction_content = match.group(3)
        action_content = match.group(4)
        think_content = "<observation>" + observation_content + "</observation><reasoning>" + reasoning_content + "</reasoning><prediction>" + prediction_content + "</prediction>"

        if special_token_list is not None:
            for special_token in special_token_list:
                observation_content = observation_content.replace(special_token, "").strip()
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                prediction_content = prediction_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<action>" + action_content.strip() + "</action>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }


# =============================================================================
# ReAct and ReflAct Parsers (from ReflAct paper: arXiv:2505.15182)
# =============================================================================

def parse_react(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in ReAct format: <think>...</think><action>...</action>

    ReAct emphasizes thinking about current condition and planning for future actions.

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <think> and <action> tags
    - think_content: the content inside <think> tag (condition + plan)
    - action_content: the content inside <action> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>", "")

    # Pattern to check for content strictly in the format <think>...</think><action>...</action>
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)

    # Pattern to extract content from think and action tags
    extraction_pattern = r'<think>(.*?)</think>\s*<action>(.*?)</action>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + "<action>" + action_content.strip() + "</action>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }


def parse_reflact(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in ReflAct format: <reflection>...</reflection><action>...</action>

    ReflAct emphasizes reflecting on agent's state (location, inventory, progress)
    in relation to the task goal before taking action.

    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <reflection> and <action> tags
    - reflection_content: the content inside <reflection> tag
    - think_content: alias for reflection_content (for compatibility)
    - action_content: the content inside <action> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>", "")

    # Pattern to check for content strictly in the format <reflection>...</reflection><action>...</action>
    strict_pattern = r'^\s*<reflection>(.*?)</reflection>\s*<action>(.*?)</action>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    if (
        len(re.findall(r"<reflection>", response)) != 1
        or len(re.findall(r"<action>", response)) != 1
    ):
        strict_match = None

    # Pattern to extract content from reflection and action tags
    extraction_pattern = r'<reflection>(.*?)</reflection>\s*<action>(.*?)</action>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match:
        reflection_content, action_content, actions = "", "", []
    else:
        reflection_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                reflection_content = reflection_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<reflection>" + reflection_content.strip() + "</reflection>" + "<action>" + action_content.strip() + "</action>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "reflection_content": reflection_content,
        "think_content": reflection_content,  # Alias for compatibility with training pipeline
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }


PARSE_FUNC_MAP = {
    "free_think": parse_freethink,
    "no_think": parse_no_think,
    "grounding": parse_grounding,
    "worldmodeling": parse_worldmodeling,
    "grounding_worldmodeling": parse_grounding_worldmodeling,
    "grounding_structured": parse_grounding,
    "worldmodeling_structured": parse_worldmodeling,
    "grounding_worldmodeling_structured": parse_grounding_worldmodeling,
    "grounding_symbolic": parse_grounding,
    "worldmodeling_symbolic": parse_worldmodeling,
    "grounding_worldmodeling_symbolic": parse_grounding_worldmodeling,
    # ReAct and ReflAct frameworks
    "react": parse_react,
    "reflact": parse_reflact,
    # reflact_diverse uses same format as reflact (<reflection>...<action>...)
    # but with multiple diverse ICL examples in the prompt
    "reflact_diverse": parse_reflact,
}
