import re


REASONING_TAGS = ("think", "reflection", "planning", "explore", "monitor")


def extract_reasoning_content(response: str, fallback_to_raw: bool = True) -> str:
    """Extract the reasoning span from a tagged LLM response.

    The response may use one of several reasoning tags depending on prompt format.
    When no supported tag is present, the raw response is returned to preserve the
    existing logging fallback behavior.
    """
    if not response:
        return ""

    for tag in REASONING_TAGS:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, flags=re.S | re.I)
        if match:
            return match.group(1)

    return response if fallback_to_raw else ""


def replace_reasoning_block(original: str, new_reasoning: str) -> str:
    """Replace the existing reasoning block while preserving the response format.

    If the original response already contains a known reasoning tag, that tag is
    replaced in place. Otherwise a reasoning block is inserted before the action or
    answer block when possible. ReflAct responses preserve the `<reflection>` tag;
    other formats default to `<think>`.
    """
    if not original:
        return original

    for tag in REASONING_TAGS:
        match = re.search(rf"(.*?)(<{tag}>.*?</{tag}>)(.*)", original, flags=re.S | re.I)
        if match:
            return match.group(1) + f"<{tag}>{new_reasoning}</{tag}>" + match.group(3)

    insert_tag = "reflection" if re.search(r"<reflection>.*?</reflection>", original, flags=re.S | re.I) else "think"

    for anchor_tag in ("action", "answer"):
        match = re.search(rf"(.*?)(<{anchor_tag}>.*?</{anchor_tag}>)(.*)", original, flags=re.S | re.I)
        if match:
            return match.group(1) + f"<{insert_tag}>{new_reasoning}</{insert_tag}>" + match.group(2) + match.group(3)

    return original + f"\n<{insert_tag}>{new_reasoning}</{insert_tag}>"