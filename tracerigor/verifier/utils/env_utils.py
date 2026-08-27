from typing import Any, Dict, List, Optional

# somewhere central
ACTION_SPACES = {
    "sokoban": ["up","down","left","right"],
    "alfworld": ["go north","go south","go east","go west","examine","take","open"],
    # ...
}

def resolve_actions(item: Dict[str, Any]) -> Optional[List[str]]:
    if item.get("admissible_actions"):
        return item["admissible_actions"]
    env = (item.get("env") or "").lower()
    return ACTION_SPACES.get(env)