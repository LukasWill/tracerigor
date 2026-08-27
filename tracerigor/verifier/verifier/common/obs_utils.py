# verifier/common/obs_utils.py
from typing import Dict, Any, List

def extract_text_from_obs(obs: Dict[str, Any]) -> str:
    """
    Pull the textual observation string from an env observation dict.
    Compatible with Sokoban's {"obs_str": "..."} shape.
    """
    return (obs or {}).get("obs_str", "") or ""

def extract_images_from_obs(obs: Dict[str, Any], placeholder: str = "<image>") -> List[Any]:
    """
    Pull a list of images from an env observation dict.
    Returns raw objects (PIL.Image, bytes, paths, or URLs). Your normalizer handles them.
    Compatible with Sokoban's {"multi_modal_data": { "<image>": [ ... ] }} shape.
    """
    mm = (obs or {}).get("multi_modal_data") or {}
    imgs = mm.get(placeholder) or []
    return list(imgs)
