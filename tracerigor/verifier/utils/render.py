
from typing import Dict, Any, Iterable

class MissingKeysError(KeyError):
    pass

def ensure_keys(data: Dict[str, Any], required: Iterable[str]) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise MissingKeysError(f"Missing required keys: {missing}")

def as_bullets(text: str, max_bullets: int = 2) -> str:
    """Normalize evidence into ≤max_bullets bullets (caller can pre-trim)."""
    lines = [ln.strip('- •\n ') for ln in text.split('\n') if ln.strip()]
    return '\n'.join(f"- {ln}" for ln in lines[:max_bullets])
