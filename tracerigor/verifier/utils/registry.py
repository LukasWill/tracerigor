
from typing import Dict, Any

_REGISTRY: Dict[str, Any] = {}

def register(name: str, obj: Any) -> None:
    if name in _REGISTRY:
        raise KeyError(f"Template already registered: {name}")
    _REGISTRY[name] = obj

def get(name: str) -> Any:
    return _REGISTRY[name]

def available() -> Dict[str, Any]:
    return dict(_REGISTRY)
