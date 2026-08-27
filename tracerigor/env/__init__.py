"""Environment registry with optional-dependency isolation.

Importing :mod:`tracerigor.env` must not require every supported environment.
Unavailable integrations are recorded in ``ENV_IMPORT_ERRORS`` and become
available automatically once their optional dependencies are installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict


_ENV_SPECS = {
    "sokoban": ("sokoban", "SokobanEnv", "SokobanEnvConfig", "SokobanService", "SokobanServiceConfig"),
    "frozenlake": ("frozenlake", "FrozenLakeEnv", "FrozenLakeEnvConfig", "FrozenLakeService", "FrozenLakeServiceConfig"),
    "navigation": ("navigation", "NavigationEnv", "NavigationEnvConfig", "NavigationService", "NavigationServiceConfig"),
    "svg": ("svg", "SVGEnv", "SvgEnvConfig", "SVGService", "SVGServiceConfig"),
    "primitive_skill": ("primitive_skill", "PrimitiveSkillEnv", "PrimitiveSkillEnvConfig", "PrimitiveSkillService", "PrimitiveSkillServiceConfig"),
    "alfworld": ("alfworld", "ALFWorldEnv", "ALFWorldEnvConfig", "ALFWorldService", "ALFWorldServiceConfig"),
    "blackjack": ("blackjack", "BlackjackEnv", "BlackjackEnvConfig", "BlackjackService", "BlackjackServiceConfig"),
    "babyai_text": ("babyai_text", "BabyAITextEnv", "BabyAITextEnvConfig", "BabyAITextService", "BabyAITextServiceConfig"),
    "sciworld": ("sciworld", "SciWorldEnv", "SciWorldEnvConfig", "SciWorldService", "SciWorldServiceConfig"),
}

REGISTERED_ENV: Dict[str, Dict[str, Any]] = {}
ENV_IMPORT_ERRORS: Dict[str, str] = {}


def register_environment(
    name: str,
    *,
    env_cls: type,
    config_cls: type,
    service_cls: type,
    service_config_cls: type,
) -> None:
    """Register an environment implementation under a stable public name."""
    REGISTERED_ENV[name] = {
        "env_cls": env_cls,
        "config_cls": config_cls,
        "service_cls": service_cls,
        "service_config_cls": service_config_cls,
    }
    ENV_IMPORT_ERRORS.pop(name, None)


def _load_environment(name: str) -> None:
    module_name, env_name, config_name, service_name, service_config_name = _ENV_SPECS[name]
    try:
        module = import_module(f"{__name__}.{module_name}")
        register_environment(
            name,
            env_cls=getattr(module, env_name),
            config_cls=getattr(module, config_name),
            service_cls=getattr(module, service_name),
            service_config_cls=getattr(module, service_config_name),
        )
    except Exception as exc:  # optional dependencies vary by integration
        ENV_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"


def get_environment(name: str) -> Dict[str, Any]:
    """Return a registered environment or raise an actionable error."""
    if name not in _ENV_SPECS:
        raise KeyError(f"Unknown environment {name!r}. Known: {sorted(_ENV_SPECS)}")
    if name not in REGISTERED_ENV:
        _load_environment(name)
    if name not in REGISTERED_ENV:
        detail = ENV_IMPORT_ERRORS.get(name, "unknown import failure")
        raise ImportError(f"Environment {name!r} is unavailable: {detail}")
    return REGISTERED_ENV[name]


def environment_status() -> Dict[str, Dict[str, str]]:
    """Return availability information without exposing local configuration."""
    return {
        name: {
            "status": "available" if name in REGISTERED_ENV else "unavailable",
            "detail": "" if name in REGISTERED_ENV else ENV_IMPORT_ERRORS.get(name, "not loaded"),
        }
        for name in _ENV_SPECS
    }


for _name in _ENV_SPECS:
    _load_environment(_name)


__all__ = [
    "REGISTERED_ENV",
    "ENV_IMPORT_ERRORS",
    "environment_status",
    "get_environment",
    "register_environment",
]
