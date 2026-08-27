"""Shared utilities with optional training helpers loaded on demand."""

from __future__ import annotations

from typing import Any


_ENV_EXPORTS = {
    "permanent_seed",
    "set_seed",
    "NoLoggerWarnings",
    "setup_logging",
    "get_train_val_env",
}

__all__ = sorted(_ENV_EXPORTS)


def __getattr__(name: str) -> Any:
    """Preserve legacy convenience imports without eagerly importing PyTorch."""
    if name not in _ENV_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tracerigor.utils import env

    return getattr(env, name)
