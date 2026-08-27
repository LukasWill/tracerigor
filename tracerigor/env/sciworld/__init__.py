"""
SciWorld Environment Package for TraceRigor.

This package provides a TraceRigor-compliant wrapper for the ScienceWorld text-based
environment for training LLM agents on science curriculum tasks.

Exports:
    - SciWorldEnv: Main environment class
    - SciWorldEnvConfig: Environment configuration
    - SciWorldService: Service for batch operations
    - SciWorldServiceConfig: Service configuration
"""

from .env_config import SciWorldEnvConfig
from .env import SciWorldEnv
from .service import SciWorldService
from .service_config import SciWorldServiceConfig

__all__ = [
    "SciWorldEnv",
    "SciWorldEnvConfig",
    "SciWorldService",
    "SciWorldServiceConfig"
]

# Environment info for registration in REGISTERED_ENV
SCIWORLD_ENV_INFO = {
    "env_cls": SciWorldEnv,
    "config_cls": SciWorldEnvConfig,
    "service_cls": SciWorldService,
    "service_config_cls": SciWorldServiceConfig
}
