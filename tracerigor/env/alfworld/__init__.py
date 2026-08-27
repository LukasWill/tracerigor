from .env import ALFWorldEnv
from .env_config import ALFWorldEnvConfig
from .service import ALFWorldService
from .service_config import ALFWorldServiceConfig

ALFWORLD_ENV_INFO = {
    "env_cls": ALFWorldEnv,
    "config_cls": ALFWorldEnvConfig,
    "service_cls": ALFWorldService,
    "service_config_cls": ALFWorldServiceConfig,
    "description": "ALFRED household task environment (TextWorld / AI2-THOR backed)",
}

__all__ = [
    "ALFWorldEnv",
    "ALFWorldEnvConfig",
    "ALFWorldService",
    "ALFWorldServiceConfig",
    "ALFWORLD_ENV_INFO",
]
