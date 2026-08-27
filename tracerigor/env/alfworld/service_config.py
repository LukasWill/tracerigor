"""Service-level configuration for the ALFWorld environment."""
from dataclasses import dataclass

from tracerigor.env.base.base_service_config import BaseServiceConfig


@dataclass
class ALFWorldServiceConfig(BaseServiceConfig):
    """Configuration for :class:`tracerigor.env.alfworld.service.ALFWorldService`.

    Attributes:
        max_workers: Max parallel worker threads for batch operations
            (inherited from BaseServiceConfig).
        use_state_reward: Enable the optional LLM-as-judge state reward path.
        top_strings_m / top_strings_k: Anti-repetition buffer parameters used
            by the state-reward wrapper.
    """

    max_workers: int = 10

    use_state_reward: bool = False
    top_strings_m: int = 1000
    top_strings_k: int = 5
