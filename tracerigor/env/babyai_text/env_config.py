from dataclasses import dataclass, field
from tracerigor.env.base.base_env_config import BaseEnvConfig

@dataclass
class BabyAITextEnvConfig(BaseEnvConfig):
    """Config for BabyAI-Text adapter."""
    env_id: str = "BabyAI-MixedTrainLocal-v0"   # e.g., any registered BabyAI id
    subtask: str | None = None                  # e.g., "goto", "pickup", ...
    format_penalty: float = 0.0
    binary_reward: bool = False
    # Passed through to BabyAI wrappers / gym.make:
    babyai_kwargs: dict = field(default_factory=dict)

    def config_id(self) -> str:
        suffix = f"/{self.subtask}" if self.subtask else ""
        return f"BabyAIText({self.env_id}{suffix},fmt={self.format_penalty},bin={int(self.binary_reward)})"