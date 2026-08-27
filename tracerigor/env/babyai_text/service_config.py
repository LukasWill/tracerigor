from dataclasses import dataclass

try:
    # Prefer the project’s base config if available
    from tracerigor.env.base.base_service_config import BaseServiceConfig
except Exception:
    # Fallback to keep imports safe in static analysis
    class BaseServiceConfig:
        max_workers: int = 10

@dataclass
class BabyAITextServiceConfig(BaseServiceConfig):
    """
    Service-level knobs for BabyAI service.
    These affect prompt formatting and minor rendering details at the service layer.
    """
    # Formatting / prompt assembly
    format_type: str = "grounding_worldmodeling"   # {"free_think","no_think","grounding","worldmodeling","grounding_worldmodeling"}
    max_actions_per_step: int = 1
    action_sep: str = ","
    add_example: bool = True
    image_tag: str = "<image>"

    # Optional mission override (otherwise derived from env config)
    mission: str | None = None

    use_state_reward: bool = False
