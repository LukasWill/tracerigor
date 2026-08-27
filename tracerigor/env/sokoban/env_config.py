from tracerigor.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from .utils import generate_seeds
from tracerigor.verifier.verifier.common.config import VerifierConfig

@dataclass
class SokobanEnvConfig(BaseEnvConfig):
    env_name: str = "sokoban"
    # Sokoban-v0 (10, 10)	3 boxes
    # Sokoban-v1 (10, 10)	4
    # Sokoban-small-v0	(7, 7)	2
    # Sokoban-small-v1	(7, 7)	3
    # Sokoban-large-v0	(13, 11)	3
    # Sokoban-large-v1	(13, 11)	4
    dim_room: tuple = (6, 6)
    max_steps: int = 100
    num_boxes: int = 1
    render_mode: str = "vision" # "vision" or "text"
    min_actions_to_succeed: int = 5
    max_actions_per_step: int = 3
    noop_invalidation: bool = True   # If True, any no-op sub-action invalidates the turn (original behaviour).
                                      # If False, no-op sub-actions simply stop execution of the remaining
                                      # sub-actions but the prior valid sub-actions are *kept*, and the turn
                                      # is not marked invalid (format reward still granted if format is correct).
    prompt_format: str = "free_think"
    turn_wise_update: bool = False
    # Supported prompt formats:
    # - Basic: "free_think", "no_think"
    # - Grounding/Worldmodeling: "grounding", "worldmodeling", "grounding_worldmodeling"
    # - Symbolic variants: "grounding_symbolic", "worldmodeling_symbolic", "grounding_worldmodeling_symbolic"
    # - Structured variants: "grounding_structured", "worldmodeling_structured", "grounding_worldmodeling_structured"
    # - ReAct/ReflAct frameworks (arXiv:2505.15182): "react", "reflact"


    # configs for process reward for grounding and world modeling
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5

    # Violation tracking — tuned for short Sokoban episodes (5-10 turns)
    enable_violation_termination: bool = True
    format_violation_threshold: int = 2     # 2 consecutive → terminate
    invalid_action_threshold: int = 3       # 3 consecutive → terminate
    repetition_threshold: int = 2           # 2 consecutive → terminate
    violation_penalty: float = -1.0         # reward penalty on violation termination

    verifier: VerifierConfig = field(default_factory=VerifierConfig)

    def config_id(self) -> str:
        id_fields = ["dim_room", "max_steps", "num_boxes", "render_mode", "min_actions_to_succeed", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"SokobanEnvConfig({id_str})"

    def generate_seeds(self,size,seed=0,n_candidate: int = 20000,) -> list:
        return generate_seeds(size=size,
                              config=self,
                              min_actions_to_succeed=self.min_actions_to_succeed,
                              seed=seed,
                              n_candidate=n_candidate,)





if __name__ == "__main__":
    config = SokobanEnvConfig()
    print(config.config_id())
    print(config.generate_seeds(10))
