from typing import Any, Dict, List, Optional, Tuple

try:
    from tracerigor.env.base.base_service import BaseService
except Exception:
    class BaseService:  # fallback for type-checkers
        pass

from tracerigor.server.serial import serialize_observation

from .env import BabyAITextEnv
from .env_config import BabyAITextEnvConfig
from .service_config import BabyAITextServiceConfig
from .prompt import (
    system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)


class BabyAITextService(BaseService):
    """
    Batch-capable service for BabyAI-Text that satisfies the BaseService interface.
    Also exposes single-env convenience helpers (build_initial_messages / build_step_messages).
    """

    def __init__(self, config: BabyAITextServiceConfig):
        self.config = config
        self.environments: Dict[Any, BabyAITextEnv] = {}   # env_id -> env instance
        self.env_configs: Dict[Any, BabyAITextEnvConfig] = {}  # env_id -> env config

        # Optional feature parity with other services
        if getattr(self.config, "use_state_reward", False):
            try:
                from tracerigor.env.utils.top_string_tracker import TopKStringTracker
                # Example field; harmless if unused
                self.top_strings_tracker = TopKStringTracker(getattr(self.config, "top_strings_m", 10))
            except Exception:
                # Silently ignore if utility is unavailable; not needed for BabyAI
                self.top_strings_tracker = None

    # ---------------------------------------------------------------------
    # Required by BaseService
    # ---------------------------------------------------------------------
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        ids2configs: env_id -> dict with at least {'env_config': {...}}
        """
        for env_id, cfg in ids2configs.items():
            env_cfg_dict = (cfg or {}).get("env_config", {})
            env_cfg = BabyAITextEnvConfig(**env_cfg_dict)
            env = BabyAITextEnv(env_cfg)

            self.environments[env_id] = env
            self.env_configs[env_id] = env_cfg

    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, Dict]]:
        """
        Returns: env_id -> (serialized_observation, info)
        """
        results: Dict[Any, Tuple[Dict, Dict]] = {}
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            obs, info = env.reset(seed=seed)
            results[env_id] = (serialize_observation(obs), info)
        return results

    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        ids2actions: env_id -> raw LLM text (the env extracts/validates action)
        Returns: env_id -> (serialized_observation, reward, done, info)
        """
        results: Dict[Any, Tuple[Dict, float, bool, Dict]] = {}
        for env_id, raw in ids2actions.items():
            env = self.environments[env_id]
            obs, reward, done, info = env.step(raw)
            results[env_id] = (serialize_observation(obs), float(reward), bool(done), info)
        return results

    def compute_reward_batch(self, env_ids: List[Any]) -> Dict[Any, float]:
        """
        Episode-end reward computation (0.0/1.0 in our env; see env.compute_reward()).
        """
        results: Dict[Any, float] = {}
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = float(env.compute_reward())
        return results

    def get_system_prompts_batch(self, env_ids: List[Any]) -> Dict[Any, str]:
        results: Dict[Any, str] = {}
        for env_id in env_ids:
            results[env_id] = self.environments[env_id].system_prompt()
        return results

    def close_batch(self, env_ids: Optional[List[Any]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())
        for env_id in env_ids:
            env = self.environments.get(env_id, None)
            if env is not None:
                env.close()
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

    # ---------------------------------------------------------------------
    # Optional single-env convenience helpers (useful for non-batch runners)
    # ---------------------------------------------------------------------
    def _format_instructions(self) -> str:
        return format_prompt[self.config.format_type](
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.add_example,
        )

    def _derive_mission(self, env: BabyAITextEnv) -> str:
        if self.config.mission:
            return self.config.mission
        env_id = getattr(env.config, "env_id", "BabyAI-MixedTrainLocal-v0")
        subtask = getattr(env.config, "subtask", None)
        return f"{env_id}{('/' + subtask) if subtask else ''}"

    def build_initial_messages(self, env_id: Any, obs: Dict) -> Dict[str, Any]:
        """
        Chat-style inputs for a single environment instance.
        """
        env = self.environments[env_id]
        sys = system_prompt(mission=self._derive_mission(env))
        fmt = self._format_instructions()

        obs_str = obs.get("obs_str", "")
        core_obs = obs_str.split("Observation:", 1)[1].strip() if "Observation:" in obs_str else obs_str

        user = init_observation_template(observation=core_obs, image_tag=self.config.image_tag)
        return {
            "system": f"{sys}\n\n{fmt}",
            "user": user,
            "images": obs.get("multi_modal_data", {}).get(self.config.image_tag, []),
        }

    def build_step_messages(self, obs: Dict, valid_action=None) -> Dict[str, Any]:
        obs_str = obs.get("obs_str", "")
        core_obs = obs_str.split("Observation:", 1)[1].strip() if "Observation:" in obs_str else obs_str
        user = action_template(
            valid_action=valid_action,
            observation=core_obs,
            image_tag=self.config.image_tag,
        )
        return {
            "user": user,
            "images": obs.get("multi_modal_data", {}).get(self.config.image_tag, []),
        }
