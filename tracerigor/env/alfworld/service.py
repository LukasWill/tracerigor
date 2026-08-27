"""Batch service layer for the ALFWorld environment.

The service follows the same lightweight pattern used by ``SciWorldService`` and
``BlackjackService``: one ALFWorldEnv instance per ``env_id``, batched
operations are simple sequential loops (the cost of batch parallelism is
dominated by the heavy alfworld env construction, not by per-call latency).
"""
from typing import Any, Dict, List, Optional, Tuple

from tracerigor.env.base.base_service import BaseService
from tracerigor.server.serial import serialize_observation

from .env import ALFWorldEnv
from .env_config import ALFWorldEnvConfig
from .service_config import ALFWorldServiceConfig


class ALFWorldService(BaseService):
    """Service that owns a pool of :class:`ALFWorldEnv` instances."""

    def __init__(self, config: ALFWorldServiceConfig):
        self.config = config
        self.environments: Dict[str, ALFWorldEnv] = {}
        self.env_configs: Dict[str, ALFWorldEnvConfig] = {}

        if getattr(self.config, "use_state_reward", False):
            try:
                from tracerigor.env.utils.top_string_tracker import TopKStringTracker

                self.top_strings_tracker_decision = TopKStringTracker(
                    self.config.top_strings_m
                )
                self.top_strings_tracker_reasoning = TopKStringTracker(
                    self.config.top_strings_m
                )
            except ImportError:
                self.top_strings_tracker_decision = None
                self.top_strings_tracker_reasoning = None

    # ------------------------------------------------------------------
    # BaseService contract
    # ------------------------------------------------------------------
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        for env_id, config in ids2configs.items():
            env_config_dict = config.get("env_config", {}) if isinstance(config, dict) else {}
            env_config = ALFWorldEnvConfig(**env_config_dict)
            env = ALFWorldEnv(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config

    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        results: Dict[str, Tuple[Any, Any]] = {}
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            results[env_id] = (serialize_observation(observation), info)
        return results

    def step_batch(
        self, ids2actions: Dict[str, Any]
    ) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        results: Dict[str, Tuple[Dict, float, bool, Dict]] = {}
        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            results[env_id] = (serialize_observation(observation), reward, done, info)
        return results

    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        return {env_id: self.environments[env_id].compute_reward() for env_id in env_ids}

    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        return {env_id: self.environments[env_id].system_prompt() for env_id in env_ids}

    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())
        for env_id in env_ids:
            env = self.environments.get(env_id)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

    # ------------------------------------------------------------------
    # Optional LLM-as-judge state reward hooks (mirrors blackjack/sciworld)
    # ------------------------------------------------------------------
    def gen_decision_reasoning_prompt(self, content: str, **kwargs) -> str:
        return f"""Evaluate the quality of this ALFWorld household-task reasoning and decision:

Context: The agent is solving a household instruction (navigation, manipulation, appliance use) in a text-based environment.

Decision/Reasoning: {content}

Please assess:
1. Does the reasoning correctly identify the goal sub-task (find / clean / heat / put ...)?
2. Does the chosen command match an admissible action in the current state?
3. Does it reflect a sensible plan toward the overall goal?

Respond with JSON: {{"decision_quality": 0.0-1.0, "reasoning_quality": 0.0-1.0}}"""

    def calculate_decision_reasoning_reward(self, **kwargs) -> float:
        response = kwargs.get("response")
        content = kwargs.get("content")
        r_type = kwargs.get("r_type", "decision")

        try:
            reward = response.get("quality", 0.0) if isinstance(response, dict) else 0.5
        except Exception:
            reward = 0.0

        tracker = None
        if getattr(self.config, "use_state_reward", False):
            tracker = (
                self.top_strings_tracker_decision
                if r_type == "decision"
                else self.top_strings_tracker_reasoning
            )
        if tracker is not None:
            top_k_strings = tracker.get_top_k(self.config.top_strings_k)
            if content in top_k_strings and reward < 0.6:
                return -0.1
        return reward
