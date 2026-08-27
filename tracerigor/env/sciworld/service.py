"""
SciWorld Service for TraceRigor.

This module implements the service layer for batch processing of SciWorld
environments, following the TraceRigor BaseService interface.
"""
from typing import Dict, List, Tuple, Optional, Any

from tracerigor.env.base.base_service import BaseService
from tracerigor.server.serial import serialize_observation

from .env import SciWorldEnv
from .env_config import SciWorldEnvConfig
from .service_config import SciWorldServiceConfig


class SciWorldService(BaseService):
    """
    Service for managing multiple SciWorld environment instances.

    This service implements batch operations for efficient parallel processing
    of multiple SciWorld environments, following the TraceRigor BaseService interface.

    The service supports:
    - Creating and managing multiple environment instances
    - Batch reset and step operations
    - System prompt retrieval
    - Optional state reward computation
    """

    def __init__(self, config: SciWorldServiceConfig):
        """
        Initialize the SciWorld service.

        Args:
            config: Service configuration
        """
        self.environments: Dict[str, SciWorldEnv] = {}
        self.env_configs: Dict[str, SciWorldEnvConfig] = {}
        self.config = config

        # Initialize state reward components if enabled
        if self.config.use_state_reward:
            try:
                from tracerigor.env.utils.top_string_tracker import TopKStringTracker
                self.top_strings_tracker_decision = TopKStringTracker(self.config.top_strings_m)
                self.top_strings_tracker_reasoning = TopKStringTracker(self.config.top_strings_m)
            except ImportError:
                self.top_strings_tracker_decision = None
                self.top_strings_tracker_reasoning = None

    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        """
        Create multiple environments in parallel.

        Args:
            ids2configs: Dictionary mapping environment IDs to their configurations.
                Each config should have structure:
                {"env_name": "sciworld", "env_config": {...}}
        """
        for env_id, config in ids2configs.items():
            env_config_dict = config.get('env_config', {})
            env_config = SciWorldEnvConfig(**env_config_dict)
            env = SciWorldEnv(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config

    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple environments in parallel.

        Args:
            ids2seeds: Dictionary mapping environment IDs to seed values

        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        results = {}

        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)

        return results

    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Execute steps across multiple environments in parallel.

        Args:
            ids2actions: Dictionary mapping environment IDs to LLM response actions

        Returns:
            Dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        results = {}

        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)

        return results

    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute final rewards for multiple environments.

        Args:
            env_ids: List of environment IDs

        Returns:
            Dictionary mapping environment IDs to reward values
        """
        results = {}

        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.compute_reward()

        return results

    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Get system prompts for multiple environments.

        Args:
            env_ids: List of environment IDs

        Returns:
            Dictionary mapping environment IDs to system prompt strings
        """
        results = {}

        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()

        return results

    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and release resources.

        Args:
            env_ids: List of environment IDs to close (None for all)
        """
        if env_ids is None:
            env_ids = list(self.environments.keys())

        for env_id in env_ids:
            if env_id in self.environments:
                env = self.environments[env_id]
                env.close()

        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

    # ==========================================================================
    # State Reward Methods (Optional - for process reward functionality)
    # ==========================================================================

    def gen_decision_reasoning_prompt(self, content: str, **kwargs) -> str:
        """
        Generate prompt for evaluating decision-making reasoning in SciWorld.

        This method is called by the state reward wrapper when use_state_reward is enabled.

        Args:
            content: The reasoning/decision content to evaluate
            **kwargs: Additional context

        Returns:
            Evaluation prompt string
        """
        return f"""Evaluate the quality of this SciWorld agent's decision and reasoning:

Task Context: The agent is solving science curriculum tasks in a text-based environment.

Decision/Reasoning: {content}

Please assess:
1. Is the reasoning logically sound given the task?
2. Does it demonstrate understanding of scientific concepts?
3. Is the chosen action appropriate for the current situation?

Respond with JSON: {{"decision_quality": 0.0-1.0, "reasoning_quality": 0.0-1.0}}"""

    def calculate_decision_reasoning_reward(self, **kwargs) -> float:
        """
        Calculate reward for SciWorld decision making.

        Args:
            response: The LLM judge response
            content: The original content being judged
            r_type: Type of reasoning being evaluated

        Returns:
            Calculated reward value (0.0 to 1.0)
        """
        response = kwargs.get("response")
        content = kwargs.get("content")
        r_type = kwargs.get("r_type", "decision")

        # Parse the LLM judge response to get quality score
        try:
            if isinstance(response, dict):
                reward = response.get("quality", 0.0)
            else:
                reward = 0.5
        except Exception:
            reward = 0.0

        # Check for repetitive responses if tracker is available
        if self.config.use_state_reward and self.top_strings_tracker_decision:
            if r_type == "decision":
                top_k_strings = self.top_strings_tracker_decision.get_top_k(self.config.top_strings_k)
            else:
                top_k_strings = self.top_strings_tracker_reasoning.get_top_k(self.config.top_strings_k)

            # Penalize repetitive low-quality responses
            if content in top_k_strings and reward < 0.6:
                return -0.1

        return reward


if __name__ == "__main__":
    # Test the service
    service_config = SciWorldServiceConfig(max_workers=2)
    service = SciWorldService(service_config)

    # Create test environments
    test_configs = {
        "test_env_1": {
            "env_name": "sciworld",
            "env_config": {
                "task_nums": [1],
                "env_step_limit": 10
            }
        }
    }

    print("Creating environments...")
    service.create_environments_batch(test_configs)
    print(f"Created {len(service.environments)} environment(s)")

    # Clean up
    service.close_batch()
    print("Service test completed")
