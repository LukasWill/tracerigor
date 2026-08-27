from typing import Dict, List, Tuple, Optional, Any, Union
from tracerigor.env.base.base_service import BaseService
from tracerigor.env.base.base_service_config import BaseServiceConfig
# from tracerigor.env.utils.state_reward_text_utils import service_state_reward_wrapper_v2 as service_state_reward_wrapper
from tracerigor.env.utils.service_llm_verifier_wrapper import service_llm_verifier_wrapper
from tracerigor.server.serial import serialize_observation
from tracerigor.verifier.verifier.common.config import VerifierConfig

from .env import SokobanEnv
from .env_config import SokobanEnvConfig
from tracerigor.env.utils.state_reward_text_utils import service_state_reward_wrapper_v3 as service_state_reward_wrapper
from .prompt import visual_reasoning_reward_prompt
from tracerigor.env.utils.state_matching import calculate_visual_reasoning_reward_bipartite, calculate_f1_with_max_matching
from tracerigor.env.utils.top_string_tracker import TopKStringTracker

from tracerigor.env.utils.load_verifier_cfg import load_verifier_cfg
from dataclasses import asdict
from pathlib import Path

_DEFAULT_VERIFIER_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "verifier"
    / "verifier"
    / "common"
    / "verifier_config.yaml"
)
global_vcfg = load_verifier_cfg(str(_DEFAULT_VERIFIER_CONFIG))

# by 15Apr, we found that the verifier config is mostly static across environments, so we can load it once globally and then merge with any per-env overrides. This also avoids the need to read from disk on every env creation, which was a bottleneck in early experiments. If in the future we want to support more dynamic configs, we can refactor this to be more flexible.
# import os as _os
# global_vcfg = load_verifier_cfg(
#     _os.path.join(_os.path.dirname(__file__), "..", "..", "verifier", "verifier", "common", "verifier_config.yaml")
# )

class SokobanService(BaseService):

    def __init__(self, config: BaseServiceConfig):
        self.environments = {}
        self.env_configs = {}
        self.config = config
        if self.config.use_state_reward:
            self.top_strings_tracker_grounding = TopKStringTracker(self.config.top_strings_m)
            self.top_strings_tracker_worldmodeling = TopKStringTracker(self.config.top_strings_m)
        self._verifier_train_step = 0  # used by service_llm_verifier_wrapper annealing

    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        for env_id, config in ids2configs.items():
            env_config_dict = config.get('env_config', {})
            env_cfg = SokobanEnvConfig(**env_config_dict)

            # merge: global defaults < per-env overrides
            merged = asdict(global_vcfg)
            merged.update(env_config_dict.get("verifier", {}) or {})
            env_cfg.verifier = VerifierConfig(**merged)

            env = SokobanEnv(env_cfg)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_cfg

    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        results = {}

        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)

        # Bump anneal step once per TRAIN rollout reset (no tick for VAL)
        # ids2seeds keys look like "train3", "val1", etc. (set by the client)
        if ids2seeds:
            # all env_ids in this reset belong to the same split in your current design
            any_id = next(iter(ids2seeds.keys()))
            if str(any_id).startswith("train"):
                self._verifier_train_step = getattr(self, "_verifier_train_step", 0) + 1

        return results

    @service_llm_verifier_wrapper
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results = {}

        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)

        return results

    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        results = {}

        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.compute_reward()

        return results

    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        results = {}

        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()

        return results

    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())

        for env_id in env_ids:
            env = self.environments[env_id]
            env.close()

        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

    def gen_visual_reasoning_prompt(self, content,**kwargs) -> str:
        return visual_reasoning_reward_prompt.format(prediction=content)

    def calculate_visual_reasoning_reward(self,**kwargs) -> float:
        """
        Calculate the visual reasoning reward based on the response and state.
        e.g. [{"object_id": "target", "vertical_relation":above,"horizontal_relation":left},
            {"object_id": "hole", "vertical_relation":above,"horizontal_relation":left}]
        Args:
            response: The output of the llm judge (structured state).
            state: The current state of the environment.
            content: The input to the llm judge (natural lanagugae state).

        Returns:
            A float representing the calculated reward.
        """
        # object_weights={"target": 0.7,"hole": 0.3}
        # return calculate_visual_reasoning_reward_bipartite(response, state,object_weights)
        r_type = kwargs.get("r_type")
        if r_type not in ["grounding", "worldmodeling"]:
            raise ValueError("r_type must be either 'grounding' or 'worldmodeling'")
        response = kwargs.get("response")
        state = kwargs.get("state")
        content = kwargs.get("content")
        target_result = calculate_f1_with_max_matching(
            [item for item in state if item['object_id'] == 'target'] if state else [],
            [item for item in response if item['object_id'] == 'target'] if response else [],
            match_func=lambda x, y: x['vertical_relation'] == y['vertical_relation'] and x['horizontal_relation'] == y['horizontal_relation']
        )
        # check hole reward
        box_result =calculate_f1_with_max_matching(
            [item for item in state if item['object_id'] == 'box'] if state else [],
            [item for item in response if item['object_id'] == 'box'] if response else [],
            match_func=lambda x, y: x['vertical_relation'] == y['vertical_relation'] and x['horizontal_relation'] == y['horizontal_relation']
        )
        target_reward = target_result['f1']
        box_reward = box_result['f1']
        if r_type=="grounding":
            top_k_strings = self.top_strings_tracker_grounding.get_top_k(self.config.top_strings_k)
        if r_type=="worldmodeling":
            top_k_strings = self.top_strings_tracker_worldmodeling.get_top_k(self.config.top_strings_k)

        if content in top_k_strings and target_reward+box_reward<0.7:
            return -0.1
        return target_reward*0.5 + box_reward*0.5
