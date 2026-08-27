from typing import Dict, Optional, Tuple
import gymnasium as gym
import minigrid
import hashlib
import io

from tracerigor.env.base.base_env import BaseEnv
from .clean_lang_wrapper import BabyAITextCleanLangWrapper
from .llm_agents_wrapper import BabyAILLMAgentsWrapper
from .prompt import get_instruction_prompt  # uses your ACTIONS map
from .env_config import BabyAITextEnvConfig


def _make_babyai_wrapped_env(cfg: BabyAITextEnvConfig, render_mode: Optional[str] = None):
    # Ensure MiniGrid/BabyAI task registry is present
    minigrid.register_minigrid_envs()

    # MixedTrainLocal goal selection (matches your Verlog logic)
    if cfg.subtask:
        base_task, goal = cfg.env_id, cfg.subtask  # goal like "goto", "pickup", ...
        while True:
            env = gym.make(base_task, render_mode=render_mode, **cfg.babyai_kwargs)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                break
    else:
        env = gym.make(cfg.env_id, render_mode=render_mode, **cfg.babyai_kwargs)

    # Stack your Verlog wrappers unchanged
    env = BabyAITextCleanLangWrapper(env, **cfg.babyai_kwargs)
    env = BabyAILLMAgentsWrapper(
        env,
        format_penalty=cfg.format_penalty,
        binary_reward=cfg.binary_reward,
        **cfg.babyai_kwargs,
    )
    return env


class BabyAITextEnv(BaseEnv):
    """
    Thin adapter mapping Verlog's BabyAI wrappers <-> TraceRigor BaseEnv contract.
    - Produces TraceRigor observations: {"obs_str": ..., "multi_modal_data": {...}}
    - Maintains metrics dict with 'turn_metrics' and 'traj_metrics' each step.
    - Leaves prompting to the Service layer (env does NOT embed system/format prompts).
    """

    def __init__(self, config: BabyAITextEnvConfig, render_mode: Optional[str] = None):
        BaseEnv.__init__(self)
        self.config = config
        self._env = _make_babyai_wrapped_env(config, render_mode)

        # System prompt (base) – service adds format instructions
        mission = f"{config.env_id}{('/' + config.subtask) if config.subtask else ''}"
        self._system_prompt = get_instruction_prompt(self._env, mission=mission)

        # Episode bookkeeping
        self.total_reward: float = 0.0
        self._last_img_hash: Optional[str] = None

        # Use a configurable image placeholder if provided by config, otherwise <image>
        self._img_tag: str = getattr(self.config, "image_placeholder", "<image>")

    # ----------------- Helpers -----------------
    @staticmethod
    def _hash_pil_image(pil_img) -> str:
        """Fast-ish content hash to detect visual state changes."""
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def _to_tracerigor_obs(self, obs: Dict, info: Dict) -> Dict:
        """
        Convert BabyAITextCleanLangWrapper obs to TraceRigor format.
        NOTE: No system/format prompts here; the Service composes those.
        """
        # Your wrapper puts:
        #   obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        #   obs["image"] = PIL.Image
        long_ctx = obs["text"]["long_term_context"]

        obs_str = f"Observation:\n{long_ctx}\n{self._img_tag}"
        # obs_str = f"{self._system_prompt}\n\nObservation:\n{long_ctx}\n<image>"

        return {
            "obs_str": obs_str,
            "multi_modal_data": {self._img_tag: [obs["image"]]},
        }

    # ----------------- BaseEnv API -----------------
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        obs, info = self._env.reset(seed=seed)
        # Initialize episode state
        self.total_reward = 0.0
        # Cache image hash for effectiveness check
        self._last_img_hash = self._hash_pil_image(obs["image"]) if obs.get("image") is not None else None

        tracerigor_obs = self._to_tracerigor_obs(obs, info)
        # No special info required on reset; keep consistent with other envs
        return tracerigor_obs, {}

    def step(self, llm_raw_response: str):
        """
        Take a raw LLM string; the Verlog wrapper extracts a canonical BabyAI action
        and returns (obs, reward, terminated, truncated, info).
        We synthesize TraceRigor-compatible metrics and info.
        """
        # Parse & validate through your robust extractor
        full_action, valid_action, is_valid, behavior_metrics = self._env.extract_action(llm_raw_response)

        # Snapshot old visual hash to judge effectiveness
        prev_hash = self._last_img_hash

        # Step underlying env (note: wrapper can penalize if !is_valid)
        obs, reward, terminated, truncated, raw_info = self._env.step(valid_action, is_valid=is_valid)
        done = bool(terminated or truncated)

        # Visual effectiveness: did pixels change?
        new_hash = self._hash_pil_image(obs["image"]) if obs.get("image") is not None else None
        self._last_img_hash = new_hash
        action_is_effective = (prev_hash is None) or (new_hash is None) or (prev_hash != new_hash)

        # Success signal: BabyAI gives positive reward on success; wrapper sets progression=1.0 too
        success = bool(reward > 0) or bool(getattr(self._env, "progression", 0.0) >= 1.0)

        # Compose standard metrics block expected by TraceRigor trainers
        metrics = {
            "turn_metrics": {
                "action_is_valid": bool(is_valid),
                "action_is_effective": bool(action_is_effective),
            },
            "traj_metrics": {
                "success": bool(success),
            },
        }

        # Build TraceRigor obs and info
        tracerigor_obs = self._to_tracerigor_obs(obs, raw_info)
        info: Dict = {
            # Keep the raw and parsed LLM outputs around
            "llm_raw_response": llm_raw_response,
            "llm_response": {
                "full_action": full_action,
                "action": valid_action,
                "is_valid": bool(is_valid),
            },
            # Standard metrics (plus behavior diagnostics from your extractor)
            "metrics": {**metrics, "behavior": behavior_metrics},
        }

        # Track episode return like other envs
        self.total_reward += float(reward)

        return tracerigor_obs, float(reward), done, info

    def system_prompt(self) -> str:
        """Only the base system prompt. The Service appends format instructions."""
        return self._system_prompt

    def compute_reward(self) -> float:
        """
        Final/episode reward hook. Mirrors wrapper's progression:
        1.0 if solved, else 0.0 (kept simple and stable for scoring).
        """
        get_stats = getattr(self._env, "get_stats", None)
        if callable(get_stats):
            stats = get_stats()  # e.g., {"mission": ..., "progression": 0.0/1.0}
            if stats.get("progression", 0.0) >= 1.0:
                return 1.0
        return 0.0

    def close(self):
        return self._env.close()
