"""TraceRigor wrapper around the ALFWorld TextWorld / THOR environment."""
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tracerigor.env.base.base_env import BaseEnv
from tracerigor.env.utils.parse_utils import PARSE_FUNC_MAP

from .alfworld_utils import (
    build_alfworld_env,
    get_thor_frame,
    load_alf_config,
    numpy_frame_to_pil,
)
from .env_config import ALFWorldEnvConfig
from .prompt import (
    action_observation_template,
    format_action_history,
    format_prompt,
    init_observation_template,
    system_prompt,
)


# =============================================================================
# Violation Types and Tracker (ALFWorld)
# =============================================================================
#
# Modeled on tracerigor/env/sciworld/env.py but extended to handle an ALFWorld-
# specific failure mode that the sciworld tracker misses: "no-progress
# oscillation", where the agent emits a *different* action string each turn
# yet the world observation is frozen (e.g., a series of inadmissible
# `look at X` variants interleaved with admissible-but-useless `look`).
#
# Empirically (audit of finegrained-alfworld-reflact runs at steps 0/10/20/30
# over 128 trajs each), this is the dominant alfworld failure pattern mid-
# training: >80% of trajectories at step_10/20 contain a 20+ step run of
# frozen observation, often spanning many distinct action strings — none of
# sciworld's REPETITION (same action + same obs) nor INVALID_ACTION
# (consecutive rejections) counters accumulate across such runs because
# admissible no-op actions reset them.

class ViolationType(Enum):
    """Types of violations that can trigger early termination."""
    FORMAT = "format_violation"          # Missing/malformed <reflection>/<action> tags
    INADMISSIBLE = "inadmissible_action" # Action not in admissible_commands
    REPETITION = "repetition"            # Same action + unchanged observation
    NO_PROGRESS = "no_progress"          # Observation unchanged across N steps regardless of action


@dataclass
class ViolationTracker:
    """Tracks consecutive ALFWorld violations to enable early termination.

    Four failure modes are monitored, each with an independent consecutive
    counter that resets the moment a step doesn't exhibit that mode:

    - FORMAT: cannot parse <reflection>...</reflection><action>...</action>
      (or the configured tag pair) from the LLM response, or parsed action
      is empty. Reset when a well-formed response with a non-empty action
      is produced.

    - INADMISSIBLE: parsed action is not in the current ``admissible_commands``
      list. ALFWorld surfaces this list explicitly, so detection is exact —
      no string-matching on observations needed. Reset when an admissible
      action is chosen.

    - REPETITION: exact same action as previous step *and* observation
      after the action equals the observation after the previous action.
      Reset when either the action or the observation changes.

    - NO_PROGRESS: observation after the action equals the observation
      after the previous action, regardless of whether the action changed.
      Reset when the observation changes. This is the failure mode
      sciworld's tracker misses; it subsumes REPETITION but with a more
      lenient threshold since "agent flailing across multiple ideas" is
      less obviously broken than "agent locked on one idea".

    Termination priority when multiple thresholds are crossed in the same
    step: FORMAT > INADMISSIBLE > REPETITION > NO_PROGRESS.
    """

    # Thresholds for each violation type.
    format_threshold: int = 3
    inadmissible_threshold: int = 5
    repetition_threshold: int = 3
    no_progress_threshold: int = 8

    # Consecutive counters.
    consecutive_format_violations: int = field(default=0, init=False)
    consecutive_inadmissible: int = field(default=0, init=False)
    consecutive_repetitions: int = field(default=0, init=False)
    consecutive_no_progress: int = field(default=0, init=False)

    # Last action / observation for repetition + no-progress detection.
    last_action: Optional[str] = field(default=None, init=False)
    last_observation: Optional[str] = field(default=None, init=False)

    # Totals for offline metrics / logging.
    total_format_violations: int = field(default=0, init=False)
    total_inadmissible: int = field(default=0, init=False)
    total_repetitions: int = field(default=0, init=False)
    total_no_progress: int = field(default=0, init=False)

    def reset(self) -> None:
        """Reset all counters and last-step state (call on env.reset())."""
        self.consecutive_format_violations = 0
        self.consecutive_inadmissible = 0
        self.consecutive_repetitions = 0
        self.consecutive_no_progress = 0
        self.last_action = None
        self.last_observation = None
        self.total_format_violations = 0
        self.total_inadmissible = 0
        self.total_repetitions = 0
        self.total_no_progress = 0

    def record_step(
        self,
        format_correct: bool,
        action: str,
        observation: str,
        action_admissible: bool,
    ) -> Tuple[bool, Optional[ViolationType]]:
        """Record a step, return (should_terminate, violation_type).

        Args:
            format_correct: Whether the LLM response parsed cleanly into the
                expected reflection/action structure.
            action: The parsed action string (empty if format failure).
            observation: The observation *after* this action (or the
                unchanged observation if the action was inadmissible — the
                ALFWorld env does not advance state in that case).
            action_admissible: Whether ``action`` was in the env's current
                ``admissible_commands`` list. Pass False when there is no
                action to evaluate (e.g., format failure with empty action),
                in which case the inadmissible counter is *reset* rather
                than incremented — the underlying issue is format, not
                action choice.
        """
        # --- FORMAT -----------------------------------------------------
        if not format_correct or not action:
            self.consecutive_format_violations += 1
            self.total_format_violations += 1
        else:
            self.consecutive_format_violations = 0

        # --- INADMISSIBLE -----------------------------------------------
        # Only meaningful when an action was actually parsed. If there is
        # no action, the issue is FORMAT, so we reset rather than penalise
        # twice.
        if action:
            if not action_admissible:
                self.consecutive_inadmissible += 1
                self.total_inadmissible += 1
            else:
                self.consecutive_inadmissible = 0
        else:
            self.consecutive_inadmissible = 0

        # --- REPETITION (same action + same obs) ------------------------
        is_repetition = (
            self.last_action is not None
            and action != ""
            and action == self.last_action
            and self._observations_equal(self.last_observation, observation)
        )
        if is_repetition:
            self.consecutive_repetitions += 1
            self.total_repetitions += 1
        else:
            self.consecutive_repetitions = 0

        # --- NO_PROGRESS (observation frozen) ---------------------------
        # Detected independently of action identity. This is what catches
        # mid-training alfworld trajectories that oscillate between many
        # inadmissible action strings while the world stays still.
        is_no_progress = (
            self.last_observation is not None
            and self._observations_equal(self.last_observation, observation)
        )
        if is_no_progress:
            self.consecutive_no_progress += 1
            self.total_no_progress += 1
        else:
            self.consecutive_no_progress = 0

        # --- Termination decision (priority order) ----------------------
        should_terminate = False
        reason: Optional[ViolationType] = None
        if self.consecutive_format_violations >= self.format_threshold:
            should_terminate, reason = True, ViolationType.FORMAT
        elif self.consecutive_inadmissible >= self.inadmissible_threshold:
            should_terminate, reason = True, ViolationType.INADMISSIBLE
        elif self.consecutive_repetitions >= self.repetition_threshold:
            should_terminate, reason = True, ViolationType.REPETITION
        elif self.consecutive_no_progress >= self.no_progress_threshold:
            should_terminate, reason = True, ViolationType.NO_PROGRESS

        self.last_action = action
        self.last_observation = observation
        return should_terminate, reason

    @staticmethod
    def _observations_equal(obs1: Optional[str], obs2: Optional[str]) -> bool:
        if obs1 is None or obs2 is None:
            return False
        return " ".join(obs1.split()) == " ".join(obs2.split())

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_format_violations": self.total_format_violations,
            "total_inadmissible": self.total_inadmissible,
            "total_repetitions": self.total_repetitions,
            "total_no_progress": self.total_no_progress,
            "consecutive_format_violations": self.consecutive_format_violations,
            "consecutive_inadmissible": self.consecutive_inadmissible,
            "consecutive_repetitions": self.consecutive_repetitions,
            "consecutive_no_progress": self.consecutive_no_progress,
        }


# Map our prompt formats to entries in PARSE_FUNC_MAP. The parsers accept either
# <answer>...</answer> or <action>...</action> as the action tag, so they work
# directly for the ALFWorld format prompts. ``reflact`` / ``reflact_diverse``
# both consume the <reflection>...<action>... structure handled by parse_reflact.
_PARSE_FUNC_KEYS = {
    "free_think": "free_think",
    "no_think": "no_think",
    "grounding": "grounding",
    "worldmodeling": "worldmodeling",
    "grounding_worldmodeling": "grounding_worldmodeling",
    "reflact": "reflact",
    "reflact_diverse": "reflact_diverse",
}


class ALFWorldEnv(BaseEnv):
    """ALFWorld single-instance environment for LLM agents.

    Wraps either ``AlfredTWEnv`` (text mode) or ``AlfredThorEnv`` (vision
    mode) from the bundled alfworld package. Each TraceRigor env corresponds to one
    underlying alfworld environment with ``batch_size=1``.
    """

    def __init__(self, config: ALFWorldEnvConfig):
        BaseEnv.__init__(self)
        self.config = config

        # Underlying alfworld env is created lazily on the first reset so that
        # construction cost (config parsing, game-file walking, dataset
        # listing) is paid once per process per env_id.
        self._env = None
        self._base_env = None
        self._env_type: Optional[str] = None

        # Per-episode state.
        self.task: str = ""
        self.current_observation: str = ""
        self.admissible_commands: List[str] = []
        self.history_buffer: List[Dict[str, Any]] = []
        self.last_action: str = ""
        self.last_action_valid: bool = True
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.is_done: bool = False
        self.task_completed: bool = False

        # Parsing / format helpers.
        parse_key = _PARSE_FUNC_KEYS.get(self.config.prompt_format, "free_think")
        self.parse_func = PARSE_FUNC_MAP[parse_key]
        self.format_prompt_func = format_prompt.get(
            self.config.prompt_format, format_prompt["free_think"]
        )

        # Pre-load the alf-config so we can fail fast on bad paths and decide
        # whether to override env.type for vision mode.
        self._alf_config = load_alf_config(self.config.alf_config_path)
        self._maybe_override_env_type()

        self._rng = random.Random()

        # Violation tracker for early termination of pathological trajectories.
        if self.config.enable_violation_termination:
            self.violation_tracker: Optional[ViolationTracker] = ViolationTracker(
                format_threshold=self.config.format_violation_threshold,
                inadmissible_threshold=self.config.inadmissible_action_threshold,
                repetition_threshold=self.config.repetition_threshold,
                no_progress_threshold=self.config.no_progress_threshold,
            )
        else:
            self.violation_tracker = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _maybe_override_env_type(self) -> None:
        wanted = "AlfredThorEnv" if self.config.render_mode == "vision" else "AlfredTWEnv"
        existing = self._alf_config.get("env", {}).get("type")
        if existing != wanted:
            self._alf_config.setdefault("env", {})["type"] = wanted

    def _ensure_env(self, seed: Optional[int]) -> None:
        if self._env is not None:
            return

        # Pass the (possibly env-type-patched) alf-config dict in-memory and
        # keep ``alf_config_path`` as the stable user-supplied path. The
        # factory cache in ``build_alfworld_env`` keys on that path, so
        # every ALFWorldEnv in this process that uses the same YAML +
        # train_eval shares one ``AlfredTWEnv`` / ``AlfredThorEnv`` factory
        # and the multi-minute ``collect_game_files`` walk happens exactly
        # once per split per process.
        self._env, self._base_env, self._env_type = build_alfworld_env(
            self.config.alf_config_path,
            train_eval=self.config.train_eval,
            batch_size=1,
            seed=seed,
            config_override=self._alf_config,
        )

    # ------------------------------------------------------------------
    # BaseEnv contract
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self._rng.seed(seed)

        # The alfworld env.seed (TW) reseeds its internal sampler so that
        # subsequent resets pick a different game; for THOR it just stashes
        # the seed for later use.
        self._ensure_env(seed=seed)
        if seed is not None and hasattr(self._env, "seed"):
            try:
                self._env.seed(seed)
            except Exception:
                pass

        raw_obs, info = self._env.reset()
        text_obs, admissible, task_desc = self._unpack_reset(raw_obs, info)

        self.current_observation = text_obs
        self.admissible_commands = admissible
        self.task = task_desc
        self.history_buffer = []
        self.last_action = ""
        self.last_action_valid = True
        self.total_reward = 0.0
        self.step_count = 0
        self.is_done = False
        self.task_completed = False

        if self.violation_tracker is not None:
            self.violation_tracker.reset()

        return self._render(init_obs=True), {
            "task": self.task,
            "admissible_commands": list(self.admissible_commands),
            "observation_text": self.current_observation,
        }

    def step(self, llm_raw_response: str) -> Tuple[Dict, float, bool, Dict]:
        rst = self.parse_func(
            response=llm_raw_response,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )
        action_list: List[str] = rst.get("actions", [])
        format_correct: bool = rst.get("format_correct", False)

        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct and len(action_list) > 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        info: Dict[str, Any] = {}
        info.update(rst)

        reward = 0.0
        done = self.is_done
        action_to_log = action_list[0] if action_list else ""

        # Format shaping is applied AFTER action execution (see below) so that
        # "gated_bonus" mode can condition on admissibility / effectiveness.
        # The old unconditional "+format_reward if format_correct" was farmable:
        # under multi-turn GAE the discounted per-turn bonus stream saturates at
        # format_reward / (1 - high_level_gamma) (= 10.0 at 0.95) ~= win_reward,
        # so a well-formatted but non-solving episode could earn a return
        # comparable to a real success.
        action_was_admissible = False
        pre_obs = self.current_observation

        if action_list:
            action = action_list[0]
            if action in self.admissible_commands:
                action_was_admissible = True
                raw_obs, scores, dones, gym_info = self._env.step([action])
                self.current_observation, self.admissible_commands, won, gc_rate = (
                    self._unpack_step(raw_obs, gym_info)
                )
                done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)

                step_reward = self.config.win_reward * float(won)
                if self.config.include_gc_reward:
                    step_reward += float(gc_rate)
                reward += step_reward

                metrics["turn_metrics"]["action_is_effective"] = (
                    self.current_observation.strip() != pre_obs.strip()
                )
                if won:
                    metrics["traj_metrics"]["success"] = True
                    self.task_completed = True

                info.update({
                    "won": bool(won),
                    "goal_condition_success_rate": float(gc_rate),
                    "observation_text": self.current_observation,
                })
            else:
                # Action parsed cleanly but not in admissible commands.
                reward += self.config.invalid_action_penalty
                metrics["turn_metrics"]["action_is_valid"] = False
                info["won"] = False
                info["observation_text"] = self.current_observation

        # --- Format shaping -------------------------------------------------
        # NOTE: ``is_format_rewarded`` keeps its original meaning (the response
        # was well-formed); the state-reward / verifier path gates on it. The
        # actual per-turn shaping contribution is reported separately as
        # ``format_shaping_term``.
        action_is_effective = bool(metrics["turn_metrics"]["action_is_effective"])
        shaping = getattr(self.config, "format_shaping", "penalty")
        if shaping == "bonus":
            # Legacy: unconditional positive bonus for any well-formed turn (farmable).
            fmt_term = self.config.format_reward if format_correct else 0.0
        elif shaping == "gated_bonus":
            # Positive bonus only for well-formed AND admissible AND effective turns.
            # (Still leaks via admissible obs-changing oscillation, e.g. open/close,
            # so "penalty" is preferred for ALFWorld where the bonus ~= win_reward.)
            fmt_term = (
                self.config.format_reward
                if (format_correct and action_was_admissible and action_is_effective)
                else 0.0
            )
        else:  # "penalty" (default, farm-proof): the best a turn can do is 0.
            fmt_term = 0.0 if format_correct else -self.config.format_reward
        reward += fmt_term
        info["is_format_rewarded"] = format_correct
        info["format_shaping_term"] = fmt_term

        # Bookkeeping.
        self._save_to_history(text_obs=pre_obs, action=action_to_log)
        self.last_action = action_to_log
        self.last_action_valid = action_was_admissible
        self.step_count += 1
        self.total_reward += reward

        if self.step_count >= self.config.max_steps:
            done = True

        # --- Violation tracking / early termination -----------------------
        # Runs after state + reward updates so we use the post-action obs
        # (which equals pre_obs when the action was inadmissible — alfworld
        # does not advance state in that case, so NO_PROGRESS will trigger).
        if self.violation_tracker is not None:
            v_terminated, v_reason = self.violation_tracker.record_step(
                format_correct=format_correct,
                action=action_to_log,
                observation=self.current_observation,
                action_admissible=action_was_admissible,
            )
            if v_terminated:
                done = True
                reward += self.config.violation_penalty
                self.total_reward += self.config.violation_penalty
                metrics["traj_metrics"]["violation_terminated"] = True
                metrics["traj_metrics"]["termination_reason"] = (
                    v_reason.value if v_reason is not None else None
                )
                info["violation_terminated"] = True
                info["termination_reason"] = (
                    v_reason.value if v_reason is not None else None
                )
            metrics["traj_metrics"]["violation_metrics"] = (
                self.violation_tracker.get_metrics()
            )

        self.is_done = done

        info["metrics"] = metrics
        info["llm_raw_response"] = llm_raw_response
        info["task"] = self.task
        info["step_count"] = self.step_count
        info["admissible_commands"] = list(self.admissible_commands)

        return self._render(init_obs=False), reward, done, info

    def system_prompt(self) -> str:
        base = system_prompt(render_mode=self.config.render_mode)
        format_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.add_example,
        )
        return base + "\n\n" + format_text

    def compute_reward(self) -> float:
        return 0.0

    def close(self) -> None:
        if self._env is None:
            return
        try:
            if hasattr(self._env, "close"):
                self._env.close()
        except Exception:
            pass
        self._env = None
        self._base_env = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _unpack_reset(self, raw_obs, info) -> Tuple[str, List[str], str]:
        """Normalise the batched reset return into a single-env tuple."""
        text = self._unbatch_text(raw_obs)
        admissible = self._unbatch_list(info.get("admissible_commands", []))
        # The alfworld TW env exposes the task description through the
        # "extra.gamefile" infos along with the goal in the obs text. As a
        # safe default we extract the goal from the obs string.
        task_desc = self._extract_task(text)
        return text, admissible, task_desc

    def _unpack_step(self, raw_obs, info) -> Tuple[str, List[str], bool, float]:
        text = self._unbatch_text(raw_obs)
        admissible = self._unbatch_list(info.get("admissible_commands", []))
        won = bool(self._unbatch_scalar(info.get("won", [False])))
        gc_rate = float(self._unbatch_scalar(info.get("goal_condition_success_rate", [0.0])))
        return text, admissible, won, gc_rate

    @staticmethod
    def _unbatch_text(value) -> str:
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        return str(value)

    @staticmethod
    def _unbatch_list(value) -> List[str]:
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, (list, tuple)):
                return list(first)
            return list(value)
        return []

    @staticmethod
    def _unbatch_scalar(value):
        if isinstance(value, (list, tuple)) and value:
            return value[0]
        return value

    @staticmethod
    def _extract_task(obs_text: str) -> str:
        """Best-effort extraction of the goal sentence from a reset obs."""
        marker = "Your task is to:"
        idx = obs_text.find(marker)
        if idx >= 0:
            return obs_text[idx + len(marker):].strip().split("\n")[0].strip()
        return obs_text.split("\n")[-1].strip()

    def _save_to_history(self, text_obs: str, action: str) -> None:
        self.history_buffer.append({"text_obs": text_obs, "action": action})

    def _render(self, init_obs: bool = False) -> Dict[str, Any]:
        admissible_str = ", ".join(self.admissible_commands)

        if init_obs or not self.config.use_history or self.config.history_length <= 0:
            text_observation = self.current_observation
            obs_str = init_observation_template(
                task=self.task,
                observation=text_observation if self.config.render_mode == "text" else self.config.image_placeholder,
                admissible_commands=admissible_str,
            )
        else:
            action_history = format_action_history(
                buffers=self.history_buffer,
                history_length=self.config.history_length,
            )
            text_observation = self.current_observation
            obs_str = action_observation_template(
                task=self.task,
                action_history=action_history,
                current_step=self.step_count + 1,
                observation=text_observation if self.config.render_mode == "text" else self.config.image_placeholder,
                admissible_commands=admissible_str,
                last_action_valid=self.last_action_valid,
            )

        # Per-turn format-prompt repetition was intentionally removed (mirrors
        # the SciWorld pattern). Rationale, in three points:
        #   1. At ROLLOUT time (where the model actually generates),
        #      ``_generate_input_for_rollout`` in rollout_manager_service.py
        #      does NOT truncate the chat, so the system prompt — which
        #      already carries the full format spec + ICL example — is always
        #      in context. The per-turn repeat is redundant for generation.
        #   2. At UPDATE time, verl's ``tokenize_and_postprocess_data`` with
        #      ``truncation='left'`` does ``input_ids[:, -max_length:]``,
        #      which can drop the system prompt for trajectories longer than
        #      ``max_trajectory_length``. But the model's own prior assistant
        #      turns — which exhibit the correct format — always survive in
        #      the kept tail window, so gradients still flow over
        #      format-correct examples without needing a textual reminder.
        #   3. With ALFWorld's long ``admissible_commands`` lists, the
        #      per-turn repeat costs ~75 tokens × ~30 turns ≈ ~2.2k tokens of
        #      the 15k trajectory budget. Reclaiming that budget for the
        #      admissible commands themselves is a much better use of context.
        if self.config.render_mode == "vision":
            obs_str += f"\nText description of the current view: {self.current_observation}"
            frame = get_thor_frame(self._env) if self._env is not None else None
            pil = numpy_frame_to_pil(frame) if frame is not None else None
            multi_modal = {
                self.config.image_placeholder: [pil] if pil is not None else []
            }
            # Ensure placeholder count in obs_str matches list length: if no
            # frame is available, drop the placeholder.
            if pil is None and self.config.image_placeholder in obs_str:
                obs_str = obs_str.replace(self.config.image_placeholder, "")
                multi_modal = None
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal,
            }

        return {"obs_str": obs_str}
