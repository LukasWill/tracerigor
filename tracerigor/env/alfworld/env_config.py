"""Configuration dataclass for the TraceRigor ALFWorld environment."""
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

from tracerigor.env.base.base_env_config import BaseEnvConfig

# Repo root, derived from this file's location: tracerigor/env/alfworld/env_config.py
# -> three parents up is the TraceRigor repo. Used to resolve relative
# ``alf_config_path`` values when the server's CWD differs from the repo root
# (e.g. Slurm submit-from-home, separate env-server processes, etc.).
_TRACERIGOR_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)


@dataclass
class ALFWorldEnvConfig(BaseEnvConfig):
    """Configuration for :class:`tracerigor.env.alfworld.env.ALFWorldEnv`.

    Attributes:
        env_name: Identifier registered in ``REGISTERED_ENV``.
        alf_config_path: Path to the alf-config.yaml file. ``$ENV`` references
            (e.g. ``$ALFWORLD_DATA``) are expanded at load time.
        train_eval: Dataset split fed to the underlying alfworld env. One of
            ``"train"``, ``"eval_in_distribution"``, ``"eval_out_of_distribution"``.
        render_mode: ``"text"`` (AlfredTWEnv) or ``"vision"`` (AlfredThorEnv).
            When ``"vision"`` the alf-config's ``env.type`` is overridden to
            ``AlfredThorEnv`` if needed.
        max_actions_per_step: Maximum number of actions parsed from one LLM
            response (currently the env executes the first one).
        prompt_format: One of the keys in
            :data:`tracerigor.env.alfworld.prompt.format_prompt`.
        use_history / history_length: Whether/how many past (obs, action)
            pairs to include in observations.
        max_steps: Hard cap on environment steps per episode.
        invalid_action_penalty: Per-step reward when the parsed action is not
            in the current admissible_commands list.
        include_gc_reward: When ``True`` (and the env exposes it), add the
            ALFRED ``goal_condition_success_rate`` as a partial step reward.
        win_reward: Reward magnitude applied on task completion.
        add_example: Include an ICL example in the format-prompt section.
    """

    env_name: str = "alfworld"

    alf_config_path: str = "SCRIPT_DIR/alf-config.yaml"
    train_eval: str = "train"

    render_mode: str = "text"
    max_actions_per_step: int = 1
    prompt_format: str = "free_think"

    use_history: bool = True
    history_length: int = 2

    max_steps: int = 50

    invalid_action_penalty: float = -0.1
    include_gc_reward: bool = False
    win_reward: float = 10.0

    add_example: bool = True

    # Per-turn format shaping mode (see ALFWorldEnv.step):
    #   "penalty"     : 0 if well-formed, -format_reward if malformed (default; farm-proof)
    #   "gated_bonus" : +format_reward only for well-formed + admissible + effective turns
    #   "bonus"       : +format_reward for any well-formed turn (legacy; farmable under GAE)
    # Rationale: under multi-turn GAE the discounted +format_reward stream
    # saturates at format_reward / (1 - high_level_gamma) (= 10.0 at 0.95) which
    # equals win_reward, so a positive bonus lets a well-formatted non-solving
    # episode accrue a return comparable to a real success. ALFWorld defaults to
    # "penalty" because even "gated_bonus" leaks via admissible obs-changing
    # oscillation (open/close, go-to loops) that the violation tracker misses.
    format_shaping: str = "penalty"

    # ------------------------------------------------------------------
    # Violation-based early termination.
    #
    # Pathological failure modes seen in finegrained-alfworld-reflact runs
    # (audit at steps 0/10/20/30, 128 trajs each):
    #   - FORMAT violations spike at step 0 (60 turns in one traj).
    #   - INADMISSIBLE runs of 5+ in 56/128 trajs at step 0.
    #   - REPETITION of identical action with frozen obs persists across
    #     training (70-113/128 trajs with 5+ consecutive repeats).
    #   - NO_PROGRESS (obs unchanged regardless of action) is the dominant
    #     mid-training mode: 80%+ of step_10/20 trajs contain 20+ step
    #     frozen-obs runs over multiple distinct (often inadmissible)
    #     action strings — admissible no-op actions reset both the
    #     INADMISSIBLE and REPETITION counters, so NO_PROGRESS is what
    #     catches this pattern.
    # ------------------------------------------------------------------
    enable_violation_termination: bool = True

    # Consecutive format violations (cannot parse <reflection>/<action> or
    # parsed action is empty).
    format_violation_threshold: int = 3

    # Consecutive inadmissible actions (action not in admissible_commands).
    inadmissible_action_threshold: int = 5

    # Consecutive identical (action, observation) pairs — agent locked in.
    repetition_threshold: int = 5

    # Consecutive steps with unchanged observation regardless of action —
    # agent flailing. Set higher than REPETITION because exploration via
    # admissible no-op actions (look/inventory) legitimately keeps obs
    # static for a few turns.
    no_progress_threshold: int = 7

    # Reward applied once when a violation threshold triggers termination.
    violation_penalty: float = -1.0

    # Mirrors the field on every other env config (sciworld, blackjack, ...).
    # The flag is consumed by ALFWorldService (and its
    # ``service_state_reward_wrapper``), not by this env directly — but it
    # has to be accepted here too so create_dataset can pass-through the
    # same ``env_config`` dict to both the env and the service constructor.
    use_state_reward: bool = False

    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "<think>", "</think>",
            "<action>", "</action>",
            "<answer>", "</answer>",
            "<reflection>", "</reflection>",
        ]
    )

    def __post_init__(self):
        # Allow ${env:VAR} / $VAR style references and ~ expansion.
        raw = self.alf_config_path
        raw = raw.replace("${env:", "$").replace("}", "")
        raw = os.path.expanduser(os.path.expandvars(raw))

        # If a relative path can't be resolved against the current CWD
        # (typical when the env-server runs from a different directory than
        # the trainer), fall back to resolving it against the TraceRigor repo
        # root. Absolute paths are left untouched.
        if not os.path.isabs(raw) and not os.path.exists(raw):
            candidate = os.path.join(_TRACERIGOR_REPO_ROOT, raw)
            if os.path.exists(candidate):
                raw = candidate

        self.alf_config_path = raw

    def config_id(self) -> str:
        id_fields = [
            "env_name",
            "alf_config_path",
            "train_eval",
            "render_mode",
            "max_actions_per_step",
            "prompt_format",
            "use_history",
            "history_length",
            "max_steps",
            "include_gc_reward",
            "format_reward",
            "format_shaping",
            "enable_violation_termination",
            "format_violation_threshold",
            "inadmissible_action_threshold",
            "repetition_threshold",
            "no_progress_threshold",
        ]
        parts = []
        for f in fields(self):
            if f.name in id_fields:
                parts.append(f"{f.name}={getattr(self, f.name)}")
        return f"ALFWorldEnvConfig({','.join(parts)})"
