"""
SciWorld Environment Configuration for TraceRigor.

This module defines the configuration dataclass for the ScienceWorld environment,
following the TraceRigor BaseEnvConfig interface.
"""
from tracerigor.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields, field
from typing import Optional, List, Union
import random
import os
import json


# Get the directory where this module is located
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SciWorldEnvConfig(BaseEnvConfig):
    """
    Configuration for SciWorld Environment.

    Attributes:
        env_name: Name of the environment (always "sciworld")
        task_nums: List of task numbers to sample from
        split: Dataset split ("train", "dev", "test")
        simplifications_preset: Simplification preset string for ScienceWorld
        env_step_limit: Maximum number of steps per episode
        jar_path: Path to ScienceWorld JAR file
        variations_idx: List of [task_id, variation_id] tuples to use
        variations_idx_path: Path to JSON file with variations_idx
        generalization_level: Level of generalization (0, 1, 2)
        render_mode: Observation rendering mode ("text" only for SciWorld)
        max_actions_per_step: Maximum number of actions to execute per step
        prompt_format: Format for LLM prompts ("free_think", "no_think", etc.)
        use_history: Whether to include action history in observations
        history_length: Number of historical steps to include
        meta_think: Whether to use meta-thinking prompts
        use_state_reward: Whether to use state-based rewards (for process reward)

        # Inherited from BaseEnvConfig:
        format_reward: Reward bonus for correct response format
        image_placeholder: Placeholder string for images
        special_token_list: List of special tokens to filter
        action_sep: Separator for multiple actions
    """
    env_name: str = "sciworld"

    # SciWorld-specific configurations
    task_nums: List[int] = field(default_factory=lambda: [1])
    split: str = "train"
    simplifications_preset: str = ""
    env_step_limit: int = 100
    jar_path: Optional[str] = None
    variations_idx: Optional[List[tuple]] = None
    variations_idx_path: Optional[str] = None
    generalization_level: int = 0

    # Rendering and prompt configuration
    render_mode: str = "text"  # SciWorld is text-only
    max_actions_per_step: int = 1
    prompt_format: str = "free_think"

    # History configuration
    use_history: bool = True
    history_length: int = 2

    # Meta-thinking mode (multi-phase reasoning)
    meta_think: bool = False

    # Whether to include ICL example in the format prompt
    add_example: bool = True

    # Per-turn format shaping mode (see SciWorldEnv.step):
    #   "penalty"     : 0 if well-formed, -format_reward if malformed (default; farm-proof)
    #   "gated_bonus" : +format_reward on a well-formed turn whose action executed
    #                   successfully (not rejected) and changed the observation
    #   "bonus"       : +format_reward for any well-formed turn (legacy; farmable under GAE)
    # SciWorld defaults to "penalty". In a free-form text env a positive bonus
    # cannot be gated faithfully: a changed observation does NOT imply progress
    # (varied wandering -- and even rejected actions -- yield fresh feedback every
    # turn), and the only fully reliable signal (a score increase) is sparse early
    # and already the graded task reward. "gated_bonus" is therefore a *liveness*
    # gate, not a progress guarantee: it denies the dominant text-env farm (runs of
    # rejected/failed actions) and stays dense early, but clean wandering still
    # leaks -- hence penalty is the default.
    format_shaping: str = "penalty"

    # State reward configuration
    use_state_reward: bool = False

    # ==========================================================================
    # Observation Enhancement Configuration
    # SciWorld's normal obs can be incomplete (e.g., "You move to kitchen" without
    # showing what's in the kitchen). Per GitHub issue #81, including look/inv
    # info provides more complete environment state for agent task-solving.
    # See: https://github.com/allenai/ScienceWorld/issues/81
    # ==========================================================================

    # Whether to include 'look' (room description) in observations
    include_look_in_obs: bool = False

    # Whether to include 'inventory' in observations
    include_inventory_in_obs: bool = False

    # ==========================================================================
    # Violation Handling Configuration
    # Controls early termination when agent repeatedly produces illegal actions
    # ==========================================================================

    # Enable/disable violation-based early termination
    enable_violation_termination: bool = True

    # Threshold for consecutive format violations (missing <reflection> or <action>)
    # Format violation = cannot parse reflection/action from LLM response
    format_violation_threshold: int = 3

    # Threshold for consecutive invalid actions ("No known action matches")
    # Invalid action = syntactically correct but action not in valid_actions
    invalid_action_threshold: int = 5

    # Threshold for consecutive repeated actions with unchanged observation
    # Repetition = exact same action + exact same observation response
    repetition_threshold: int = 5

    # Penalty applied when terminated due to violations (negative reward)
    violation_penalty: float = -1.0

    # Override special tokens for SciWorld format
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<action>", "</action>",
                                  "<planning>", "</planning>", "<explore>", "</explore>",
                                  "<reflection>", "</reflection>", "<monitor>", "</monitor>"]
    )

    def __post_init__(self):
        """
        Post-initialization to automatically load variations_idx from JSON file.

        Priority:
        1. If variations_idx is already provided, use it directly
        2. If variations_idx_path is provided, load from that path
        3. If generalization_level is set (0, 1, 2), auto-load from bundled JSON files
        """
        # If variations_idx is already set, no need to load
        if self.variations_idx is not None:
            return

        # Determine the JSON path to load from
        json_path = None

        if self.variations_idx_path:
            # Use explicitly provided path
            json_path = self.variations_idx_path
        elif self.generalization_level in [0, 1, 2]:
            # Use bundled JSON file based on generalization level
            json_path = os.path.join(
                _MODULE_DIR,
                "variations_idx",
                f"L{self.generalization_level}_idx.json"
            )

        # Load variations_idx from JSON if path is determined
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Select based on split (train/test/dev)
                if self.split in data:
                    self.variations_idx = [tuple(item) for item in data[self.split]]
                elif 'train' in data and self.split == 'dev':
                    # Fallback: use a portion of train for dev if dev not available
                    self.variations_idx = [tuple(item) for item in data['train']]
                else:
                    # Default to train if split not found
                    self.variations_idx = [tuple(item) for item in data.get('train', [])]

                if self.variations_idx:
                    print(f"[SciWorldEnvConfig] Loaded {len(self.variations_idx)} variations "
                          f"for split='{self.split}' from {json_path}")
            except Exception as e:
                print(f"[SciWorldEnvConfig] Warning: Failed to load variations_idx from {json_path}: {e}")
                self.variations_idx = None

    def config_id(self) -> str:
        """
        Generate a unique identifier for this configuration.
        Used for environment pooling and logging.
        """
        id_fields = [
            "task_nums",
            "split",
            "simplifications_preset",
            "env_step_limit",
            "render_mode",
            "max_actions_per_step",
            "use_history",
            "history_length",
            "meta_think",
            "format_reward",
            "format_shaping",
            "prompt_format"
        ]
        id_parts = []
        for field_obj in fields(self):
            if field_obj.name in id_fields:
                val = getattr(self, field_obj.name)
                # Handle list types
                if isinstance(val, list):
                    val = str(val)
                id_parts.append(f"{field_obj.name}={val}")

        id_str = ",".join(id_parts)
        return f"SciWorldEnvConfig({id_str})"

    def generate_seeds(self, size: int, seed: int = 0, n_candidate: int = 20000) -> list:
        """
        Generate a list of seeds for environment resets.

        For SciWorld, this method has two modes:

        Mode 1 (variations_idx available): Returns indices into self.variations_idx.
            Each "seed" is an index that will be used to select a specific
            (task_id, variation_id) tuple from variations_idx during reset.
            This ensures deterministic task/variation selection.

        Mode 2 (no variations_idx): Returns random integer seeds that will be
            used for random task/variation selection during reset.

        Args:
            size: Number of seeds/indices to generate
            seed: Random seed for reproducibility
            n_candidate: Pool size for seed sampling (only used in Mode 2)

        Returns:
            List of integer seeds or indices
        """
        random.seed(seed)

        if self.variations_idx:
            # Mode 1: Sample indices into variations_idx
            # This ensures we use the curated (task, variation) pairs
            num_variations = len(self.variations_idx)
            if size <= num_variations:
                # Sample without replacement if we have enough variations
                indices = random.sample(range(num_variations), size)
            else:
                # Sample with replacement if we need more than available
                indices = [random.randint(0, num_variations - 1) for _ in range(size)]
            return indices
        else:
            # Mode 2: Generate random seeds for random task/variation selection
            seeds = random.sample(range(0, n_candidate + size), size)
            return seeds

    def get_variation_for_seed(self, seed_or_idx: int) -> Optional[tuple]:
        """
        Get the (task_id, variation_id) tuple for a given seed/index.

        When variations_idx is available, the seed is actually an index into
        variations_idx. This method returns the corresponding tuple.

        Args:
            seed_or_idx: Either a random seed or an index into variations_idx

        Returns:
            (task_id, variation_id) tuple if variations_idx is available, else None
        """
        if self.variations_idx and 0 <= seed_or_idx < len(self.variations_idx):
            return self.variations_idx[seed_or_idx]
        return None


if __name__ == "__main__":
    # Test the configuration
    config = SciWorldEnvConfig()
    print(config.config_id())

    config_custom = SciWorldEnvConfig(
        task_nums=[1, 2, 3],
        env_step_limit=50,
        meta_think=True
    )
    print(config_custom.config_id())
