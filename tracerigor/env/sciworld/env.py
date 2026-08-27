"""
SciWorld Environment for TraceRigor.

This module implements the ScienceWorld environment wrapper following the TraceRigor
BaseEnv interface. It bridges raw LLM responses to the underlying ScienceWorld
game logic.
"""
import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from tracerigor.env.base.base_env import BaseEnv
from tracerigor.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import SciWorldEnvConfig


# ScienceWorld emits several distinct strings when an action is rejected or
# cannot be executed. Matching only "No known action" (as the violation tracker
# historically did) misses most failed actions; this fuller set is used to gate
# the non-default "gated_bonus" format shaping on *successful execution*.
# Lowercased substrings; extend as new rejection phrasings are observed.
_SCIWORLD_REJECTION_PATTERNS = (
    "no known action",            # "No known action matches that input."
    "unknown action",             # "Unknown action. Type 'help' ..."
    "ambiguous request",          # "Ambiguous request: Please enter the number ..."
    "not something that can be",  # "The thermometer is not something that can be activated."
    "not clear how to",           # "Its not clear how to go to/through a counter." / "... how to get there"
)


def _is_rejection(feedback: str) -> bool:
    """True if SciWorld feedback indicates the action was rejected / could not
    execute. Shared by the violation tracker (invalid-action detection) and the
    "gated_bonus" liveness gate so both use the same broadened pattern set."""
    f = (feedback or "").lower()
    return any(p in f for p in _SCIWORLD_REJECTION_PATTERNS)


# =============================================================================
# Violation Types and Tracker
# =============================================================================

class ViolationType(Enum):
    """Types of violations that can trigger early termination."""
    FORMAT = "format_violation"      # Missing <reflection> or <action> tags
    INVALID_ACTION = "invalid_action"  # Action not in valid_actions list
    REPETITION = "repetition"        # Same action with unchanged observation


@dataclass
class ViolationTracker:
    """
    Tracks consecutive violations to enable early termination of problematic trajectories.

    This mechanism addresses the issue where LLM agents fall into failure modes
    (e.g., repeatedly outputting invalid actions or the same action without progress),
    which:
    1. Pollutes training data with uninformative interactions
    2. Makes LLM judge evaluation ambiguous (hard to track state)
    3. Wastes compute on trajectories that won't improve

    The tracker monitors three violation types:
    - FORMAT: Cannot parse <reflection>/<action> from response
    - INVALID_ACTION: Action not available in current state
    - REPETITION: Same (action, observation) pair repeated

    Each violation type has its own consecutive counter that resets when
    a valid, different action is taken.
    """

    # Thresholds for each violation type
    format_threshold: int = 3
    invalid_action_threshold: int = 5
    repetition_threshold: int = 3

    # Consecutive violation counters
    consecutive_format_violations: int = field(default=0, init=False)
    consecutive_invalid_actions: int = field(default=0, init=False)
    consecutive_repetitions: int = field(default=0, init=False)

    # Track last action and observation for repetition detection
    last_action: Optional[str] = field(default=None, init=False)
    last_observation: Optional[str] = field(default=None, init=False)

    # Track total violations for metrics
    total_format_violations: int = field(default=0, init=False)
    total_invalid_actions: int = field(default=0, init=False)
    total_repetitions: int = field(default=0, init=False)

    def reset(self):
        """Reset all violation counters (called on env.reset())."""
        self.consecutive_format_violations = 0
        self.consecutive_invalid_actions = 0
        self.consecutive_repetitions = 0
        self.last_action = None
        self.last_observation = None
        self.total_format_violations = 0
        self.total_invalid_actions = 0
        self.total_repetitions = 0

    def record_step(
        self,
        format_correct: bool,
        action: str,
        observation: str,
        action_feedback: str = ""
    ) -> Tuple[bool, Optional[ViolationType]]:
        """
        Record a step and check for violation threshold breach.

        Args:
            format_correct: Whether the LLM response was correctly formatted
            action: The parsed action (empty string if format violation)
            observation: The observation AFTER the action (for repetition detection)
            action_feedback: Raw feedback from environment (for detecting "No known action")

        Returns:
            Tuple of (should_terminate, violation_type)
            - should_terminate: True if any violation threshold exceeded
            - violation_type: Which violation caused termination (None if not terminating)

        Note: Violation types are checked independently, not mutually exclusive.
        A single step can have multiple violation types (e.g., format + repetition).
        The termination priority is: FORMAT > INVALID_ACTION > REPETITION
        """
        should_terminate = False
        termination_reason = None

        # Track current violations for this step
        has_format_violation = False
        has_invalid_action = False
        has_repetition = False

        # Check 1: Format violation (missing tags or empty action)
        if not format_correct or not action:
            has_format_violation = True
            self.consecutive_format_violations += 1
            self.total_format_violations += 1
        else:
            self.consecutive_format_violations = 0

        # Check 2: Invalid action - when the env rejects the action.
        # NOTE: SciWorld has an internal syntax interpreter that accepts many actions
        # not in the explicit valid_actions list, so we detect invalidity from the
        # env's rejection feedback rather than a valid_actions lookup. We match the
        # full rejection set (_SCIWORLD_REJECTION_PATTERNS) -- "No known action",
        # "Unknown action", "Ambiguous request", "not something that can be",
        # "not clear how to ..." -- since matching only "No known action" misses
        # most failed actions and lets error-spam trajectories run to the cap.
        # Only check if we have an action to validate
        if action:
            is_invalid = _is_rejection(action_feedback)
            if is_invalid:
                has_invalid_action = True
                self.consecutive_invalid_actions += 1
                self.total_invalid_actions += 1
            else:
                self.consecutive_invalid_actions = 0
        else:
            # No action (format violation) - reset invalid counter since
            # the issue is format, not invalid action choice
            self.consecutive_invalid_actions = 0

        # Check 3: Repetition (same action + same/similar observation)
        # Check this regardless of other violations - agent can repeat malformed responses too
        is_repetition = (
            self.last_action is not None and
            self.last_action == action and
            self._observations_similar(self.last_observation, observation)
        )

        if is_repetition:
            has_repetition = True
            self.consecutive_repetitions += 1
            self.total_repetitions += 1
        else:
            self.consecutive_repetitions = 0

        # Determine termination (priority: FORMAT > INVALID > REPETITION)
        if self.consecutive_format_violations >= self.format_threshold:
            should_terminate = True
            termination_reason = ViolationType.FORMAT
        elif self.consecutive_invalid_actions >= self.invalid_action_threshold:
            should_terminate = True
            termination_reason = ViolationType.INVALID_ACTION
        elif self.consecutive_repetitions >= self.repetition_threshold:
            should_terminate = True
            termination_reason = ViolationType.REPETITION

        # Update last action/observation for next step's repetition check
        self.last_action = action
        self.last_observation = observation

        return should_terminate, termination_reason

    def _observations_similar(self, obs1: Optional[str], obs2: str) -> bool:
        """
        Check if two observations are similar enough to count as repetition.

        Currently uses exact match, but could be extended to use fuzzy matching
        or semantic similarity if needed.
        """
        if obs1 is None:
            return False
        # Normalize whitespace for comparison
        obs1_normalized = ' '.join(obs1.split())
        obs2_normalized = ' '.join(obs2.split())
        return obs1_normalized == obs2_normalized

    def get_metrics(self) -> Dict[str, Any]:
        """Return violation metrics for logging."""
        return {
            "total_format_violations": self.total_format_violations,
            "total_invalid_actions": self.total_invalid_actions,
            "total_repetitions": self.total_repetitions,
            "consecutive_format_violations": self.consecutive_format_violations,
            "consecutive_invalid_actions": self.consecutive_invalid_actions,
            "consecutive_repetitions": self.consecutive_repetitions,
        }


# Import prompt utilities after ViolationTracker to avoid circular imports
from .prompt import (
    system_prompt,
    init_observation_template,
    action_observation_template,
    format_prompt,
    format_action_history
)


class SciWorldEnv(BaseEnv):
    """
    ScienceWorld Environment for training and evaluating language models as agents.

    This environment wraps the ScienceWorld text-based game where an agent must
    complete science curriculum tasks through natural language interactions.

    The environment follows the TraceRigor BaseEnv interface, supporting:
    - Multi-turn interactions with history tracking
    - Flexible prompt formats (free_think, no_think, grounding, etc.)
    - Meta-thinking mode for multi-phase reasoning
    - Format reward for correct response structure
    """

    def __init__(self, config: SciWorldEnvConfig):
        """
        Initialize the SciWorld environment.

        Args:
            config: Environment configuration containing task settings and prompt options
        """
        BaseEnv.__init__(self)
        self.config = config

        # Lazy initialization - the actual ScienceWorld env will be created on reset
        self._gym_env = None
        self._initialized = False

        # Episode state
        self.total_reward = 0.0
        self.current_step = 0
        self.task_description = ""
        self.available_actions = ""
        self.possible_actions = []
        self.current_observation = ""

        # History buffer for multi-turn context
        self.history_buffer: List[Dict] = []
        self.planning: str = "No plan."

        # Initialize violation tracker for early termination of problematic trajectories
        if self.config.enable_violation_termination:
            self.violation_tracker = ViolationTracker(
                format_threshold=self.config.format_violation_threshold,
                invalid_action_threshold=self.config.invalid_action_threshold,
                repetition_threshold=self.config.repetition_threshold,
            )
        else:
            self.violation_tracker = None

        # Setup parsing function based on prompt format
        if self.config.prompt_format == "meta_think":
            self.parse_func = self._parse_meta_think
        elif self.config.prompt_format in ("reflact", "reflact_diverse"):
            # Both reflact and reflact_diverse use <reflection>...<action>... format
            self.parse_func = self._create_reflact_parser()
        elif self.config.prompt_format == "no_think":
            self.parse_func = self._create_no_think_parser()
        else:
            # Formats using <think>...</think><action>...</action>:
            # free_think, react, grounding, worldmodeling, grounding_worldmodeling
            self.parse_func = self._create_action_parser()

        # Format prompt function
        self.format_prompt_func = format_prompt.get(
            self.config.prompt_format,
            format_prompt["free_think"]
        )

    def _create_no_think_parser(self):
        """
        Create a parser for no_think format: <action>...</action>
        """
        def parse_no_think_response(response: str, special_token_list=None, action_sep=',', max_actions=1) -> Dict:
            """Parse response with <action>...</action> format only."""
            response = response.replace("<image>", "")

            # Pattern for strict format check
            strict_pattern = r'^\s*<action>(.*?)</action>\s*$'
            strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)

            # Extraction pattern (more lenient)
            extraction_pattern = r'<action>(.*?)</action>'
            match = re.search(extraction_pattern, response, re.DOTALL)
            format_correct = strict_match is not None

            if not match:
                action_content, actions = "", []
            else:
                action_content = match.group(1).strip()
                if special_token_list is not None:
                    for special_token in special_token_list:
                        action_content = action_content.replace(special_token, "").strip()
                actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
                if len(actions) > max_actions:
                    actions = actions[:max_actions]
                    action_content = (" " + action_sep + " ").join(actions)

            llm_response = f"<action>{action_content}</action>"
            return {
                "llm_raw_response": response,
                "llm_response": llm_response,
                "think_content": "",
                "action_content": action_content,
                "actions": actions,
                "format_correct": format_correct
            }

        return parse_no_think_response

    def _create_action_parser(self):
        """
        Create a parser that handles <action> tags instead of <answer> tags.
        SciWorld uses <action>...</action> while standard TraceRigor uses <answer>...</answer>.
        """
        def parse_action_response(response: str, special_token_list=None, action_sep=',', max_actions=1) -> Dict:
            """Parse response with <think>...</think><action>...</action> format."""
            response = response.replace("<image>", "")

            # Pattern for strict format check
            strict_pattern = r'^\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*$'
            strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)

            # Extraction pattern (more lenient)
            extraction_pattern = r'<think>(.*?)</think>\s*<action>(.*?)</action>'
            match = re.search(extraction_pattern, response, re.DOTALL)
            format_correct = strict_match is not None

            if not match:
                think_content, action_content, actions = "", "", []
            else:
                think_content, action_content = match.group(1).strip(), match.group(2).strip()
                if special_token_list is not None:
                    for special_token in special_token_list:
                        action_content = action_content.replace(special_token, "").strip()
                        think_content = think_content.replace(special_token, "").strip()
                actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
                if len(actions) > max_actions:
                    actions = actions[:max_actions]
                    action_content = (" " + action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><action>{action_content}</action>"
            return {
                "llm_raw_response": response,
                "llm_response": llm_response,
                "think_content": think_content,
                "action_content": action_content,
                "actions": actions,
                "format_correct": format_correct
            }

        return parse_action_response

    def _create_reflact_parser(self):
        """
        Create a parser for ReflAct format: <reflection>...</reflection><action>...</action>

        ReflAct (from arXiv:2505.15182) emphasizes reflecting on agent's state
        (location, inventory, progress) in relation to the task goal.
        """
        def parse_reflact_response(response: str, special_token_list=None, action_sep=',', max_actions=1) -> Dict:
            """Parse response with <reflection>...</reflection><action>...</action> format."""
            response = response.replace("<image>", "")

            # Pattern for strict format check
            strict_pattern = r'^\s*<reflection>(.*?)</reflection>\s*<action>(.*?)</action>\s*$'
            strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
            if (
                len(re.findall(r"<reflection>", response)) != 1
                or len(re.findall(r"<action>", response)) != 1
            ):
                strict_match = None

            # Extraction pattern (more lenient)
            extraction_pattern = r'<reflection>(.*?)</reflection>\s*<action>(.*?)</action>'
            match = re.search(extraction_pattern, response, re.DOTALL)
            format_correct = strict_match is not None

            if not strict_match:
                reflection_content, action_content, actions = "", "", []
            else:
                reflection_content, action_content = match.group(1).strip(), match.group(2).strip()
                if special_token_list is not None:
                    for special_token in special_token_list:
                        action_content = action_content.replace(special_token, "").strip()
                        reflection_content = reflection_content.replace(special_token, "").strip()
                actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
                if len(actions) > max_actions:
                    actions = actions[:max_actions]
                    action_content = (" " + action_sep + " ").join(actions)

            llm_response = f"<reflection>{reflection_content}</reflection><action>{action_content}</action>"
            return {
                "llm_raw_response": response,
                "llm_response": llm_response,
                "reflection_content": reflection_content,
                "think_content": reflection_content,  # Alias for compatibility
                "action_content": action_content,
                "actions": actions,
                "format_correct": format_correct
            }

        return parse_reflact_response

    def _parse_meta_think(self, response: str, special_token_list=None, action_sep=',', max_actions=1) -> Dict:
        """
        Parse response with meta-thinking format.
        Supports <planning>, <explore>, <reflection>, <monitor> tags.
        """
        response = response.replace("<image>", "")

        # Check for any of the meta-thinking tags
        meta_tags = ['planning', 'explore', 'reflection', 'monitor']
        think_content = ""
        meta_type = None

        for tag in meta_tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                think_content = match.group(1).strip()
                meta_type = tag
                break

        # Extract action
        action_pattern = r'<action>(.*?)</action>'
        action_match = re.search(action_pattern, response, re.DOTALL)

        if action_match:
            action_content = action_match.group(1).strip()
            actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
            if len(actions) > max_actions:
                actions = actions[:max_actions]
                action_content = (" " + action_sep + " ").join(actions)
        else:
            action_content = ""
            actions = []

        # Format correct if has both meta tag and action
        format_correct = meta_type is not None and len(actions) > 0

        # Store planning if detected
        planning = think_content if meta_type == 'planning' else None

        llm_response = f"<{meta_type or 'think'}>{think_content}</{meta_type or 'think'}><action>{action_content}</action>"
        return {
            "llm_raw_response": response,
            "llm_response": llm_response,
            "think_content": think_content,
            "action_content": action_content,
            "actions": actions,
            "format_correct": format_correct,
            "meta_type": meta_type,
            "planning": planning
        }

    def _init_sciworld_env(self, seed: int):
        """
        Lazily initialize the ScienceWorld environment.

        This is called on first reset() to avoid import issues and allow
        configuration of JAR path at runtime.
        """
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError:
            raise ImportError(
                "ScienceWorld is not installed. Please install it with: "
                "pip install scienceworld"
            )

        # JAR path handling - None means use default bundled JAR
        jar_path = self.config.jar_path

        try:
            # ScienceWorld expects: ScienceWorldEnv(taskName, jarPath, envStepLimit)
            # Empty string for taskName means we'll load a task later via load()
            self._gym_env = ScienceWorldEnv("", jar_path, envStepLimit=self.config.env_step_limit)
            self._task_names = self._gym_env.get_task_names()
            self._initialized = True
        except ValueError as e:
            # Common issue: Java not installed or JAR path issues
            raise RuntimeError(
                f"Failed to initialize ScienceWorld environment. "
                f"This is often caused by: \n"
                f"1. Java not being installed (ScienceWorld requires Java)\n"
                f"2. Invalid JAR path: {jar_path}\n"
                f"3. ScienceWorld package not properly installed\n"
                f"Original error: {e}"
            ) from e

    def reset(self, seed: int = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility. When variations_idx is used,
                  this can also be an index into variations_idx (from generate_seeds).

        Returns:
            Tuple of (observation dict, info dict)
        """
        if not self._initialized:
            self._init_sciworld_env(seed or 0)

        # Set random seed for any randomness
        if seed is not None:
            random.seed(seed)

        # Select task and variation
        if self.config.variations_idx:
            # Mode 1: Use pre-defined variations_idx
            # Check if seed is a valid index into variations_idx
            variation_tuple = self.config.get_variation_for_seed(seed) if seed is not None else None

            if variation_tuple is not None:
                # Seed is an index from generate_seeds - use the specific variation
                task_id, task_variation = variation_tuple
            else:
                # Fallback: random choice from variations_idx
                task_id, task_variation = random.choice(self.config.variations_idx)
        else:
            # Select random task
            task_id = random.choice(self.config.task_nums)
            task_name = self._task_names[task_id]

            # Must load a task first before getting variations
            # Load with variation 0 initially to get available variations
            simplification_str = self.config.simplifications_preset or ""
            self._gym_env.load(task_name, 0, simplification_str)

            # Now get variations for this task based on split
            split = self.config.split
            try:
                if split == "train":
                    variations = self._gym_env.get_variations_train()
                elif split == "dev":
                    variations = self._gym_env.get_variations_dev()
                else:  # test
                    variations = self._gym_env.get_variations_test()
                task_variation = random.choice(variations) if variations else 0
            except Exception:
                # Fallback if variations can't be retrieved
                task_variation = 0

        task_name = self._task_names[task_id]

        # Load and reset
        simplification_str = self.config.simplifications_preset or ""
        self._gym_env.load(task_name, task_variation, simplification_str)
        observation, gym_reset_info = self._gym_env.reset()

        # Store task info
        self.task_description = self._gym_env.get_task_description()
        self._update_available_actions()

        # Build enhanced observation with look/inventory if enabled
        # For initial observation, look == observation, so we only add inventory
        self.current_observation = self._build_enhanced_observation(observation, gym_reset_info)

        # Reset episode state
        self.total_reward = 0.0
        self.current_step = 0
        self.history_buffer = []
        self.planning = "No plan."

        # Reset violation tracker
        if self.violation_tracker is not None:
            self.violation_tracker.reset()

        # Store task info for trajectory replay in offline evaluation
        self._current_task_name = task_name
        self._current_task_variation = task_variation
        self._current_task_id = task_id

        # Build info dict (note: info is a new dict, not the gym_reset_info)
        info = {
            "task_description": self.task_description,
            "available_actions": self.available_actions,
            "observation_text": self.current_observation,  # Use enhanced observation
            "possible_actions": self.possible_actions,
            "won": False,
            "task_num": task_id,
            # Task info for trajectory replay in offline LLM judge evaluation
            "task_name": task_name,
            "task_variation": task_variation,
            "metrics": {
                "turn_metrics": {},
                "traj_metrics": {"success": False}
            }
        }

        return self._render(init_obs=True), info

    def _update_available_actions(self):
        """Update the list of available actions from the environment."""
        valid_actions = self._gym_env.get_possible_actions()
        valid_objs = self._gym_env.get_possible_objects()
        self.available_actions = (
            f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of "
            f"the following objects: {valid_objs}\n example: <action>focus on door</action>"
        )
        self.possible_actions = self._gym_env.get_valid_action_object_combinations()

    def _build_enhanced_observation(self, raw_obs: str, gym_info: dict) -> str:
        """
        Build enhanced observation by optionally including look and inventory info.

        SciWorld's normal observation can be incomplete (e.g., "You move to kitchen"
        without showing what's in the kitchen). Per GitHub issue #81, including
        look/inv info provides more complete environment state for agent task-solving.
        See: https://github.com/allenai/ScienceWorld/issues/81

        Args:
            raw_obs: The raw observation from SciWorld step/reset
            gym_info: The info dict from SciWorld containing 'look' and 'inv' keys

        Returns:
            Enhanced observation string with look and/or inventory info appended
        """
        enhanced_obs = raw_obs

        # Append look (room description) if enabled and available
        if self.config.include_look_in_obs:
            look_info = gym_info.get('look', '')
            if look_info and look_info.strip():
                # Only append if look provides different info than raw obs
                # (for initial obs, look == raw_obs, so we don't duplicate)
                if look_info.strip() != raw_obs.strip():
                    enhanced_obs += f"\n\nCurrent room: {look_info.strip()}"

        # Append inventory if enabled and available
        if self.config.include_inventory_in_obs:
            inv_info = gym_info.get('inv', '')
            if inv_info and inv_info.strip():
                enhanced_obs += f"\n\n{inv_info.strip()}"

        return enhanced_obs

    def step(self, llm_raw_response: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute a step in the environment based on the LLM's response.

        Args:
            llm_raw_response: Raw text response from the LLM

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Store the observation BEFORE executing the action (for history)
        pre_action_observation = self.current_observation

        # Parse the LLM response
        rst = self.parse_func(
            response=llm_raw_response,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )

        action_list = rst.get('actions', [])
        format_correct = rst.get('format_correct', False)

        # Update planning if meta_think mode
        if self.config.meta_think and rst.get('planning'):
            self.planning = rst['planning']

        # Initialize metrics
        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct and len(action_list) > 0,
                "action_is_effective": False,
                "action_is_available": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        # Initialize step variables
        reward = 0.0
        done = False
        info = {}
        info.update(rst)

        # Liveness signals used to gate the non-default "gated_bonus" shaping below.
        action_rejected = False
        obs_changed = False

        # Execute action
        if action_list:
            action = action_list[0]  # Take first action

            # Check if action is in available actions
            action_available = action in self.possible_actions
            metrics["turn_metrics"]["action_is_available"] = action_available

            # Execute in ScienceWorld
            observation, step_reward, is_completed, gym_info = self._gym_env.step(action)

            # Build enhanced observation with look/inventory if enabled
            # This addresses the incomplete observation issue per GitHub issue #81
            enhanced_observation = self._build_enhanced_observation(observation, gym_info)

            # Update state
            self.current_observation = enhanced_observation
            self._update_available_actions()

            # Calculate reward
            reward = step_reward
            done = is_completed

            # Liveness signals for the "gated_bonus" mode. In a free-form text env
            # an observation change does NOT imply progress: varied wandering and
            # even rejected actions each yield fresh feedback. So we gate on
            # *successful execution* -- the env did not reject the action -- using
            # the full SciWorld rejection set, not just "No known action" (which
            # alone misses most failed actions). obs_changed additionally drops
            # immediate exact-repeat no-ops.
            action_rejected = _is_rejection(observation)
            obs_changed = enhanced_observation.strip() != pre_action_observation.strip()

            if step_reward > 0:
                metrics["turn_metrics"]["action_is_effective"] = True

            # Check success
            # Per ReflAct paper (arXiv:2505.15182), ScienceWorld's isCompleted flag
            # can be buggy. Using score >= 70 as success threshold is more accurate.
            score = gym_info.get('score', 0.0)
            if is_completed and score > 0:
                metrics["traj_metrics"]["success"] = True
                info["won"] = True
            else:
                info["won"] = False

            # Update gym info
            info.update({
                "score": score,
                "task_score": score,
                "observation_text": observation,
                "available_actions": self.available_actions,
                "possible_actions": self.possible_actions,
            })
        else:
            # No valid action - provide current observation again
            info["won"] = False
            info["observation_text"] = self.current_observation
            info["available_actions"] = self.available_actions
            info["possible_actions"] = self.possible_actions

        # Check for Chinese characters (invalid format)
        if re.search(r'[\u4e00-\u9fff]', llm_raw_response):
            format_correct = False
            metrics["turn_metrics"]["action_is_valid"] = False

        # --- Format shaping -------------------------------------------------
        # A positive per-turn bonus is farmable under multi-turn GAE (the
        # discounted +format_reward stream saturates at
        # format_reward / (1 - high_level_gamma) ~= 10 at 0.95). SciWorld defaults
        # to "penalty" because in a free-form text env a positive bonus cannot be
        # gated faithfully: a changed observation does not imply progress, and the
        # only fully reliable progress signal (a score increase) is sparse early
        # and already the graded task reward. The non-default "gated_bonus" is a
        # *liveness* gate (not a progress guarantee): it pays a well-formed turn
        # whose action executed successfully (not rejected) and changed the
        # observation. This denies the dominant text-env farm -- runs of rejected
        # / failed actions -- while staying dense early; clean wandering still
        # leaks, which is why "penalty" is the default. ``is_format_rewarded``
        # keeps its original meaning (well-formed response) so the state-reward
        # path still gates on it.
        action_was_effective = (
            bool(action_list) and (not action_rejected) and obs_changed
        )
        shaping = getattr(self.config, "format_shaping", "penalty")
        if shaping == "bonus":
            fmt_term = self.config.format_reward if format_correct else 0.0
        elif shaping == "gated_bonus":  # well-formed + executed successfully + obs changed
            fmt_term = self.config.format_reward if (format_correct and action_was_effective) else 0.0
        else:  # "penalty" (default, farm-proof): the best a turn can do is 0
            fmt_term = 0.0 if format_correct else -self.config.format_reward
        reward += fmt_term
        info["is_format_rewarded"] = format_correct
        info["format_shaping_term"] = fmt_term

        # =================================================================
        # Violation Tracking and Early Termination
        # =================================================================
        violation_terminated = False
        termination_reason = None

        if self.violation_tracker is not None:
            # Get the action and observation for violation checking
            action_taken = action_list[0] if action_list else ""
            observation_after = info.get("observation_text", self.current_observation)

            # Check for violations
            # Note: invalid action is detected from the env's rejection feedback
            # (_is_rejection / _SCIWORLD_REJECTION_PATTERNS), not a valid_actions
            # lookup (SciWorld has an internal syntax interpreter).
            violation_terminated, termination_reason = self.violation_tracker.record_step(
                format_correct=format_correct,
                action=action_taken,
                observation=observation_after,
                action_feedback=observation_after,  # rejection feedback, matched by _is_rejection
            )

            if violation_terminated:
                done = True
                reward += self.config.violation_penalty

                # Add violation info to metrics
                metrics["traj_metrics"]["violation_terminated"] = True
                metrics["traj_metrics"]["termination_reason"] = termination_reason.value if termination_reason else None

                info["violation_terminated"] = True
                info["termination_reason"] = termination_reason.value if termination_reason else None

            # Always add violation metrics
            metrics["traj_metrics"]["violation_metrics"] = self.violation_tracker.get_metrics()

        # Update history buffer with observation BEFORE action
        # Each (Observation N, Action N) pair
        # represents "what the agent saw → what it did"
        self._save_to_history(
            text_obs=pre_action_observation,  # observation BEFORE action
            action=action_list[0] if action_list else "",
            full_output=llm_raw_response
        )

        # Update step counter and total reward
        self.current_step += 1
        self.total_reward += reward

        info["metrics"] = metrics
        info["llm_raw_response"] = llm_raw_response

        return self._render(init_obs=False), reward, done, info

    def _save_to_history(self, text_obs: str, action: str, full_output: str):
        """
        Save current step to history buffer.

        Note: text_obs should be the observation BEFORE the action was taken,
        following the convention that (Observation N, Action N) represents
        "what the agent saw → what it did".
        """
        self.history_buffer.append({
            'text_obs': text_obs,
            'action': action,
            'full_output': full_output
        })

    def _render(self, init_obs: bool = False) -> Dict:
        """
        Render the environment observation for the LLM.

        Args:
            init_obs: Whether this is the initial observation

        Returns:
            Observation dictionary with 'obs_str' key
        """
        if init_obs or not self.config.use_history or self.config.history_length <= 0:
            obs_str = init_observation_template(
                task_description=self.task_description,
                observation=self.current_observation,
                available_actions=self.available_actions
            )
        else:
            action_history = format_action_history(
                buffers=self.history_buffer,
                history_length=self.config.history_length,
                include_full_output=self.config.meta_think
            )

            obs_str = action_observation_template(
                task_description=self.task_description,
                step_count=len(self.history_buffer),
                history_length=min(self.config.history_length, len(self.history_buffer)),
                action_history=action_history,
                current_step=self.current_step + 1,
                observation=self.current_observation,
                available_actions=self.available_actions,
                planning=self.planning if self.config.meta_think else None
            )

        return {
            'obs_str': obs_str,
            # SciWorld is text-only, no multi_modal_data needed
        }

    def system_prompt(self) -> str:
        """
        Get the system prompt for the environment.

        Returns:
            Complete system prompt string including format instructions
        """
        base_prompt = system_prompt(meta_think=self.config.meta_think)
        format_instructions = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.add_example
        )
        return base_prompt + '\n\n' + format_instructions

    def compute_reward(self) -> float:
        """
        Compute final episode reward.

        Returns:
            Final reward value (typically 0.0 as step rewards are accumulated)
        """
        return 0.0

    def close(self):
        """Close the environment and release resources."""
        if self._gym_env is not None:
            self._gym_env.close()
            self._gym_env = None
            self._initialized = False


if __name__ == "__main__":
    # Test the environment (requires scienceworld to be installed)
    config = SciWorldEnvConfig(
        task_nums=[1],
        env_step_limit=10,
        prompt_format="free_think"
    )

    try:
        env = SciWorldEnv(config)
        obs, info = env.reset(seed=42)
        print("System prompt:")
        print(env.system_prompt())
        print("\n" + "=" * 50 + "\n")
        print("Initial observation:")
        print(obs["obs_str"])

        for step_idx in range(5):
            user_action = input(f"\nStep {step_idx + 1} action (without tags): ").strip()
            if not user_action:
                print("Empty action, skipping this step.")
                continue
            llm_response = f"<think></think><action>{user_action}</action>"
            obs, reward, done, info = env.step(llm_response)
            print(f"Reward: {reward}, Done: {done}, Won: {info.get('won')}")
            print("Observation:")
            print(obs["obs_str"])
            if done:
                print("Episode finished early.")
                break
    except ImportError as e:
        print(f"ScienceWorld not installed: {e}")
