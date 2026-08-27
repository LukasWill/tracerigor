"""
Sokoban ViolationTracker — early termination for problematic trajectories.

Modeled on SciWorld's ViolationTracker but tuned for Sokoban's shorter episodes
(typically 5-10 max turns).

Violation types:
  FORMAT        — Response cannot be parsed (<action> tag missing, empty actions)
  INVALID_ACTION — Action not in ACTION_LOOKUP or causes no movement (no-op)
  REPETITION    — Same action + unchanged board observation (stuck loop)

Each violation type has a consecutive counter that resets when a valid,
effective action is taken.  API mirrors SciWorld's ViolationTracker for
consistency across envs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class SokobanViolationType(Enum):
    """Types of violations that can trigger early termination."""
    FORMAT = "format_violation"          # Missing <action> tag or unparseable
    INVALID_ACTION = "invalid_action"    # Not in ACTION_LOOKUP or no-op into wall
    REPETITION = "repetition"            # Same action + same board observation


@dataclass
class SokobanViolationTracker:
    """
    Tracks consecutive violations to enable early termination of problematic
    Sokoban trajectories during RL training.

    Tuned for Sokoban's short episodes (5-10 turns). Default thresholds are
    intentionally lower than SciWorld's since there are fewer steps to recover.

    API mirrors SciWorld's ViolationTracker: ``record_step`` accepts the parsed
    action string and the post-action observation string (for repetition
    detection via text similarity).
    """

    # Thresholds (lower than SciWorld due to shorter episodes)
    format_threshold: int = 2          # 2 consecutive format failures → terminate
    invalid_action_threshold: int = 3  # 3 consecutive invalid/no-op → terminate
    repetition_threshold: int = 2      # 2 consecutive repeats with same obs → terminate

    # Consecutive violation counters
    consecutive_format_violations: int = field(default=0, init=False)
    consecutive_invalid_actions: int = field(default=0, init=False)
    consecutive_repetitions: int = field(default=0, init=False)

    # Track last action and observation for repetition detection
    last_action: Optional[str] = field(default=None, init=False)
    last_observation: Optional[str] = field(default=None, init=False)

    # Total violation counts for metrics
    total_format_violations: int = field(default=0, init=False)
    total_invalid_actions: int = field(default=0, init=False)
    total_repetitions: int = field(default=0, init=False)

    def reset(self) -> None:
        """Reset all counters (called on env.reset())."""
        self.consecutive_format_violations = 0
        self.consecutive_invalid_actions = 0
        self.consecutive_repetitions = 0
        self.last_action = None
        self.last_observation = None
        self.total_format_violations = 0
        self.total_invalid_actions = 0
        self.total_repetitions = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _observations_similar(obs_a: Optional[str], obs_b: Optional[str]) -> bool:
        """Whitespace-normalised comparison (mirrors SciWorld)."""
        if obs_a is None or obs_b is None:
            return False
        return " ".join(obs_a.split()) == " ".join(obs_b.split())

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def record_step(
        self,
        format_correct: bool,
        action: str,
        observation: str,
        action_is_valid: bool = True,
        action_is_effective: bool = True,
    ) -> Tuple[bool, Optional[SokobanViolationType]]:
        """
        Record a step and check for violation threshold breach.

        Args:
            format_correct:      Whether the LLM response had valid format.
            action:              The first parsed action string (e.g. "Up").
                                 Empty string when format fails.
            observation:         Post-action observation text (for repetition).
            action_is_valid:     Action recognised in ACTION_LOOKUP.
            action_is_effective: Action caused actual movement (not a no-op).

        Returns:
            (should_terminate, violation_type)
        """
        # Check 1: Format violation
        if not format_correct or not action:
            self.consecutive_format_violations += 1
            self.total_format_violations += 1
        else:
            self.consecutive_format_violations = 0

        # Check 2: Invalid / no-op action (only meaningful when format is ok)
        if action and format_correct:
            if not action_is_valid or not action_is_effective:
                self.consecutive_invalid_actions += 1
                self.total_invalid_actions += 1
            else:
                self.consecutive_invalid_actions = 0
        else:
            # Format violation → don't count as invalid action
            self.consecutive_invalid_actions = 0

        # Check 3: Repetition (same action + same/similar observation)
        if action and self.last_action is not None:
            is_repetition = (
                self.last_action == action
                and self._observations_similar(self.last_observation, observation)
            )
            if is_repetition:
                self.consecutive_repetitions += 1
                self.total_repetitions += 1
            else:
                self.consecutive_repetitions = 0
        else:
            self.consecutive_repetitions = 0

        # Update last action / observation for next step
        self.last_action = action
        self.last_observation = observation

        # Determine termination (priority: FORMAT > INVALID > REPETITION)
        if self.consecutive_format_violations >= self.format_threshold:
            return True, SokobanViolationType.FORMAT
        if self.consecutive_invalid_actions >= self.invalid_action_threshold:
            return True, SokobanViolationType.INVALID_ACTION
        if self.consecutive_repetitions >= self.repetition_threshold:
            return True, SokobanViolationType.REPETITION

        return False, None

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
