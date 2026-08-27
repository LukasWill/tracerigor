"""
SciWorld Trajectory Replay for Ground Truth State Extraction.

This module provides utilities to replay recorded trajectories on a live SciWorld
environment to extract ground truth state information (location, inventory, etc.)
that isn't available in offline evaluation.

The key insight is that recorded trajectories contain the sequence of actions taken,
and we can replay these actions on a fresh environment instance to obtain:
1. Ground truth agent location (from object_tree)
2. Ground truth inventory state
3. Valid actions at each step (for definitive action validity checking)

Usage:
    from tracerigor.verifier.verifier.sciworld_trajectory_replay import (
        TrajectoryReplayer,
        extract_ground_truth_for_samples
    )

    # Method 1: Replay a full trajectory
    replayer = TrajectoryReplayer(task_name="boil", task_variation=0)
    states = replayer.replay_actions(["go to kitchen", "look around", "pick up beaker"])

    # Method 2: Enrich evaluation samples with ground truth
    enriched_samples = extract_ground_truth_for_samples(samples, task_info)
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class GroundTruthState:
    """Ground truth state extracted from SciWorld environment."""
    step_index: int
    location: Optional[str] = None
    inventory: List[str] = field(default_factory=list)
    object_tree: Optional[Dict] = None
    valid_actions: List[str] = field(default_factory=list)
    valid_objects: List[str] = field(default_factory=list)
    action_executed: str = ""
    action_accepted: bool = True  # False if "No known action matches"
    observation_after_action: str = ""


def _find_agent_in_tree(obj: Dict, parent_name: str = "root") -> Optional[str]:
    """Recursively find the agent's location in the object tree."""
    if not isinstance(obj, dict):
        return None

    if obj.get("name") == "agent":
        return parent_name

    contents = obj.get("contents", {})
    if isinstance(contents, dict):
        for child_obj in contents.values():
            result = _find_agent_in_tree(child_obj, obj.get("name", parent_name))
            if result:
                return result
    return None


def _get_agent_inventory(obj: Dict) -> Optional[List[str]]:
    """Recursively find the agent and return its inventory."""
    if not isinstance(obj, dict):
        return None

    if obj.get("name") == "agent":
        contents = obj.get("contents", {})
        if isinstance(contents, dict):
            return [
                item.get("name", key)
                for key, item in contents.items()
                if item.get("name") not in ("terminal 1", "terminal 2", "inventory")
            ]
        return []

    contents = obj.get("contents", {})
    if isinstance(contents, dict):
        for child_obj in contents.values():
            result = _get_agent_inventory(child_obj)
            if result is not None:
                return result
    return None


class TrajectoryReplayer:
    """
    Replays recorded trajectories on a live SciWorld environment to extract ground truth.

    This class manages a SciWorld environment instance and allows stepping through
    a recorded action sequence while capturing ground truth state at each step.
    """

    def __init__(
        self,
        task_name: str,
        task_variation: int = 0,
        simplification_str: str = "easy",
        env_step_limit: int = 100,
    ):
        """
        Initialize the replayer with a specific task.

        Args:
            task_name: Name of the SciWorld task (e.g., "boil", "melt", etc.)
            task_variation: Variation index for the task
            simplification_str: Simplification preset (e.g., "easy")
            env_step_limit: Maximum steps allowed
        """
        self.task_name = task_name
        self.task_variation = task_variation
        self.simplification_str = simplification_str
        self.env_step_limit = env_step_limit

        self._env = None
        self._initialized = False

    def _init_env(self):
        """Lazily initialize the SciWorld environment."""
        if self._initialized:
            return

        try:
            from scienceworld import ScienceWorldEnv
        except ImportError:
            raise ImportError(
                "scienceworld package not installed. "
                "Install with: pip install scienceworld"
            )

        self._env = ScienceWorldEnv('', serverPath=None, envStepLimit=self.env_step_limit)
        self._env.load(self.task_name, self.task_variation, self.simplification_str)
        self._initialized = True

    def reset(self) -> GroundTruthState:
        """Reset the environment and return initial ground truth state."""
        self._init_env()
        obs, info = self._env.reset()

        object_tree = self._env.getObjectTree()
        location = _find_agent_in_tree(object_tree)
        inventory = _get_agent_inventory(object_tree) or []

        return GroundTruthState(
            step_index=0,
            location=location,
            inventory=inventory,
            object_tree=object_tree,
            valid_actions=self._env.get_possible_actions(),
            valid_objects=self._env.get_possible_objects(),
            action_executed="",
            action_accepted=True,
            observation_after_action=obs,
        )

    def step(self, action: str, step_index: int = 0) -> GroundTruthState:
        """
        Execute an action and return the resulting ground truth state.

        Args:
            action: The action string to execute
            step_index: Current step index (for tracking)

        Returns:
            GroundTruthState with ground truth information after the action
        """
        self._init_env()

        # Execute the action
        obs, reward, done, info = self._env.step(action)

        # Check if action was accepted
        action_accepted = "no known action matches" not in obs.lower()

        # Get ground truth state
        object_tree = self._env.getObjectTree()
        location = _find_agent_in_tree(object_tree)
        inventory = _get_agent_inventory(object_tree) or []

        return GroundTruthState(
            step_index=step_index,
            location=location,
            inventory=inventory,
            object_tree=object_tree,
            valid_actions=self._env.get_possible_actions(),
            valid_objects=self._env.get_possible_objects(),
            action_executed=action,
            action_accepted=action_accepted,
            observation_after_action=obs,
        )

    def replay_actions(self, actions: List[str]) -> List[GroundTruthState]:
        """
        Replay a sequence of actions and return ground truth states for each step.

        Args:
            actions: List of action strings to replay

        Returns:
            List of GroundTruthState objects, one for each step (including initial state)
        """
        states = []

        # Get initial state
        initial_state = self.reset()
        states.append(initial_state)

        # Replay each action
        for i, action in enumerate(actions):
            state = self.step(action, step_index=i + 1)
            states.append(state)

        return states

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._initialized = False


def extract_task_info_from_description(task_description: str) -> Optional[str]:
    """
    Try to extract task name from task description.

    SciWorld task descriptions typically start with "Your task is to..."
    This function attempts to map the description to a task name.

    Note: This is a heuristic and may not always work perfectly.
    For reliable replay, task_name and task_variation should be stored
    in the samples.jsonl during rollout.
    """
    task_description = task_description.lower()

    # Map common task descriptions to task names
    task_mappings = {
        "boil": ["boil", "bring water to a boil", "heat water"],
        "melt": ["melt", "melting"],
        "freeze": ["freeze", "freezing", "cool water"],
        "change state": ["change the state of matter"],
        "friction": ["friction", "inclined plane"],
        "conductivity": ["conduct", "conductivity"],
        "grow plant": ["grow a plant", "grow plant"],
        "chemistry": ["chemical", "mix", "react"],
    }

    for task_name, keywords in task_mappings.items():
        for keyword in keywords:
            if keyword in task_description:
                return task_name

    return None


def enrich_sample_with_ground_truth(
    sample: Any,  # EvalSample from eval_sciworld_samples.py
    replayer: TrajectoryReplayer,
    step_states: List[GroundTruthState],
) -> Dict[str, Any]:
    """
    Enrich an evaluation sample with ground truth information.

    Args:
        sample: An EvalSample object
        replayer: Active TrajectoryReplayer instance
        step_states: List of ground truth states from replay

    Returns:
        Dictionary with ground truth enrichment information
    """
    step_idx = sample.step_index

    if step_idx < len(step_states):
        gt_state = step_states[step_idx]
        return {
            "ground_truth_location": gt_state.location,
            "ground_truth_inventory": gt_state.inventory,
            "ground_truth_action_accepted": gt_state.action_accepted,
            "ground_truth_valid_objects": gt_state.valid_objects[:50],  # Limit size
        }

    return {
        "ground_truth_location": None,
        "ground_truth_inventory": [],
        "ground_truth_action_accepted": None,
        "ground_truth_valid_objects": [],
    }


# =============================================================================
# Task Name Registry for Trajectory Replay
# =============================================================================

# Complete mapping of task indices to task names
# Based on ScienceWorld's task list
TASK_INDEX_TO_NAME = {
    0: "boil",
    1: "melt",
    2: "freeze",
    3: "change-the-state-of-matter-of",
    4: "use-thermometer",
    5: "measure-melting-point",
    6: "measure-boiling-point",
    7: "understand-thermometer-calibration",
    8: "understand-thermometer-calibration-2",
    9: "measure-liquid-temperature",
    10: "identify-life-stages",
    11: "identify-life-stages-nonliving",
    12: "identify-life-stages-2",
    13: "identify-life-stages-2-nonliving",
    14: "grow-plant",
    15: "grow-fruit",
    16: "chemistry-mix",
    17: "chemistry-mix-paint",
    18: "lifespan",
    19: "lifespan-longest-lived",
    20: "lifespan-shortest-lived",
    21: "test-conductivity",
    22: "test-conductivity-of-unknown",
    23: "test-friction",
    24: "test-friction-unknown",
    25: "use-magnet",
    26: "find-living-thing",
    27: "find-non-living-thing",
    28: "find-plant",
    29: "find-animal",
}


def get_task_name_from_index(task_index: int) -> Optional[str]:
    """Get task name from task index."""
    return TASK_INDEX_TO_NAME.get(task_index)


# =============================================================================
# Integration Example
# =============================================================================

if __name__ == "__main__":
    # Example: Replay a simple trajectory
    print("=== Trajectory Replay Example ===\n")

    try:
        replayer = TrajectoryReplayer(
            task_name="boil",
            task_variation=0,
            simplification_str="easy"
        )

        actions = [
            "look around",
            "go to kitchen",
            "look around",
            "pick up thermometer",
        ]

        print(f"Replaying {len(actions)} actions...\n")
        states = replayer.replay_actions(actions)

        for i, state in enumerate(states):
            print(f"Step {state.step_index}:")
            print(f"  Location: {state.location}")
            print(f"  Inventory: {state.inventory}")
            print(f"  Action: {state.action_executed or '(initial)'}")
            print(f"  Accepted: {state.action_accepted}")
            print()

        replayer.close()
        print("Done!")

    except ImportError as e:
        print(f"ScienceWorld not installed: {e}")
