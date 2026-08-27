from tracerigor.env.base.base_env import BaseEnv
import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from tracerigor.env.sokoban.utils import generate_room
from typing import Dict
from tracerigor.env.utils.env_utils import NoLoggerWarnings, set_seed
from tracerigor.env.utils.context_utils import convert_numpy_to_PIL
import numpy as np
from tracerigor.env.utils.parse_utils import PARSE_FUNC_MAP
from tracerigor.env.sokoban.prompt import (
    system_prompt,
    init_observation_template,
    action_template,
    format_prompt
)
from tracerigor.env.sokoban.env_config import SokobanEnvConfig
from tracerigor.env.utils.state_reward_text_utils import env_state_reward_wrapper
from tracerigor.env.sokoban.utils import sokoban_state_to_sentences, convert_sokoban_state_to_relative_list
from tracerigor.verifier.verifier.common.verifier_memory import VerifierMemory
from tracerigor.verifier.verifier.common.obs_utils import *
from tracerigor.env.utils.verifier_probe_wrapper import llm_verifier_probe_wrapper
from tracerigor.env.sokoban.violation_tracker import SokobanViolationTracker
import hashlib
import numpy as np

def stable_hash_seed(seed: int | str, mod: int = 2**32) -> int:
    s = str(seed).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "little") % mod

class SokobanEnv(BaseEnv):
    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

    ACTION_LOOKUP = {
        "Up":1,
        "Down":2,
        "Left":3,
        "Right":4,
    }

    def __init__(self, config:SokobanEnvConfig):
        BaseEnv.__init__(self)
        self.config=config
        self.env=GymSokobanEnv(
            dim_room=self.config.get('dim_room', (6, 6)),
            max_steps=self.config.get('max_steps', 100),
            num_boxes=self.config.get('num_boxes', 3),
        )

        # Get the appropriate format prompt function
        self.format_prompt_func = format_prompt[self.config.prompt_format]

        # Call the function with add_example=True for system prompt

        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]

        self._verifier_mem = VerifierMemory(max_history=getattr(self.config.verifier, 'max_history_len', 5))
        self._last_obs = None  # track last observation dict returned by _render()

        # Violation tracker for early termination of problematic trajectories
        if self.config.enable_violation_termination:
            self.violation_tracker = SokobanViolationTracker(
                format_threshold=self.config.format_violation_threshold,
                invalid_action_threshold=self.config.invalid_action_threshold,
                repetition_threshold=self.config.repetition_threshold,
            )
        else:
            self.violation_tracker = None

    def reset(self, seed=None):
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.env.room_fixed, self.env.room_state, self.env.box_mapping, action_sequence = generate_room(
                        dim=self.env.dim_room,
                        num_steps=self.env.num_gen_steps,
                        num_boxes=self.env.num_boxes,
                        search_depth=self.config.get('search_depth', 100),
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                # next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                next_seed = stable_hash_seed(seed) if seed is not None else None
                return self.reset(next_seed)

            self.env.player_position = np.argwhere(self.env.room_state == 5)[0]
            self.env.num_env_steps = self.env.reward_last = self.env.boxes_on_target = 0
        self.total_reward = 0
        obs = self._render(init_obs=True)
        info = {"state_id": self._get_env_state_id()}

        # reset per-episode verifier memory and seed with initial observation
        self._verifier_mem.reset()
        # obs_text = extract_text_from_obs(obs)  # obs doesn't contain raw text description of the current scene.
        obs_text = "; ".join(sokoban_state_to_sentences(self.get_env_state(to_relative=False)))
        obs_images = extract_images_from_obs(obs, placeholder=self.config.get("image_placeholder", "<image>"))
        self._verifier_mem.add_step(
            observation_text=obs_text,
            observation_images=obs_images,
            reasoning_tokens=None,
            action_tokens=None,
        )
        self._last_obs = {**obs, "obs_str": obs_text, "state_id": self._get_env_state_id()}

        # Reset violation tracker
        if self.violation_tracker is not None:
            self.violation_tracker.reset()

        return obs, info

    @llm_verifier_probe_wrapper
    @env_state_reward_wrapper
    def step(self, action_str: str):  # action_str is the raw LLM response (completion)
        rst=self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 3)
        )
        #print("rst:", rst)
        action_list=rst['actions']
        prev_player_position = self.env.player_position

        metrics={
            "turn_metrics":{
                "action_is_valid": len(action_list) != 0,
                "action_is_effective": False,},
            "traj_metrics": {
                "success": False,
            }
        }

        self.reward=0
        self.valid_actions=[]
        done=False
        info={}
        info.update(rst)

        # Track success index & last step reward for optional discounting
        success_index = None
        last_step_reward_value = 0.0

        ## -- REDEFINE invalidation: any no-op subaction invalidates the turn
        noop_invalidation = self.config.get('noop_invalidation', True)
        for action in action_list:
            if action not in self.ACTION_LOOKUP:
                metrics['turn_metrics']['action_is_valid'] = False
                break

            action_int = self.ACTION_LOOKUP[action]

            # check movement for this sub-action
            pre_pos = self.env.player_position.copy()
            _, step_reward, _, _ = self.env.step(action_int)
            last_step_reward_value = step_reward

            moved_now = not np.array_equal(pre_pos, self.env.player_position)
            if not moved_now:
                self.reward += step_reward
                if noop_invalidation:
                    # Strict: any no-op in the sequence → whole turn invalid
                    metrics['turn_metrics']['action_is_valid'] = False
                # In both modes, stop executing remaining sub-actions
                break
            # otherwise, we keep it
            self.valid_actions.append(action)
            self.reward += step_reward

            done = self._success()
            if done:
                metrics['traj_metrics']['success'] = True
                if success_index is None:
                    success_index = len(self.valid_actions)  # 1-indexed position in sequence
                break

        # Net effectiveness: did the agent end up in a new position?
        metrics['turn_metrics']['action_is_effective'] = not np.array_equal(
            prev_player_position, self.env.player_position
        )

        # # (Optional) action-index-based discount of terminal reward (very light-touch)
        # if done and self.config.get("success_discount_by_action_index", False):
        #     N = max(1, len(action_list))
        #     k = min(success_index or N, N)  # if missing, treat as last
        #     factor = (N - k + 1) / N  # k=1 => 1.0; k=2 of 3 => 2/3; k=3 of 3 => 1/3
        #     # Adjust only the final-step contribution we just added
        #     self.reward += (factor - 1.0) * last_step_reward_value

        # (Optional) action-index-based discount of terminal reward (redundancy penalty)
        if done and self.config.get("success_discount_by_action_index", False):
            N = max(1, len(action_list))
            k = min(success_index or N, N)  # 1-based index of the action that achieved success
            factor = k / N                  # k=1 of 3 => 1/3, k=2 of 3 => 2/3, k=3 of 3 => 1
            self.reward += (factor - 1.0) * last_step_reward_value  # rescale ONLY the final contribution

        # --- after executing actions & computing metrics ---
        if metrics['turn_metrics']['action_is_valid'] and rst["format_correct"]:
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
            ## optional tiny extra when effective (default 0.0 to keep current behavior)
            # eff_bonus = getattr(self.config, "format_effective_bonus", 0.0)
            # if metrics['turn_metrics']['action_is_effective'] and eff_bonus > 0:
            #     self.reward += eff_bonus
        else:
            info["is_format_rewarded"] = False

        info["metrics"] = metrics
        self.total_reward += self.reward
        info["state_id"] = self._get_env_state_id()
        ## -- END

        ## original
        # for action in action_list:
        #     if action in self.ACTION_LOOKUP:
        #         action_int=self.ACTION_LOOKUP[action]
        #         _,step_reward, _, _=self.env.step(action_int)
        #         done=self._success()
        #         self.reward+=step_reward
        #         self.valid_actions.append(action)
        #         if done:
        #             metrics['traj_metrics']['success'] = True
        #             break
        #     else:
        #         metrics['turn_metrics']['action_is_valid'] = False
        #         break
        # if metrics['turn_metrics']['action_is_valid'] and rst["format_correct"]:
        #     self.reward += self.config.format_reward
        #     info["is_format_rewarded"] = True
        # else:
        #     info["is_format_rewarded"] = False  # invalid/empty/format-bad -> is_format_rewarded=False
        # info["metrics"] = metrics
        # metrics['turn_metrics']['action_is_effective'] = not np.array_equal(prev_player_position, self.env.player_position)
        # self.total_reward += self.reward
        # info["state_id"] = self._get_env_state_id()
        ## END

        # =================================================================
        # Violation Tracking and Early Termination
        # =================================================================
        if self.violation_tracker is not None and not done:
            action_taken = action_list[0] if action_list else ""
            obs_text_pre = "; ".join(sokoban_state_to_sentences(self.get_env_state(to_relative=False)))

            violation_terminated, termination_reason = self.violation_tracker.record_step(
                format_correct=rst["format_correct"],
                action=action_taken,
                observation=obs_text_pre,
                action_is_valid=metrics['turn_metrics']['action_is_valid'],
                action_is_effective=metrics['turn_metrics']['action_is_effective'],
            )

            if violation_terminated:
                done = True
                self.reward += self.config.violation_penalty
                self.total_reward += self.config.violation_penalty
                metrics["traj_metrics"]["violation_terminated"] = True
                metrics["traj_metrics"]["termination_reason"] = termination_reason.value if termination_reason else None
                info["violation_terminated"] = True
                info["termination_reason"] = termination_reason.value if termination_reason else None

            # Always add violation metrics
            metrics["traj_metrics"]["violation_metrics"] = self.violation_tracker.get_metrics()

        # Update per-episode memory with the *pre-step* observation and the parsed tokens.
        # We captured the pre-step observation as self._last_obs in the wrapper below.
        # Here we only move the "pointer" to the current obs (post-step); the record is added by the wrapper.
        obs = self._render(init_obs=False)
        obs_text = "; ".join(sokoban_state_to_sentences(self.get_env_state(to_relative=False)))
        self._last_obs = {**obs, "obs_str": obs_text, "state_id": self._get_env_state_id()}

        return obs, self.reward, done, info

    def system_prompt(self):
        format_prompt=self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True if not self.config.turn_wise_update else False  # Always true for system prompt
        )
        return system_prompt() + "\n" + format_prompt

    def close(self):
        self.env.close()

    def _render(self, init_obs=True):
        assert self.config.render_mode in ['text', 'vision']
        multi_modal_data = None

        # Get the appropriate format prompt function for action/init templates (with add_example=False)

        format_prompt = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False  # No examples for action and init obs
        )

        if self.config.render_mode == 'vision':
            img_placeholder=self.config.get("image_placeholder", "<image>")
            multi_modal_data={
                img_placeholder: [convert_numpy_to_PIL(self.env.render(mode='rgb_array'))],
                }
            img_str=img_placeholder
        else:
            room_state = np.where((self.env.room_state == 5) & (self.env.room_fixed == 2), 6, self.env.room_state).tolist()
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            img_str = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)

        # concise textual summary (opt-in by config if you want)
        append_state_summary = self.config.get("append_state_summary", False)
        state_summary = "; ".join(sokoban_state_to_sentences(self.get_env_state(to_relative=False))) if append_state_summary else ""

        if init_obs:
            obs_str = init_observation_template(img_str=img_str,
                                                turn_wise_update=self.config.get("turn_wise_update", False),
                                                state_summary=state_summary,
                                                ) + "\n" + format_prompt
        else:
            # Only show the reflect hint when explicitly enabled (e.g. when
            # a verifier is actively injecting textual feedback into the
            # observation).  By default it is suppressed because no code path
            # currently prepends feedback text into the observation.
            obs_str = action_template(
                turn_wise_update=self.config.get("turn_wise_update", False),
                valid_action=self.valid_actions,
                img_str=img_str,
                state_summary=state_summary,
                show_reflect_hint=self.config.get("show_reflect_hint", False),
            ) + "\n" + format_prompt

        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }

    def _success(self):
        return self.env.boxes_on_target == self.env.num_boxes

    def _get_env_state_id(self):
        """
        Generate a unique identifier for the current environment state using numpy arrays.

        Uses the raw room_state and room_fixed arrays directly for maximum efficiency.
        # use for gigpo
        Returns:
            str: A unique identifier (SHA-256 hash) for the current state
        """
        # Get the raw numpy arrays
        room_state = self.env.room_state
        room_fixed = self.env.room_fixed

        # Convert to bytes for hashing
        state_bytes = room_state.tobytes()
        fixed_bytes = room_fixed.tobytes()

        # Combine both arrays' bytes
        combined_bytes = state_bytes + fixed_bytes

        # Generate SHA-256 hash
        state_hash = hashlib.sha256(combined_bytes).hexdigest()

        return state_hash

    def get_env_state(self, to_relative: bool = True):
        """
        Get the basic positional state of the Sokoban environment.

        Args:
            to_relative (bool): If True, return relative-list format; otherwise return raw dict.

        Returns:
            Dict: Contains player position, box positions, target positions, and wall positions
                as simple coordinate tuples with standard Python types for JSON serialization.
        """
        # Extract positions from room_state and room_fixed
        room_state = self.env.room_state

        # Find player position (codes 5: player on floor, 6: player on target)
        player_pos = tuple(map(int, np.argwhere(np.logical_or(room_state == 5, room_state == 6))[0]))

        # Find box positions (codes 3: box on target, 4: box not on target)
        box_positions = [tuple(map(int, pos)) for pos in np.argwhere(np.logical_or(room_state == 3, room_state == 4))]

        # Find target positions (codes 2: empty target, 3: box on target, 6: player on target)
        # For targets, we need to check both room_state and room_fixed
        target_positions = [tuple(map(int, pos)) for pos in np.argwhere(self.env.room_fixed == 2)]

        # Find wall positions (code 0: wall)
        wall_positions = [tuple(map(int, pos)) for pos in np.argwhere(room_state == 0)]

        # Convert grid size dimensions to standard Python integers
        grid_size = tuple(map(int, room_state.shape))

        state_dict = {
            "player_position": player_pos,
            "box_positions": box_positions,
            "target_positions": target_positions,
            "wall_positions": wall_positions,
            "grid_size": grid_size
        }
        return convert_sokoban_state_to_relative_list(state_dict) if to_relative else state_dict

if __name__ == "__main__":
    kwargs = {
        'render_mode': 'vision',
    }
    config = SokobanEnvConfig(**kwargs)
    env = SokobanEnv(config)
    print(env.system_prompt())
    obs, info = env.reset()
    print(obs["obs_str"], info["state_id"])
    i=0
    import os
    if config.render_mode == 'vision':
        os.makedirs("./test_sokoban", exist_ok=True)
        img = obs["multi_modal_data"][config.image_placeholder][0]
        # img.save(f"./test_sokoban/sokoban_{i}.png")
    while True:
        i += 1
        action = input("Enter action (Left, Down, Right, Up): ")
        action = f"<think>Let me try this direction.</think><action>{action}</action>"
        obs, reward, done, info = env.step(action)
        print(obs["obs_str"], info["state_id"])
        if config.render_mode == 'vision':
            # save the image
            img = obs["multi_modal_data"][config.image_placeholder][0]
            # img.save(f"./test_sokoban/sokoban_{i}.png")
        if done:
            break

    print(f"Total reward: {env.compute_reward()}")
    print(info)
    env.close()