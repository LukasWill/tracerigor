"""Regenerate ``sample_turns.jsonl`` from a seeded Sokoban environment.

Observations and ground-truth state sentences come from the real environment
(``SokobanEnv(render_mode="text")``, seed 7). The agent responses are
hand-written illustrations of distinct trace failure modes; they are not
model outputs.

    pip install -e '.[envs]'
    python examples/judge/make_sample_turns.py
"""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from tracerigor.env import get_environment  # noqa: E402
from tracerigor.env.sokoban.utils import sokoban_state_to_sentences  # noqa: E402

ACTIONS = ["Up", "Down", "Left", "Right"]
OUT = Path(__file__).with_name("sample_turns.jsonl")


def state_sentences(env):
    return sokoban_state_to_sentences(env.get_env_state(to_relative=False))


def main() -> None:
    spec = get_environment("sokoban")
    env = spec["env_cls"](spec["config_cls"](render_mode="text"))
    obs, _ = env.reset(seed=7)
    step1 = {"observation": obs["obs_str"], "ground_truth_state": {"state_sentences": state_sentences(env)}}

    first_response = (
        "<think>The box is directly to my right and the target is one row below and two "
        "columns to the right of me. Pushing the box right puts it in the target's column, "
        "then I can push it down.</think><action>Right</action>"
    )
    obs2, reward, _, _ = env.step(first_response)
    step2 = {
        "observation": obs2["obs_str"],
        "ground_truth_state": {"state_sentences": state_sentences(env)},
        "history": [
            {"observation_text": step1["observation"], "action_tokens": "<action>Right</action>", "reward": reward}
        ],
    }
    env.close()

    turns = [
        dict(id="grounded", scenario="consistent reasoning and action", episode_step=1, **step1,
             response=first_response),
        dict(id="hallucinated_state", scenario="reasoning misdescribes the observation", episode_step=1, **step1,
             response="<think>The box is to my left and the target is above me, so I should go up "
                      "and around the box first.</think><action>Up</action>"),
        dict(id="incoherent_action", scenario="action contradicts the stated plan", episode_step=1, **step1,
             response="<think>The box is directly to my right, so I will push it right toward the "
                      "target column.</think><action>Left</action>"),
        dict(id="invalid_action", scenario="action outside the action space", episode_step=1, **step1,
             response="<think>The box is to my right and I want to reach the target quickly, so I "
                      "will jump over it.</think><action>Jump</action>"),
        dict(id="format_violation", scenario="no parseable action", episode_step=1, **step1,
             response="I think moving right is best here."),
        dict(id="stale_belief", scenario="reasoning ignores the previous turn", episode_step=2, **step2,
             response="<think>Last turn I pushed the box onto the target, so the box is already "
                      "solved and I only need to walk down.</think><action>Down</action>"),
    ]
    with OUT.open("w", encoding="utf-8") as handle:
        for turn in turns:
            turn = {"task_name": "sokoban", "admissible_actions": ACTIONS, **turn}
            handle.write(json.dumps(turn) + "\n")
    print(f"wrote {len(turns)} turns to {OUT}")


if __name__ == "__main__":
    main()
