# The trace judge

`tracerigor.judge` scores individual agent turns on three rubrics:

| Rubric | Fails when… |
| --- | --- |
| `observation_grounding` | the reasoning makes claims about the current state that the observation (or supplied ground truth) contradicts |
| `action_coherence` | the chosen action contradicts the plan stated in the reasoning |
| `temporal_consistency` | the reasoning ignores or misstates what earlier turns revealed (failed actions, changed state, stuck loops) |

## Routing

Every turn is first checked by deterministic Layer-0 heuristics
(`tracerigor/judge/heuristics.py`). Critical checks short-circuit the turn
without an LLM call; informational checks are recorded and passed on.

| Check | Fires when | Effect |
| --- | --- | --- |
| `format_violation` | the env parser failed, no action was parsed, the raw trace has no `<action>`/`<answer>` tag, or a long trace has no `<think>`/`<reflection>` tag | short-circuit |
| `invalid_action` | env feedback contains "No known action", or the action is not in `admissible_actions` | short-circuit |
| `repetition` | same action as the most recent history entry (`history[0]`) with an unchanged observation | informational |
| `empty_trace` | reasoning shorter than `heuristics.min_trace_tokens` words (default 8) | informational |

Remaining turns are rendered with an environment-specific template when one is
registered (`sokoban`, `sciworld`) or a generic universal template otherwise,
and sent to the Layer-1 judge in one batch. Responses are parsed from
structured JSON (`{"observation_grounding": {"yes_no": "YES", "evidence": "…"}, …}`),
with fallbacks for fenced JSON and `<answer>YES|NO</answer>` outputs.

## Command line

```bash
# Heuristics + rendered prompts only; no network, base install.
tracerigor judge turns.jsonl --dry-run --show-prompt -o routed.jsonl

# Self-hosted OpenAI-compatible server (pip install -e '.[judge]').
tracerigor judge turns.jsonl --base-url http://127.0.0.1:8001/v1 --model <name> -o judged.jsonl

# Hosted provider; the key is read from the named environment variable.
tracerigor judge turns.jsonl --base-url https://api.openai.com/v1 --model <name> \
  --api-key-env OPENAI_API_KEY -o judged.jsonl

tracerigor analyze judged.jsonl --group-by training_step
```

`--config` loads a judge YAML in the format of
[`tracerigor/judge/default_config.yaml`](https://github.com/LukasWill/tracerigor/blob/main/tracerigor/judge/default_config.yaml);
`${VAR}` references in provider fields are expanded from the environment.
Command-line flags override the file. The CLI disables the audit queue.

### Input records

One JSON object per turn (JSON array or JSONL). Only `response` is required.

| Field | Meaning |
| --- | --- |
| `id` | unique turn identifier (default: position in the file) |
| `task_name` | selects a registered template, e.g. `sokoban`, `sciworld` (default `generic`) |
| `episode_step` | 1-indexed step within the episode |
| `observation` | observation text the agent saw before acting |
| `response` | the agent's raw output, e.g. `<think>…</think><action>…</action>` |
| `admissible_actions` | list of valid actions, used by the `invalid_action` check |
| `history` | earlier turns, **most recent first**: `{"observation_text", "action_tokens", "reward"}` |
| `ground_truth_state` | simulator state for the prompt, e.g. `{"state_sentences": [...]}` (Sokoban) or `{"location", "inventory"}` (ScienceWorld) |
| `action_feedback`, `task_description`, `format_correct` | optional environment metadata |

Any other field (for example `training_step` or `checkpoint`) is copied to the
output unchanged, so results can be grouped by it in `tracerigor analyze`.
[`examples/judge/sample_turns.jsonl`](https://github.com/LukasWill/tracerigor/blob/main/examples/judge/sample_turns.jsonl)
is a complete example.

### Output records

Each output line holds the pass-through fields, `id`, `task_name`,
`episode_step`, `chosen_action`, and a `response` object: per-rubric `label`,
`score` (0, 0.5, or 1), `confidence`, and `evidence`, plus `short_circuited`,
`short_circuit_reason`, `query_success`, `parse_success`, `process_reward`,
the raw judge text, and the Layer-0 `heuristics`. Dry runs replace
`response` with `route`, `short_circuit_reason`, `heuristics`, and optionally
`prompt`.

## Python API

```python
from tracerigor.judge import JudgeConfig, register_all_templates
from tracerigor.judge.router import JudgeRouter
from tracerigor.judge.schema import TurnJudgePacket

register_all_templates()
cfg = JudgeConfig.from_yaml()             # packaged defaults
cfg.provider.base_url = "http://127.0.0.1:8001/v1"
cfg.provider.model = "<served-model-name>"

packet = TurnJudgePacket(env_id="ep0_t1", episode_step=1, task_name="sokoban")
packet.current_observation_text = "..."
packet.raw_trace = "<think>...</think><action>Right</action>"
packet.reasoning_tokens = "<think>...</think>"
packet.action_tokens = "<action>Right</action>"
packet.chosen_action = "Right"
packet.admissible_actions = ["Up", "Down", "Left", "Right"]

router = JudgeRouter(cfg)
verdicts = router.score_batch([packet])   # {env_id: JudgeResponse}
router.close()
```

`tracerigor.judge.batch.judge_turns` wraps this for lists of dictionaries in
the input format above. `JudgeIntegration` is the step-batch interface for a
rollout loop. It is not wired into the rollout manager in this release; see
[Known issues](known-issues.md).

## Process reward mapping

`compute_process_reward` (`tracerigor/judge/reward.py`) maps a verdict to a
bounded scalar:

1. Short-circuited turns receive `heuristics.short_circuit_penalty`
   (−0.20 by default), clamped to `reward.reward_range`, and skip the steps
   below.
2. Otherwise `r_raw` is the weighted mean of the available rubric scores
   (default weights 0.35 / 0.40 / 0.25), minus
   `insufficient_evidence_penalty` if the judge flagged insufficient evidence.
3. With annealing enabled, a scale `w_t` interpolates linearly from
   `anneal_hi_scale` to `anneal_lo_scale` over
   `[anneal_start_step, anneal_end_step]`; otherwise `w_t = 1`.
4. By default shaping is asymmetric, `r = w_t · max(0, 2·r_raw − 1)`: turns
   with `r_raw > 0.5` earn a bonus and the rest earn 0, so the default never
   produces a negative reward for a judged turn. `symmetric_shaping: true`
   uses `r = w_t · (2·r_raw − 1)`, which penalises `r_raw < 0.5`.
   `gate_mul` switches to multiplicative/additive gating around
   `big_threshold`. `alpha_small` is accepted for parity with the
   training-path verifier config but is not used by the judge.
5. `r` is clamped to `reward_range` (default `[-0.25, 0.25]`).
   `JudgeIntegration` adds `lambda_proc · r` (default 0.10) to the
   environment reward.

`JudgeIntegration` passes rollout history to the judge most recent first,
the same order as the `history` input field.

## Self-hosted judge

[`scripts/launch_judge_server.sh`](https://github.com/LukasWill/tracerigor/blob/main/scripts/launch_judge_server.sh)
starts a vLLM OpenAI-compatible server, and
[`scripts/setup_qwen35_judge_env.sh`](https://github.com/LukasWill/tracerigor/blob/main/scripts/setup_qwen35_judge_env.sh)
creates an isolated environment for it. A local server accepts the
placeholder API key `EMPTY`.
