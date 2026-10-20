# Public examples

These examples are intentionally small and contain no private model, dataset,
cluster, or experiment identifiers.

## Judge agent turns

`judge/sample_turns.jsonl` holds six Sokoban turns. The observations and
ground-truth spatial state come from `SokobanEnv(render_mode="text")` with
seed 7; the agent responses are hand-written to illustrate distinct failure
modes and are not model outputs. Regenerate it with
`python judge/make_sample_turns.py` (requires the `envs` extra).

```bash
tracerigor judge examples/judge/sample_turns.jsonl --dry-run --show-prompt
tracerigor judge examples/judge/sample_turns.jsonl \
  --base-url http://127.0.0.1:8001/v1 --model <served-model-name> -o judged.jsonl
tracerigor analyze judged.jsonl
```

## Analyze trajectory reliability

```bash
tracerigor analyze examples/analysis/sample_records.jsonl --group-by training_step
```

`analysis/aggregate_step_controlled_partials/` contains per-rubric aggregate
correlation statistics exported from earlier runs. The code that produced them
is not included in this release, so treat them as schema examples rather than
reproducible results.

## Generate environment datasets

Install the data and environment extras, then run:

```bash
pip install -e '.[data,envs]'
tracerigor data examples/data/blackjack.yaml \
  --train-path data/blackjack/train.parquet \
  --test-path data/blackjack/test.parquet
```

## Training

Training requires a compatible external VERL checkout and the `train` extra.
The example defaults to two steps but still requires model weights and a GPU:

```bash
export MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
export TRAIN_FILE=data/blackjack/train.parquet
export VAL_FILE=data/blackjack/test.parquet
bash examples/train/run.sh
```

The command is a smoke configuration, not a performance-reproduction recipe.
