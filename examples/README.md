# Public examples

These examples are intentionally small and contain no private model, dataset,
cluster, or experiment identifiers.

## Analyze trajectory reliability

```bash
tracerigor analyze examples/analysis/sample_records.jsonl --group-by training_step
```

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
