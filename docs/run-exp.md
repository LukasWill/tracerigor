# Installation and workflows

## Base installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
tracerigor --help
```

The base installation includes the record-analysis command. Optional extras are
available for `analysis`, `data`, `envs`, `judge`, `train`, and `dev`.

## Hardware expectations

- Record analysis, environment inspection, and the test suite are CPU-safe.
- The bundled Blackjack seed-data example is intended as a lightweight CPU
  workflow; other environments may bring their own simulator requirements.
- A hosted judge needs provider credentials but no local accelerator.
- A self-hosted judge needs enough GPU capacity for the model served by vLLM or
  Ollama.
- Rollout and RL training normally require Linux/CUDA, one or more GPUs, VERL,
  model weights, and running environment services.

There is deliberately no fixed VRAM claim: model size, precision, context
length, batch size, optimizer state, and sharding determine the requirement.
Validate the two-step smoke configuration before scaling a run.

## Analyze records

```bash
tracerigor analyze examples/analysis/sample_records.jsonl \
  --group-by training_step \
  --output /tmp/tracerigor-summary.json
```

## Inspect environments

```bash
python -m pip install -e '.[envs]'
tracerigor envs --verbose
```

An unavailable optional environment is reported individually and does not make
the package registry unusable.

## Generate data

```bash
python -m pip install -e '.[data,envs]'
tracerigor data examples/data/blackjack.yaml \
  --train-path data/blackjack/train.parquet \
  --test-path data/blackjack/test.parquet
```

## Train

The trainer requires a compatible VERL runtime, model weights, environment
services, and GPU capacity. The public launcher checks its required variables
and defaults to two steps:

```bash
export MODEL_PATH=/path/to/a/compatible/model
export TRAIN_FILE=data/blackjack/train.parquet
export VAL_FILE=data/blackjack/test.parquet
bash examples/train/run.sh
```

Do not place API keys, private paths, or experiment-tracking credentials in a
tracked launcher or YAML file.
