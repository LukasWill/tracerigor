# General configuration

TraceRigor uses Python dataclasses for environment/service configuration and
Hydra overrides for training. The active trainer configuration is
`tracerigor/trainer/config/ppo_trainer.yaml`.

## Environment datasets

Each top-level YAML entry specifies an environment name, constructor overrides,
and train/evaluation sizes:

```yaml
blackjack:
  env_name: blackjack
  env_config:
    render_mode: text
  train_size: 16
  test_size: 8
```

Use `tracerigor envs --verbose` before generation to identify missing optional
dependencies.

## Runtime overrides

Training settings are passed as Hydra `key=value` arguments. Prefer a checked-in
generic config plus command-line overrides for paths, model identifiers, GPU
counts, and short smoke-run limits. The public example in
`examples/train/run.sh` demonstrates the supported pattern.

## Judge configuration

The default judge schema is in `tracerigor/judge/default_config.yaml`. Configure
provider URL, model, gating, rubric selection, and process-reward weight there
or via Hydra overrides. Supply real API credentials only through environment
variables or an untracked secrets manager.
