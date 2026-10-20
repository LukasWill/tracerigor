# Contributing

Bug reports, environment integrations, judge templates, and documentation
fixes are welcome. Please open an issue before starting a large change.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,judge,envs,data]'
python -m pytest
```

The test suite runs on CPU in a few seconds. Tests that need optional
packages skip when those packages are missing. The judge tests start a local
stub OpenAI-compatible server, so no API key or network access is required.
CI runs on Python 3.10 and 3.11, once with the base package plus the judge
client and once with the environment extras.

## Guidelines

- Keep optional dependencies optional: import simulator or provider SDKs
  inside the integration that needs them, and make sure `tracerigor envs`
  still reports other environments correctly when one is missing.
- Changes to rewards, judge prompts, parsing, or advantage estimation change
  experimental results. Describe the expected effect in the pull request and
  add a regression test that pins the new behaviour.
- Never commit credentials, machine-specific paths, checkpoints, or raw run
  outputs. CI rejects common secret patterns and absolute home-directory paths.
- New environments: follow [docs/envs/create-env.md](docs/envs/create-env.md),
  register the integration in `tracerigor/env/__init__.py`, and add a seeded
  reset/step test.
- New judge templates: register them in `tracerigor/judge/env_templates.py`
  and add a dry-run test that checks the rendered prompt has no unfilled
  placeholders (see `tests/test_judge_batch.py`).
