# Scripts

This directory contains generic public utilities only:

- `install.sh` installs the package with selected extras;
- `check_service_health.sh` waits for configured service health URLs;
- `launch_judge_server.sh` launches a local OpenAI-compatible vLLM judge; and
- `setup_qwen35_judge_env.sh` creates an isolated optional vLLM environment.

Run-specific sweeps and cluster launchers are intentionally not part of the
public tree. The former benchmark wrappers were also excluded because their
referenced Python benchmark module was not present. Prefer environment
variables and Hydra overrides to copying a launcher per experiment.
