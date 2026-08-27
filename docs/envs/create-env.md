# Create an environment

An environment integration normally contains `env.py`, `env_config.py`,
`service.py`, `service_config.py`, prompts, and package exports. The environment
class implements the `BaseEnv` contract and returns observations shaped like:

```python
{"obs_str": "...", "multi_modal_data": {"<image>": [image]}}
```

Text-only environments may omit `multi_modal_data`. A step returns observation,
reward, termination state, and an info dictionary containing turn and
trajectory metrics.

Register the integration in `_ENV_SPECS` inside `tracerigor/env/__init__.py`.
Imports must tolerate absent optional dependencies: the registry records an
integration-specific error and `tracerigor envs --verbose` exposes it to users.

Use `tracerigor/env/blackjack/` as the compact reference implementation. Add a
small text-mode dataset config under `examples/data/` and test both registration
and one seeded reset where dependencies permit.
