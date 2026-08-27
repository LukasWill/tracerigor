# Create an environment service

Services batch environment instances behind the interface used by the rollout
manager. A service integration pairs a `BaseService` implementation with a
configuration dataclass and the corresponding environment classes.

Keep transport and lifecycle behavior generic:

- accept environment configuration rather than local file paths;
- make host and port runtime settings;
- return stable, serializable observations and metrics;
- close environment resources explicitly; and
- avoid importing optional SDKs at package-import time.

Register the service class alongside the environment class in
`tracerigor/env/__init__.py`. Use `scripts/check_service_health.sh` for a generic
HTTP readiness check by setting `ENV_URL` and, optionally, `JUDGE_URL`.
