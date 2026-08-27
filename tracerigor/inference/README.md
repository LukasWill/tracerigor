# Inference clients

This package contains provider adapters used by rollout and evaluation code.
Supported adapters include local vLLM/OpenAI-compatible endpoints and optional
hosted providers.

Install the SDK required by the selected provider and supply credentials only at
runtime. Common variables are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, and `TOGETHER_API_KEY`. Configuration dataclasses default to
`None` and must not contain real keys in source control.

Environment and judge services can be checked with:

```bash
ENV_URL=http://127.0.0.1:8000/health \
JUDGE_URL=http://127.0.0.1:8001/health \
bash scripts/check_service_health.sh
```
