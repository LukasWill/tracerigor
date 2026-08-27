#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-qwen35-judge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VLLM_INSTALL_CHANNEL="${VLLM_INSTALL_CHANNEL:-stable}"
VLLM_VERSION="${VLLM_VERSION:-0.18.0}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly}"

find_conda_bin() {
  if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    printf '%s\n' "$CONDA_EXE"
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  return 1
}

CONDA_BIN="$(find_conda_bin)" || {
  echo "ERROR: conda is required to create the judge serving environment"
  exit 1
}

run_in_env() {
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"
}

echo "Creating judge env '$ENV_NAME' with Python $PYTHON_VERSION"
"$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"

echo "Installing uv into '$ENV_NAME' so vLLM can resolve prebuilt wheels"
run_in_env python -m pip install -U pip uv

ENV_PYTHON="$(run_in_env python -c 'import sys; print(sys.executable)')"

install_stable_vllm() {
  echo "Installing stable vLLM==$VLLM_VERSION with prebuilt wheels only"
  if ! run_in_env python -m uv pip install \
    --python "$ENV_PYTHON" \
    --torch-backend=auto \
    --only-binary vllm \
    "vllm==$VLLM_VERSION"; then
    cat <<EOF
ERROR: no compatible prebuilt wheel was found for vllm==$VLLM_VERSION on this platform.
This script intentionally refuses to build vLLM from source because that requires
a full CUDA toolkit and CUDA_HOME.

Next options:
  1. Retry with nightly wheels:
   VLLM_INSTALL_CHANNEL=nightly $0 $ENV_NAME
  2. If your cluster is too old for the published wheels, use a containerized judge server.
EOF
    exit 1
  fi

  echo "Installing the pinned stable helper packages for Qwen3.5 serving"
  run_in_env python -m pip install \
    "transformers==5.3.0" \
    "tokenizers==0.22.2" \
    "openai==2.29.0" \
    "huggingface_hub>=0.34.0" \
    "hf-transfer>=0.1.9"
}

install_nightly_vllm() {
  echo "Installing nightly vLLM from $VLLM_NIGHTLY_INDEX"
  run_in_env python -m uv pip install \
    --python "$ENV_PYTHON" \
    -U vllm \
    --torch-backend=auto \
    --extra-index-url "$VLLM_NIGHTLY_INDEX"

  echo "Installing judge client helper packages"
  run_in_env python -m pip install \
    "openai==2.29.0" \
    "huggingface_hub>=0.34.0" \
    "hf-transfer>=0.1.9"
}

case "$VLLM_INSTALL_CHANNEL" in
  stable)
    install_stable_vllm
    ;;
  nightly)
    install_nightly_vllm
    ;;
  *)
    echo "ERROR: unsupported VLLM_INSTALL_CHANNEL=$VLLM_INSTALL_CHANNEL (expected stable or nightly)"
    exit 1
    ;;
esac

cat <<EOF

Judge env ready: $ENV_NAME

Install channel: $VLLM_INSTALL_CHANNEL

Stable pinned versions for the recommended stable path:
  vllm==$VLLM_VERSION
  transformers==5.3.0
  tokenizers==0.22.2
  openai==2.29.0

If you need newer model support, recreate or upgrade only this env with nightly wheels:

  VLLM_INSTALL_CHANNEL=nightly $0 $ENV_NAME

EOF
