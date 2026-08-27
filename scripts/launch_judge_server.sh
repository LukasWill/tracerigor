#!/usr/bin/env bash
# ============================================================================
# Co-launch vLLM Judge Server alongside TraceRigor training
#
# This script fragment can be sourced or integrated into your run_slurm.sh.
# It launches a vLLM server for the judge LLM (e.g., Qwen3.5-9B) on a
# dedicated GPU, separate from the training GPUs.
#
# Usage:
#   1) Standalone:  bash scripts/launch_judge_server.sh
#   2) In run_slurm.sh:  source scripts/launch_judge_server.sh
#   3) Copy the relevant sections into your training script
#
# Requirements:
#   - vLLM installed (pip install vllm)
#   - At least one spare GPU for the judge server
# ============================================================================
set -euo pipefail

# -----------------------
# Configuration
# -----------------------
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-judge}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-9B}"
JUDGE_GPU="${JUDGE_GPU:-0}"                # GPU index for the judge server
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"        # tensor parallel size
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-32768}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-64}"  # max concurrent requests
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.85}"
JUDGE_DTYPE="${JUDGE_DTYPE:-auto}"

# Log directory (reuse LOG_DIR if set by run_slurm.sh)
LOG_DIR="${LOG_DIR:-.}"
JUDGE_LOG="$LOG_DIR/judge_server_${SLURM_JOB_ID:-local}.log"

find_conda_bin() {
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        printf '%s\n' "$CONDA_EXE"
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

CONDA_BIN="$(find_conda_bin)" || {
    echo "ERROR: could not find a conda executable for judge env launch"
    exit 1
}

echo "============================================"
echo "  Judge vLLM Server Configuration"
echo "============================================"
echo "  Conda env:     $JUDGE_CONDA_ENV"
echo "  Model:         $JUDGE_MODEL"
echo "  Port:          $JUDGE_PORT"
echo "  GPU(s):        $JUDGE_GPU"
echo "  TP size:       $JUDGE_TP_SIZE"
echo "  Max model len: $JUDGE_MAX_MODEL_LEN"
echo "  Max seqs:      $JUDGE_MAX_NUM_SEQS"
echo "  GPU mem util:  $JUDGE_GPU_MEMORY_UTILIZATION"
echo "  Log:           $JUDGE_LOG"
echo "============================================"

# -----------------------
# Launch judge vLLM server
# -----------------------
# Pin to specific GPU(s).  For multi-GPU training, pick GPUs that
# are NOT used by the actor/critic/rollout.
# Example: 4-GPU node, GPUs 1-3 for training, GPU 0 for judge.
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" nohup "$CONDA_BIN" run --no-capture-output -n "$JUDGE_CONDA_ENV" \
    vllm serve "$JUDGE_MODEL" \
    --host 127.0.0.1 \
    --port "$JUDGE_PORT" \
    --tensor-parallel-size "$JUDGE_TP_SIZE" \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" \
    --max-num-seqs "$JUDGE_MAX_NUM_SEQS" \
    --reasoning-parser qwen3 \
    --language-model-only \
    --disable-log-requests \
    --trust-remote-code \
    --dtype "$JUDGE_DTYPE" \
    --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
    > "$JUDGE_LOG" 2>&1 &

JUDGE_PID=$!
echo "Judge server started (pid=$JUDGE_PID) on port $JUDGE_PORT"

# -----------------------
# Cleanup trap
# -----------------------
# If sourced into run_slurm.sh, append to the existing cleanup function.
# If standalone, set our own trap.
_cleanup_judge() {
    echo "Stopping judge server (PID: $JUDGE_PID)..."
    if kill -0 "$JUDGE_PID" 2>/dev/null; then
        kill "$JUDGE_PID"
        wait "$JUDGE_PID" 2>/dev/null || true
    fi
}

# Prefer delegating cleanup registration to the parent script when available.
if type register_cleanup_pid >/dev/null 2>&1; then
    register_cleanup_pid "$JUDGE_PID"
else
    trap _cleanup_judge EXIT INT TERM
fi

# -----------------------
# Wait for readiness
# -----------------------
echo "Waiting for judge server to be ready on port $JUDGE_PORT..."
MAX_WAIT=120
WAITED=0
while ! curl -sf "http://localhost:$JUDGE_PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Judge server did not start within ${MAX_WAIT}s. Check $JUDGE_LOG"
        exit 1
    fi
done
echo "Judge server ready after ${WAITED}s."

# -----------------------
# Integration with training
# -----------------------
# When launching training, pass the judge config:
#
#   python3 -m tracerigor.trainer.main_ppo \
#       ... \
#       +judge.enabled=true \
#       +judge.provider.base_url="http://localhost:$JUDGE_PORT/v1" \
#       +judge.provider.model="$JUDGE_MODEL" \
#       +judge.provider.api_key=EMPTY \
#       +judge.provider.max_completion_tokens=256 \
#       +judge.provider.temperature=0.0 \
#       +judge.provider.use_structured_output=true \
#       +judge.gating.enable_after_step=5 \
#       +judge.gating.run_every_k_steps=2 \
#       +judge.reward.lambda_proc=0.10 \
#       ...
#
# For non-thinking mode (recommended for literal consistency checking):
#   The judge client sends extra_body={"chat_template_kwargs": {"enable_thinking": false}}
#   This is handled automatically by the JudgeClient when using Qwen3.5.
