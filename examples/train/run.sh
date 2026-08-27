#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local or Hugging Face model identifier}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the generated training parquet file}"
: "${VAL_FILE:?Set VAL_FILE to the generated evaluation parquet file}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-tracerigor-smoke}"
NUM_GPUS="${NUM_GPUS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"

python -m tracerigor.trainer.main_ppo \
  algorithm.adv_estimator=bi_level_gae \
  algorithm.high_level_gamma=1.0 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=8 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  critic.model.path="${MODEL_PATH}" \
  trainer.logger='[console]' \
  trainer.project_name=tracerigor \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  rollout_manager.max_turns=3 \
  rollout_manager.window_size=5 \
  rollout_manager.use_multi_turn_reward=true \
  rollout_manager.use_loss_mask=true \
  rollout_manager.use_gae_mask=true
