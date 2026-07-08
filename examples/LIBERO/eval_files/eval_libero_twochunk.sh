#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/home/liuyuyan/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-./results/Checkpoints/159_qwen3.5-0.8b-twochunk-v2-20260707_183630/checkpoints/steps_50000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6696}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-./tmp/libero/eval_logs}"
TWOCHUNK_DEBUG="${TWOCHUNK_DEBUG:-0}"
TWOCHUNK_ATTENTION_DEBUG="${TWOCHUNK_ATTENTION_DEBUG:-0}"
TWOCHUNK_SHORT_CHUNKS_PER_LONG_WINDOW="${TWOCHUNK_SHORT_CHUNKS_PER_LONG_WINDOW:-8}"

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
VIDEO_OUT_PATH="${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${FOLDER_NAME}"

ARGS=(
  --args.pretrained-path "${CKPT}"
  --args.host "${HOST}"
  --args.port "${PORT}"
  --args.task-suite-name "${TASK_SUITE_NAME}"
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --args.video-out-path "${VIDEO_OUT_PATH}"
  --args.eval-log-dir "${EVAL_LOG_DIR}"
  --args.twochunk
  --args.twochunk-short-chunks-per-long-window "${TWOCHUNK_SHORT_CHUNKS_PER_LONG_WINDOW}"
)
if [[ "${TWOCHUNK_DEBUG}" == "1" ]]; then
  ARGS+=(--args.twochunk-debug)
fi
if [[ "${TWOCHUNK_ATTENTION_DEBUG}" == "1" ]]; then
  ARGS+=(--args.twochunk-attention-debug)
fi

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py "${ARGS[@]}"
