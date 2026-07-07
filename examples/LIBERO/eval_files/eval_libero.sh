#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/home/liuyuyan/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-/15T/liuyuyan/lyyvla/results/Checkpoints/libero10_qwen3.5-0.8b-qwenpi-20260706_232419/checkpoints/steps_50000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-./tmp/libero/eval_logs}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0"
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
VIDEO_OUT_PATH="${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${FOLDER_NAME}"

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  --args.eval-log-dir "${EVAL_LOG_DIR}"
