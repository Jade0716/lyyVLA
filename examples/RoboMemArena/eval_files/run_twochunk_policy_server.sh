#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:-/15T/liuyuyan/lyyvla/results/Checkpoints/robomemarena_qwen3.5-0.8b-twochunk-dctmemory-20260704_134931/checkpoints/steps_100000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-6697}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy_twochunk.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
