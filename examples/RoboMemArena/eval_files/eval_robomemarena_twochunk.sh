#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ROBOMEMARENA_ROOT="${ROBOMEMARENA_ROOT:-/home/liuyuyan/RoboMemArena}"
ROBOMEMARENA_PYTHON="${ROBOMEMARENA_PYTHON:-python}"
CKPT="${CKPT:-/15T/liuyuyan/lyyvla/results/Checkpoints/robomemarena_qwen3.5-0.8b-twochunk-dctmemory-20260704_134931/checkpoints/steps_100000_pytorch_model.pt}"  # only used for naming output dirs; server must already be running
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6697}"
UNNORM_KEY="${UNNORM_KEY:-}"
SUBSET="${SUBSET:-occlusion}"
TASK_IDS="${TASK_IDS:-}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-100}"
RESIZE_SIZE="${RESIZE_SIZE:-256}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-3000}"
POST_GOAL_STEPS="${POST_GOAL_STEPS:-200}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
TWOCHUNK_DEBUG="${TWOCHUNK_DEBUG:-0}"
FAIL_ON_EXTRA_POUR="${FAIL_ON_EXTRA_POUR:-1}"
PROMPT_SOURCE="${PROMPT_SOURCE:-dataset}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/15T/liuyuyan/robomemarena_lerobot}"
OUT_ROOT="${OUT_ROOT:-}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${ROBOMEMARENA_ROOT}/evaluation_benchmark/scripts:${ROBOMEMARENA_ROOT}/evaluation_benchmark/libero_fork:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ -z "${OUT_ROOT}" ]]; then
  if [[ -n "${CKPT}" ]]; then
    folder_name="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
    model_root="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
    OUT_ROOT="${model_root}/results/robomemarena_${SUBSET}/${folder_name}"
  else
    OUT_ROOT="tmp/robomemarena/eval_logs/twochunk_${SUBSET}_$(date +%Y%m%d_%H%M%S)"
  fi
fi

args=(
  --robomemarena-root "${ROBOMEMARENA_ROOT}"
  --host "${HOST}"
  --port "${PORT}"
  --subset "${SUBSET}"
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --seed "${SEED}"
  --resize-size "${RESIZE_SIZE}"
  --replan-steps "${REPLAN_STEPS}"
  --max-steps "${MAX_STEPS}"
  --post-goal-steps "${POST_GOAL_STEPS}"
  --num-steps-wait "${NUM_STEPS_WAIT}"
  --prompt-source "${PROMPT_SOURCE}"
  --lerobot-root "${LEROBOT_ROOT}"
  --out-root "${OUT_ROOT}"
)

if [[ -n "${UNNORM_KEY}" ]]; then
  args+=(--unnorm-key "${UNNORM_KEY}")
fi
if [[ -n "${TASK_IDS}" ]]; then
  args+=(--task-ids "${TASK_IDS}")
fi
if [[ "${TWOCHUNK_DEBUG}" == "1" ]]; then
  args+=(--twochunk-debug)
fi
if [[ "${FAIL_ON_EXTRA_POUR}" == "0" ]]; then
  args+=(--no-fail-on-extra-pour)
fi

"${ROBOMEMARENA_PYTHON}" examples/RoboMemArena/eval_files/eval_robomemarena.py "${args[@]}"
