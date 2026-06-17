#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
calvin_python="${calvin_python:-/home/liuyuyan/miniconda3/envs/calvin_venv/bin/python}"
host="${host:-127.0.0.1}"
port="${port:-5694}"
unnorm_key="${unnorm_key:-franka}"
your_ckpt="${your_ckpt:-results/Checkpoints/calvin_gr00t_action2chunk-0.8B_20260609_182929/checkpoints/steps_70000_pytorch_model.pt}"
num_sequences="${num_sequences:-1000}"
dataset_path="${dataset_path:-/16T/liuyuyan/calvin_test}"
eval_sequences_path="${eval_sequences_path:-./examples/calvin/eval_files/eval_sequences.json}"
twochunk_debug="${twochunk_debug:-false}"
twochunk_debug_every_step="${twochunk_debug_every_step:-false}"

folder_name="$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"

args=(
  --args.pretrained-path "${your_ckpt}"
  --args.unnorm-key "${unnorm_key}"
  --args.host "${host}"
  --args.port "${port}"
  --args.dataset-path "${dataset_path}"
  --args.eval-sequences-path "${eval_sequences_path}"
  --args.eval-log-dir "tmp/calvin/eval_logs/twochunk_${folder_name}"
  --args.num-sequences "${num_sequences}"
)

if [[ "${twochunk_debug}" == "true" ]]; then
  args+=(--args.twochunk-debug)
fi

if [[ "${twochunk_debug_every_step}" == "true" ]]; then
  args+=(--args.twochunk-debug-every-step)
fi

"${calvin_python}" ./examples/calvin/eval_files/eval_calvin_twochunk.py "${args[@]}"
