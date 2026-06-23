#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
calvin_python="${calvin_python:-/home/liuyuyan/miniconda3/envs/calvin_venv/bin/python}"
host="${host:-127.0.0.1}"
port="${port:-5695}"
unnorm_key="${unnorm_key:-franka}"
your_ckpt="${your_ckpt:-/16T/liuyuyan/lyyvla/results/Checkpoints/calvin_gr00t_4actiontoken_8features-0.8B_20260621_044327/checkpoints/steps_40000_pytorch_model.pt}"
num_sequences="${num_sequences:-1000}"
dataset_path="${dataset_path:-/16T/liuyuyan/calvin_test}"
eval_sequences_path="${eval_sequences_path:-./examples/calvin/eval_files/eval_sequences.json}"

folder_name="$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"

"${calvin_python}" ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.host "${host}" \
    --args.port "${port}" \
    --args.dataset-path "${dataset_path}" \
    --args.eval-sequences-path "${eval_sequences_path}" \
    --args.eval-log-dir "tmp/calvin/eval_logs/${folder_name}" \
    --args.num-sequences "${num_sequences}"
