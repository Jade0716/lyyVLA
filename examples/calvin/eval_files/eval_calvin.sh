#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Calvin client find websocket tools from main repo
export calvin_python=/home/liuyuyan/miniconda3/envs/calvin_venv/bin/python
export CUDA_VISIBLE_DEVICES=1
host="127.0.0.1"
base_port=5694
unnorm_key="franka" #franka
your_ckpt=results/Checkpoints/calvin_qwen3.5_gr00t-0.8B_20260602_153154/final_model/pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################


${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset_path /16T/liuyuyan/calvin_test \
    --args.eval_log_dir "tmp/calvin/eval_logs/${folder_name}" \
    --args.num_sequences 1000
