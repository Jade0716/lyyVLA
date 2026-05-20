#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/liuyuyan/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/liuyuyan/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/home/liuyuyan/starVLA/results/Checkpoints/0414_liberogoal_qwen3.5-0.8b-pi/checkpoints/steps_100000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=2
video_out_path="results/${task_suite_name}/${folder_name}"

# Heatmap settings
generate_heatmaps=true
heatmap_dir="attention_heatmaps/${folder_name}"
heatmap_layer=10  # -1 means last layer, 0 means first layer, etc.
heatmap_interval=10


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero_attention.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.generate-heatmaps \
    --args.heatmap-dir "${heatmap_dir}" \
    --args.heatmap-layer ${heatmap_layer} \
    --args.heatmap-interval ${heatmap_interval}