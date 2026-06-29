#!/bin/bash
set -e

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_DEBUG=WARN
# # used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
config_yaml=${config_yaml:-./examples/LIBERO/train_files/starvla_cotrain_libero_twochunk_dct_memory.yaml}
run_id=${run_id:-liberogoal_qwen3.5-0.8b-twochunk-$(date +%Y%m%d_%H%M%S)}
LOG_TO_FILE=${LOG_TO_FILE:-1}
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled
run_root_dir=${run_root_dir:-/16T/liuyuyan/lyyvla/results/Checkpoints}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"

if [ "${LOG_TO_FILE}" = "1" ]; then
  LOG_FILE=${LOG_FILE:-${output_dir}/train_$(date +%Y%m%d_%H%M%S).log}

  echo "============================================"
  echo "Training started at $(date)"
  echo "Output dir: ${output_dir}"
  echo "Log file: ${LOG_FILE}"
  echo "View with: tail -f ${LOG_FILE}"
  echo "============================================"

  export config_yaml run_root_dir run_id LOG_FILE
  LOG_TO_FILE=0 script -q -c "bash $0" "${LOG_FILE}"

  echo "============================================"
  echo "Training finished at $(date)"
  echo "Log file: ${LOG_FILE}"
  echo "============================================"
  exit 0
fi

# mv this script to the output dir
cp "$0" "${output_dir}/"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port 29501 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  # --is_debug True

  # --main_process_port 0 \

##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
