#!/bin/bash

###########################################################################################
# Training wrapper script - save all output to log file, view with: tail -f <log_file>
###########################################################################################

# === 环境变量配置 (需要与 run_oxe_train.sh 保持一致) ===
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1000

run_root_dir=/16T/liuyuyan/lyyvla/results/Checkpoints
run_id=calvin_gr00t_4actiontoken_8features-0.8B_$(date +%Y%m%d_%H%M%S)
# run_id=calvin_adapter-0.8B_$(date +%Y%m%d_%H%M%S)
export run_root_dir
export run_id
# === End ===

# Log file path
LOG_FILE="${run_root_dir}/${run_id}/train_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if not exists
mkdir -p "${run_root_dir}/${run_id}"

echo "============================================"
echo "Training started at $(date)"
echo "Log file: $LOG_FILE"
echo "View with: tail -f $LOG_FILE"
echo "============================================"

# Run the training script, redirect ALL output to log file
# 使用 script 命令可以捕获所有子进程的输出（包括 accelerate 的多进程输出）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
script -q -c "bash ${SCRIPT_DIR}/run_calvin_train.sh" "$LOG_FILE"

echo "============================================"
echo "Training finished at $(date)"
echo "Log file: $LOG_FILE"
echo "============================================"