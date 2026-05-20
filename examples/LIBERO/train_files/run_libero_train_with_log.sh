#!/bin/bash

###########################################################################################
# Training wrapper script - save all output to log file, view with: tail -f <log_file>
###########################################################################################

# === 环境变量配置 (需要与 run_libero_train.sh 保持一致) ===
run_root_dir=./results/Checkpoints
run_id=liberoall_qwen3.5-0.8b-onlylanguage-$(date +%Y%m%d_%H%M%S)
export run_id  # 透传给 run_libero_train.sh
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
script -q -c "bash ${SCRIPT_DIR}/run_libero_train.sh" "$LOG_FILE"

echo "============================================"
echo "Training finished at $(date)"
echo "Log file: $LOG_FILE"
echo "============================================"