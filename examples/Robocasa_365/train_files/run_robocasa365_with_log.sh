#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# Training wrapper script - save all output to log file, view with: tail -f <log_file>
###########################################################################################

# Only define values needed by this wrapper. Training knobs are passed through to
# run_robocasa365.sh and keep their defaults there unless already exported.
TASK=${TASK:-OpenDrawer}
run_root_dir=${run_root_dir:-./results/Checkpoints}
run_id=robocasa_qwen3.5-cs-0.8B_$(date +%Y%m%d_%H%M%S)
export TASK
export run_root_dir
export run_id

LOG_FILE="${run_root_dir}/${run_id}/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${run_root_dir}/${run_id}"

echo "============================================"
echo "Training started at $(date)"
echo "Log file: $LOG_FILE"
echo "View with: tail -f $LOG_FILE"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
script -q -c "bash \"${SCRIPT_DIR}/run_robocasa365.sh\"" "$LOG_FILE"

echo "============================================"
echo "Training finished at $(date)"
echo "Log file: $LOG_FILE"
echo "============================================"
