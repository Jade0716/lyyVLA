#!/usr/bin/env bash
set -euo pipefail

echo "$(which python)"

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
sim_python="${sim_python:-python}"
SimplerEnv_PATH="${SimplerEnv_PATH:-/home/liuyuyan/SimplerEnv}"
SIMPLER_ENV_LIB_DIR="${SIMPLER_ENV_LIB_DIR:-}"
port="${port:-6678}"

your_ckpt="${your_ckpt:-./results/Checkpoints/oxe_qwen3.5-0.8B_20260525_150212/final_model/pytorch_model.pt}"

MODEL_PATH=${1:-"${your_ckpt}"}
port=${2:-"${port}"}

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
if [[ -n "${SIMPLER_ENV_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SIMPLER_ENV_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

#### build output directory #####
ckpt_path=${MODEL_PATH}
ckpt_dir=$(dirname "${ckpt_path}")
ckpt_parent=$(basename "${ckpt_dir}")
ckpt_base=$(basename "${ckpt_path}")
ckpt_name="${ckpt_base%.*}"
if [[ "${ckpt_parent}" == "final_model" || "${ckpt_parent}" == "checkpoints" ]]; then
  eval_run_dir=$(dirname "${ckpt_dir}")
else
  eval_run_dir="${ckpt_dir}"
fi
eval_run_id=$(basename "${eval_run_dir}")
eval_name="${eval_run_id}"
if [[ "${ckpt_parent}" == "checkpoints" ]]; then
  eval_name="${eval_run_id}_${ckpt_name}"
fi

# Create output directories
output_server_dir="${eval_run_dir}/output_server"
output_eval_dir="${eval_run_dir}/output_eval"
mkdir -p "${output_server_dir}"
mkdir -p "${output_eval_dir}"
#### build output directory #####

TSET_NUM=1
# export DEBUG=1

export CUDA_VISIBLE_DEVICES=1
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#CUDA_DEVICES[@]} 

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

RUNNER=()
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  RUNNER=(xvfb-run -a -s "-screen 0 1280x1024x24")
  echo "DISPLAY is not set; using xvfb-run for offscreen rendering."
fi

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
if [[ ! -f "${rgb_overlay_path}" ]]; then
  echo "rgb_overlay_path not found: ${rgb_overlay_path}" >&2
  exit 1
fi
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${eval_name}_${env}_run${run_idx}.log"
    task_summary="${output_eval_dir}/${eval_name}_${env}_run${run_idx}_summary.json"
    echo "▶️ Launching task [${env}] run#${run_idx}, log → ${task_log}"

    "${RUNNER[@]}" "${sim_python}" examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --summary-path "${task_summary}" \
      --logging-dir "${output_eval_dir}" \
      --eval-save-name "${eval_name}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &

    sleep 6

  done
done

declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
if [[ ! -f "${rgb_overlay_path}" ]]; then
  echo "rgb_overlay_path not found: ${rgb_overlay_path}" >&2
  exit 1
fi
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${eval_name}_${env}_run${run_idx}.log"
    task_summary="${output_eval_dir}/${eval_name}_${env}_run${run_idx}_summary.json"
    echo "▶️ Launching V2 task [${env}] run#${run_idx}, log → ${task_log}"

    "${RUNNER[@]}" "${sim_python}" examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --summary-path "${task_summary}" \
      --logging-dir "${output_eval_dir}" \
      --eval-save-name "${eval_name}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 
      # \
      # > "${task_log}" 2>&1

    sleep 6
  done
done

# echo "✅ Finished"
