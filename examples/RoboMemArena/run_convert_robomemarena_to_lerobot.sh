#!/usr/bin/env bash
set -euo pipefail

# One-command RoboMemArena full-trajectory -> LeRobot v2 conversion.
#
# Optional overrides:
#   SOURCE_ROOT=/16T/liuyuyan
#   OUTPUT_DIR=/16T/liuyuyan/robomemarena_lerobot
#   ROBOMEMARENA_REPO=/home/liuyuyan/RoboMemArena
#   CONDA_ENV=lingbot
#   TASKS=auto
#   DATASET_NAMES=RoboMemArena-Multi-Object-Counting,RoboMemArena-Multi-Object-Occlusion,...
#   PROMPT_SOURCE=hdf5
#   VIDEO_CODEC=libx264
#   VIDEO_PRESET=veryfast
#   VIDEO_CRF=18
#   VIDEO_SAMPLES=26
#   OVERWRITE=1
#   HASH_SOURCES=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LYYVLA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/15T/liuyuyan/RoboMemArena}"
OUTPUT_DIR="${OUTPUT_DIR:-/15T/liuyuyan/robomemarena_lerobot}"
ROBOMEMARENA_REPO="${ROBOMEMARENA_REPO:-/home/liuyuyan/RoboMemArena}"
CONDA_ENV="${CONDA_ENV:-starVLA}"
TASKS="${TASKS:-auto}"
DATASET_NAMES="${DATASET_NAMES:-Multi-Object_Counting,Multi-Object_Occlusion,Multi-Object_Sequence,Multi-Object_Transferring}"
PROMPT_SOURCE="${PROMPT_SOURCE:-hdf5}"
VIDEO_CODEC="${VIDEO_CODEC:-libx264}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
VIDEO_CRF="${VIDEO_CRF:-18}"
VIDEO_SAMPLES="${VIDEO_SAMPLES:-26}"
OVERWRITE="${OVERWRITE:-0}"
HASH_SOURCES="${HASH_SOURCES:-0}"

CONVERTER="${SCRIPT_DIR}/data_preparation/convert_hdf5_to_lerobot.py"
VALIDATOR="${SCRIPT_DIR}/data_preparation/validate_dataset.py"
BDDL_DIR="${ROBOMEMARENA_REPO}/evaluation_benchmark/bddl"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffmpeg and ffprobe are required." >&2
  exit 1
fi

for path in "${CONVERTER}" "${VALIDATOR}" "${BDDL_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path does not exist: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR: source root does not exist: ${SOURCE_ROOT}" >&2
  exit 1
fi

echo "Checking conversion environment: ${CONDA_ENV}"
conda run -n "${CONDA_ENV}" python -c \
  "import cv2, h5py, numpy, pandas, pyarrow; print('dependencies OK')"

discover_tasks() {
  local dataset_dir="$1"
  find "${dataset_dir}" \
    -path '*/._____temp/*' -prune \
    -o -path '*/full_trajectory/*.hdf5' -print \
    | sed -n 's/.*_task\([0-9][0-9]*\)\.hdf5$/\1/p' \
    | sort -n \
    | uniq \
    | paste -sd, -
}

IFS=',' read -r -a dataset_names <<< "${DATASET_NAMES}"

echo "Starting CPU conversion into one LeRobot dataset per source dataset"
echo "  lyyVLA root: ${LYYVLA_ROOT}"
echo "  source:      ${SOURCE_ROOT}"
echo "  output root: ${OUTPUT_DIR}"
echo "  BDDL:        ${BDDL_DIR}"
echo "  tasks:       ${TASKS}"
echo "  datasets:    ${DATASET_NAMES}"
echo "  prompts:     ${PROMPT_SOURCE}"
echo "  codec:       ${VIDEO_CODEC}, preset=${VIDEO_PRESET}, crf=${VIDEO_CRF}"

mkdir -p "${OUTPUT_DIR}"

converted=()

for dataset_name in "${dataset_names[@]}"; do
  dataset_name="${dataset_name#"${dataset_name%%[![:space:]]*}"}"
  dataset_name="${dataset_name%"${dataset_name##*[![:space:]]}"}"
  if [[ -z "${dataset_name}" ]]; then
    continue
  fi

  dataset_source="${SOURCE_ROOT}/${dataset_name}"
  dataset_output="${OUTPUT_DIR}/${dataset_name}"

  if [[ ! -d "${dataset_source}" ]]; then
    echo "ERROR: source dataset does not exist: ${dataset_source}" >&2
    exit 1
  fi

  dataset_tasks="${TASKS}"
  if [[ "${TASKS}" == "auto" ]]; then
    dataset_tasks="$(discover_tasks "${dataset_source}")"
  fi
  if [[ -z "${dataset_tasks}" ]]; then
    echo "ERROR: no full_trajectory HDF5 tasks discovered under: ${dataset_source}" >&2
    exit 1
  fi

  if [[ -e "${dataset_output}" && "${OVERWRITE}" != "1" ]]; then
    echo "ERROR: output already exists: ${dataset_output}" >&2
    echo "Set OVERWRITE=1 only if you intend to replace it." >&2
    exit 1
  fi

  convert_args=(
    "${CONVERTER}"
    --source-root "${dataset_source}"
    --output-dir "${dataset_output}"
    --bddl-dir "${BDDL_DIR}"
    --prompt-source "${PROMPT_SOURCE}"
    --tasks "${dataset_tasks}"
    --strict-task-coverage
    --video-codec "${VIDEO_CODEC}"
    --video-preset "${VIDEO_PRESET}"
    --video-crf "${VIDEO_CRF}"
    --progress
  )

  if [[ "${OVERWRITE}" == "1" ]]; then
    convert_args+=(--overwrite)
  fi

  if [[ "${HASH_SOURCES}" == "1" ]]; then
    convert_args+=(--hash-sources)
  fi

  echo "Converting ${dataset_name}"
  echo "  source: ${dataset_source}"
  echo "  output: ${dataset_output}"
  echo "  tasks:  ${dataset_tasks}"

  conda run --no-capture-output -n "${CONDA_ENV}" python "${convert_args[@]}"

  echo "Validating ${dataset_name}"
  conda run --no-capture-output -n "${CONDA_ENV}" python "${VALIDATOR}" \
    --dataset-dir "${dataset_output}" \
    --expected-tasks "${dataset_tasks}" \
    --strict-task-coverage \
    --video-samples "${VIDEO_SAMPLES}" \
    --source-frame-samples 3

  converted+=("${dataset_output}")
done

echo "Conversion and validation completed successfully:"
for path in "${converted[@]}"; do
  echo "  ${path}"
done
