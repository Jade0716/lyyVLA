#!/usr/bin/env bash
set -euo pipefail

# One-command RoboMemArena full-trajectory -> LeRobot v2 conversion.
#
# Optional overrides:
#   SOURCE_ROOT=/16T/liuyuyan
#   OUTPUT_DIR=/16T/liuyuyan/robomemarena_lerobot
#   ROBOMEMARENA_REPO=/home/liuyuyan/RoboMemArena
#   CONDA_ENV=lingbot
#   TASKS=1-26
#   VIDEO_CODEC=libx264
#   VIDEO_PRESET=veryfast
#   VIDEO_CRF=18
#   VIDEO_SAMPLES=26
#   OVERWRITE=1
#   HASH_SOURCES=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LYYVLA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/16T/liuyuyan}"
OUTPUT_DIR="${OUTPUT_DIR:-/16T/liuyuyan/robomemarena_lerobot}"
ROBOMEMARENA_REPO="${ROBOMEMARENA_REPO:-/home/liuyuyan/RoboMemArena}"
CONDA_ENV="${CONDA_ENV:-lingbot}"
TASKS="${TASKS:-1-26}"
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

if [[ -e "${OUTPUT_DIR}" && "${OVERWRITE}" != "1" ]]; then
  echo "ERROR: output already exists: ${OUTPUT_DIR}" >&2
  echo "Set OVERWRITE=1 only if you intend to replace it." >&2
  exit 1
fi

echo "Checking conversion environment: ${CONDA_ENV}"
conda run -n "${CONDA_ENV}" python -c \
  "import cv2, h5py, numpy, pandas, pyarrow; print('dependencies OK')"

convert_args=(
  "${CONVERTER}"
  --source-root "${SOURCE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --bddl-dir "${BDDL_DIR}"
  --prompt-source bddl
  --tasks "${TASKS}"
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

echo "Starting CPU conversion"
echo "  lyyVLA root: ${LYYVLA_ROOT}"
echo "  source:      ${SOURCE_ROOT}"
echo "  output:      ${OUTPUT_DIR}"
echo "  BDDL:        ${BDDL_DIR}"
echo "  tasks:       ${TASKS}"
echo "  codec:       ${VIDEO_CODEC}, preset=${VIDEO_PRESET}, crf=${VIDEO_CRF}"
echo "The converter will stop before writing output if any selected task is missing."

conda run --no-capture-output -n "${CONDA_ENV}" python "${convert_args[@]}"

echo "Running strict dataset validation"
conda run --no-capture-output -n "${CONDA_ENV}" python "${VALIDATOR}" \
  --dataset-dir "${OUTPUT_DIR}" \
  --expected-tasks "${TASKS}" \
  --strict-task-coverage \
  --video-samples "${VIDEO_SAMPLES}" \
  --source-frame-samples 3

echo "Conversion and validation completed successfully:"
echo "  ${OUTPUT_DIR}"
