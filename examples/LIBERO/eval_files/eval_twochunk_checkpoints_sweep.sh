#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
EXP_DIR="${EXP_DIR:-${STARVLA_DIR}/results/Checkpoints/159_qwen3.5-0.8b-twochunk-v2-20260707_183630}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${EXP_DIR}/checkpoints}"
CKPT_PATTERN="${CKPT_PATTERN:-steps_*_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6696}"
GPU_ID="${GPU_ID:-2}"
USE_BF16="${USE_BF16:-1}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${EXP_DIR}/eval_sweep_logs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-300}"
SERVER_POLL_INTERVAL_S="${SERVER_POLL_INTERVAL_S:-2}"
ALLOW_PORT_IN_USE="${ALLOW_PORT_IN_USE:-0}"

SERVER_SCRIPT="${SERVER_SCRIPT:-${STARVLA_DIR}/examples/LIBERO/eval_files/run_twochunk_policy_server.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${STARVLA_DIR}/examples/LIBERO/eval_files/eval_libero_twochunk.sh}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/liuyuyan/miniconda3/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${STARVLA_PYTHON}}"

mkdir -p "${SWEEP_LOG_DIR}"
cd "${STARVLA_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[sweep] checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

mapfile -t CKPTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "${CKPT_PATTERN}" | sort -Vr)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[sweep] no checkpoints found: ${CHECKPOINT_DIR}/${CKPT_PATTERN}" >&2
  exit 1
fi

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[sweep] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

port_open() {
  "${STARVLA_PYTHON}" - <<PY2 >/dev/null 2>&1
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
try:
    s.connect(("${HOST}", int("${PORT}")))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY2
}

wait_for_server() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[sweep] server exited before becoming ready" >&2
      return 1
    fi
    if port_open; then
      return 0
    fi
    sleep "${SERVER_POLL_INTERVAL_S}"
  done
  echo "[sweep] server did not open ${HOST}:${PORT} within ${SERVER_START_TIMEOUT_S}s" >&2
  return 1
}

FAILURES=0
TOTAL=${#CKPTS[@]}
echo "[sweep] experiment: ${EXP_DIR}"
echo "[sweep] checkpoints: ${TOTAL} from ${CHECKPOINT_DIR}/${CKPT_PATTERN}"
echo "[sweep] logs: ${SWEEP_LOG_DIR}"

for idx in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$idx]}"
  CKPT_BASENAME="$(basename "${CKPT}" .pt)"
  SERVER_LOG="${SWEEP_LOG_DIR}/${CKPT_BASENAME}.server.log"
  CLIENT_LOG="${SWEEP_LOG_DIR}/${CKPT_BASENAME}.client.log"

  echo "[sweep] [$((idx + 1))/${TOTAL}] ckpt=${CKPT}"
  cleanup_server

  if port_open && [[ "${ALLOW_PORT_IN_USE}" != "1" ]]; then
    echo "[sweep] ${HOST}:${PORT} is already open before starting server. Stop the old server or set ALLOW_PORT_IN_USE=1." >&2
    exit 1
  fi

  echo "[sweep] starting server on gpu=${GPU_ID}, port=${PORT}; log=${SERVER_LOG}"
  STARVLA_DIR="${STARVLA_DIR}" \
  STARVLA_PYTHON="${STARVLA_PYTHON}" \
  CKPT="${CKPT}" \
  GPU_ID="${GPU_ID}" \
  PORT="${PORT}" \
  USE_BF16="${USE_BF16}" \
    bash "${SERVER_SCRIPT}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "[sweep] server failed for ${CKPT}; see ${SERVER_LOG}" >&2
    cleanup_server
    FAILURES=$((FAILURES + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
    continue
  fi

  echo "[sweep] running client eval; log=${CLIENT_LOG}"
  set +e
  STARVLA_DIR="${STARVLA_DIR}" \
  LIBERO_PYTHON="${LIBERO_PYTHON}" \
  CKPT="${CKPT}" \
  HOST="${HOST}" \
  PORT="${PORT}" \
    bash "${EVAL_SCRIPT}" >"${CLIENT_LOG}" 2>&1
  CLIENT_STATUS=$?
  set -e

  cleanup_server

  if [[ ${CLIENT_STATUS} -ne 0 ]]; then
    echo "[sweep] client eval failed for ${CKPT}; status=${CLIENT_STATUS}; see ${CLIENT_LOG}" >&2
    FAILURES=$((FAILURES + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${CLIENT_STATUS}"
    fi
  else
    echo "[sweep] finished ${CKPT}"
  fi

done

trap - EXIT INT TERM
cleanup_server

if [[ ${FAILURES} -ne 0 ]]; then
  echo "[sweep] completed with ${FAILURES}/${TOTAL} failures" >&2
  exit 1
fi

echo "[sweep] completed all ${TOTAL} checkpoints successfully"
