#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <num_drones> [steps=auto:3episodes] [log_every=1] [action_source=dmpc] [seed=None] [num_envs=1] [episode_length_s=auto:60]" >&2
  exit 2
fi

NUM_DRONES="$1"
STEPS="${2:-}"
LOG_EVERY="${3:-1}"
ACTION_SOURCE="${4:-dmpc}"
SEED="${5:-}"
NUM_ENVS="${6:-1}"
EPISODE_LENGTH_S="${7:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif [[ -x "${ISAACLAB_SH}" ]]; then
  PYTHON_CMD=("${ISAACLAB_SH}" -p)
elif [[ -x "/workspace/my_project/isaacsim_ws/IsaacLab/isaaclab.sh" ]]; then
  PYTHON_CMD=("/workspace/my_project/isaacsim_ws/IsaacLab/isaaclab.sh" -p)
else
  echo "Could not find python, python3, or isaaclab.sh. Set ISAACLAB_SH=/path/to/isaaclab.sh" >&2
  exit 1
fi

RUN_DIR="${SCRIPT_DIR}/runs/online_bc_dmpc"
LOG_PATH="${RUN_DIR}/dmpc_debug_${NUM_DRONES}drone.npz"
mkdir -p "${RUN_DIR}"

SEED_ARGS=()
if [[ -n "${SEED}" && "${SEED}" != "None" && "${SEED}" != "none" ]]; then
  SEED_ARGS+=(--seed "${SEED}")
fi

cd "${PROJECT_ROOT}"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/online_bc_dmpc.py" \
  --num_envs "${NUM_ENVS}" \
  --num_drones "${NUM_DRONES}" \
  --expert_test_only \
  ${STEPS:+--expert_test_steps "${STEPS}"} \
  ${STEPS:+--steps_per_batch "${STEPS}"} \
  --eval_every_rounds 0 \
  ${EPISODE_LENGTH_S:+--episode_length_s "${EPISODE_LENGTH_S}"} \
  --no_randomize_episode_start \
  --no_terminate_on_bounds \
  --action_source "${ACTION_SOURCE}" \
  --headless \
  --livestream 2 \
  "${SEED_ARGS[@]}" \
  --dmpc_log_path "${LOG_PATH}" \
  --dmpc_log_every "${LOG_EVERY}"

echo "DMPC log saved to: ${LOG_PATH}"
