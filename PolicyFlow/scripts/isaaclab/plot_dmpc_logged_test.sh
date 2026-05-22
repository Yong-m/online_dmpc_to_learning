#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <num_drones> [agent=0] [horizon_stride=10]" >&2
  exit 2
fi

NUM_DRONES="$1"
AGENT="${2:-0}"
HORIZON_STRIDE="${3:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
FIG_PATH="${RUN_DIR}/dmpc_debug_${NUM_DRONES}drone_agent${AGENT}.png"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "Log file not found: ${LOG_PATH}" >&2
  echo "Run: ${SCRIPT_DIR}/run_dmpc_logged_test.sh ${NUM_DRONES}" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/plot_dmpc_log.py" \
  "${LOG_PATH}" \
  --agent "${AGENT}" \
  --horizon_stride "${HORIZON_STRIDE}" \
  --out "${FIG_PATH}"

echo "Figure saved to: ${FIG_PATH}"
