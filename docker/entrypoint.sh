#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — Container startup script
# =============================================================================
# Starts Python detector and C++ visualizer as parallel background processes.
# Both write to the shared/ and logs/ volumes.
# The script waits for either process to exit, then cleans up the other.
# =============================================================================

set -e

APP_DIR="/app"
SHARED_DIR="${APP_DIR}/shared"
LOGS_DIR="${APP_DIR}/logs"

mkdir -p "${SHARED_DIR}" "${LOGS_DIR}"

echo "======================================================"
echo "  Yoga Mudra Detection System"
echo "  Starting hybrid Python + C++ pipeline..."
echo "======================================================"
echo ""
echo "  Shared IPC : ${SHARED_DIR}/output.json"
echo "  Logs       : ${LOGS_DIR}/"
echo ""

# ── Start Python detector ─────────────────────────────────────────────────────
echo "[1/2] Starting Python / MediaPipe detector..."
cd "${APP_DIR}"
python3 python/pose_detector.py &
PYTHON_PID=$!
echo "      PID: ${PYTHON_PID}"

# Give Python a moment to initialise the camera and write first JSON
sleep 2

# ── Start C++ visualizer ──────────────────────────────────────────────────────
echo "[2/2] Starting C++ / OpenCV visualizer..."
visualizer &
CPP_PID=$!
echo "      PID: ${CPP_PID}"

echo ""
echo "Both components running. Press ESC in either window to stop."
echo "------------------------------------------------------"

# ── Wait for either process to exit, then stop both ──────────────────────────
wait -n "${PYTHON_PID}" "${CPP_PID}" 2>/dev/null || true

echo ""
echo "One component exited. Stopping the other..."
kill "${PYTHON_PID}" 2>/dev/null || true
kill "${CPP_PID}"    2>/dev/null || true
wait  2>/dev/null || true

echo "======================================================"
echo "  Session complete."
echo "  Log files are in: ${LOGS_DIR}/"
echo "======================================================"
