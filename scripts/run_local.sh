#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Run the system locally WITHOUT Docker
# =============================================================================
# Pre-requisites (Ubuntu/Debian):
#   sudo apt install libopencv-dev cmake build-essential python3-pip
#   pip3 install -r python/requirements.txt
# =============================================================================

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p shared logs

# ── Build C++ ─────────────────────────────────────────────────────────────────
echo "[BUILD] Compiling C++ visualizer..."
mkdir -p cpp/build
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build cpp/build --parallel 4
echo "[BUILD] Done → cpp/build/visualizer"

# ── Download json.hpp if not present ─────────────────────────────────────────
if [ ! -f cpp/json.hpp ]; then
    echo "[FETCH] Downloading nlohmann/json.hpp..."
    curl -sL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
         -o cpp/json.hpp
fi

# ── Start Python ──────────────────────────────────────────────────────────────
echo "[START] Python / MediaPipe detector..."
python3 python/pose_detector.py &
PYTHON_PID=$!

sleep 2

# ── Start C++ ─────────────────────────────────────────────────────────────────
echo "[START] C++ / OpenCV visualizer..."
cpp/build/visualizer &
CPP_PID=$!

echo ""
echo "Both running. Press ESC in either window to stop."
wait -n "${PYTHON_PID}" "${CPP_PID}" 2>/dev/null || true
kill "${PYTHON_PID}" "${CPP_PID}" 2>/dev/null || true
wait 2>/dev/null || true
echo "Done. See logs/ for results."
