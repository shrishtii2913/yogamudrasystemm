#!/usr/bin/env bash
# =============================================================================
# fetch_deps.sh — Download header-only C++ dependencies
# =============================================================================
# nlohmann/json is a single-header JSON library used by main.cpp
# Run this once before building C++ locally or in Docker.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="${SCRIPT_DIR}/../cpp"

JSON_URL="https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
JSON_OUT="${CPP_DIR}/json.hpp"

if [ -f "${JSON_OUT}" ]; then
    echo "[INFO] json.hpp already present at ${JSON_OUT}"
    exit 0
fi

echo "[FETCH] Downloading nlohmann/json v3.11.3..."
curl -fsSL "${JSON_URL}" -o "${JSON_OUT}"
echo "[FETCH] Saved to ${JSON_OUT}"
echo "[FETCH] Size: $(wc -c < "${JSON_OUT}") bytes"
