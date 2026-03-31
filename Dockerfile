# =============================================================================
# Dockerfile — Yoga Mudra Detection System
# =============================================================================
# Place this file at the repository root.
# Build context must include: python/, cpp/, scripts/, docker/entrypoint.sh
#
# Multi-stage build:
#   Stage 1 (builder) : Compiles C++ OpenCV visualizer
#   Stage 2 (runtime) : Python + MediaPipe + C++ binary
#
# Build:
#   docker build -t yoga-mudra-system .
#
# Run (with webcam + X11):
#   docker run --rm \
#     --device /dev/video0 \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v $(pwd)/logs:/app/logs \
#     -v $(pwd)/shared:/app/shared \
#     --network host \
#     yoga-mudra-system
# =============================================================================

# ─── Stage 1: C++ Build ───────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential    \
    cmake              \
    libopencv-dev      \
    ca-certificates    \
    curl               \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy C++ sources
COPY cpp/main.cpp       ./
COPY cpp/CMakeLists.txt ./

# Download nlohmann/json if not bundled
RUN if [ ! -f json.hpp ]; then \
      curl -fsSL \
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
        -o json.hpp; \
    fi

# Build Release binary
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --parallel "$(nproc)"

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11              \
    python3-pip             \
    libopencv-dev           \
    libx11-6                \
    libxext6                \
    v4l-utils               \
    ca-certificates         \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3    /usr/bin/python

# Install Python dependencies
COPY python/requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy compiled C++ binary
COPY --from=builder /build/build/visualizer /usr/local/bin/visualizer

# Copy Python source and scripts
WORKDIR /app
COPY python/pose_detector.py ./python/
COPY docker/entrypoint.sh    ./entrypoint.sh

RUN chmod +x entrypoint.sh \
 && mkdir -p shared logs

VOLUME ["/app/shared", "/app/logs"]

ENTRYPOINT ["./entrypoint.sh"]
