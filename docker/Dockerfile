# =============================================================================
# Dockerfile — Yoga Mudra Detection System
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder)  : Compiles the C++ OpenCV visualizer
#   Stage 2 (runtime)  : Installs Python + MediaPipe + copies the C++ binary
#
# Base image: ubuntu:22.04  — well-supported, stable, slim
# =============================================================================

# ─── Stage 1: C++ Build ───────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

LABEL stage="cpp-builder"

ENV DEBIAN_FRONTEND=noninteractive

# Build tools + OpenCV development libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential          \
    cmake                    \
    libopencv-dev            \
    libopencv-highgui-dev    \
    pkg-config               \
    ca-certificates          \
    && rm -rf /var/lib/apt/lists/*

# Copy C++ source into the build stage
WORKDIR /build
COPY cpp/main.cpp        ./
COPY cpp/CMakeLists.txt  ./
COPY cpp/json.hpp        ./    

# Configure and compile
RUN cmake -B build_dir -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build_dir --parallel 4

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS runtime

LABEL maintainer="Yoga Mudra Detection System — Academic Project"
LABEL description="Hybrid Python/MediaPipe + C++/OpenCV Mudra Detection"

ENV DEBIAN_FRONTEND=noninteractive
# Prevent Python from buffering stdout (important for real-time log streaming)
ENV PYTHONUNBUFFERED=1
# X11 display forwarding for GUI windows
ENV DISPLAY=:0

# ── Runtime system packages ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python runtime
    python3.11                  \
    python3-pip                 \
    python3.11-dev              \
    # OpenCV runtime (shared libraries needed by C++ binary)
    libopencv-dev               \
    libopencv-highgui-dev       \
    # Camera/video device access
    v4l-utils                   \
    # X11 forwarding (GUI over SSH or host display)
    libx11-6                    \
    libxext6                    \
    x11-apps                    \
    # Utility
    ca-certificates             \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3    /usr/bin/python

# ── Python packages ───────────────────────────────────────────────────────────
COPY python/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── Copy C++ binary from builder stage ────────────────────────────────────────
COPY --from=builder /build/build_dir/visualizer /usr/local/bin/visualizer

# ── Copy application source ───────────────────────────────────────────────────
WORKDIR /app
COPY python/pose_detector.py ./python/
COPY scripts/                ./scripts/
COPY docker/entrypoint.sh    ./

# Create shared + log directories (runtime mounts may override)
RUN mkdir -p shared logs

# Make entry-point executable
RUN chmod +x entrypoint.sh scripts/*.sh 2>/dev/null || true

# Expose nothing — this is a desktop GUI application
# Volumes: /app/shared  (JSON IPC), /app/logs (output)
VOLUME ["/app/shared", "/app/logs"]

ENTRYPOINT ["./entrypoint.sh"]
