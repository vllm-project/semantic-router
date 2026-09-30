# =============================================================================
# Dockerfile for OpenVINO binding build and test
# Builds the C++ shared library (libopenvino_semantic_router.so) and runs Go tests
# =============================================================================
# Usage:
#   docker build --platform linux/amd64 --target tester \
#     -f openvino-binding/Dockerfile .
# =============================================================================

# Use golang image to copy Go toolchain (avoids network download)
FROM golang:1.25-alpine AS go-src

# =============================================================================
# Builder stage
# =============================================================================
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Go from golang image
COPY --from=go-src /usr/local/go /usr/local/go
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV PATH="/go/bin:${PATH}"

# Install OpenVINO and tokenizers via pip
RUN pip install --no-cache-dir \
    openvino==2025.1.0 \
    openvino-tokenizers==2025.1.0

WORKDIR /app

# Copy CMakeLists and source for C++ library build
COPY openvino-binding/CMakeLists.txt ./openvino-binding/
COPY openvino-binding/cmake/ ./openvino-binding/cmake/
COPY openvino-binding/cpp/ ./openvino-binding/cpp/

# Build C++ library
RUN mkdir -p openvino-binding/build && \
    cd openvino-binding/build && \
    cmake .. && \
    make -j$(nproc) && \
    echo "=== Verifying library ===" && \
    test -s libopenvino_semantic_router.so

# Copy Go module and source files
COPY openvino-binding/go.mod ./openvino-binding/
COPY openvino-binding/*.go ./openvino-binding/

# Download Go dependencies
RUN cd openvino-binding && go mod download

# Set library path for OpenVINO runtime
ENV LD_LIBRARY_PATH=/app/openvino-binding/build
ENV CGO_ENABLED=1

# Build Go binding to verify compilation succeeds
RUN cd openvino-binding && \
    go build -v . && \
    echo "=== Go binding compiled successfully ==="

# =============================================================================
# Test stage: convert models and run tests
# =============================================================================
FROM builder AS tester

# The binding image tests owned handles and published Vela weights. Router-owned
# runtime integration additionally runs in native CI with the shared libraries.
COPY openvino-binding/requirements-test.txt ./openvino-binding/
RUN pip install --no-cache-dir -r openvino-binding/requirements-test.txt
COPY . .
ENV HF_HUB_DISABLE_XET=1
RUN make verify-openvino-published

# =============================================================================
# Final stage: minimal runtime with built library
# =============================================================================
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    openvino==2025.1.0 \
    openvino-tokenizers==2025.1.0

WORKDIR /app

# Copy built shared library
COPY --from=builder /app/openvino-binding/build/libopenvino_semantic_router.so /app/lib/

ENV LD_LIBRARY_PATH=/app/lib

CMD ["echo", "OpenVINO binding built successfully. Use as build stage in Dockerfile.extproc."]
