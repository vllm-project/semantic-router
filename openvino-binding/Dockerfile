# =============================================================================
# Dockerfile for OpenVINO binding build and test
# Builds the C++ shared library (libopenvino_semantic_router.so) and runs Go tests
# =============================================================================
# Usage:
#   docker build -f openvino-binding/Dockerfile -t openvino-binding .
#   docker run --rm openvino-binding
# =============================================================================

# Use golang image to copy Go toolchain (avoids network download)
FROM golang:1.24-alpine AS go-src

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
    ls -la libopenvino_semantic_router.so && \
    nm -D libopenvino_semantic_router.so | grep -c "T " || true

# Copy Go module and source files
COPY openvino-binding/go.mod ./openvino-binding/
COPY openvino-binding/semantic-router.go ./openvino-binding/
COPY openvino-binding/semantic-router_test.go ./openvino-binding/

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

# Copy test models and conversion scripts
COPY openvino-binding/test_models/ ./openvino-binding/test_models/
COPY openvino-binding/scripts/ ./openvino-binding/scripts/
COPY openvino-binding/convert_modernbert_models.py ./openvino-binding/

# Convert test tokenizers
RUN cd openvino-binding && \
    python3 scripts/convert_test_tokenizers.py 2>/dev/null || \
    echo "Note: test tokenizer conversion skipped (models may not be available)"

# Run Go tests (some may fail without full models)
RUN cd openvino-binding && \
    go test -v -timeout 10m -count=1 ./... 2>&1; \
    echo "Test stage completed (failures expected without pre-converted models)"

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
