# Multi-stage Dockerfile for vLLM Semantic Router
# Cross-compiles arm64 on BUILDPLATFORM (amd64) to avoid slow QEMU emulation.
# NOTE: TARGETARCH and BUILDPLATFORM are automatic BuildKit variables.
# Do NOT set defaults — they would override BuildKit's platform detection.
ARG TARGETARCH
ARG BUILDPLATFORM
ARG GIT_SSL_NO_VERIFY=0
ARG IMAGE_REGISTRY=docker.io/
ARG RUST_RUNTIME_COMPAT_IMAGE=rustlang/rust:nightly-bullseye
ARG GO_RUNTIME_COMPAT_IMAGE=library/golang:1.25-bookworm

# Stage 1: Build Rust candle-binding (CPU-only, cross-compiled for TARGETARCH)
#
# The router runtime image is Debian bookworm-slim. Keep the Rust and Go build
# stages on an older libc baseline so the copied router binary and shared
# objects do not pick up GLIBC_2.39+ requirements that fail at runtime.
FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${RUST_RUNTIME_COMPAT_IMAGE} AS rust-builder
ARG TARGETARCH
ARG GIT_SSL_NO_VERIFY
WORKDIR /build
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Install cross-compilation toolchain and OpenSSL for arm64 (needed by hf-hub -> openssl-sys, esaxx-rs -> g++)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      dpkg --add-architecture arm64 && \
      apt-get update && apt-get install -y \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        libssl-dev:arm64 \
        libssl-dev \
        pkg-config && \
      rm -rf /var/lib/apt/lists/* && \
      rustup target add aarch64-unknown-linux-gnu; \
    fi

# Configure cross-compilation environment for arm64
# NOTE: Do NOT set OPENSSL_DIR globally — it breaks amd64 pkg-config discovery.
# The pinned runtime-compatible Rust image (Debian bullseye) has libssl-dev pre-installed and
# pkg-config finds it automatically for amd64. For arm64, we set OPENSSL_LIB_DIR
# per-command below.
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
ENV PKG_CONFIG_ALLOW_CROSS=1
ENV PKG_CONFIG_PATH_aarch64_unknown_linux_gnu=/usr/lib/aarch64-linux-gnu/pkgconfig

# Copy Cargo manifest first for dependency caching
COPY candle-binding/Cargo.toml candle-binding/Cargo.loc[k] ./

# Pre-build dependencies with a dummy source for layer caching
RUN [ "$GIT_SSL_NO_VERIFY" = "1" ] && git config --global http.sslVerify false || true; \
    mkdir -p src && echo "pub fn _dummy() {}" > src/lib.rs && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu \
      cargo build --release --no-default-features --target aarch64-unknown-linux-gnu; \
    else \
      cargo build --release --no-default-features; \
    fi && \
    rm -rf src

# Copy real source and rebuild (dependencies are cached from above)
COPY candle-binding/src/ ./src/
COPY candle-binding/go.mod candle-binding/semantic-router.go ./

# Delete stale library from dummy build so cargo relinks with real FFI symbols
RUN find target -name "libcandle_semantic_router.so" -delete 2>/dev/null; \
    find target -name "libcandle_semantic_router.a" -delete 2>/dev/null; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu \
      cargo build --release --no-default-features --target aarch64-unknown-linux-gnu; \
    else \
      cargo build --release --no-default-features; \
    fi

# Normalize output: copy built artifacts to /build/out/
RUN mkdir -p /build/out && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cp target/aarch64-unknown-linux-gnu/release/libcandle_semantic_router.so /build/out/ && \
      (cp target/aarch64-unknown-linux-gnu/release/libcandle_semantic_router.a /build/out/ 2>/dev/null || true); \
    else \
      cp target/release/libcandle_semantic_router.so /build/out/ && \
      (cp target/release/libcandle_semantic_router.a /build/out/ 2>/dev/null || true); \
    fi && \
    echo "=== Built library ===" && \
    ls -la /build/out/ && \
    file /build/out/libcandle_semantic_router.so && \
    echo "Exported symbol count:" && \
    nm -D /build/out/libcandle_semantic_router.so 2>/dev/null | grep -c " T " || true

# Stage 1a: Build Rust ml-binding (Linfa ML algorithms, cross-compiled for TARGETARCH)
FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${RUST_RUNTIME_COMPAT_IMAGE} AS ml-builder
ARG TARGETARCH
ARG GIT_SSL_NO_VERIFY
WORKDIR /build
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Reuse cross-compilation toolchain setup (same as rust-builder)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      dpkg --add-architecture arm64 && \
      apt-get update && apt-get install -y \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu && \
      rm -rf /var/lib/apt/lists/* && \
      rustup target add aarch64-unknown-linux-gnu; \
    fi

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

# Copy Cargo manifest for dependency caching
COPY ml-binding/Cargo.toml ml-binding/Cargo.loc[k] ./

# Pre-build dependencies with dummy source
RUN [ "$GIT_SSL_NO_VERIFY" = "1" ] && git config --global http.sslVerify false || true; \
    mkdir -p src && echo "pub fn _dummy() {}" > src/lib.rs && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cargo build --release --target aarch64-unknown-linux-gnu; \
    else \
      cargo build --release; \
    fi && \
    rm -rf src

# Copy real source and rebuild
COPY ml-binding/src/ ./src/
COPY ml-binding/go.mod ml-binding/ml_binding.go ./

RUN find target -name "libml_semantic_router.so" -delete 2>/dev/null; \
    find target -name "libml_semantic_router.a" -delete 2>/dev/null; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cargo build --release --target aarch64-unknown-linux-gnu; \
    else \
      cargo build --release; \
    fi

# Normalize output
RUN mkdir -p /build/out && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cp target/aarch64-unknown-linux-gnu/release/libml_semantic_router.so /build/out/ && \
      (cp target/aarch64-unknown-linux-gnu/release/libml_semantic_router.a /build/out/ 2>/dev/null || true); \
    else \
      cp target/release/libml_semantic_router.so /build/out/ && \
      (cp target/release/libml_semantic_router.a /build/out/ 2>/dev/null || true); \
    fi && \
    echo "=== Built ml-binding library ===" && \
    ls -la /build/out/ && \
    file /build/out/libml_semantic_router.so

# Stage 1b: Build Rust nlp-binding (BM25 + N-gram keyword classification, cross-compiled)
FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${RUST_RUNTIME_COMPAT_IMAGE} AS nlp-builder
ARG TARGETARCH
ARG GIT_SSL_NO_VERIFY
WORKDIR /build
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      dpkg --add-architecture arm64 && \
      apt-get update && apt-get install -y \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu && \
      rm -rf /var/lib/apt/lists/* && \
      rustup target add aarch64-unknown-linux-gnu; \
    fi

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

COPY nlp-binding/Cargo.toml nlp-binding/Cargo.loc[k] ./

RUN [ "$GIT_SSL_NO_VERIFY" = "1" ] && git config --global http.sslVerify false || true; \
    mkdir -p src && echo "pub fn _dummy() {}" > src/lib.rs && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cargo build --release --target aarch64-unknown-linux-gnu; \
    else \
      cargo build --release; \
    fi && \
    rm -rf src

COPY nlp-binding/src/ ./src/

RUN find target -name "libnlp_binding.so" -delete 2>/dev/null; \
    find target -name "libnlp_binding.a" -delete 2>/dev/null; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cargo build --release --target aarch64-unknown-linux-gnu; \
    else \
      cargo build --release; \
    fi

RUN mkdir -p /build/out && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cp target/aarch64-unknown-linux-gnu/release/libnlp_binding.so /build/out/ && \
      (cp target/aarch64-unknown-linux-gnu/release/libnlp_binding.a /build/out/ 2>/dev/null || true); \
    else \
      cp target/release/libnlp_binding.so /build/out/ && \
      (cp target/release/libnlp_binding.a /build/out/ 2>/dev/null || true); \
    fi && \
    echo "=== Built nlp-binding library ===" && \
    ls -la /build/out/

# Build independent ORT instances using the existing CPU cross toolchain.
FROM rust-builder AS onnx-builder
ARG TARGETARCH
WORKDIR /build/onnx
COPY onnx-binding/Cargo.toml onnx-binding/Cargo.loc[k] ./
COPY onnx-binding/src/ ./src/
RUN mkdir -p /build/onnx-out && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu cargo build --release --lib --locked --no-default-features --features dynamic --target aarch64-unknown-linux-gnu && \
      cp target/aarch64-unknown-linux-gnu/release/libonnx_semantic_router.so /build/onnx-out/; \
    else \
      cargo build --release --lib --locked --no-default-features --features dynamic && \
      cp target/release/libonnx_semantic_router.so /build/onnx-out/; \
    fi

# ort rc.10 uses ORT API 22; keep the shared CPU runtime on that version.
ARG ONNXRUNTIME_VERSION=1.22.0
RUN case "$TARGETARCH" in arm64) ort_arch=aarch64 ;; amd64) ort_arch=x64 ;; *) exit 1 ;; esac && \
    mkdir -p /build/onnx-runtime && \
    curl --fail --location --retry 3 \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ort_arch}-${ONNXRUNTIME_VERSION}.tgz" \
      -o /tmp/onnxruntime.tgz && \
    tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /build/onnx-runtime && \
    cp -a /build/onnx-runtime/lib/libonnxruntime*.so* /build/onnx-out/ && \
    rm /tmp/onnxruntime.tgz

# Stage 2: Build Go semantic router
FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${GO_RUNTIME_COMPAT_IMAGE} AS go-builder
ARG TARGETARCH
WORKDIR /build

# Copy Rust libraries from all builders
COPY --from=rust-builder /build/out/ /usr/local/lib/
COPY --from=onnx-builder /build/onnx-out/ /usr/local/lib/
COPY --from=ml-builder /build/out/ /usr/local/lib/
COPY --from=nlp-builder /build/out/ /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy candle-binding Go files (needed for go.mod replace directive)
COPY candle-binding/go.mod candle-binding/*.go /build/../candle-binding/
COPY --from=rust-builder /build/out/ /build/../candle-binding/target/release/

COPY onnx-binding/go.mod /build/../onnx-binding/
COPY onnx-binding/instance/ /build/../onnx-binding/instance/
COPY --from=onnx-builder /build/onnx-out/ /build/../onnx-binding/target/release/

# Copy ml-binding Go files (needed for go.mod replace directive)
# Also copy the built library to ml-binding/target/release/ for CGO linking
COPY ml-binding/go.mod ml-binding/ml_binding.go /build/../ml-binding/
COPY --from=ml-builder /build/out/ /build/../ml-binding/target/release/

# Copy nlp-binding Go files (needed for go.mod replace directive)
COPY nlp-binding/go.mod nlp-binding/nlp_binding.go nlp-binding/nlp_binding_mock.go /build/../nlp-binding/
COPY --from=nlp-builder /build/out/ /build/../nlp-binding/target/release/

# Create openvino-binding stub (needed for go.mod replace directive)
RUN mkdir -p /build/../openvino-binding && \
    printf 'module github.com/vllm-project/semantic-router/openvino-binding\n\ngo 1.24.1\n' > /build/../openvino-binding/go.mod && \
    printf 'package openvino_binding\n' > /build/../openvino-binding/stub.go

# Copy Go source (go.mod has local replace directives, so go mod download
# requires candle-binding and ml-binding to be present first)
COPY src/semantic-router/ .
COPY tools/docker/check-native-abi.sh /build/check-native-abi.sh
RUN bash /build/check-native-abi.sh \
    /usr/local/lib/libcandle_semantic_router.so \
    /usr/local/lib/libonnx_semantic_router.so


# Build router (cross-compile for arm64 when TARGETARCH=arm64)
# arm64: need libssl-dev:arm64 because libcandle_semantic_router.so dynamically links OpenSSL
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      dpkg --add-architecture arm64 && \
      apt-get update && apt-get install -y \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        libssl-dev:arm64 && \
      rm -rf /var/lib/apt/lists/* && \
      CC=aarch64-linux-gnu-gcc \
      CGO_ENABLED=1 GOOS=linux GOARCH=arm64 \
      CGO_LDFLAGS="-L/usr/lib/aarch64-linux-gnu -lssl -lcrypto" \
      go build -buildvcs=false -ldflags="-w -s" -o router ./cmd; \
    else \
      CGO_ENABLED=1 go build -buildvcs=false -ldflags="-w -s" -o router ./cmd; \
    fi

# Stage 3: Final router runtime image
FROM ${IMAGE_REGISTRY}library/debian:bookworm-slim

ENV VIRTUAL_ENV=/opt/vllm-sr-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        libssl3 \
        python3 \
        python3-pip \
        python3-venv \
        python3-yaml; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir \
      'huggingface_hub==1.16.4' \
      'click>=8.1.7' \
      'pyyaml>=6.0.2'

COPY --from=go-builder /build/router /usr/local/bin/router
COPY --from=rust-builder /build/out/libcandle_semantic_router.so /usr/local/lib/
COPY --from=onnx-builder /build/onnx-out/ /usr/local/lib/
COPY --from=ml-builder /build/out/libml_semantic_router.so /usr/local/lib/
COPY --from=nlp-builder /build/out/libnlp_binding.so /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

WORKDIR /app
RUN mkdir -p /app /app/.vllm-sr /app/config /app/models

COPY config/knowledge_bases/ /app/config/knowledge_bases/
COPY src/vllm-sr/cli/ /app/cli/
COPY src/semantic-router/pkg/configschema/router-config-v0.3.schema.json /app/cli/config_schema/router-config-v0.3.schema.json
COPY config/recipes/built-in/ /app/cli/model_assets/
COPY src/vllm-sr/start-router.sh /app/start-router.sh
RUN chmod +x /app/start-router.sh

EXPOSE 50051 8080 9190

VOLUME ["/app/models"]

ENTRYPOINT ["/app/start-router.sh"]
CMD ["/app/config.yaml"]
