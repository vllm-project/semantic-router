# Multi-stage Dockerfile for vLLM Semantic Router
# Cross-compiles arm64 on BUILDPLATFORM (amd64) to avoid slow QEMU emulation.
# NOTE: TARGETARCH and BUILDPLATFORM are automatic BuildKit variables.
# Do NOT set defaults — they would override BuildKit's platform detection.
ARG TARGETARCH
ARG BUILDPLATFORM
ARG GIT_SSL_NO_VERIFY=0
ARG RUST_RUNTIME_COMPAT_IMAGE=rustlang/rust:nightly-bullseye
ARG GO_RUNTIME_COMPAT_IMAGE=golang:1.24-bullseye

# Stage 1: Build Rust candle-binding (CPU-only, cross-compiled for TARGETARCH)
#
# The router runtime image is Debian bookworm-slim. Keep the Rust and Go build
# stages on an older libc baseline so the copied router binary and shared
# objects do not pick up GLIBC_2.39+ requirements that fail at runtime.
FROM --platform=$BUILDPLATFORM ${RUST_RUNTIME_COMPAT_IMAGE} AS rust-builder
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
COPY candle-binding/go.mod candle-binding/semantic-router.go candle-binding/backend_contract.go ./

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
FROM --platform=$BUILDPLATFORM ${RUST_RUNTIME_COMPAT_IMAGE} AS ml-builder
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
FROM --platform=$BUILDPLATFORM ${RUST_RUNTIME_COMPAT_IMAGE} AS nlp-builder
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

# Stage 2: Build Go semantic router
FROM --platform=$BUILDPLATFORM ${GO_RUNTIME_COMPAT_IMAGE} AS go-builder
ARG TARGETARCH
WORKDIR /build

# Copy Rust libraries from all builders
COPY --from=rust-builder /build/out/ /usr/local/lib/
COPY --from=ml-builder /build/out/ /usr/local/lib/
COPY --from=nlp-builder /build/out/ /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy candle-binding Go files (needed for go.mod replace directive)
COPY --from=rust-builder /build/go.mod /build/semantic-router.go /build/backend_contract.go /build/../candle-binding/

# Copy ml-binding Go files (needed for go.mod replace directive)
# Also copy the built library to ml-binding/target/release/ for CGO linking
COPY ml-binding/go.mod ml-binding/ml_binding.go /build/../ml-binding/
COPY --from=ml-builder /build/out/ /build/../ml-binding/target/release/

# Copy nlp-binding Go files (needed for go.mod replace directive)
COPY nlp-binding/go.mod nlp-binding/nlp_binding.go nlp-binding/nlp_binding_mock.go /build/../nlp-binding/
COPY --from=nlp-builder /build/out/ /build/../nlp-binding/target/release/

# Copy Go source (go.mod has local replace directives, so go mod download
# requires candle-binding and ml-binding to be present first)
COPY src/semantic-router/ .

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
FROM debian:bookworm-slim

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
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir 'huggingface_hub[cli]==1.5.0'

COPY --from=go-builder /build/router /usr/local/bin/router
COPY --from=rust-builder /build/out/libcandle_semantic_router.so /usr/local/lib/
COPY --from=ml-builder /build/out/libml_semantic_router.so /usr/local/lib/
COPY --from=nlp-builder /build/out/libnlp_binding.so /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /app
RUN mkdir -p /app /app/.vllm-sr /app/config /app/models

COPY config/knowledge_bases/ /app/config/knowledge_bases/
COPY src/vllm-sr/start-router.sh /app/start-router.sh
RUN chmod +x /app/start-router.sh

EXPOSE 50051 8080 9190

VOLUME ["/app/models"]

ENTRYPOINT ["/app/start-router.sh"]
CMD ["/app/config.yaml"]
