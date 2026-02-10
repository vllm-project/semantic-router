# Multi-stage Dockerfile for vLLM Semantic Router
# Cross-compiles arm64 on BUILDPLATFORM (amd64) to avoid slow QEMU emulation.
# NOTE: TARGETARCH and BUILDPLATFORM are automatic BuildKit variables.
# Do NOT set defaults — they would override BuildKit's platform detection.
ARG TARGETARCH
ARG BUILDPLATFORM
ARG GIT_SSL_NO_VERIFY=0

# Stage 1: Build Rust candle-binding (CPU-only, cross-compiled for TARGETARCH)
FROM --platform=$BUILDPLATFORM rustlang/rust:nightly AS rust-builder
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
# The rustlang/rust:nightly image (Debian) has libssl-dev pre-installed and
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

# Stage 1b: Build dashboard frontend with Node.js
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json ./
# Use official npm registry to avoid mirror timeouts
RUN npm config set registry https://registry.npmjs.org/ && \
    npm ci --include=dev
COPY dashboard/frontend/ ./
RUN npm run build

# Stage 1c: Build dashboard backend with Go
FROM --platform=$BUILDPLATFORM golang:1.24 AS dashboard-builder
ARG TARGETARCH
WORKDIR /app

ENV GOPROXY=https://proxy.golang.org,direct

COPY src/semantic-router/go.mod src/semantic-router/go.sum /app/src/semantic-router/
COPY src/semantic-router/pkg/config/ /app/src/semantic-router/pkg/config/
COPY src/semantic-router/pkg/observability/ /app/src/semantic-router/pkg/observability/

COPY dashboard/backend/go.mod dashboard/backend/go.sum /app/dashboard/backend/
WORKDIR /app/dashboard/backend
RUN go mod download

COPY dashboard/backend/ /app/dashboard/backend/
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      apt-get update && apt-get install -y gcc-aarch64-linux-gnu && rm -rf /var/lib/apt/lists/* && \
      CC=aarch64-linux-gnu-gcc CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -ldflags="-w -s" -o /app/dashboard-backend main.go; \
    else \
      CGO_ENABLED=1 GOOS=linux go build -ldflags="-w -s" -o /app/dashboard-backend main.go; \
    fi

# Stage 2: Build Go semantic router
FROM --platform=$BUILDPLATFORM golang:1.24 AS go-builder
ARG TARGETARCH
WORKDIR /build

# Copy from normalized /build/out/ (set in rust stages)
COPY --from=rust-builder /build/out/ /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy candle-binding Go files (needed for go.mod replace directive)
COPY --from=rust-builder /build/go.mod /build/semantic-router.go /build/../candle-binding/

# Copy Go source (go.mod has local replace directives, so go mod download
# requires candle-binding to be present first)
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
      go build -ldflags="-w -s" -o router cmd/main.go; \
    else \
      CGO_ENABLED=1 go build -ldflags="-w -s" -o router cmd/main.go; \
    fi

# Stage 3: Get Envoy binary
FROM envoyproxy/envoy:v1.34-latest AS envoy

# Stage 4: Final runtime image
FROM python:3.12-slim
ARG GIT_SSL_NO_VERIFY

# Install system dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        supervisor \
        curl \
        ca-certificates; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Copy binaries from builders (rust artifacts from normalized /build/out/)
COPY --from=go-builder /build/router /usr/local/bin/router
COPY --from=rust-builder /build/out/libcandle_semantic_router.so /usr/local/lib/
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=dashboard-builder /app/dashboard-backend /usr/local/bin/dashboard-backend
COPY --from=frontend-builder /app/frontend/dist /app/frontend

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib

# Create necessary directories
RUN mkdir -p /app /app/models /app/config /etc/envoy /var/log/supervisor /var/log

# Copy Python CLI (includes templates in cli/templates/)
WORKDIR /app
COPY src/vllm-sr/cli/ /app/cli/
COPY src/vllm-sr/requirements.txt /app/requirements.txt

# Copy tools database to config directory
COPY src/vllm-sr/cli/templates/tools_db.json /app/config/tools_db.json

# Copy bench directory for evaluation benchmarks
COPY bench/ /app/bench/

# Copy training/model_eval directory for signal evaluation
COPY src/training/model_eval/ /app/src/training/model_eval/

# GIT_SSL_NO_VERIFY=1: add --trusted-host for pypi (proxy)
RUN if [ "$GIT_SSL_NO_VERIFY" = "1" ]; then PIP_EXTRA="--trusted-host pypi.org --trusted-host files.pythonhosted.org"; else PIP_EXTRA=""; fi && \
    pip install $PIP_EXTRA --no-cache-dir -r requirements.txt && \
    pip install $PIP_EXTRA --no-cache-dir -r /app/bench/requirements.txt && \
    pip install $PIP_EXTRA --no-cache-dir -r /app/src/training/model_eval/requirements.txt

# Copy start scripts
COPY src/vllm-sr/start-router.sh /app/start-router.sh
COPY src/vllm-sr/start-dashboard.sh /app/start-dashboard.sh
RUN chmod +x /app/start-router.sh /app/start-dashboard.sh

# Copy supervisord configuration
COPY src/vllm-sr/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log files and directories with proper permissions for non-root
RUN touch /var/log/envoy_access.log /var/log/supervisor/supervisord.log && \
    chmod 666 /var/log/envoy_access.log /var/log/supervisor/supervisord.log && \
    chmod 777 /var/log /var/log/supervisor /var/run

# Expose ports
# Note: Listener ports are configured in config.yaml and mapped dynamically
# 50051: Router gRPC (ExtProc)
# 9190: Prometheus metrics
# 9901: Envoy admin
# 8700: Dashboard UI
EXPOSE 50051 9190 9901 8700

# Volume for model cache
# Mount host's ./models directory to persist downloaded models
VOLUME ["/app/models"]

# Run supervisord
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
