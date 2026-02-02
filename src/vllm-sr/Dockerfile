# Multi-stage Dockerfile for vLLM Semantic Router
# Stage 1: Build Rust candle-binding (CPU-only)
# Use nightly Rust because candle-core uses unstable features
FROM rustlang/rust:nightly AS rust-builder
WORKDIR /build
COPY candle-binding/ .
# Build with CPU-only (no CUDA) - disable default features which include cuda
RUN cargo build --release --no-default-features && \
    echo "Checking built library:" && \
    find target -name "*.so" -type f && \
    ls -la target/release/

# Stage 1a: Build Rust ml-binding (for ML model selection: KNN, KMeans, SVM)
FROM rustlang/rust:nightly AS ml-rust-builder
WORKDIR /build
COPY ml-binding/ .
# Build ml-binding library
RUN cargo build --release && \
    echo "Checking ml-binding library:" && \
    find target -name "*.so" -type f && \
    ls -la target/release/

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
# Note: dashboard/backend/go.mod has a replace directive for src/semantic-router
# Use golang image (not alpine) for CGO support - go-sqlite3 requires CGO
FROM golang:1.24 AS dashboard-builder
WORKDIR /app

# Use Go proxy for reliable downloads
ENV GOPROXY=https://proxy.golang.org,direct

# Copy src/semantic-router first (required by dashboard/backend/go.mod replace directive)
COPY src/semantic-router/go.mod src/semantic-router/go.sum /app/src/semantic-router/
COPY src/semantic-router/pkg/config/ /app/src/semantic-router/pkg/config/
COPY src/semantic-router/pkg/observability/ /app/src/semantic-router/pkg/observability/

# Copy go.mod and go.sum first for better caching
COPY dashboard/backend/go.mod dashboard/backend/go.sum /app/dashboard/backend/
WORKDIR /app/dashboard/backend
RUN go mod download

# Copy source code and build
# CGO_ENABLED=1 is required for go-sqlite3 (evaluation database)
COPY dashboard/backend/ /app/dashboard/backend/
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-w -s" -o /app/dashboard-backend main.go

# Stage 2: Build Go semantic router
FROM golang:1.24 AS go-builder
WORKDIR /build

# Copy candle-binding library for CGO linking
COPY --from=rust-builder /build/target/release/libcandle_semantic_router.so /usr/local/lib/
COPY --from=rust-builder /build/target/release/libcandle_semantic_router.a /usr/local/lib/

# Copy ml-binding library for CGO linking (ML model selection: KNN, KMeans, SVM)
# Note: ml-binding only produces .so (shared library), not .a (static library)
COPY --from=ml-rust-builder /build/target/release/libml_semantic_router.so /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy candle-binding Go files (needed for go.mod replace directive)
COPY --from=rust-builder /build/go.mod /build/semantic-router.go /build/../candle-binding/

# Copy ml-binding Go files (needed for ML model selection - go.mod replace directive)
COPY --from=ml-rust-builder /build/go.mod /build/ml_binding.go /build/../ml-binding/

# Copy Go source
COPY src/semantic-router/ .

# Build router
RUN CGO_ENABLED=1 go build -o router cmd/main.go

# Stage 3: Get Envoy binary
FROM envoyproxy/envoy:v1.34-latest AS envoy

# Stage 4: Final runtime image
FROM python:3.12-slim

# Install system dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        supervisor \
        curl \
        ca-certificates; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Copy binaries from builders
COPY --from=go-builder /build/router /usr/local/bin/router
COPY --from=rust-builder /build/target/release/libcandle_semantic_router.so /usr/local/lib/
COPY --from=ml-rust-builder /build/target/release/libml_semantic_router.so /usr/local/lib/
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

# Install Python dependencies (CLI + bench)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r /app/bench/requirements.txt

# Copy start scripts
COPY src/vllm-sr/start-router.sh /app/start-router.sh
COPY src/vllm-sr/start-dashboard.sh /app/start-dashboard.sh
RUN chmod +x /app/start-router.sh /app/start-dashboard.sh

# Copy supervisord configuration
COPY src/vllm-sr/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log files
RUN touch /var/log/envoy_access.log /var/log/supervisor/supervisord.log

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
