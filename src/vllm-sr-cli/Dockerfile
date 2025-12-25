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

# Stage 2: Build Go semantic router
FROM golang:1.24 AS go-builder
WORKDIR /build

# Copy candle-binding library for CGO linking
COPY --from=rust-builder /build/target/release/libcandle_semantic_router.so /usr/local/lib/
COPY --from=rust-builder /build/target/release/libcandle_semantic_router.a /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy candle-binding Go files (needed for go.mod replace directive)
COPY --from=rust-builder /build/go.mod /build/semantic-router.go /build/../candle-binding/

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
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib

# Create necessary directories
RUN mkdir -p /app /app/models /app/config /etc/envoy /var/log/supervisor /var/log

# Copy Python CLI (includes templates in cli/templates/)
WORKDIR /app
COPY src/vllm-sr-cli/cli/ /app/cli/
COPY src/vllm-sr-cli/requirements.txt /app/requirements.txt

# Copy tools database to config directory
COPY src/vllm-sr-cli/cli/templates/tools_db.json /app/config/tools_db.json

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy start scripts
COPY src/vllm-sr-cli/start-router.sh /app/start-router.sh
RUN chmod +x /app/start-router.sh

# Copy supervisord configuration
COPY src/vllm-sr-cli/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log files
RUN touch /var/log/envoy_access.log /var/log/supervisor/supervisord.log

# Expose ports
# Note: Listener ports are configured in config.yaml and mapped dynamically
# 50051: Router gRPC (ExtProc)
# 9190: Prometheus metrics
# 9901: Envoy admin
EXPOSE 50051 9190 9901

# Volume for model cache
# Mount host's ./models directory to persist downloaded models
VOLUME ["/app/models"]

# Run supervisord
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
