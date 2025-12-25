# vLLM Semantic Router
# Envoy + semantic-router in a single container

# Stage 1: Rust - Build candle binding
FROM rust:1.90 AS rust-builder
WORKDIR /build/candle-binding
COPY candle-binding/Cargo.toml candle-binding/Cargo.lock ./
COPY candle-binding/src/ src/
RUN cargo build --release --no-default-features

# Stage 2: Go - Build semantic-router
FROM golang:1.24 AS go-builder
WORKDIR /build

# Download dependencies first (cache layer)
COPY src/semantic-router/go.mod src/semantic-router/go.sum src/semantic-router/
COPY candle-binding/go.mod candle-binding/
RUN cd src/semantic-router && go mod download

# Copy source and build
COPY src/semantic-router/ src/semantic-router/
COPY candle-binding/semantic-router.go candle-binding/
COPY --from=rust-builder /build/candle-binding/target/release/libcandle_semantic_router.so candle-binding/target/release/

ENV CGO_ENABLED=1 \
    LD_LIBRARY_PATH=/build/candle-binding/target/release \
    GOOS=linux
RUN cd src/semantic-router && go build -ldflags="-w -s" -o /build/router cmd/main.go

FROM envoyproxy/envoy:v1.31.7 AS envoy

FROM quay.io/centos/centos:stream10
RUN dnf -y install gettext python3 python3-pip ca-certificates curl \
    && dnf clean all \
    && pip3 install --no-cache-dir supervisor huggingface_hub[cli] \
    && mkdir -p /var/log/supervisor /app/lib /app/config /app/models /etc/envoy

COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=go-builder /build/router /app/semantic-router
COPY --from=rust-builder /build/candle-binding/target/release/libcandle_semantic_router.so /app/lib/

COPY config/config.yaml /app/config/
COPY deploy/docker-compose/addons/envoy.yaml /etc/envoy/envoy.template.yaml
COPY deploy/docker-compose/supervisord.conf /etc/supervisor/supervisord.conf
RUN chmod +x /app/semantic-router

ENV LD_LIBRARY_PATH=/app/lib \
    CONFIG_FILE=/app/config/config.yaml \
    ENVOY_LISTEN_PORT=8801 \
    ENVOY_ADMIN_PORT=19000 \
    EXTPROC_HOST=localhost \
    EXTPROC_PORT=50051 \
    VLLM_BACKEND_HOST=localhost \
    VLLM_BACKEND_PORT=8000

EXPOSE 8801 19000 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
