# ML Training Service - Python sidecar for benchmark & training operations
#
# This image is used alongside the Go dashboard backend.
# The dashboard calls this service's HTTP API instead of spawning Python subprocesses.
#
# Build:
#   docker build -t ml-training-service -f src/training/model_selection/ml_model_selection/Dockerfile .
#
# Run:
#   docker run -p 8686:8686 -v /data:/app/data ml-training-service

# ---- Stage 1: Build dependencies (CPU-only PyTorch for smaller image) ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY src/training/model_selection/ml_model_selection/requirements.txt .

# Install CPU-only PyTorch first (much smaller than GPU version)
RUN pip install --no-cache-dir --target=/build/deps \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir --target=/build/deps \
    -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages/

# Copy the ML training scripts
COPY src/training/model_selection/ml_model_selection/*.py /app/
COPY src/training/model_selection/ml_model_selection/requirements.txt /app/

# Create non-root user
RUN useradd -m -u 65532 nonroot && \
    mkdir -p /app/data && \
    chown -R nonroot:nonroot /app

USER nonroot

# Pre-download will happen at runtime (embedding models are large)
# Set HuggingFace cache to a persistent volume mount point
ENV HF_HOME=/app/data/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV ML_SERVICE_PORT=8686

EXPOSE 8686

HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8686/api/health || exit 1

ENTRYPOINT ["python", "server.py"]
CMD ["--port", "8686"]
