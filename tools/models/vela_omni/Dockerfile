# Build from repository root. The final image is a data artifact, not a server:
# docker build -f tools/models/vela_omni/Dockerfile --output type=local,dest=./artifacts .
# Add --build-arg VELA_OMNI_VARIANTS="nano mini" to prepare both public models.
ARG BUILDPLATFORM
FROM --platform=$BUILDPLATFORM python:3.12-slim-bookworm AS vela-omni-builder
ENV HF_HOME=/tmp/vela-hf-cache PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build/vela-omni
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY tools/models/vela_omni/requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0 && \
    pip install --no-cache-dir -r requirements.txt
COPY tools/models/vela_omni/ ./
RUN OMP_NUM_THREADS=4 python -m unittest discover -s . -p "test_*.py"
ARG VELA_OMNI_VARIANTS=nano
# E2E images retain official-reference inputs/outputs for native parity checks.
ARG VELA_OMNI_KEEP_GOLDEN=0
RUN case "$VELA_OMNI_KEEP_GOLDEN" in 0) set -- ;; 1) set -- --keep-golden ;; *) exit 1 ;; esac && \
    python prepare.py --variants ${VELA_OMNI_VARIANTS} "$@" --work /tmp/vela-export --output /opt/router-model-artifacts && \
    rm -rf /tmp/vela-export /tmp/vela-hf-cache

FROM scratch
COPY --from=vela-omni-builder /opt/router-model-artifacts/ /
