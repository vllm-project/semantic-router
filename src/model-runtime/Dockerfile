# CPU image of the vLLM Semantic Router model runtime.
#
#   docker build -t vllm-srun src/model-runtime
#   docker run -p 8100:8100 vllm-srun \
#     serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --host 0.0.0.0
FROM python:3.12-slim

ARG TORCH_VERSION=2.10.0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/var/cache/vllm-srun

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

WORKDIR /opt/vllm-srun
COPY pyproject.toml README.md requirements-lock.txt ./
COPY vllm_srun ./vllm_srun
RUN pip install -c requirements-lock.txt ".[multimodal]" \
    && useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin runtime \
    && mkdir -p "$HF_HOME" \
    && chown runtime "$HF_HOME"

USER 65532
EXPOSE 8100
ENTRYPOINT ["vllm-srun"]
CMD ["--help"]
