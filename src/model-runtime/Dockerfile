# CPU image of the vLLM Semantic Router model runtime.
#
#   docker build -t vllm-sr-runtime src/model-runtime
#   docker run -p 8100:8100 vllm-sr-runtime \
#     serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --host 0.0.0.0
FROM python:3.12-slim

ARG TORCH_VERSION=2.10.0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/var/cache/vllm-sr-runtime

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

WORKDIR /opt/vllm-sr-runtime
COPY pyproject.toml README.md ./
COPY vllm_sr_runtime ./vllm_sr_runtime
RUN pip install ".[multimodal]" \
    && useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin runtime \
    && mkdir -p "$HF_HOME" \
    && chown runtime "$HF_HOME"

USER 65532
EXPOSE 8100
ENTRYPOINT ["vllm-sr-runtime"]
CMD ["--help"]
