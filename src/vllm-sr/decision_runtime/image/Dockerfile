# syntax=docker/dockerfile:1
# Decision-only OCI image. The caller supplies a digest-qualified base image;
# the ROCm base must already contain the Torch/HIP/Triton stack bound by the
# selected release artifact. No model data or credentials enter this build.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS wheel-builder
ARG PYTHON_BIN=python3
USER root
WORKDIR /source
COPY src/vllm-sr/ /source/src/vllm-sr/
COPY src/semantic-router/pkg/configschema/router-config-v0.3.schema.json /source/src/semantic-router/pkg/configschema/router-config-v0.3.schema.json
COPY config/recipes/built-in/ /source/config/recipes/built-in/
COPY tools/release/stage_model_catalog_package.py /source/tools/release/stage_model_catalog_package.py
RUN "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12)' && \
    "${PYTHON_BIN}" -m pip install --no-cache-dir \
      setuptools==80.9.0 wheel==0.45.1 PyYAML==6.0.3 && \
    "${PYTHON_BIN}" /source/tools/release/stage_model_catalog_package.py && \
    "${PYTHON_BIN}" -m pip wheel --no-deps --no-cache-dir --no-build-isolation \
      --wheel-dir /wheels /source/src/vllm-sr

FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG PYTHON_BIN=python3
ARG BACKEND
ARG TORCH_VERSION=2.12.0
ARG TORCH_INDEX_URL
ARG SOURCE_REVISION=uncommitted
ARG SOURCE_STATE=dirty
USER root
WORKDIR /opt/vllm-sr
LABEL org.opencontainers.image.source="https://github.com/vllm-project/semantic-router" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      ai.vllm-sr.decision.source-state="${SOURCE_STATE}" \
      ai.vllm-sr.decision.backend="${BACKEND}"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=wheel-builder /wheels/ /wheels/
COPY src/vllm-sr/decision_runtime/image/install.py /opt/vllm-sr/install-image.py
COPY src/vllm-sr/decision_runtime/image/requirements-vela.txt /opt/vllm-sr/requirements-vela.txt
COPY src/vllm-sr/decision_runtime/image/requirements-qwen35.txt /opt/vllm-sr/requirements-qwen35.txt
RUN set -eu; \
    case "${BASE_IMAGE}" in \
      *@sha256:*) ;; \
      *) echo "BASE_IMAGE must be digest-qualified" >&2; exit 2 ;; \
    esac; \
    case "${BACKEND}" in \
      cpu|cuda) \
        test -n "${TORCH_INDEX_URL}"; \
        "${PYTHON_BIN}" -m pip install --no-cache-dir \
          --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" ;; \
      rocm) \
        test -z "${TORCH_INDEX_URL}" ;; \
      *) echo "BACKEND must be cpu, cuda, or rocm" >&2; exit 2 ;; \
    esac; \
    "${PYTHON_BIN}" /opt/vllm-sr/install-image.py \
      --backend "${BACKEND}" --wheel-dir /wheels; \
    rm -rf /wheels /opt/vllm-sr/install-image.py \
      /opt/vllm-sr/requirements-vela.txt /opt/vllm-sr/requirements-qwen35.txt
EXPOSE 8000
# decision serve supplies the complete command. This default only supports image inspection.
ENTRYPOINT []
CMD ["/opt/vllm-sr/venvs/vela/bin/python", "-m", "decision_runtime.entrypoint", "--help"]
