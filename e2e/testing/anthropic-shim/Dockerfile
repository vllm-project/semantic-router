# Anthropic-shape shim in front of llama.cpp's llama-server.
# Tiny FastAPI service used by the e2e suite.
FROM python:3.11-slim

LABEL maintainer="vLLM Semantic Router Team"
LABEL description="Anthropic Messages API shim that fronts llama.cpp's llama-server for vLLM SR e2e tests"
LABEL version="0.1.0"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY anthropic_shim/ ./anthropic_shim/
COPY pyproject.toml ./
COPY README.md ./

RUN pip install --no-cache-dir -e .

RUN useradd --create-home --shell /bin/bash shim
USER shim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANTHROPIC_SHIM_HOST=0.0.0.0 \
    ANTHROPIC_SHIM_PORT=9080 \
    ANTHROPIC_SHIM_UPSTREAM_URL=http://127.0.0.1:8080

EXPOSE 9080

CMD ["python", "-m", "anthropic_shim"]
