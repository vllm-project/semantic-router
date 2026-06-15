FROM python:3.12-slim
ARG GIT_SSL_NO_VERIFY=0

WORKDIR /app

COPY src/fleet-sim/ /app/

RUN if [ "$GIT_SSL_NO_VERIFY" = "1" ]; then PIP_EXTRA="--trusted-host pypi.org --trusted-host files.pythonhosted.org"; else PIP_EXTRA=""; fi && \
    python -m pip install --upgrade pip && \
    python -m pip install $PIP_EXTRA --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["vllm-sr-sim", "serve", "--host", "0.0.0.0", "--port", "8000"]
