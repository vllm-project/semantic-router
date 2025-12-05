# LLM Katan Dockerfile
# Lightweight LLM Server for Testing
FROM python:3.11-slim

LABEL maintainer="vLLM Semantic Router Team"
LABEL description="LLM Katan - Lightweight LLM Server for Testing"
LABEL version="0.1.8"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt ./
# Install PyTorch CPU-only version to save space (no CUDA for testing server)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the llm_katan package
COPY llm_katan/ ./llm_katan/
COPY pyproject.toml ./
COPY README.md ./

# Install the package in development mode
RUN pip install -e .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash llmkatan
USER llmkatan

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the default port
EXPOSE 8000

# Default command - can be overridden
CMD ["llm-katan", "--model", "Qwen/Qwen3-0.6B", "--host", "0.0.0.0", "--port", "8000"]
