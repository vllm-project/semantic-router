FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./  

EXPOSE 8000

# Environment variables for configuration
ENV MODEL=Qwen/Qwen2-0.5B-Instruct
ENV SERVED_MODEL_NAME=Qwen/Qwen2-0.5B-Instruct
ENV LLM_KATAN_URL=http://localhost:8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
