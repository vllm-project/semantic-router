FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Uncomment to Configure pip to use a China mirror for faster installs
# RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
#     python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
