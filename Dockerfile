FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt

# 1) Upgrade pip
RUN pip install --upgrade pip

# 2) Install torch CPU from official PyTorch index (smaller + no CUDA)
RUN pip install --retries 10 --timeout 300 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

# 3) Install the rest
RUN pip install --retries 10 --timeout 300 -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/start.sh

ENV PORT=7860
EXPOSE 7860

CMD ["bash", "/app/start.sh"]
