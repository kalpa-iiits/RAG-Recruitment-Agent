FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is fully pinned (generated from uv.lock), so pip does no
# dependency backtracking — keeps memory low on small instances.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/health || exit 1

# Single worker: per-browser sessions (and their FAISS indexes) live in memory.
ENTRYPOINT ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8501", "--workers=1", "--proxy-headers", "--forwarded-allow-ips=*"]
