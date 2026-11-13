# GCP Orchestrator Dockerfile (e2-micro optimized, slim)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY gcp/requirements.txt ./gcp/requirements.txt
RUN pip install --upgrade pip && pip install -r gcp/requirements.txt

COPY shared ./shared
COPY gcp ./gcp

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "--timeout", "120", "wsgi:application"]
