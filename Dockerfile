# Multi-Agent Autonomous Problem Solving System
# Dockerfile for production deployment

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    python-multipart==0.0.6 \
    aiohttp==3.9.1 \
    aiofiles==23.2.1 \
    redis==5.0.1 \
    sqlalchemy==2.0.25 \
    asyncpg==0.29.0 \
    alembic==1.13.1 \
    prometheus-client==0.19.0 \
    structlog==24.1.0 \
    python-json-logger==2.0.7 \
    sentry-sdk==1.39.1 \
    opentelemetry-api==1.22.0 \
    opentelemetry-sdk==1.22.0 \
    opentelemetry-exporter-otlp==1.22.0 \
    opentelemetry-instrumentation-fastapi==0.42b0 \
    opentelemetry-instrumentation-aiohttp==0.42b0 \
    numpy==1.26.3 \
    scipy==1.12.0 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    click==8.1.7 \
    rich==13.7.0 \
    tqdm==4.66.1 \
    tenacity==8.2.3 \
    cachetools==5.3.3 \
    xxhash==3.4.1 \
    lz4==4.3.3 \
    zstandard==0.22.0 \
    msgpack==1.0.8 \
    orjson==3.9.10

# Install ML dependencies
RUN pip install \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    sentence-transformers==2.2.2 \
    huggingface-hub==0.20.3 \
    tokenizers==0.15.0 \
    safetensors==0.4.2 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    auto-gptq==0.5.1 \
    awq==0.1.0

# Install vector search dependencies
RUN pip install \
    faiss-cpu==1.7.4 \
    hnswlib==0.7.0 \
    annoy==1.17.3

# Install data processing
RUN pip install \
    beautifulsoup4==4.12.2 \
    lxml==4.9.3 \
    feedparser==6.0.10 \
    requests==2.31.0 \
    httpx==0.26.0

# Install monitoring
RUN pip install \
    psutil==5.9.6 \
    GPUtil==1.4.0 \
    pynvml==11.5.0

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py ./

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/cache /app/logs /app/checkpoints && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose ports
EXPOSE 8000 8080 8081 8082 8083 8084 8085 8086 8087 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "--mode", "api"]