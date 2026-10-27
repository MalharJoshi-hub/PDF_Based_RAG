# ==============================================================================
# Enterprise PDF RAG Application Dockerfile
# Multi-stage optimized production container with PyTorch, FAISS & Flask
# ==============================================================================

FROM python:3.11-slim as base

# Prevent Python from writing .pyc files & enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install essential system dependencies for FAISS, SQLite, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-cov

# Copy application source code and assets
COPY . .

# Ensure data and static directories exist
RUN mkdir -p /app/data /app/static/eda /app/data/uploads

EXPOSE 5000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Launch application server
CMD ["python", "app.py"]
