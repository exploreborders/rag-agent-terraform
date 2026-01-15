# Multi-stage Docker build for RAG Agent Application

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies with caching
COPY ../src/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install prometheus-client==0.19.0 starlette==0.27.0 starlette-exporter==0.15.0

# Production stage
FROM python:3.11-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # PostgreSQL client
    postgresql-client \
    # Image processing dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # PDF processing
    poppler-utils \
    # Image libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # System utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    # Ollama configuration (will be overridden by Docker networking)
    OLLAMA_BASE_URL="http://host.docker.internal:11434"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app/data/uploads /app/data/processed /app/logs && \
    chown -R raguser:raguser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY ../src/ /app/

# Change ownership of application files
RUN chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]