# ECU Agent Docker Image
# Simple MLflow model serving container

FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy and install local me_ecu_agent_mlflow package
COPY src/ /app/src/
COPY setup.py /app/
RUN pip install -e .

# Copy the exported model
COPY model_export/ /app/model/

# Expose port
EXPOSE 8080

# Set environment variables
ENV MODEL_PATH=/app/model
ENV HOST=0.0.0.0
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start MLflow model serving
CMD mlflow models serve \
    -m ${MODEL_PATH} \
    -h ${HOST} \
    -p ${PORT} \
    --no-conda