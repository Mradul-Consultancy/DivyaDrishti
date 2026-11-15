# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

WORKDIR /app

# Run as a non-root user: the service ingests uploads and writes audit logs.
RUN useradd --create-home --uid 10001 appuser

# Install system dependencies (OpenCV, libGL, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure the writable directories exist and belong to the service user.
RUN mkdir -p models/checkpoints reports data logs \
    && chown -R appuser:appuser /app
USER appuser

# Expose ports for both Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Default command runs the Streamlit UI; override to run FastAPI if needed
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
