# Syntax: docker/dockerfile:1
FROM python:3.10-slim

LABEL maintainer="Adão de Jesus Braga <braganetx@gmail.com>"
LABEL description="Predictive Digital Twin with Hardware-Aware Multi-Objective Optimization for Building Energy Efficiency"
LABEL paper.doi="10.2139/ssrn.7645518"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/outputs /tmp/matplotlib && \
    chown -R appuser:appuser /app /tmp/matplotlib

# Copy project source
COPY --chown=appuser:appuser . .

USER appuser

# Default execution: run the complete closed-loop digital twin simulation
CMD ["python", "main.py"]
