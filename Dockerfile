# Use a slim Python base
FROM python:3.11-slim

# Metadata
LABEL maintainer="neerajmehta@test.com"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create app user and working dir
RUN adduser --disabled-password --gecos "" appuser
WORKDIR /app

# Install system deps needed to build some wheels (like xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Fix permissions
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
