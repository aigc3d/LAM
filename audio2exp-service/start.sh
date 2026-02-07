#!/bin/bash

# Simple startup script - model download is handled by Python app
# This ensures proper logging and authentication in Cloud Run

echo "[Startup] Starting Audio2Expression service..."
echo "[Startup] Model download will be handled by Python's google-cloud-storage client"

# Start uvicorn directly
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
