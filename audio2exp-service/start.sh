#!/bin/bash
set -e

echo "[Startup] Starting Audio2Expression service..."
echo "[Startup] Checking FUSE mount contents:"
ls -l /mnt/models/audio2exp/ || echo "[Startup] WARNING: FUSE mount not available"

exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
