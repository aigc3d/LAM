#!/bin/bash
set -e
echo "[Startup] Starting Audio2Expression service..."
echo "[Startup] Checking FUSE mount contents:"
ls -l /mnt/models/audio2exp/ || echo "[Startup] WARNING: FUSE mount not available"
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout ${GUNICORN_TIMEOUT:-300} --workers 1 --threads 4
