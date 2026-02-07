#!/bin/bash
set -e

# Download models from GCS if not already present
MODEL_DIR="/app/models"
mkdir -p "$MODEL_DIR"

# GCS bucket path (set via environment variable)
GCS_BUCKET="${GCS_MODEL_BUCKET:-gs://hp-support-477512-models/audio2exp}"

# Download wav2vec2 model if not exists
if [ ! -d "$MODEL_DIR/wav2vec2-base-960h" ]; then
    echo "[Startup] Downloading wav2vec2-base-960h from GCS..."
    gsutil -m cp -r "$GCS_BUCKET/wav2vec2-base-960h" "$MODEL_DIR/"
    echo "[Startup] wav2vec2-base-960h downloaded"
fi

# Download LAM model weights if not exists
if [ ! -f "$MODEL_DIR/lam_audio2exp_streaming.tar" ]; then
    echo "[Startup] Downloading lam_audio2exp_streaming.tar from GCS..."
    gsutil cp "$GCS_BUCKET/lam_audio2exp_streaming.tar" "$MODEL_DIR/lam_audio2exp_streaming.tar"
    echo "[Startup] lam_audio2exp_streaming.tar downloaded"
fi

# Verify model file integrity (expected size: ~390MB)
MODEL_FILE="$MODEL_DIR/lam_audio2exp_streaming.tar"
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE")
    echo "[Startup] Model file size: $FILE_SIZE bytes"
    # Warn if file seems too small (less than 350MB = 367001600 bytes)
    if [ "$FILE_SIZE" -lt 367001600 ]; then
        echo "[WARNING] Model file seems too small! Expected ~390MB, got $(echo "scale=1; $FILE_SIZE/1048576" | bc)MB"
        echo "[WARNING] The model file may be corrupted. Re-upload with fix_gcs_model.sh"
    fi
fi

echo "[Startup] Models ready, starting server..."

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
