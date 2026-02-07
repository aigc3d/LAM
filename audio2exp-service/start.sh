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
MODEL_FILE="$MODEL_DIR/lam_audio2exp_streaming.tar"
if [ ! -f "$MODEL_FILE" ]; then
    echo "[Startup] Downloading lam_audio2exp_streaming.tar from GCS..."
    gsutil cp "$GCS_BUCKET/lam_audio2exp_streaming.tar" "$MODEL_FILE"
    echo "[Startup] lam_audio2exp_streaming.tar downloaded"

    # Check if file is gzip compressed and decompress if needed
    # HuggingFace/OSS provides gzipped file (~356MB) that needs decompression to ~390MB
    if file "$MODEL_FILE" | grep -q "gzip compressed"; then
        echo "[Startup] File is gzip compressed, decompressing..."
        gunzip -c "$MODEL_FILE" > "$MODEL_FILE.tmp"
        mv "$MODEL_FILE.tmp" "$MODEL_FILE"
        echo "[Startup] Decompression complete"
    fi
fi

# Verify model file integrity (expected size: ~390MB after decompression)
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE")
    echo "[Startup] Model file size: $FILE_SIZE bytes"
    # After decompression, file should be ~390MB (408538564 bytes)
    if [ "$FILE_SIZE" -lt 400000000 ]; then
        echo "[WARNING] Model file too small! Expected ~390MB after decompression"
        echo "[WARNING] File may still be compressed or corrupted"
    else
        echo "[Startup] Model file size OK"
    fi
fi

echo "[Startup] Models ready, starting server..."

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
