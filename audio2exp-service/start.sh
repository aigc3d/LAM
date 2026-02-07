#!/bin/bash

# Model directory
MODEL_DIR="/app/models"
mkdir -p "$MODEL_DIR"

# GCS bucket path (set via environment variable)
GCS_BUCKET="${GCS_MODEL_BUCKET:-gs://hp-support-477512-models/audio2exp}"

# Function to download models
download_models() {
    echo "[Download] Starting model download from $GCS_BUCKET ..."

    # Download wav2vec2 model if not exists
    if [ ! -d "$MODEL_DIR/wav2vec2-base-960h" ]; then
        echo "[Download] Downloading wav2vec2-base-960h from GCS..."
        if gsutil -m cp -r "$GCS_BUCKET/wav2vec2-base-960h" "$MODEL_DIR/" 2>&1; then
            echo "[Download] wav2vec2-base-960h downloaded successfully"
        else
            echo "[Download] ERROR: Failed to download wav2vec2-base-960h"
            return 1
        fi
    else
        echo "[Download] wav2vec2-base-960h already exists"
    fi

    # Download LAM model weights if not exists
    MODEL_FILE="$MODEL_DIR/lam_audio2exp_streaming.tar"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "[Download] Downloading lam_audio2exp_streaming.tar from GCS..."
        if gsutil cp "$GCS_BUCKET/lam_audio2exp_streaming.tar" "$MODEL_FILE" 2>&1; then
            echo "[Download] lam_audio2exp_streaming.tar downloaded successfully"
        else
            echo "[Download] ERROR: Failed to download lam_audio2exp_streaming.tar"
            return 1
        fi

        # Check if file is gzip compressed and decompress if needed
        if file "$MODEL_FILE" | grep -q "gzip compressed"; then
            echo "[Download] File is gzip compressed, decompressing..."
            gunzip -c "$MODEL_FILE" > "$MODEL_FILE.tmp"
            mv "$MODEL_FILE.tmp" "$MODEL_FILE"
            echo "[Download] Decompression complete"
        fi
    else
        echo "[Download] lam_audio2exp_streaming.tar already exists"
    fi

    # Verify model file integrity
    if [ -f "$MODEL_FILE" ]; then
        FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE")
        echo "[Download] Model file size: $FILE_SIZE bytes"
        if [ "$FILE_SIZE" -lt 400000000 ]; then
            echo "[Download] WARNING: Model file too small!"
        else
            echo "[Download] Model file size OK"
        fi
    fi

    # Signal that models are ready
    touch "$MODEL_DIR/.models_ready"
    echo "[Download] Models ready, signaling app to initialize..."
}

echo "[Startup] Starting server immediately..."

# Start uvicorn in background first
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} &
UVICORN_PID=$!

echo "[Startup] Server started (PID: $UVICORN_PID), now downloading models..."

# Download models (this will log to stdout which Cloud Run captures)
download_models

echo "[Startup] Download complete, server running..."

# Wait for uvicorn to exit
wait $UVICORN_PID
