#!/bin/bash
set -e

# Model directory
MODEL_DIR="/app/models"
mkdir -p "$MODEL_DIR"

# GCS bucket path (set via environment variable)
GCS_BUCKET="${GCS_MODEL_BUCKET:-gs://hp-support-477512-models/audio2exp}"

# Function to download models in background
download_models() {
    echo "[Background] Starting model download..."

    # Download wav2vec2 model if not exists
    if [ ! -d "$MODEL_DIR/wav2vec2-base-960h" ]; then
        echo "[Background] Downloading wav2vec2-base-960h from GCS..."
        gsutil -m cp -r "$GCS_BUCKET/wav2vec2-base-960h" "$MODEL_DIR/"
        echo "[Background] wav2vec2-base-960h downloaded"
    else
        echo "[Background] wav2vec2-base-960h already exists"
    fi

    # Download LAM model weights if not exists
    MODEL_FILE="$MODEL_DIR/lam_audio2exp_streaming.tar"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "[Background] Downloading lam_audio2exp_streaming.tar from GCS..."
        gsutil cp "$GCS_BUCKET/lam_audio2exp_streaming.tar" "$MODEL_FILE"
        echo "[Background] lam_audio2exp_streaming.tar downloaded"

        # Check if file is gzip compressed and decompress if needed
        if file "$MODEL_FILE" | grep -q "gzip compressed"; then
            echo "[Background] File is gzip compressed, decompressing..."
            gunzip -c "$MODEL_FILE" > "$MODEL_FILE.tmp"
            mv "$MODEL_FILE.tmp" "$MODEL_FILE"
            echo "[Background] Decompression complete"
        fi
    else
        echo "[Background] lam_audio2exp_streaming.tar already exists"
    fi

    # Verify model file integrity
    if [ -f "$MODEL_FILE" ]; then
        FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE")
        echo "[Background] Model file size: $FILE_SIZE bytes"
        if [ "$FILE_SIZE" -lt 400000000 ]; then
            echo "[Background] WARNING: Model file too small!"
        else
            echo "[Background] Model file size OK"
        fi
    fi

    # Signal that models are ready
    touch "$MODEL_DIR/.models_ready"
    echo "[Background] Models ready, signaling app to initialize..."
}

# Start model download in background
download_models &

echo "[Startup] Starting server immediately (models downloading in background)..."

# Start the application immediately - Cloud Run health check will pass
# App will initialize model when models are ready
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
