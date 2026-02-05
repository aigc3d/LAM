FROM python:3.10-slim

WORKDIR /app

# System dependencies for audio processing and GCS access
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy LAM_Audio2Expression code (required for inference)
COPY LAM_Audio2Expression/ ./LAM_Audio2Expression/

# Copy application code
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# Environment variables for model paths
ENV LAM_A2E_PATH=/app/LAM_Audio2Expression
ENV LAM_WEIGHT_PATH=/app/models/lam_audio2exp_streaming.tar
ENV WAV2VEC_PATH=/app/models/wav2vec2-base-960h

# GCS bucket for models (override with --set-env-vars)
ENV GCS_MODEL_BUCKET=gs://hp-support-477512-models/audio2exp

# Cloud Run uses PORT environment variable
ENV PORT=8080

CMD ["./start.sh"]
