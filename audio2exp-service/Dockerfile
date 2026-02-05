FROM python:3.10-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy LAM_Audio2Expression code
COPY LAM_Audio2Expression/ ./LAM_Audio2Expression/

# Copy model files
COPY models/ ./models/

# Copy application code
COPY app.py .

# Environment variables for model paths
ENV LAM_A2E_PATH=/app/LAM_Audio2Expression
ENV LAM_WEIGHT_PATH=/app/models/lam_audio2exp_streaming.tar
ENV WAV2VEC_PATH=/app/models/wav2vec2-base-960h

# Cloud Run uses PORT environment variable
ENV PORT=8080

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
