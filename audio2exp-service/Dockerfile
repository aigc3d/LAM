FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy LAM_Audio2Expression code
COPY LAM_Audio2Expression/ ./LAM_Audio2Expression/

# Copy models directory (downloaded during Cloud Build, fallback for FUSE)
COPY models/ ./models/

# Copy application code
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# Fixed paths - models served via GCS FUSE mount (primary), Docker-baked (fallback)
ENV LAM_A2E_PATH=/app/LAM_Audio2Expression
ENV LAM_WEIGHT_PATH=/app/models/lam_audio2exp_streaming.pth
ENV WAV2VEC_PATH=/app/models/wav2vec2-base-960h

ENV PORT=8080

CMD ["./start.sh"]
