FROM python:3.11-slim

# ffmpeg (pydub dependency), libsndfile (librosa dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch + torchaudio (CUDA不要、イメージ ~700MB 軽量化、import 高速化)
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# INFER ログ出力先
RUN mkdir -p /tmp/audio2exp_logs/model

ENV PORT=8080
ENV MODEL_DIR=/app/models
ENV DEVICE=cpu

EXPOSE 8080

# Shell form so env vars are expanded at runtime (Cloud Run injects PORT=8080)
# GUNICORN_TIMEOUT: リクエスト処理タイムアウト (エンジンロードはバックグラウンド)
CMD gunicorn --bind "0.0.0.0:${PORT}" --timeout ${GUNICORN_TIMEOUT:-300} --workers 1 --threads 4 app:app
