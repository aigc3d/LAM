FROM python:3.11-slim

# ffmpeg (pydub dependency), libsndfile (librosa dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# モデルディレクトリ (ボリュームマウントまたはビルド時にコピー)
# models/ に以下を配置:
#   - LAM_audio2exp_streaming.tar (A2Eチェックポイント)
#   - wav2vec2-base-960h/ (Wav2Vec2モデル)
RUN mkdir -p /app/models

# INFER ログ出力先
RUN mkdir -p /tmp/audio2exp_logs/model

ENV PORT=8081
ENV MODEL_DIR=/app/models
ENV DEVICE=cpu

EXPOSE 8081

CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--timeout", "120", "--workers", "1", "--threads", "4", "app:app"]
