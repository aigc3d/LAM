FROM python:3.11-slim

# ffmpeg (pydub dependency), libsndfile (librosa dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch を先にインストール (GPU版の無駄なダウンロードを回避)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# torch は既にインストール済みなので requirements.txt からは除外してインストール
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# モデルディレクトリ
RUN mkdir -p /app/models

# wav2vec2-base-960h をビルド時にダウンロード (ランタイムでのハングを防止)
RUN python -c "\
from transformers import Wav2Vec2Model, Wav2Vec2Processor; \
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h', cache_dir='/app/models/wav2vec2-base-960h-cache'); \
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h', cache_dir='/app/models/wav2vec2-base-960h-cache'); \
print('wav2vec2-base-960h downloaded successfully')"

# INFER ログ出力先
RUN mkdir -p /tmp/audio2exp_logs/model

ENV PORT=8080
ENV MODEL_DIR=/app/models
ENV DEVICE=cpu
# ランタイムで HuggingFace へのアクセスを無効化 (ハング防止)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080

CMD gunicorn --bind "0.0.0.0:${PORT}" --timeout 120 --workers 1 --threads 4 app:app
