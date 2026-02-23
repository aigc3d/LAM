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

# wav2vec2-base-960h をビルド時にダウンロードし、標準形式で保存
# cache_dir ではなく save_pretrained を使用 (config.json が直下に必要)
RUN python -c "\
from transformers import Wav2Vec2Model, Wav2Vec2Processor; \
p = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'); \
m = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h'); \
p.save_pretrained('/app/models/wav2vec2-base-960h'); \
m.save_pretrained('/app/models/wav2vec2-base-960h'); \
print('wav2vec2-base-960h saved to /app/models/wav2vec2-base-960h')"

# INFER ログ出力先
RUN mkdir -p /tmp/audio2exp_logs/model

ENV PORT=8080
ENV MODEL_DIR=/app/models
ENV DEVICE=cpu
# ランタイムで HuggingFace へのアクセスを無効化 (ハング防止)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080

# --timeout 300: モデルロードに十分な時間を確保
# --threads 1: signal 関連の thread-safety 問題を回避
CMD gunicorn --bind "0.0.0.0:${PORT}" --timeout 300 --workers 1 --threads 1 app:app
