FROM python:3.11-slim

# ffmpeg (pydub dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# モデルディレクトリ (ボリュームマウントまたはビルド時にコピー)
RUN mkdir -p /app/models

ENV PORT=8081
ENV MODEL_DIR=/app/models
ENV DEVICE=cpu

EXPOSE 8081

CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--timeout", "120", "--workers", "1", "--threads", "4", "app:app"]
