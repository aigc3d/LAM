FROM python:3.11-slim

WORKDIR /app

# 依存パッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコード (support_base パッケージとして配置)
COPY . ./support_base/

ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

# uvicorn で起動（Cloud Run は PORT 環境変数を注入する）
CMD uvicorn support_base.server:app --host 0.0.0.0 --port ${PORT}
