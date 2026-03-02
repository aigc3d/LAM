"""
プラットフォーム設定

環境変数から読み込み。.env ファイルも対応。
"""

import os

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LIVE_API_MODEL = os.getenv("LIVE_API_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
REST_API_MODEL = os.getenv("REST_API_MODEL", "gemini-2.5-flash")

# Audio
SEND_SAMPLE_RATE = 16000    # マイク入力 (PCM 16kHz)
RECEIVE_SAMPLE_RATE = 24000  # Live API 出力 (PCM 24kHz)

# Live API 再接続設定 (stt_stream.py L372-373 から移植)
MAX_AI_CHARS_BEFORE_RECONNECT = 800
LONG_SPEECH_THRESHOLD = 500
RECONNECT_DELAY_SECONDS = 3

# audio2exp-service
A2E_SERVICE_URL = os.getenv("A2E_SERVICE_URL", "https://audio2exp-service-XXXXX.run.app")
A2E_TIMEOUT_SECONDS = 10

# Google Cloud TTS
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# 既存 gourmet-support (Phase 1 プロキシ用)
LEGACY_BACKEND_URL = os.getenv("LEGACY_BACKEND_URL", "")
