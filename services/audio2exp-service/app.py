"""
Audio2Expression マイクロサービス

gourmet-support バックエンドから呼び出される A2E 推論サービス。
MP3音声を受け取り、52次元ARKitブレンドシェイプ係数を返す。

アーキテクチャ:
    MP3 audio (base64) → PCM 16kHz → Wav2Vec2 → A2E Decoder → 52-dim ARKit blendshapes

エンドポイント:
    POST /api/audio2expression
    GET  /health

環境変数:
    MODEL_DIR: モデルディレクトリ (default: ./models)
    PORT: サーバーポート (default: 8081)
    DEVICE: cpu or cuda (default: auto)
"""

import os
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from a2e_engine import Audio2ExpressionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# A2Eエンジンの初期化
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DEVICE = os.getenv("DEVICE", "auto")

logger.info(f"[Audio2Exp] Initializing engine: model_dir={MODEL_DIR}, device={DEVICE}")
engine = Audio2ExpressionEngine(model_dir=MODEL_DIR, device=DEVICE)
logger.info("[Audio2Exp] Engine initialized successfully")


@app.route('/api/audio2expression', methods=['POST'])
def audio2expression():
    """
    音声から表情係数を生成

    Request JSON:
        {
            "audio_base64": "...",       # base64エンコードされた音声データ
            "session_id": "...",         # セッションID (ログ用)
            "is_start": true,            # ストリームの開始フラグ
            "is_final": true,            # ストリームの終了フラグ
            "audio_format": "mp3"        # 音声フォーマット (mp3, wav, pcm)
        }

    Response JSON:
        {
            "names": ["eyeBlinkLeft", ...],  # 52個のARKitブレンドシェイプ名
            "frames": [[0.0, ...], ...],     # フレームごとの52次元係数
            "frame_rate": 30                  # フレームレート (fps)
        }
    """
    try:
        data = request.json
        audio_base64 = data.get('audio_base64', '')
        session_id = data.get('session_id', 'unknown')
        audio_format = data.get('audio_format', 'mp3')

        if not audio_base64:
            return jsonify({'error': 'audio_base64 is required'}), 400

        logger.info(f"[Audio2Exp] Processing: session={session_id}, "
                    f"format={audio_format}, size={len(audio_base64)} bytes")

        t0 = time.time()
        result = engine.process(audio_base64, audio_format=audio_format)
        elapsed = time.time() - t0

        frame_count = len(result.get('frames', []))
        logger.info(f"[Audio2Exp] Done: {frame_count} frames in {elapsed:.2f}s, "
                    f"session={session_id}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"[Audio2Exp] Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'engine_ready': engine.is_ready(),
        'mode': engine.get_mode(),
        'device': engine.device_name,
        'model_dir': MODEL_DIR
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    logger.info(f"[Audio2Exp] Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, load_dotenv=False)
