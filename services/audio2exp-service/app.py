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
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# A2Eエンジンの遅延初期化
# gunicorn が即座にポートをバインドできるよう、モデルロードはバックグラウンドで実行
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DEVICE = os.getenv("DEVICE", "auto")

engine = None
_engine_error = None
_engine_lock = threading.Lock()


def _load_engine():
    """バックグラウンドスレッドでエンジンをロード"""
    global engine, _engine_error
    try:
        from a2e_engine import Audio2ExpressionEngine
        logger.info(f"[Audio2Exp] Loading engine: model_dir={MODEL_DIR}, device={DEVICE}")
        t0 = time.time()
        eng = Audio2ExpressionEngine(model_dir=MODEL_DIR, device=DEVICE)
        elapsed = time.time() - t0
        with _engine_lock:
            engine = eng
        logger.info(f"[Audio2Exp] Engine ready in {elapsed:.1f}s")
    except Exception as e:
        with _engine_lock:
            _engine_error = str(e)
        logger.error(f"[Audio2Exp] Engine failed to load: {e}", exc_info=True)


_loader_thread = threading.Thread(target=_load_engine, daemon=True)
_loader_thread.start()
logger.info("[Audio2Exp] Server started, engine loading in background...")


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
    if engine is None:
        msg = _engine_error or 'Engine is still loading, please retry shortly'
        status = 500 if _engine_error else 503
        return jsonify({'error': msg}), status

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
    """ヘルスチェック - エンジンロード中でも200を返す（Cloud Run起動判定用）"""
    if engine is None:
        return jsonify({
            'status': 'loading',
            'engine_ready': False,
            'error': _engine_error,
            'model_dir': MODEL_DIR
        })
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
