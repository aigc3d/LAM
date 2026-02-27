"""
gourmet-support 連携用コード

このファイルを gourmet-support バックエンドに追加し、
TTS処理後に audio2exp-service へ音声を転送します。

使用方法:
  1. このファイルを gourmet-support/app/ にコピー
  2. TTS処理部分で send_audio_to_expression() を呼び出し
"""

import asyncio
import aiohttp
import os
from typing import Optional

# Audio2Expression Service URL (環境変数で設定)
AUDIO2EXP_SERVICE_URL = os.getenv(
    "AUDIO2EXP_SERVICE_URL",
    "https://audio2exp-service-xxxxx-an.a.run.app"  # Cloud Run URL
)


async def send_audio_to_expression(
    audio_base64: str,
    session_id: str,
    is_final: bool = False
) -> Optional[dict]:
    """
    TTS音声を Audio2Expression サービスに送信

    Args:
        audio_base64: Base64エンコードされた音声データ (PCM 16-bit, サンプルレート任意)
        session_id: セッションID (WebSocket接続と紐付け)
        is_final: 音声ストリームの最終チャンクかどうか

    Returns:
        表情データ (成功時) または None (失敗時)

    Usage:
        # TTS処理後
        tts_audio_base64 = synthesize_tts(text)
        await send_audio_to_expression(tts_audio_base64, session_id)
    """
    if not AUDIO2EXP_SERVICE_URL:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{AUDIO2EXP_SERVICE_URL}/api/audio2expression",
                json={
                    "audio_base64": audio_base64,
                    "session_id": session_id,
                    "is_final": is_final
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"[Audio2Exp] Error: {response.status}")
                    return None
    except Exception as e:
        print(f"[Audio2Exp] Request failed: {e}")
        return None


def send_audio_to_expression_sync(
    audio_base64: str,
    session_id: str,
    is_final: bool = False
) -> Optional[dict]:
    """
    同期版: TTS音声を Audio2Expression サービスに送信
    """
    import requests

    if not AUDIO2EXP_SERVICE_URL:
        return None

    try:
        response = requests.post(
            f"{AUDIO2EXP_SERVICE_URL}/api/audio2expression",
            json={
                "audio_base64": audio_base64,
                "session_id": session_id,
                "is_final": is_final
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[Audio2Exp] Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"[Audio2Exp] Request failed: {e}")
        return None


# ============================================
# 既存の TTS 処理への組み込み例
# ============================================
#
# Before (既存コード):
# ```python
# @app.post("/api/tts/synthesize")
# async def synthesize_tts(request: TTSRequest):
#     audio_base64 = await gcp_tts_synthesize(request.text)
#     return {"success": True, "audio": audio_base64}
# ```
#
# After (変更後):
# ```python
# from gourmet_support_integration import send_audio_to_expression
#
# @app.post("/api/tts/synthesize")
# async def synthesize_tts(request: TTSRequest):
#     audio_base64 = await gcp_tts_synthesize(request.text)
#
#     # Audio2Expression に送信 (非同期、レスポンスを待たない)
#     if request.session_id:
#         asyncio.create_task(
#             send_audio_to_expression(audio_base64, request.session_id)
#         )
#
#     return {"success": True, "audio": audio_base64}
# ```
