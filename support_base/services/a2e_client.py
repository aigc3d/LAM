"""
audio2exp-service クライアント

audio2exp-service (Cloud Run) への HTTP リクエスト。
音声データを送信し、52次元ARKitブレンドシェイプ係数を取得する。
REST経路とLive API経路の両方で使用。
"""

import logging
from dataclasses import dataclass

import httpx

from support_base.config.settings import A2E_SERVICE_URL, A2E_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class A2EResult:
    """A2E推論結果"""
    names: list[str]           # 52個のARKit名
    frames: list[list[float]]  # N×52
    frame_rate: int            # 通常30


class A2EClient:
    """
    audio2exp-service クライアント

    確認済みAPI仕様 (a2e_engine.py L381-401):
      POST /api/audio2expression
      Request:  { audio_base64, session_id, audio_format }
      Response: { names: [52], frames: [N][52], frame_rate: 30 }
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or A2E_SERVICE_URL).rstrip("/")
        self._client = httpx.AsyncClient(timeout=A2E_TIMEOUT_SECONDS)

    async def process_audio(
        self,
        audio_base64: str,
        session_id: str = "unknown",
        audio_format: str = "mp3",
    ) -> A2EResult | None:
        """
        音声 → 52次元ARKitブレンドシェイプ

        Args:
            audio_base64: base64エンコードされた音声データ
            session_id: セッションID（ログ用）
            audio_format: 音声フォーマット (mp3, wav, pcm)

        Returns:
            A2EResult or None (エラー時)
        """
        url = f"{self.base_url}/api/audio2expression"
        payload = {
            "audio_base64": audio_base64,
            "session_id": session_id,
            "audio_format": audio_format,
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            result = A2EResult(
                names=data["names"],
                frames=data["frames"],
                frame_rate=data.get("frame_rate", 30),
            )

            frame_count = len(result.frames)
            logger.info(
                f"[A2E] OK: {frame_count} frames, "
                f"session={session_id}"
            )
            return result

        except httpx.TimeoutException:
            logger.warning(f"[A2E] Timeout: session={session_id}")
            return None
        except Exception as e:
            logger.error(f"[A2E] Error: {e}, session={session_id}")
            return None

    async def health_check(self) -> dict | None:
        """ヘルスチェック (GET /health)"""
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            logger.error(f"[A2E] Health check failed: {e}")
            return None

    async def close(self):
        """HTTPクライアントのクローズ"""
        await self._client.aclose()
