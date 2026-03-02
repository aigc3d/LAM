"""
Live API 再接続管理

stt_stream.py L372-373, L624-643, L714-796 から移植。
Gemini FLASH版の累積トークン制限を回避するための自動再接続ロジック。
"""

import logging

from lam_platform.live.speech_detector import SpeechDetector
from lam_platform.config.settings import (
    MAX_AI_CHARS_BEFORE_RECONNECT,
    LONG_SPEECH_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ReconnectManager:
    """
    累積文字数制限の回避ロジック

    FLASH版 (gemini-2.5-flash-native-audio-preview) には
    セッション内の累積入出力トークンに制限がある。
    これを超えるとAPIエラー(1011/1008)が発生しセッションが切断される。

    回避策:
    1. AI発話の累積文字数を追跡
    2. 閾値に達したら再接続フラグを立てる
    3. 発話途切れを検知したら即時再接続
    4. 再接続時に会話コンテキストを引き継ぐ
    """

    def __init__(
        self,
        max_chars: int = MAX_AI_CHARS_BEFORE_RECONNECT,
        long_speech_threshold: int = LONG_SPEECH_THRESHOLD,
    ):
        self.max_chars = max_chars
        self.long_speech_threshold = long_speech_threshold
        self.ai_char_count = 0
        self.needs_reconnect = False
        self.reconnect_reason: str | None = None
        self.session_count = 0

    def on_ai_speech_complete(self, text: str, language: str = "ja") -> None:
        """
        AI発話完了時に呼び出す。再接続判定を行う。

        stt_stream.py L624-643 のロジックを移植:
        1. 発話途切れ → 即時再接続
        2. 長文発話(500文字超) → 次ターン前に再接続
        3. 累積800文字超 → 再接続
        """
        char_count = len(text)
        self.ai_char_count += char_count
        remaining = self.max_chars - self.ai_char_count

        logger.info(
            f"[Reconnect] chars={char_count}, "
            f"cumulative={self.ai_char_count}, remaining={remaining}"
        )

        # 1. 発話途切れ → 即時再接続
        if SpeechDetector.is_incomplete(text, language):
            self.needs_reconnect = True
            self.reconnect_reason = "incomplete"
            logger.warning("[Reconnect] Speech incomplete, reconnecting")
            return

        # 2. 長文発話 → 次ターン前に再接続
        if char_count >= self.long_speech_threshold:
            self.needs_reconnect = True
            self.reconnect_reason = "long_speech"
            logger.info(
                f"[Reconnect] Long speech ({char_count} chars), reconnecting"
            )
            return

        # 3. 累積上限 → 再接続
        if self.ai_char_count >= self.max_chars:
            self.needs_reconnect = True
            self.reconnect_reason = "char_limit"
            logger.info("[Reconnect] Char limit reached, reconnecting")

    def reset_for_new_session(self) -> None:
        """新セッション開始時にカウンターをリセット"""
        self.ai_char_count = 0
        self.needs_reconnect = False
        self.reconnect_reason = None
        self.session_count += 1
        logger.info(f"[Reconnect] Session #{self.session_count} started")

    @staticmethod
    def is_retriable_error(error: Exception) -> bool:
        """
        再接続可能なエラーか判定

        stt_stream.py L786-796, L902 から移植
        """
        msg = str(error).lower()
        retriable_keywords = [
            "1011", "1008", "internal error", "disconnected",
            "closed", "websocket", "deadline", "policy",
        ]
        return any(kw in msg for kw in retriable_keywords)
