"""
セッション管理

セッションのライフサイクル管理。モード割当、短期記憶保持、接続状態追跡。
"""

import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field

from lam_platform.memory.session_memory import SessionMemory

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """1つの対話セッションを表現"""

    session_id: str
    mode: str                          # "gourmet", "support", "interview"
    language: str                      # "ja", "en", "ko", "zh"
    dialogue_type: str = "live"        # "rest", "live", "hybrid"
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    memory: SessionMemory = field(default_factory=SessionMemory)
    live_session_count: int = 0        # Live API の再接続回数


class SessionManager:
    """セッション管理"""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(
        self,
        mode: str,
        language: str = "ja",
        dialogue_type: str = "live",
        user_id: str | None = None,
    ) -> Session:
        """新しいセッションを作成"""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = Session(
            session_id=session_id,
            mode=mode,
            language=language,
            dialogue_type=dialogue_type,
            user_id=user_id,
        )
        self._sessions[session_id] = session
        logger.info(
            f"[Session] Created: {session_id}, mode={mode}, "
            f"lang={language}, type={dialogue_type}"
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        """セッションを取得"""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """セッションを終了"""
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info(f"[Session] Ended: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """アクティブなセッションIDのリスト"""
        return list(self._sessions.keys())
