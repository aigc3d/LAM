"""
プラットフォーム FastAPI サーバー

エントリーポイント。WebSocket (Live API中継) と REST エンドポイントを提供する。

エンドポイント:
  POST /api/v2/session/start    - セッション開始
  POST /api/v2/session/end      - セッション終了
  WS   /api/v2/live/{session_id} - Live API WebSocket 中継
  GET  /api/v2/modes            - 利用可能モード一覧
  GET  /api/v2/health           - ヘルスチェック
"""

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lam_platform.config.settings import HOST, PORT, CORS_ORIGINS, A2E_SERVICE_URL
from lam_platform.modes.registry import ModeRegistry
from lam_platform.modes.gourmet.plugin import GourmetModePlugin
from lam_platform.services.a2e_client import A2EClient
from lam_platform.session.manager import SessionManager
from lam_platform.live.relay import LiveRelay

logger = logging.getLogger(__name__)

# --- グローバルインスタンス ---
mode_registry = ModeRegistry()
session_manager = SessionManager()
a2e_client: A2EClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション起動・終了処理"""
    global a2e_client

    # --- 起動 ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # モードプラグイン登録
    mode_registry.register(GourmetModePlugin())
    logger.info(f"[Server] Modes registered: {mode_registry.list_modes()}")

    # A2E クライアント初期化
    if A2E_SERVICE_URL and not A2E_SERVICE_URL.endswith("XXXXX.run.app"):
        a2e_client = A2EClient()
        health = await a2e_client.health_check()
        if health:
            logger.info(f"[Server] A2E service healthy: {health}")
        else:
            logger.warning("[Server] A2E service unreachable (avatar expressions disabled)")
    else:
        logger.info("[Server] A2E service not configured (avatar expressions disabled)")

    logger.info(f"[Server] Platform ready on {HOST}:{PORT}")

    yield

    # --- 終了 ---
    if a2e_client:
        await a2e_client.close()
    logger.info("[Server] Shutdown complete")


app = FastAPI(
    title="LAM Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === リクエスト/レスポンスモデル ===

class SessionStartRequest(BaseModel):
    mode: str = "gourmet"
    language: str = "ja"
    dialogue_type: str = "live"
    user_id: str | None = None


class SessionStartResponse(BaseModel):
    session_id: str
    mode: str
    language: str
    dialogue_type: str
    greeting: str
    ws_url: str


class SessionEndResponse(BaseModel):
    session_id: str
    ended: bool


class HealthResponse(BaseModel):
    status: str
    modes: list[dict]
    a2e_available: bool
    active_sessions: int


# === REST エンドポイント ===

@app.post("/api/v2/session/start", response_model=SessionStartResponse)
async def start_session(req: SessionStartRequest):
    """
    セッション開始

    モードプラグインを検証し、新しいセッションを作成。
    初回挨拶メッセージとWebSocket URLを返す。
    """
    # モード検証
    plugin = mode_registry.get(req.mode)
    if not plugin:
        available = [m["name"] for m in mode_registry.list_modes()]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode: '{req.mode}'. Available: {available}",
        )

    # dialogue_type はモードのデフォルトを尊重
    dialogue_type = req.dialogue_type or plugin.default_dialogue_type

    # セッション作成
    session = session_manager.create_session(
        mode=req.mode,
        language=req.language,
        dialogue_type=dialogue_type,
        user_id=req.user_id,
    )

    # 初回挨拶
    greeting = plugin.get_initial_greeting(language=req.language)

    return SessionStartResponse(
        session_id=session.session_id,
        mode=req.mode,
        language=req.language,
        dialogue_type=dialogue_type,
        greeting=greeting,
        ws_url=f"/api/v2/live/{session.session_id}",
    )


@app.post("/api/v2/session/end", response_model=SessionEndResponse)
async def end_session(session_id: str):
    """セッション終了"""
    ended = session_manager.end_session(session_id)
    if not ended:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return SessionEndResponse(session_id=session_id, ended=True)


@app.get("/api/v2/modes")
async def list_modes():
    """利用可能モード一覧"""
    return {"modes": mode_registry.list_modes()}


@app.get("/api/v2/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェック"""
    a2e_available = False
    if a2e_client:
        result = await a2e_client.health_check()
        a2e_available = result is not None

    return HealthResponse(
        status="healthy",
        modes=mode_registry.list_modes(),
        a2e_available=a2e_available,
        active_sessions=len(session_manager.list_sessions()),
    )


# === WebSocket エンドポイント ===

@app.websocket("/api/v2/live/{session_id}")
async def live_websocket(websocket: WebSocket, session_id: str):
    """
    Live API WebSocket 中継

    ブラウザ ↔ サーバー ↔ Gemini Live API の3者間を中継する。
    セッション開始後、クライアントはこのエンドポイントに WebSocket 接続する。
    """
    # セッション検証
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason=f"Session not found: {session_id}")
        return

    # モードプラグイン取得
    plugin = mode_registry.get(session.mode)
    if not plugin:
        await websocket.close(code=4005, reason=f"Mode not found: {session.mode}")
        return

    # LiveRelay 生成・実行
    relay = LiveRelay(
        session=session,
        mode_plugin=plugin,
        a2e_client=a2e_client,
    )

    logger.info(
        f"[Server] WebSocket /api/v2/live/{session_id} "
        f"mode={session.mode} lang={session.language}"
    )

    await relay.handle_client_ws(websocket)


# === エントリーポイント ===

def main():
    """uvicorn で起動"""
    import uvicorn

    uvicorn.run(
        "lam_platform.server:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
