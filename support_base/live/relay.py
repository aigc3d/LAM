"""
Live API WebSocket 中継 (LiveRelay)

stt_stream.py の GeminiLiveApp を Web 向けに再構成。
ブラウザ ↔ サーバー ↔ Gemini Live API の WebSocket 中継を行う。

主要な責務:
1. ブラウザからの PCM 16kHz 音声を Gemini に中継
2. Gemini からの PCM 24kHz 音声をブラウザに中継
3. 累積文字数制限の回避（自動再接続 + コンテキスト引き継ぎ）
4. AI音声を A2E サービスに送信し、Expression をブラウザに中継（アバター連携）
5. 割り込み（barge-in）処理
6. transcription のブラウザ中継

プロトコル (クライアント ↔ サーバー WebSocket):
  クライアント → サーバー:
    { "type": "audio", "data": "<base64 PCM 16kHz>" }
    { "type": "text",  "data": "テキスト入力" }
    { "type": "stop" }

  サーバー → クライアント:
    { "type": "audio",         "data": "<base64 PCM 24kHz>" }
    { "type": "transcription", "role": "user"|"ai", "text": "..." }
    { "type": "expression",    "data": { names, frames, frame_rate } }
    { "type": "interrupted" }
    { "type": "reconnecting",  "reason": "..." }
    { "type": "reconnected",   "session_count": N }
    { "type": "error",         "message": "..." }
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from fastapi import WebSocket, WebSocketDisconnect

from support_base.config.settings import (
    GEMINI_API_KEY,
    LIVE_API_MODEL,
    RECONNECT_DELAY_SECONDS,
)
from support_base.i18n.language_config import get_language_profile
from support_base.live.reconnect import ReconnectManager
from support_base.modes.base_mode import BaseModePlugin
from support_base.services.a2e_client import A2EClient
from support_base.session.manager import Session

logger = logging.getLogger(__name__)


@dataclass
class LiveRelayState:
    """LiveRelay の内部状態"""
    user_transcript_buffer: str = ""
    ai_transcript_buffer: str = ""
    ai_audio_buffer: bytearray = field(default_factory=bytearray)
    is_running: bool = True


class LiveRelay:
    """
    Gemini Live API WebSocket 中継

    stt_stream.py の GeminiLiveApp.run() + _session_loop() +
    receive_audio() を Web 向けに再構成。
    """

    def __init__(
        self,
        session: Session,
        mode_plugin: BaseModePlugin,
        a2e_client: A2EClient | None = None,
    ):
        self.session = session
        self.mode_plugin = mode_plugin
        self.a2e_client = a2e_client
        self.reconnect_mgr = ReconnectManager()
        self.state = LiveRelayState()
        self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    async def handle_client_ws(self, websocket: WebSocket) -> None:
        """
        クライアント WebSocket ハンドラ — メインエントリーポイント

        stt_stream.py L714-796 の run() に相当。
        再接続ループを管理する。
        """
        await websocket.accept()
        logger.info(f"[LiveRelay] Client connected: session={self.session.session_id}")

        try:
            while self.state.is_running:
                try:
                    await self._run_gemini_session(websocket)
                    if not self.reconnect_mgr.needs_reconnect:
                        break
                except WebSocketDisconnect:
                    logger.info("[LiveRelay] Client disconnected")
                    break
                except Exception as e:
                    if ReconnectManager.is_retriable_error(e):
                        logger.warning(f"[LiveRelay] Retriable error: {e}")
                        await self._send_json(websocket, {
                            "type": "reconnecting",
                            "reason": "error",
                        })
                        await asyncio.sleep(RECONNECT_DELAY_SECONDS)
                        self.reconnect_mgr.needs_reconnect = True
                        continue
                    logger.error(f"[LiveRelay] Fatal error: {e}", exc_info=True)
                    await self._send_json(websocket, {
                        "type": "error",
                        "message": str(e),
                    })
                    break
        finally:
            logger.info(f"[LiveRelay] Session ended: {self.session.session_id}")

    async def _run_gemini_session(self, client_ws: WebSocket) -> None:
        """
        1つの Gemini セッションを実行

        stt_stream.py L741-783 の1ループ反復に相当。
        """
        # コンテキスト引き継ぎ (stt_stream.py L747-752)
        context = None
        if self.reconnect_mgr.session_count > 0:
            context = self.session.memory.get_context_summary()
            logger.info(
                f"[LiveRelay] Reconnecting with context: "
                f"{context[:80] if context else 'none'}..."
            )

        config = self._build_live_config(context)
        self.reconnect_mgr.reset_for_new_session()
        self.session.live_session_count = self.reconnect_mgr.session_count

        # 状態リセット
        self.state.user_transcript_buffer = ""
        self.state.ai_transcript_buffer = ""
        self.state.ai_audio_buffer = bytearray()

        async with self._gemini_client.aio.live.connect(
            model=LIVE_API_MODEL,
            config=config,
        ) as gemini_session:

            if self.reconnect_mgr.session_count > 1:
                # 再接続通知 (stt_stream.py L766-776)
                try:
                    await gemini_session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text="続きをお願いします")],
                        ),
                        turn_complete=True,
                    )
                    logger.info("[LiveRelay] Reconnection prompt sent")
                except Exception as e:
                    logger.warning(f"[LiveRelay] Reconnection prompt failed: {e}")

                await self._send_json(client_ws, {
                    "type": "reconnected",
                    "session_count": self.reconnect_mgr.session_count,
                })

            # 3つの非同期タスクを並行実行 (stt_stream.py L926-930)
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        self._relay_client_to_gemini(client_ws, gemini_session)
                    )
                    tg.create_task(
                        self._relay_gemini_to_client(gemini_session, client_ws)
                    )
            except* WebSocketDisconnect:
                raise
            except* Exception as eg:
                for e in eg.exceptions:
                    if isinstance(e, WebSocketDisconnect):
                        raise e
                    if ReconnectManager.is_retriable_error(e):
                        self.reconnect_mgr.needs_reconnect = True
                        logger.warning(f"[LiveRelay] Task error (retriable): {e}")
                    else:
                        raise e

    async def _relay_client_to_gemini(
        self, client_ws: WebSocket, gemini_session
    ) -> None:
        """
        ブラウザ → Gemini 中継

        stt_stream.py の listen_audio() + send_audio() に相当。
        ブラウザからは JSON メッセージで音声/テキストを受け取り、Gemini に転送。
        """
        while not self.reconnect_mgr.needs_reconnect:
            try:
                raw = await asyncio.wait_for(
                    client_ws.receive_text(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                self.state.is_running = False
                self.reconnect_mgr.needs_reconnect = True
                raise

            try:
                import json
                msg = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            msg_type = msg.get("type")

            if msg_type == "audio":
                # base64 PCM 16kHz → Gemini
                audio_bytes = base64.b64decode(msg["data"])
                await gemini_session.send_realtime_input(
                    audio={"data": audio_bytes, "mime_type": "audio/pcm"}
                )

            elif msg_type == "text":
                # テキスト入力 → Gemini
                text = msg.get("data", "")
                if text:
                    await gemini_session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=text)],
                        ),
                        turn_complete=True,
                    )
                    self.session.memory.add("ユーザー", text)

            elif msg_type == "stop":
                self.state.is_running = False
                self.reconnect_mgr.needs_reconnect = True
                return

    async def _relay_gemini_to_client(
        self, gemini_session, client_ws: WebSocket
    ) -> None:
        """
        Gemini → ブラウザ 中継

        stt_stream.py の receive_audio() (L579-683) を Web向けに再構成。
        音声データ、transcription、割り込みをブラウザに中継し、
        A2Eサービスに並行リクエストして Expression もブラウザに送信する。
        """
        while not self.reconnect_mgr.needs_reconnect:
            turn = gemini_session.receive()
            async for response in turn:
                if self.reconnect_mgr.needs_reconnect:
                    return

                sc = response.server_content
                if not sc:
                    # tool_call 等の処理 (将来拡張)
                    continue

                # --- 割り込み検知 (stt_stream.py L650-662) ---
                if hasattr(sc, "interrupted") and sc.interrupted:
                    logger.info("[LiveRelay] Barge-in detected")
                    # AI音声バッファをフラッシュ
                    if self.state.ai_transcript_buffer.strip():
                        self.session.memory.add(
                            "AI", self.state.ai_transcript_buffer.strip()
                        )
                    self.state.ai_transcript_buffer = ""
                    self.state.ai_audio_buffer = bytearray()
                    await self._send_json(client_ws, {"type": "interrupted"})
                    continue

                # --- 入力 transcription (stt_stream.py L665-669) ---
                if hasattr(sc, "input_transcription") and sc.input_transcription:
                    user_text = sc.input_transcription.text
                    if user_text:
                        self.state.user_transcript_buffer += user_text
                        await self._send_json(client_ws, {
                            "type": "transcription",
                            "role": "user",
                            "text": user_text,
                            "is_partial": True,
                        })

                # --- 出力 transcription (stt_stream.py L672-676) ---
                if hasattr(sc, "output_transcription") and sc.output_transcription:
                    ai_text = sc.output_transcription.text
                    if ai_text:
                        self.state.ai_transcript_buffer += ai_text
                        await self._send_json(client_ws, {
                            "type": "transcription",
                            "role": "ai",
                            "text": ai_text,
                            "is_partial": True,
                        })

                # --- 音声データ (stt_stream.py L679-683) ---
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            audio_data = part.inline_data.data
                            if isinstance(audio_data, bytes):
                                # ブラウザに音声を即時送信
                                await self._send_json(client_ws, {
                                    "type": "audio",
                                    "data": base64.b64encode(audio_data).decode(),
                                })
                                # A2E用にバッファに蓄積
                                self.state.ai_audio_buffer.extend(audio_data)

                # --- ターン完了 (stt_stream.py L600-645) ---
                if hasattr(sc, "turn_complete") and sc.turn_complete:
                    await self._on_turn_complete(client_ws)

    async def _on_turn_complete(self, client_ws: WebSocket) -> None:
        """
        ターン完了時の処理

        stt_stream.py L600-645 に相当:
        1. ユーザー transcription を確定
        2. AI transcription を確定
        3. 累積文字数チェック → 再接続判定
        4. A2E → Expression をブラウザに送信（アバター連携）
        """
        # ユーザー発言の確定
        user_text = self.state.user_transcript_buffer.strip()
        if user_text:
            self.session.memory.add("ユーザー", user_text)
            await self._send_json(client_ws, {
                "type": "transcription",
                "role": "user",
                "text": user_text,
                "is_partial": False,
            })
        self.state.user_transcript_buffer = ""

        # AI発言の確定
        ai_text = self.state.ai_transcript_buffer.strip()
        if ai_text:
            self.session.memory.add("AI", ai_text)
            await self._send_json(client_ws, {
                "type": "transcription",
                "role": "ai",
                "text": ai_text,
                "is_partial": False,
            })

            # 累積文字数チェック → 再接続判定 (stt_stream.py L624-643)
            self.reconnect_mgr.on_ai_speech_complete(
                ai_text, self.session.language
            )
            if self.reconnect_mgr.needs_reconnect:
                await self._send_json(client_ws, {
                    "type": "reconnecting",
                    "reason": self.reconnect_mgr.reconnect_reason,
                })
        self.state.ai_transcript_buffer = ""

        # --- A2E → Expression（アバター連携） ---
        # AI音声をA2Eサービスに送信し、表情データをブラウザに中継
        # LAMAvatarController の frameBuffer に投入される
        if self.a2e_client and len(self.state.ai_audio_buffer) > 0:
            asyncio.create_task(
                self._process_a2e_and_send(
                    client_ws,
                    bytes(self.state.ai_audio_buffer),
                )
            )
        self.state.ai_audio_buffer = bytearray()

    async def _process_a2e_and_send(
        self, client_ws: WebSocket, audio_pcm: bytes
    ) -> None:
        """
        A2E推論 + Expression送信（非同期・音声再生と並行）

        Live API 出力の PCM 24kHz 音声を A2E サービスに送信し、
        52次元ARKitブレンドシェイプをブラウザに中継する。
        ブラウザ側の LAMAvatarController がこれを frameBuffer にキューして
        音声再生と同期してアバターを動かす。
        """
        try:
            audio_b64 = base64.b64encode(audio_pcm).decode()
            result = await self.a2e_client.process_audio(
                audio_base64=audio_b64,
                session_id=self.session.session_id,
                audio_format="pcm",
            )
            if result and result.frames:
                await self._send_json(client_ws, {
                    "type": "expression",
                    "data": {
                        "names": result.names,
                        "frames": result.frames,
                        "frame_rate": result.frame_rate,
                    },
                })
                logger.info(
                    f"[LiveRelay] Expression sent: {len(result.frames)} frames"
                )
        except Exception as e:
            logger.warning(f"[LiveRelay] A2E failed (non-fatal): {e}")

    def _build_live_config(self, context: str | None = None) -> dict:
        """
        Live API 設定を構築

        stt_stream.py L410-468 の _build_config() を移植。
        モードプラグインからシステムプロンプトを取得し、
        言語設定と再接続コンテキストを適用する。
        """
        lang_profile = get_language_profile(self.session.language)

        # モードプラグインからシステムプロンプト取得
        system_instruction = self.mode_plugin.get_system_prompt(
            language=self.session.language,
            context=context,
        )

        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_instruction,
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "speech_config": {
                "language_code": lang_profile.live_api_language_code,
            },
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
                    "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
                    "prefix_padding_ms": 100,
                    "silence_duration_ms": 500,
                }
            },
            "context_window_compression": {
                "sliding_window": {
                    "target_tokens": 32000,
                }
            },
        }

        # モード固有のツール定義
        tools = self.mode_plugin.get_live_api_tools()
        if tools:
            config["tools"] = tools

        return config

    @staticmethod
    async def _send_json(ws: WebSocket, data: dict) -> None:
        """WebSocket に JSON を送信（接続切れを安全に処理）"""
        try:
            import json
            await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass  # クライアント切断時は無視
