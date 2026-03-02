# -*- coding: utf-8 -*-
"""
REST API ルーター (gourmet-support 互換)

既存の gourmet-support (Flask) の REST エンドポイントを FastAPI ルーターに変換。
dialogue_type="rest" のセッション向け。

エンドポイント:
  POST /api/v2/rest/session/start    - セッション開始
  POST /api/v2/rest/chat             - チャット処理
  POST /api/v2/rest/finalize         - セッション完了
  POST /api/v2/rest/cancel           - 処理中止
  POST /api/v2/rest/tts/synthesize   - 音声合成 (TTS)
  POST /api/v2/rest/stt/transcribe   - 音声認識 (STT)
  POST /api/v2/rest/stt/stream       - 音声認識 (Streaming STT)
  GET  /api/v2/rest/session/{id}     - セッション取得
"""

import os
import time
import base64
import logging
from datetime import datetime

import requests as http_requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from support_base.core.support_core import (
    SYSTEM_PROMPTS,
    INITIAL_GREETINGS,
    SupportSession,
    SupportAssistant,
)
from support_base.core.api_integrations import (
    enrich_shops_with_photos,
    extract_area_from_text,
    GOOGLE_PLACES_API_KEY,
)

logger = logging.getLogger(__name__)

# --- オプション依存: 長期記憶 ---
try:
    from support_base.core.long_term_memory import LongTermMemory
    LONG_TERM_MEMORY_ENABLED = True
except Exception as e:
    logger.warning(f"[REST] Long term memory not available: {e}")
    LONG_TERM_MEMORY_ENABLED = False

# --- オプション依存: Google Cloud TTS/STT ---
try:
    from google.cloud import texttospeech, speech
    tts_client = texttospeech.TextToSpeechClient()
    stt_client = speech.SpeechClient()
    TTS_STT_ENABLED = True
except Exception as e:
    logger.warning(f"[REST] Google Cloud TTS/STT not available: {e}")
    tts_client = None
    stt_client = None
    TTS_STT_ENABLED = False

# --- Audio2Expression ---
AUDIO2EXP_SERVICE_URL = os.getenv("AUDIO2EXP_SERVICE_URL", "")


# === Pydantic モデル ===

class RestSessionStartRequest(BaseModel):
    user_info: dict = {}
    language: str = "ja"
    mode: str = "chat"  # "chat" or "concierge"


class RestSessionStartResponse(BaseModel):
    session_id: str
    initial_message: str
    user_profile: dict | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    stage: str = "conversation"
    language: str = "ja"
    mode: str = "chat"


class FinalizeRequest(BaseModel):
    session_id: str


class CancelRequest(BaseModel):
    session_id: str


class TTSRequest(BaseModel):
    text: str
    language_code: str = "ja-JP"
    voice_name: str = "ja-JP-Chirp3-HD-Leda"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    session_id: str = ""


class STTRequest(BaseModel):
    audio: str  # base64
    language_code: str = "ja-JP"


# === ルーター ===

router = APIRouter(prefix="/api/v2/rest", tags=["REST API"])


# === ヘルパー ===

def _get_expression_frames(audio_base64: str, session_id: str, audio_format: str = "mp3"):
    """Audio2Expression サービスから表情フレームを取得"""
    if not AUDIO2EXP_SERVICE_URL or not session_id:
        return None
    try:
        resp = http_requests.post(
            f"{AUDIO2EXP_SERVICE_URL}/api/audio2expression",
            json={
                "audio_base64": audio_base64,
                "session_id": session_id,
                "is_start": True,
                "is_final": True,
                "audio_format": audio_format,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            logger.info(f"[Audio2Exp] OK: {len(result.get('frames', []))} frames")
            return result
        logger.warning(f"[Audio2Exp] Failed: status={resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"[Audio2Exp] Error: {e}")
        return None


# === エンドポイント ===

@router.post("/session/start", response_model=RestSessionStartResponse)
async def rest_start_session(req: RestSessionStartRequest):
    """
    REST セッション開始

    gourmet-support の /api/session/start と互換。
    SupportSession + SupportAssistant を使ってセッションを初期化する。
    """
    try:
        # 1. セッション初期化
        session = SupportSession()
        session.initialize(req.user_info, language=req.language, mode=req.mode)

        # 2. アシスタント作成
        assistant = SupportAssistant(session, SYSTEM_PROMPTS)

        # 3. 初回メッセージ生成
        initial_message = assistant.get_initial_message()

        # 4. 履歴に追加
        session.add_message("model", initial_message, "chat")

        logger.info(
            f"[REST] Session started: {session.session_id}, "
            f"lang={req.language}, mode={req.mode}"
        )

        # コンシェルジュモードのみ: プロファイル情報を返す
        user_profile = None
        if req.mode == "concierge":
            session_data = session.get_data()
            profile = session_data.get("long_term_profile") if session_data else None
            if profile:
                user_profile = {
                    "preferred_name": profile.get("preferred_name"),
                    "name_honorific": profile.get("name_honorific"),
                }

        return RestSessionStartResponse(
            session_id=session.session_id,
            initial_message=initial_message,
            user_profile=user_profile,
        )

    except Exception as e:
        logger.error(f"[REST] Session start error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def rest_chat(req: ChatRequest):
    """
    チャット処理

    gourmet-support の /api/chat と互換。
    5ステップフロー: 状態更新 → 記録 → アシスタント生成 → Gemini → 記録
    """
    try:
        session = SupportSession(req.session_id)
        session_data = session.get_data()

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # 1. 状態確定
        session.update_language(req.language)
        session.update_mode(req.mode)

        # 2. ユーザー入力を記録
        session.add_message("user", req.message, "chat")

        # 3. アシスタント作成
        assistant = SupportAssistant(session, SYSTEM_PROMPTS)

        # 4. 推論開始
        result = assistant.process_user_message(req.message, req.stage)

        # 5. アシスタント応答を記録
        session.add_message("model", result["response"], "chat")
        if result["summary"]:
            session.add_message("model", result["summary"], "summary")

        # ショップデータ処理
        shops = result.get("shops") or []
        response_text = result["response"]
        is_followup = result.get("is_followup", False)

        # 多言語メッセージ
        shop_messages = {
            "ja": {
                "intro": lambda c: f"ご希望に合うお店を{c}件ご紹介します。\n\n",
                "not_found": "申し訳ございません。条件に合うお店が見つかりませんでした。別の条件でお探しいただけますか?",
            },
            "en": {
                "intro": lambda c: f"Here are {c} restaurant recommendations for you.\n\n",
                "not_found": "Sorry, we couldn't find any restaurants matching your criteria. Would you like to search with different conditions?",
            },
            "zh": {
                "intro": lambda c: f"为您推荐{c}家餐厅。\n\n",
                "not_found": "很抱歉,没有找到符合条件的餐厅。要用其他条件搜索吗?",
            },
            "ko": {
                "intro": lambda c: f"고객님께 {c}개의 식당을 추천합니다.\n\n",
                "not_found": "죄송합니다. 조건에 맞는 식당을 찾을 수 없었습니다. 다른 조건으로 찾으시겠습니까?",
            },
        }
        current_messages = shop_messages.get(req.language, shop_messages["ja"])

        if shops and not is_followup:
            original_count = len(shops)
            area = extract_area_from_text(req.message, req.language)

            # Places API でエンリッチ
            shops = enrich_shops_with_photos(shops, area, req.language) or []

            if shops:
                shop_list = []
                for i, shop in enumerate(shops, 1):
                    name = shop.get("name", "")
                    shop_area = shop.get("area", "")
                    description = shop.get("description", "")
                    if shop_area:
                        shop_list.append(f"{i}. **{name}**({shop_area}): {description}")
                    else:
                        shop_list.append(f"{i}. **{name}**: {description}")
                response_text = current_messages["intro"](len(shops)) + "\n\n".join(shop_list)
            else:
                response_text = current_messages["not_found"]

        # 長期記憶: action 処理
        if LONG_TERM_MEMORY_ENABLED:
            try:
                user_id = session_data.get("user_id")

                # LLM action 処理
                action = result.get("action")
                if action and action.get("type") == "update_user_profile":
                    updates = action.get("updates", {})
                    if updates and user_id:
                        ltm = LongTermMemory()
                        ltm.update_profile(user_id, updates)
                        logger.info(f"[LTM] Profile updated via action: {updates}")

                # ショップ提案サマリー保存 (concierge のみ)
                if shops and not is_followup and user_id and req.mode == "concierge":
                    try:
                        shop_names = [s.get("name", "") for s in shops]
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                        shop_summary = (
                            f"[{timestamp}] 検索条件: {req.message[:100]}\n"
                            f"提案店舗: {', '.join(shop_names)}"
                        )
                        ltm = LongTermMemory()
                        ltm.append_conversation_summary(user_id, shop_summary)
                    except Exception as e:
                        logger.error(f"[LTM] Shop summary save error: {e}")

            except Exception as e:
                logger.error(f"[LTM] Processing error: {e}")

        return {
            "response": response_text,
            "summary": result["summary"],
            "shops": shops,
            "should_confirm": result["should_confirm"],
            "is_followup": is_followup,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REST] Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finalize")
async def rest_finalize(req: FinalizeRequest):
    """セッション完了"""
    try:
        session = SupportSession(req.session_id)
        session_data = session.get_data()

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        assistant = SupportAssistant(session, SYSTEM_PROMPTS)
        final_summary = assistant.generate_final_summary()

        # 長期記憶: セッション終了サマリー追記 (concierge のみ)
        if LONG_TERM_MEMORY_ENABLED and session_data.get("mode") == "concierge":
            user_id = session_data.get("user_id")
            if user_id and final_summary:
                try:
                    ltm = LongTermMemory()
                    ltm.append_conversation_summary(user_id, final_summary)
                    logger.info(f"[LTM] Final summary appended: user_id={user_id}")
                except Exception as e:
                    logger.error(f"[LTM] Summary save error: {e}")

        return {"summary": final_summary, "session_id": req.session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REST] Finalize error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def rest_cancel(req: CancelRequest):
    """処理中止"""
    try:
        session = SupportSession(req.session_id)
        session_data = session.get_data()
        if session_data:
            session.update_status("cancelled")

        return {"success": True, "message": "処理を中止しました"}

    except Exception as e:
        logger.error(f"[REST] Cancel error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts/synthesize")
async def rest_tts_synthesize(req: TTSRequest):
    """
    音声合成 (Google Cloud TTS) + Audio2Expression

    gourmet-support の /api/tts/synthesize と互換。
    """
    if not TTS_STT_ENABLED:
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        text = req.text
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        MAX_CHARS = 1000
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "..."

        synthesis_input = texttospeech.SynthesisInput(text=text)

        try:
            voice = texttospeech.VoiceSelectionParams(
                language_code=req.language_code, name=req.voice_name
            )
        except Exception:
            voice = texttospeech.VoiceSelectionParams(
                language_code=req.language_code, name="ja-JP-Neural2-B"
            )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=req.speaking_rate,
            pitch=req.pitch,
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")

        # Audio2Expression (同期)
        expression_data = None
        if AUDIO2EXP_SERVICE_URL and req.session_id:
            try:
                expression_data = _get_expression_frames(
                    audio_base64, req.session_id, "mp3"
                )
            except Exception as e:
                logger.warning(f"[Audio2Exp] Error: {e}")

        result = {"success": True, "audio": audio_base64}
        if expression_data:
            result["expression"] = expression_data

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REST] TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stt/transcribe")
async def rest_stt_transcribe(req: STTRequest):
    """音声認識 (Google Cloud STT)"""
    if not TTS_STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT service not available")

    try:
        if not req.audio:
            raise HTTPException(status_code=400, detail="audio is required")

        audio_content = base64.b64decode(req.audio)
        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=req.language_code,
            enable_automatic_punctuation=True,
            model="default",
        )

        response = stt_client.recognize(config=config, audio=audio)

        transcript = ""
        if response.results:
            transcript = response.results[0].alternatives[0].transcript

        return {"success": True, "transcript": transcript}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REST] STT error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stt/stream")
async def rest_stt_stream(req: STTRequest):
    """音声認識 (Streaming STT)"""
    if not TTS_STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT service not available")

    try:
        if not req.audio:
            raise HTTPException(status_code=400, detail="audio is required")

        audio_content = base64.b64decode(req.audio)

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=req.language_code,
            enable_automatic_punctuation=True,
            model="default",
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False,
            single_utterance=True,
        )

        CHUNK_SIZE = 1024 * 16

        def audio_generator():
            for i in range(0, len(audio_content), CHUNK_SIZE):
                chunk = audio_content[i : i + CHUNK_SIZE]
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        responses = stt_client.streaming_recognize(streaming_config, audio_generator())

        transcript = ""
        confidence = 0.0
        for response in responses:
            if not response.results:
                continue
            for result in response.results:
                if result.is_final and result.alternatives:
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    break
            if transcript:
                break

        return {"success": True, "transcript": transcript, "confidence": confidence}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REST] Streaming STT error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def rest_get_session(session_id: str):
    """セッション情報取得"""
    try:
        session = SupportSession(session_id)
        data = session.get_data()

        if not data:
            raise HTTPException(status_code=404, detail="Session not found")

        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REST] Session get error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
