# -*- coding: utf-8 -*-
"""
汎用カスタマーサポートシステム (Gemini API版) - 改善版
モジュール分割版(3ファイル構成)

分割構成:
- api_integrations.py: 外部API連携
- support_core.py: ビジネスロジック・コアクラス
- app_customer_support.py: Webアプリケーション層(本ファイル)
"""
import os
import re
import json
import base64
import logging
import threading
import queue
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from google import genai
from google.genai import types
from google.cloud import texttospeech
from google.cloud import speech

# 新しいモジュールからインポート
from api_integrations import (
    enrich_shops_with_photos,
    extract_area_from_text,
    GOOGLE_PLACES_API_KEY
)
from support_core import (
    load_system_prompts,
    INITIAL_GREETINGS,
    SYSTEM_PROMPTS,
    SupportSession,
    SupportAssistant
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 長期記憶モジュールをインポート
try:
    from long_term_memory import LongTermMemory, PreferenceExtractor, extract_name_from_text
    LONG_TERM_MEMORY_ENABLED = True
except Exception as e:
    logger.warning(f"[LTM] 長期記憶モジュールのインポート失敗: {e}")
    LONG_TERM_MEMORY_ENABLED = False

# ========================================
# Audio2Expression Service 設定
# ========================================
AUDIO2EXP_SERVICE_URL = os.getenv("AUDIO2EXP_SERVICE_URL", "")
if AUDIO2EXP_SERVICE_URL:
    logger.info(f"[Audio2Exp] サービスURL設定済み: {AUDIO2EXP_SERVICE_URL}")
else:
    logger.info("[Audio2Exp] サービスURL未設定（リップシンク無効）")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # UTF-8エンコーディングを有効化

# ========================================
# CORS & SocketIO 設定 (Claudeアドバイス適用版)
# ========================================

# 許可するオリジン(末尾のスラッシュなし)
allowed_origins = [
    "https://gourmet-sp-two.vercel.app",
    "https://gourmet-sp.vercel.app",
    "http://localhost:4321"
]

# SocketIO初期化 (cors_allowed_originsを明示的に指定)
socketio = SocketIO(
    app,
    cors_allowed_origins=allowed_origins,
    async_mode='threading',
    logger=False,
    engineio_logger=False
)

# Flask-CORS初期化 (supports_credentials=True)
CORS(app, resources={
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# 【重要】全レスポンスに強制的にCORSヘッダーを注入するフック
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    # UTF-8エンコーディングを明示
    if response.content_type and 'application/json' in response.content_type:
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# Google Cloud TTS/STT初期化
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

# プロンプト読み込み
SYSTEM_PROMPTS = load_system_prompts()


# ========================================
# Audio2Expression 送信関数
# ========================================
def send_to_audio2exp(audio_base64_pcm: str, session_id: str):
    """
    Audio2Expression サービスに音声を送信（同期版）
    別スレッドで呼び出されることを想定
    """
    if not AUDIO2EXP_SERVICE_URL or not session_id:
        return

    try:
        response = requests.post(
            f"{AUDIO2EXP_SERVICE_URL}/api/audio2expression",
            json={
                "audio_base64": audio_base64_pcm,
                "session_id": session_id,
                "is_final": True
            },
            timeout=10
        )
        if response.status_code == 200:
            logger.info(f"[Audio2Exp] 送信成功: session={session_id}")
        else:
            logger.warning(f"[Audio2Exp] 送信失敗: status={response.status_code}")
    except Exception as e:
        logger.warning(f"[Audio2Exp] 送信エラー: {e}")


@app.route('/')
def index():
    """フロントエンド表示"""
    return render_template('support.html')


@app.route('/api/session/start', methods=['POST', 'OPTIONS'])
def start_session():
    """
    セッション開始 - モード対応

    【重要】改善されたフロー:
    1. セッション初期化(モード・言語設定)
    2. アシスタント作成(最新の状態で)
    3. 初回メッセージ生成
    4. 履歴に追加
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json or {}
        user_info = data.get('user_info', {})
        language = data.get('language', 'ja')
        mode = data.get('mode', 'chat')

        # 1. セッション初期化
        session = SupportSession()
        session.initialize(user_info, language=language, mode=mode)
        logger.info(f"[Start Session] 新規セッション作成: {session.session_id}")

        # 2. アシスタント作成(最新の状態で)
        assistant = SupportAssistant(session, SYSTEM_PROMPTS)

        # 3. 初回メッセージ生成
        initial_message = assistant.get_initial_message()

        # 4. 履歴に追加（roleは'model')
        session.add_message('model', initial_message, 'chat')

        logger.info(f"[API] セッション開始: {session.session_id}, 言語: {language}, モード: {mode}")

        # レスポンス作成
        response_data = {
            'session_id': session.session_id,
            'initial_message': initial_message
        }

        # コンシェルジュモードのみ、名前情報を返す
        if mode == 'concierge':
            session_data = session.get_data()
            profile = session_data.get('long_term_profile') if session_data else None
            if profile:
                response_data['user_profile'] = {
                    'preferred_name': profile.get('preferred_name'),
                    'name_honorific': profile.get('name_honorific')
                }
                logger.info(f"[API] user_profile を返却: {response_data['user_profile']}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"[API] セッション開始エラー: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """
    チャット処理 - 改善版

    【重要】改善されたフロー(順序を厳守):
    1. 状態確定 (State First): モード・言語を更新
    2. ユーザー入力を記録: メッセージを履歴に追加
    3. 知能生成 (Assistant作成): 最新の状態でアシスタントを作成
    4. 推論開始: Gemini APIを呼び出し
    5. アシスタント応答を記録: 履歴に追加
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        session_id = data.get('session_id')
        user_message = data.get('message')
        stage = data.get('stage', 'conversation')
        language = data.get('language', 'ja')
        mode = data.get('mode', 'chat')

        if not session_id or not user_message:
            return jsonify({'error': 'session_idとmessageが必要です'}), 400

        session = SupportSession(session_id)
        session_data = session.get_data()

        if not session_data:
            return jsonify({'error': 'セッションが見つかりません'}), 404

        logger.info(f"[Chat] セッション: {session_id}, モード: {mode}, 言語: {language}")

        # 1. 状態確定 (State First)
        session.update_language(language)
        session.update_mode(mode)

        # 2. ユーザー入力を記録
        session.add_message('user', user_message, 'chat')

        # 3. 知能生成 (Assistant作成)
        assistant = SupportAssistant(session, SYSTEM_PROMPTS)

        # 4. 推論開始
        result = assistant.process_user_message(user_message, stage)

        # 5. アシスタント応答を記録
        session.add_message('model', result['response'], 'chat')

        if result['summary']:
            session.add_message('model', result['summary'], 'summary')

        # ショップデータ処理
        shops = result.get('shops') or []  # None対策
        response_text = result['response']
        is_followup = result.get('is_followup', False)

        # 多言語メッセージ辞書
        shop_messages = {
            'ja': {
                'intro': lambda count: f"ご希望に合うお店を{count}件ご紹介します。\n\n",
                'not_found': "申し訳ございません。条件に合うお店が見つかりませんでした。別の条件でお探しいただけますか?"
            },
            'en': {
                'intro': lambda count: f"Here are {count} restaurant recommendations for you.\n\n",
                'not_found': "Sorry, we couldn't find any restaurants matching your criteria. Would you like to search with different conditions?"
            },
            'zh': {
                'intro': lambda count: f"为您推荐{count}家餐厅。\n\n",
                'not_found': "很抱歉,没有找到符合条件的餐厅。要用其他条件搜索吗?"
            },
            'ko': {
                'intro': lambda count: f"고객님께 {count}개의 식당을 추천합니다.\n\n",
                'not_found': "죄송합니다. 조건에 맞는 식당을 찾을 수 없었습니다. 다른 조건으로 찾으시겠습니까?"
            }
        }

        current_messages = shop_messages.get(language, shop_messages['ja'])

        if shops and not is_followup:
            original_count = len(shops)
            area = extract_area_from_text(user_message, language)
            logger.info(f"[Chat] 抽出エリア: '{area}' from '{user_message}'")

            # Places APIで写真を取得
            shops = enrich_shops_with_photos(shops, area, language)

            if shops:
                shop_list = []
                for i, shop in enumerate(shops, 1):
                    name = shop.get('name', '')
                    shop_area = shop.get('area', '')
                    description = shop.get('description', '')
                    if shop_area:
                        shop_list.append(f"{i}. **{name}**({shop_area}): {description}")
                    else:
                        shop_list.append(f"{i}. **{name}**: {description}")

                response_text = current_messages['intro'](len(shops)) + "\n\n".join(shop_list)
                logger.info(f"[Chat] {len(shops)}件のショップデータを返却(元: {original_count}件, 言語: {language})")
            else:
                response_text = current_messages['not_found']
                logger.warning(f"[Chat] 全店舗が除外されました(元: {original_count}件)")

        elif is_followup:
            logger.info(f"[Chat] 深掘り質問への回答: {response_text[:100]}...")

        # ========================================
        # 長期記憶: LLMからのaction処理（新設計版）
        # ========================================
        if LONG_TERM_MEMORY_ENABLED:
            try:
                # user_id をセッションデータから取得
                user_id = session_data.get('user_id')

                # ========================================
                # LLMからのaction指示を処理
                # ========================================
                # 初回訪問時の名前登録も、名前変更も、すべてLLMのactionで統一
                action = result.get('action')
                if action and action.get('type') == 'update_user_profile':
                    updates = action.get('updates', {})
                    if updates and user_id:
                        ltm = LongTermMemory()
                        # user_id をキーにしてプロファイルを更新（UPSERT動作）
                        success = ltm.update_profile(user_id, updates)
                        if success:
                            logger.info(f"[LTM] LLMからの指示でプロファイル更新成功: updates={updates}, user_id={user_id}")
                        else:
                            logger.error(f"[LTM] LLMからの指示でプロファイル更新失敗: updates={updates}, user_id={user_id}")
                    elif not user_id:
                        logger.warning(f"[LTM] user_id が空のためプロファイル更新をスキップ: action={action}")

                # ========================================
                # ショップカード提示時にサマリーを保存（マージ）
                # ========================================
                if shops and not is_followup and user_id and mode == 'concierge':
                    try:
                        # 提案した店舗名を取得
                        shop_names = [s.get('name', '不明') for s in shops]
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

                        # サマリーを生成
                        shop_summary = f"[{timestamp}] 検索条件: {user_message[:100]}\n提案店舗: {', '.join(shop_names)}"

                        ltm = LongTermMemory()
                        if ltm.append_conversation_summary(user_id, shop_summary):
                            logger.info(f"[LTM] ショップ提案サマリー保存成功: {len(shops)}件")
                        else:
                            logger.warning(f"[LTM] ショップ提案サマリー保存失敗")
                    except Exception as e:
                        logger.error(f"[LTM] ショップサマリー保存エラー: {e}")

            except Exception as e:
                logger.error(f"[LTM] 処理エラー: {e}")

        # 【デバッグ】最終的なshopsの内容を確認
        logger.info(f"[Chat] 最終shops配列: {len(shops)}件")
        if shops:
            logger.info(f"[Chat] shops[0] keys: {list(shops[0].keys())}")
        return jsonify({
            'response': response_text,
            'summary': result['summary'],
            'shops': shops,
            'should_confirm': result['should_confirm'],
            'is_followup': is_followup
        })

    except Exception as e:
        logger.error(f"[API] チャットエラー: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/finalize', methods=['POST', 'OPTIONS'])
def finalize_session():
    """セッション完了"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'error': 'session_idが必要です'}), 400

        session = SupportSession(session_id)
        session_data = session.get_data()

        if not session_data:
            return jsonify({'error': 'セッションが見つかりません'}), 404

        assistant = SupportAssistant(session, SYSTEM_PROMPTS)
        final_summary = assistant.generate_final_summary()

        # ========================================
        # 長期記憶: セッション終了時にサマリーを追記（マージ）
        # ========================================
        if LONG_TERM_MEMORY_ENABLED and session_data.get('mode') == 'concierge':
            user_id = session_data.get('user_id')
            if user_id and final_summary:
                try:
                    ltm = LongTermMemory()
                    # 既存サマリーにマージ（過去セッションの記録を保持）
                    ltm.append_conversation_summary(user_id, final_summary)
                    logger.info(f"[LTM] セッション終了サマリー追記成功: user_id={user_id}")
                except Exception as e:
                    logger.error(f"[LTM] サマリー保存エラー: {e}")

        logger.info(f"[LTM] セッション終了: {session_id}")

        return jsonify({
            'summary': final_summary,
            'session_id': session_id
        })

    except Exception as e:
        logger.error(f"[API] 完了処理エラー: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cancel', methods=['POST', 'OPTIONS'])
def cancel_processing():
    """処理中止"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'error': 'session_idが必要です'}), 400

        logger.info(f"[API] 処理中止リクエスト: {session_id}")

        # セッションのステータスを更新
        session = SupportSession(session_id)
        session_data = session.get_data()

        if session_data:
            session.update_status('cancelled')

        return jsonify({
            'success': True,
            'message': '処理を中止しました'
        })

    except Exception as e:
        logger.error(f"[API] 中止処理エラー: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts/synthesize', methods=['POST', 'OPTIONS'])
def synthesize_speech():
    """
    音声合成 - Audio2Expression対応版

    session_id が指定された場合、Audio2Expressionサービスにも音声を送信
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        text = data.get('text', '')
        language_code = data.get('language_code', 'ja-JP')
        voice_name = data.get('voice_name', 'ja-JP-Chirp3-HD-Leda')
        speaking_rate = data.get('speaking_rate', 1.0)
        pitch = data.get('pitch', 0.0)
        session_id = data.get('session_id', '')  # ★追加: リップシンク用セッションID

        if not text:
            return jsonify({'success': False, 'error': 'テキストが必要です'}), 400

        MAX_CHARS = 1000
        if len(text) > MAX_CHARS:
            logger.warning(f"[TTS] テキストが長すぎるため切り詰めます: {len(text)} → {MAX_CHARS} 文字")
            text = text[:MAX_CHARS] + '...'

        logger.info(f"[TTS] 合成開始: {len(text)} 文字, session_id={session_id}")

        synthesis_input = texttospeech.SynthesisInput(text=text)

        try:
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
        except Exception as voice_error:
            logger.warning(f"[TTS] 指定音声が無効、デフォルトに変更: {voice_error}")
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name='ja-JP-Neural2-B'
            )

        # ★ MP3形式（フロントエンド再生用）
        audio_config_mp3 = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch
        )

        response_mp3 = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config_mp3
        )

        audio_base64 = base64.b64encode(response_mp3.audio_content).decode('utf-8')
        logger.info(f"[TTS] MP3合成成功: {len(audio_base64)} bytes (base64)")

        # ========================================
        # ★ Audio2Expression用にLINEAR16形式も生成して送信
        # ========================================
        if AUDIO2EXP_SERVICE_URL and session_id:
            try:
                audio_config_pcm = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    speaking_rate=speaking_rate,
                    pitch=pitch
                )

                response_pcm = tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config_pcm
                )

                audio_base64_pcm = base64.b64encode(response_pcm.audio_content).decode('utf-8')
                logger.info(f"[TTS] PCM合成成功: {len(audio_base64_pcm)} bytes (base64)")

                # 非同期で送信（レスポンスを待たない）
                thread = threading.Thread(
                    target=send_to_audio2exp,
                    args=(audio_base64_pcm, session_id)
                )
                thread.start()

                logger.info(f"[TTS] Audio2Exp送信開始: session={session_id}")

            except Exception as e:
                logger.warning(f"[TTS] Audio2Exp送信準備エラー: {e}")

        return jsonify({
            'success': True,
            'audio': audio_base64
        })

    except Exception as e:
        logger.error(f"[TTS] エラー: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stt/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    """音声認識"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        audio_base64 = data.get('audio', '')
        language_code = data.get('language_code', 'ja-JP')

        if not audio_base64:
            return jsonify({'success': False, 'error': '音声データが必要です'}), 400

        logger.info(f"[STT] 認識開始: {len(audio_base64)} bytes (base64)")

        audio_content = base64.b64decode(audio_base64)
        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model='default'
        )

        response = stt_client.recognize(config=config, audio=audio)

        transcript = ''
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            logger.info(f"[STT] 認識成功: '{transcript}' (信頼度: {confidence:.2f})")
        else:
            logger.warning("[STT] 音声が認識されませんでした")

        return jsonify({
            'success': True,
            'transcript': transcript
        })

    except Exception as e:
        logger.error(f"[STT] エラー: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stt/stream', methods=['POST', 'OPTIONS'])
def transcribe_audio_streaming():
    """音声認識 (Streaming)"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        audio_base64 = data.get('audio', '')
        language_code = data.get('language_code', 'ja-JP')

        if not audio_base64:
            return jsonify({'success': False, 'error': '音声データが必要です'}), 400

        logger.info(f"[STT Streaming] 認識開始: {len(audio_base64)} bytes (base64)")

        audio_content = base64.b64decode(audio_base64)

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model='default'
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False,
            single_utterance=True
        )

        CHUNK_SIZE = 1024 * 16

        def audio_generator():
            for i in range(0, len(audio_content), CHUNK_SIZE):
                chunk = audio_content[i:i + CHUNK_SIZE]
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        responses = stt_client.streaming_recognize(streaming_config, audio_generator())

        transcript = ''
        confidence = 0.0

        for response in responses:
            if not response.results:
                continue

            for result in response.results:
                if result.is_final and result.alternatives:
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    logger.info(f"[STT Streaming] 認識成功: '{transcript}' (信頼度: {confidence:.2f})")
                    break

            if transcript:
                break

        if not transcript:
            logger.warning("[STT Streaming] 音声が認識されませんでした")

        return jsonify({
            'success': True,
            'transcript': transcript,
            'confidence': confidence
        })

    except Exception as e:
        logger.error(f"[STT Streaming] エラー: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/session/<session_id>', methods=['GET', 'OPTIONS'])
def get_session(session_id):
    """セッション情報取得"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        session = SupportSession(session_id)
        data = session.get_data()

        if not data:
            return jsonify({'error': 'セッションが見つかりません'}), 404

        return jsonify(data)

    except Exception as e:
        logger.error(f"[API] セッション取得エラー: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """ヘルスチェック"""
    if request.method == 'OPTIONS':
        return '', 204

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'gemini': 'ok',
            'ram_session': 'ok',
            'tts': 'ok',
            'stt': 'ok',
            'places_api': 'ok' if GOOGLE_PLACES_API_KEY else 'not configured',
            'audio2exp': 'ok' if AUDIO2EXP_SERVICE_URL else 'not configured'
        }
    })


# ========================================
# WebSocket Streaming STT
# ========================================

active_streams = {}

@socketio.on('connect')
def handle_connect():
    logger.info(f"[WebSocket STT] クライアント接続: {request.sid}")
    emit('connected', {'status': 'ready'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"[WebSocket STT] クライアント切断: {request.sid}")
    if request.sid in active_streams:
        stream_data = active_streams[request.sid]
        if 'stop_event' in stream_data:
            stream_data['stop_event'].set()
        del active_streams[request.sid]

@socketio.on('start_stream')
def handle_start_stream(data):
    language_code = data.get('language_code', 'ja-JP')
    sample_rate = data.get('sample_rate', 16000)  # フロントエンドから受け取る
    client_sid = request.sid
    logger.info(f"[WebSocket STT] ストリーム開始: {client_sid}, 言語: {language_code}, サンプルレート: {sample_rate}Hz")

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,  # 動的に設定
        language_code=language_code,
        enable_automatic_punctuation=True,
        model='latest_long'  # より高精度なモデルに変更
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,
        single_utterance=False
    )

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    active_streams[client_sid] = {
        'audio_queue': audio_queue,
        'stop_event': stop_event,
        'streaming_config': streaming_config
    }

    def audio_generator():
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def recognition_thread():
        try:
            logger.info(f"[WebSocket STT] 認識スレッド開始: {client_sid}")
            responses = stt_client.streaming_recognize(streaming_config, audio_generator())

            for response in responses:
                if stop_event.is_set():
                    break

                if not response.results:
                    continue

                result = response.results[0]

                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence if result.is_final else 0.0

                    socketio.emit('transcript', {
                        'text': transcript,
                        'is_final': result.is_final,
                        'confidence': confidence
                    }, room=client_sid)

                    if result.is_final:
                        logger.info(f"[WebSocket STT] 最終認識: '{transcript}' (信頼度: {confidence:.2f})")
                    else:
                        logger.debug(f"[WebSocket STT] 途中認識: '{transcript}'")

        except Exception as e:
            logger.error(f"[WebSocket STT] 認識エラー: {e}", exc_info=True)
            socketio.emit('error', {'message': str(e)}, room=client_sid)

    thread = threading.Thread(target=recognition_thread, daemon=True)
    thread.start()

    emit('stream_started', {'status': 'streaming'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    if request.sid not in active_streams:
        logger.warning(f"[WebSocket STT] 未初期化のストリーム: {request.sid}")
        return

    try:
        chunk_base64 = data.get('chunk', '')
        if not chunk_base64:
            return

        # ★★★ sample_rateを取得(16kHzで受信) ★★★
        sample_rate = data.get('sample_rate', 16000)

        # ★★★ 統計情報を取得してログ出力(必ず出力) ★★★
        stats = data.get('stats')
        logger.info(f"[audio_chunk受信] sample_rate: {sample_rate}Hz, stats: {stats}")

        if stats:
            logger.info(f"[AudioWorklet統計] サンプルレート: {sample_rate}Hz, "
                       f"サンプル総数: {stats.get('totalSamples')}, "
                       f"送信チャンク数: {stats.get('chunksSent')}, "
                       f"空入力回数: {stats.get('emptyInputCount')}, "
                       f"process呼び出し回数: {stats.get('processCalls')}, "
                       f"オーバーフロー回数: {stats.get('overflowCount', 0)}")  # ★ オーバーフロー追加

        audio_chunk = base64.b64decode(chunk_base64)

        # ★★★ 16kHzそのままGoogle STTに送る ★★★
        stream_data = active_streams[request.sid]
        stream_data['audio_queue'].put(audio_chunk)

    except Exception as e:
        logger.error(f"[WebSocket STT] チャンク処理エラー: {e}", exc_info=True)

@socketio.on('stop_stream')
def handle_stop_stream():
    logger.info(f"[WebSocket STT] ストリーム停止: {request.sid}")

    if request.sid in active_streams:
        stream_data = active_streams[request.sid]
        stream_data['audio_queue'].put(None)
        stream_data['stop_event'].set()
        del active_streams[request.sid]

    emit('stream_stopped', {'status': 'stopped'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
