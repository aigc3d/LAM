"""
gourmet-support TTS エンドポイント修正パッチ

app_customer_support.py の synthesize_speech() 関数を以下のように修正してください。

【変更点】
1. session_id パラメータを追加
2. LINEAR16形式でも音声生成（Audio2Expression用）
3. audio2exp-service への非同期送信を追加
"""

# ============================================
# 追加するインポート (ファイル先頭に追加)
# ============================================
"""
import asyncio
import aiohttp
import os

AUDIO2EXP_SERVICE_URL = os.getenv("AUDIO2EXP_SERVICE_URL", "")
"""

# ============================================
# 追加する関数 (ファイル内に追加)
# ============================================
"""
async def send_to_audio2exp_async(audio_base64_pcm: str, session_id: str):
    '''Audio2Expression サービスに非同期で音声を送信'''
    if not AUDIO2EXP_SERVICE_URL or not session_id:
        return

    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{AUDIO2EXP_SERVICE_URL}/api/audio2expression",
                json={
                    "audio_base64": audio_base64_pcm,
                    "session_id": session_id,
                    "is_final": True
                },
                timeout=aiohttp.ClientTimeout(total=10)
            )
            logger.info(f"[Audio2Exp] 送信成功: session={session_id}")
    except Exception as e:
        logger.warning(f"[Audio2Exp] 送信失敗: {e}")


def send_to_audio2exp(audio_base64_pcm: str, session_id: str):
    '''Audio2Expression サービスに音声を送信（同期ラッパー）'''
    if not AUDIO2EXP_SERVICE_URL or not session_id:
        return

    try:
        # 新しいイベントループで非同期実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_to_audio2exp_async(audio_base64_pcm, session_id))
        loop.close()
    except Exception as e:
        logger.warning(f"[Audio2Exp] 送信失敗: {e}")
"""

# ============================================
# synthesize_speech() 関数の修正版
# ============================================
MODIFIED_SYNTHESIZE_SPEECH = '''
@app.route('/api/tts/synthesize', methods=['POST', 'OPTIONS'])
def synthesize_speech():
    """音声合成 - Audio2Expression対応版"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        text = data.get('text', '')
        language_code = data.get('language_code', 'ja-JP')
        voice_name = data.get('voice_name', 'ja-JP-Chirp3-HD-Leda')
        speaking_rate = data.get('speaking_rate', 1.0)
        pitch = data.get('pitch', 0.0)
        session_id = data.get('session_id', '')  # ★追加: セッションID

        if not text:
            return jsonify({'success': False, 'error': 'テキストが必要です'}), 400

        MAX_CHARS = 1000
        if len(text) > MAX_CHARS:
            logger.warning(f"[TTS] テキストが長すぎるため切り詰めます: {len(text)} → {MAX_CHARS} 文字")
            text = text[:MAX_CHARS] + '...'

        logger.info(f"[TTS] 合成開始: {len(text)} 文字")

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

        # ★ Audio2Expression用にLINEAR16形式も生成して送信
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

                # 非同期で送信（レスポンスを待たない）
                import threading
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
        return jsonify({'success': False, 'error': str(e)}), 500
'''

# ============================================
# フロントエンド側の変更 (core-controller.ts)
# ============================================
FRONTEND_CHANGE = '''
// TTS呼び出し時に session_id を追加

// Before:
const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: cleanText,
        language_code: langConfig.tts,
        voice_name: langConfig.voice
    })
});

// After:
const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: cleanText,
        language_code: langConfig.tts,
        voice_name: langConfig.voice,
        session_id: this.sessionId  // ★追加
    })
});
'''

if __name__ == "__main__":
    print("=== gourmet-support TTS 修正パッチ ===")
    print("\n1. インポート追加")
    print("2. send_to_audio2exp 関数追加")
    print("3. synthesize_speech() 関数を修正版に置換")
    print("4. フロントエンド (core-controller.ts) に session_id 追加")
    print("\n詳細はこのファイルを参照してください。")
