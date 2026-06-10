# -*- coding: utf-8 -*-
"""
Gemini Live API ベースの会議アシスタント（Function Callingハイブリッド方式）
- 短い応答 → Live API（低遅延）
- 長い応答（要約・質問・検索・資料参照）→ REST API + TTS
- Function Callingで明示的に切り替え
"""

import pyaudio
import os
import argparse
import asyncio
import struct
import math
import re
from datetime import datetime

from google import genai
from google.genai import types
from google.cloud import texttospeech

# ============================================================
# 設定
# ============================================================

# Gemini API Key
GEMINI_API_KEY_OVERRIDE = "AIzaSyCwwDBpyk2K6btr5TOQBU60FUgDvPitnMg"

# GCP Project ID（TTS用）
GCP_PROJECT_ID = "ai-meet-486502"

# Voicemeeter デバイス設定
INPUT_DEVICE_NAME = "Voicemeeter Out B1 (VB-Audio Vo"
TTS_OUTPUT_DEVICE_NAME = "Voicemeeter AUX Input (VB-Audio"

# Live API 設定
LIVE_API_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
REST_API_MODEL = "gemini-2.5-flash"
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
TTS_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ファイルパス
INTERVIEW_SCRIPT_FILE_PATH = "interview_script.txt"
MEETING_SUMMARY_FILE_PATH = "meeting_summary.txt"
REFERENCE_PDF_FILE_PATH = "reference.pdf"
TRANSCRIPT_FILE_PATH = "meeting_transcript_log.md"

# ============================================================
# Function Calling ツール定義
# ============================================================

def get_interview_tools():
    """Live API用のツール定義 - 一旦無効化"""
    # ポリシーエラーが発生するため、ツールを無効化
    return []

# ============================================================
# システムインストラクション（モード別）
# ============================================================

STANDARD_SYSTEM_INSTRUCTION = """あなたは会議をサポートするフレンドリーなAIアシスタントです。
必ず日本語で話してください。

【役割】
- 会議の流れを温かくサポートする
- 参加者の発言を尊重し、必要な時だけ介入する

【重要：発話の長さ制限】
あなたの音声出力には厳しい制限があります。
1回の発話は2〜3文以内、10秒以内に収めてください。

【発話の例】
- 「なるほど、そうなんですね。」
- 「はい、承知しました。」
- 「ええ、よくわかります。」
"""

SILENT_SYSTEM_INSTRUCTION = """あなたは会議の書記役AIアシスタントです。
必ず日本語で話してください。

【重要】
- 基本的に発言しない（黙って聞いている）
- 「アシスタントさん」と呼ばれた時だけ応答する
- 呼ばれたら「はい、何でしょうか？」と応答し、指示を待つ
- 応答は短く簡潔に（2〜3文以内）
"""

INTERVIEW_SYSTEM_INSTRUCTION = """あなたはプロのインタビュアーです。
必ず日本語で話してください。

【あなたの役割】
インタビュースクリプトに従って、話者に質問をしていきます。

【インタビュー開始時】
1. 話者を紹介する（「本日は〜様にお話を伺います」）
2. 自己紹介を依頼する（「まずは自己紹介をお願いします」）
3. 話者の発言を待つ

【通常のインタビュー進行】
発話パターン：相槌 + 次の質問
「〜ですね。では、〜でしょうか？」

【重要：話者から説明・検索を依頼されたら】
話者が「説明してもらえますか？」「調べてもらえますか？」と依頼したら：

1. まず即答する（これが最優先！）
   「はい、承知しました。」
   「はい、確認しますので少々お待ちください。」
   「はい、お調べします。」

2. その後、4〜6文程度でしっかり説明する
   - 資料やスクリプトの内容を参考に
   - 具体的な数字や特徴を含める
   - 聞き手にわかりやすく

3. 説明が終わったら「以上です」で締める

4. 話者の反応を待ってから次の質問へ

【説明の例】
話者「ポイントモールについて説明してもらえますか？」

良い例：
「はい、承知しました。ポイントモールは、ウェルテクトを購入した金額と同額のポイントがもらえる仕組みです。このポイントで日用品を卸価格で購入できます。企業にとっては福利厚生費として計上でき、従業員にとっては健康管理をしながらお得に買い物ができるメリットがあります。健康経営を促進する狙いで作られました。以上です。」

悪い例（短すぎ）：
「ポイントモールは購入金額と同額のポイントがもらえます。以上です。」

【禁止事項】
- 説明後すぐに次の質問をすること（話者の反応を待つ）
- 説明を2文以内で終わらせること（しっかり説明する）
"""

# REST API用のシステムインストラクション
REST_API_SYSTEM_INSTRUCTION = """あなたはプロのインタビュアーです。
必ず日本語で回答してください。

【役割】
- ユーザーの発言を簡潔に要約する
- インタビュースクリプトに沿った次の質問を行う
- 資料を参照して詳しく説明する

【回答スタイル】
- 音声で読み上げられることを意識して、聞きやすい表現を使う
- マークダウン記法（**太字**、# 見出し、- リストなど）は使わない
- 質問は必ず「〜でしょうか？」「〜ですか？」で丁寧に終える

【フォーマット】
1. まず短い相槌（省略可）
2. ユーザーの発言の簡潔な要約（1文）
3. 次の質問（1〜2文）
"""

# ============================================================
# 効果音・つなぎ音声生成
# ============================================================

def generate_beep_sound(frequency=800, duration=0.3, sample_rate=24000):
    """ビープ音を生成"""
    num_samples = int(sample_rate * duration)
    audio_data = []
    for i in range(num_samples):
        value = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
        if i < num_samples * 0.1:
            value = int(value * (i / (num_samples * 0.1)))
        elif i > num_samples * 0.9:
            value = int(value * ((num_samples - i) / (num_samples * 0.1)))
        audio_data.append(struct.pack('<h', value))
    return b''.join(audio_data)

def generate_thinking_sound(sample_rate=24000):
    """考え中の音（2つのビープ音）"""
    beep1 = generate_beep_sound(frequency=500, duration=0.15, sample_rate=sample_rate)
    silence = b'\x00\x00' * int(sample_rate * 0.1)  # 0.1秒の無音
    beep2 = generate_beep_sound(frequency=600, duration=0.15, sample_rate=sample_rate)
    return beep1 + silence + beep2

SEARCHING_BEEP = generate_beep_sound(frequency=600, duration=0.2)
THINKING_SOUND = generate_thinking_sound()

# ============================================================
# 議事録
# ============================================================

def initialize_transcript(filepath: str, mode: str):
    """議事録ファイルを初期化"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_names = {
        'standard': 'スタンダードモード',
        'silent': 'サイレントモード', 
        'interview': 'インタビューモード',
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# 会議議事録\n")
        f.write(f"開始日時: {timestamp}\n")
        f.write(f"モード: {mode_names.get(mode, mode)} (ハイブリッド)\n\n")
        f.write("---\n\n")

def log_transcript(text: str, filepath: str):
    """議事録に追記"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {text}\n")

# ============================================================
# TTS（Google Cloud Text-to-Speech）
# ============================================================

class TTSPlayer:
    """Google Cloud TTSで音声を再生"""
    
    def __init__(self, output_device_index: int):
        self.output_device_index = output_device_index
        self.tts_client = texttospeech.TextToSpeechClient()
        # Wavenet音声（より自然な声）
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name="ja-JP-Wavenet-D",  # 男性、自然な声
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=TTS_SAMPLE_RATE,
            speaking_rate=1.05,  # 少し速めに
            pitch=0.0,
        )
    
    def synthesize_and_play(self, text: str, p: pyaudio.PyAudio):
        """テキストを音声合成して再生（事前バッファリング方式）"""
        if not text.strip():
            return
        
        # マークダウン記法を除去
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        
        # 長いテキストは分割
        sentences = self._split_text(text)
        
        # 先に全文を合成（バッファリング）
        audio_buffers = []
        print(f"   🔊 TTS合成中... ({len(sentences)}文)")
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            try:
                synthesis_input = texttospeech.SynthesisInput(text=sentence)
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=self.voice,
                    audio_config=self.audio_config,
                )
                # WAVヘッダをスキップ
                audio_data = response.audio_content[44:]
                audio_buffers.append(audio_data)
            except Exception as e:
                print(f"   ⚠️ TTS合成エラー: {e}")
        
        if not audio_buffers:
            return
        
        # 合成完了後、連続再生
        print(f"   🔊 TTS再生中...")
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=TTS_SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device_index,
        )
        
        try:
            for audio_data in audio_buffers:
                stream.write(audio_data)
        finally:
            stream.stop_stream()
            stream.close()
    
    def _split_text(self, text: str, max_length: int = 200) -> list:
        """テキストを文単位で分割"""
        sentences = re.split(r'(?<=[。！？])', text)
        result = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) > max_length:
                if current:
                    result.append(current)
                current = sentence
            else:
                current += sentence
        
        if current:
            result.append(current)
        
        return result

# ============================================================
# REST API処理（検索・資料参照用）
# ============================================================

class RestAPIHandler:
    """REST APIで検索・資料参照を処理"""
    
    def __init__(self, mode: str):
        self.mode = mode
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY_OVERRIDE
        self.client = genai.Client(api_key=api_key)
        self.chat = None
        self.pdf_file = None
        self._init_chat()
    
    def _init_chat(self):
        """チャットセッションを初期化"""
        # PDFがあればアップロード
        if os.path.exists(REFERENCE_PDF_FILE_PATH):
            try:
                self.pdf_file = self.client.files.upload(file=REFERENCE_PDF_FILE_PATH)
                print(f"📄 参照PDF読み込み完了: {REFERENCE_PDF_FILE_PATH}")
            except Exception as e:
                print(f"⚠️ PDF読み込みエラー: {e}")
        
        # チャットセッション作成
        system_instruction = REST_API_SYSTEM_INSTRUCTION
        
        # モード別の追加コンテキスト
        if self.mode == 'interview' and os.path.exists(INTERVIEW_SCRIPT_FILE_PATH):
            with open(INTERVIEW_SCRIPT_FILE_PATH, 'r', encoding='utf-8') as f:
                script = f.read()
            system_instruction += f"\n\n--- インタビュースクリプト ---\n{script}\n---"
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        
        self.chat = self.client.chats.create(
            model=REST_API_MODEL,
            config=config,
        )
    
    def query(self, prompt: str) -> str:
        """REST APIでクエリを実行"""
        try:
            # PDFがある場合は参照として追加
            if self.pdf_file:
                prompt += "\n\n※参考資料（PDF）も参照して回答してください。"
            
            response = self.chat.send_message(prompt)
            
            if response and response.text:
                return response.text.strip()
            return "申し訳ありません、回答を生成できませんでした。"
        
        except Exception as e:
            print(f"❌ REST APIエラー: {e}")
            return f"エラーが発生しました: {e}"

# ============================================================
# Gemini Live API アプリケーション（Function Callingハイブリッド）
# ============================================================

class GeminiLiveApp:
    """Gemini Live APIを使用した会議アシスタント"""
    
    # セッション再接続の閾値（大幅に緩和）
    MAX_AI_CHARS_BEFORE_RECONNECT = 800  # 累積上限を大幅に上げる
    LONG_SPEECH_THRESHOLD = 500  # 長い発話の閾値も上げる
    
    def __init__(self, mode: str, input_device_index: int, output_device_index: int):
        self.mode = mode
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        
        self.p = pyaudio.PyAudio()
        self.audio_queue_output = None
        self.audio_queue_mic = None
        self.mic_stream = None
        self.speaker_stream = None
        
        # トランスクリプトバッファ
        self.user_transcript_buffer = ""
        self.ai_transcript_buffer = ""
        self.conversation_history = []  # 会話履歴
        
        # セッション再接続用カウンター
        self.ai_char_count = 0  # AI発話文字数の累積
        self.needs_reconnect = False  # 再接続フラグ
        self.session_count = 0  # セッション番号
        
        # Gen AI クライアント
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY_OVERRIDE
        self.client = genai.Client(api_key=api_key)
        
        # REST APIハンドラとTTSプレイヤー
        self.rest_handler = RestAPIHandler(mode)
        self.tts_player = TTSPlayer(output_device_index)
        
        # システムインストラクション構築
        self.system_instruction = self._build_system_instruction()
        
        # Live API 設定（Function Calling有効）
        self.config = self._build_config()
    
    def _build_config(self, with_context: str = None) -> dict:
        """Live API設定を構築"""
        instruction = self.system_instruction
        
        # 再接続時は会話履歴を追加し、相槌を指示
        if with_context:
            # 直前のユーザー発言を抽出
            last_user_message = ""
            for h in reversed(self.conversation_history):
                if h['role'] == 'ユーザー':
                    last_user_message = h['text'][:100]
                    break
            
            # 次の質問を取得
            next_q = self._get_next_question_from_script()
            
            instruction += f"""

【これまでの会話の要約】
{with_context}

【重要：必ず守ること】
1. 直前の話者の発言「{last_user_message}」に対して短い相槌を入れる
2. 既に聞いた質問は絶対に繰り返さない
3. 次に聞くべき質問：「{next_q[:100]}」

【応答パターン】
「〜ということですね。では、{next_q[:50]}」
"""
        
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": instruction,
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "speech_config": {
                "language_code": "ja-JP",
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
        
        # ツールがある場合のみ追加
        tools = get_interview_tools()
        if tools:
            config["tools"] = tools
        
        return config
    
    def _build_system_instruction(self) -> str:
        """モードに応じたシステムインストラクションを構築"""
        if self.mode == 'interview':
            instruction = INTERVIEW_SYSTEM_INSTRUCTION
            if os.path.exists(INTERVIEW_SCRIPT_FILE_PATH):
                with open(INTERVIEW_SCRIPT_FILE_PATH, 'r', encoding='utf-8') as f:
                    script = f.read()
                instruction += f"\n\n--- インタビュースクリプト ---\n{script}\n---"
            return instruction
        elif self.mode == 'silent':
            return SILENT_SYSTEM_INSTRUCTION
        else:
            instruction = STANDARD_SYSTEM_INSTRUCTION
            if os.path.exists(MEETING_SUMMARY_FILE_PATH):
                with open(MEETING_SUMMARY_FILE_PATH, 'r', encoding='utf-8') as f:
                    summary = f.read()
                instruction += f"\n\n--- 会議の背景情報 ---\n{summary}\n---"
            return instruction
    
    def _add_to_history(self, role: str, text: str):
        """会話履歴に追加"""
        self.conversation_history.append({"role": role, "text": text})
        # 直近20ターンを保持
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def _get_history_string(self) -> str:
        """会話履歴を文字列で取得"""
        return "\n".join([f"{h['role']}: {h['text']}" for h in self.conversation_history])
    
    def _is_speech_incomplete(self, text: str) -> bool:
        """発言が途中で切れているかチェック"""
        if not text:
            return False
        
        text = text.strip()
        
        # 正常な終わり方のパターン
        normal_endings = ['。', '？', '?', '！', '!', 'か?', 'か？', 'ます', 'です', 'ね', 'よ', 'した', 'ください']
        
        for ending in normal_endings:
            if text.endswith(ending):
                return False
        
        # 途中で切れている可能性が高いパターン
        incomplete_patterns = ['、', 'の', 'を', 'が', 'は', 'に', 'で', 'と', 'も', 'や']
        
        for pattern in incomplete_patterns:
            if text.endswith(pattern):
                return True
        
        # 最後の文字がひらがな・カタカナで、文末っぽくない場合
        last_char = text[-1]
        if last_char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン':
            # 「ね」「よ」「か」などは正常
            if last_char not in 'ねよかなわ':
                return True
        
        return False
    
    def _get_next_question_from_script(self) -> str:
        """インタビュースクリプトから、まだ聞いていない次の質問を取得"""
        try:
            if not os.path.exists(INTERVIEW_SCRIPT_FILE_PATH):
                return "次の質問に進んでください"
            
            with open(INTERVIEW_SCRIPT_FILE_PATH, 'r', encoding='utf-8') as f:
                script = f.read()
            
            # 質問を抽出（[質問N]パターン）
            import re
            questions = re.findall(r'\[質問\d+\]\s*\n([^\[]+)', script)
            
            if not questions:
                return "次の質問に進んでください"
            
            # 会話履歴から既に聞いた質問のキーワードを抽出
            history_text = self._get_history_string()
            
            for q in questions:
                q_clean = q.strip()
                if not q_clean:
                    continue
                
                # 質問の主要キーワードを抽出（最初の名詞・キーワード）
                keywords = []
                for word in ['健康診断', '健康経営', 'ウェルテクト', '常温', 'ビッグデータ', 
                            'ポイントモール', '導入', 'メリット', '経営者', '人事', '締めくくり']:
                    if word in q_clean:
                        keywords.append(word)
                
                # キーワードが履歴に含まれているかチェック
                found_in_history = False
                for kw in keywords:
                    if kw in history_text:
                        found_in_history = True
                        break
                
                if not found_in_history:
                    return q_clean
            
            # 全部聞いた場合
            return "本日のインタビューはこれで終了です。ありがとうございました。"
            
        except Exception as e:
            print(f"⚠️ スクリプト読み込みエラー: {e}")
            return "次の質問に進んでください"
    
    async def receive_audio(self, session):
        """Live APIからの応答を受信"""
        while not self.needs_reconnect:
            turn = session.receive()
            async for response in turn:
                # 再接続フラグが立ったら終了
                if self.needs_reconnect:
                    return
                
                # tool_callイベントを検知（Live API形式）
                if hasattr(response, 'tool_call') and response.tool_call:
                    await self._handle_tool_call(response.tool_call, session)
                    continue
                
                if response.server_content:
                    sc = response.server_content
                    
                    # model_turn内のfunction_callは無視（tool_callで処理する）
                    # ※ tool_callにはidがあるが、function_callにはない場合があるため
                    
                    # ターン完了
                    if hasattr(sc, 'turn_complete') and sc.turn_complete:
                        if self.user_transcript_buffer.strip():
                            user_text = self.user_transcript_buffer.strip()
                            print(f"👤 ユーザー: {user_text}")
                            log_transcript(f"[ユーザー] {user_text}", TRANSCRIPT_FILE_PATH)
                            self._add_to_history("ユーザー", user_text)
                            self.user_transcript_buffer = ""
                            
                            # ユーザー発言後に再接続が必要なら実行
                            if self.needs_reconnect:
                                print("🔄 再接続を実行します...")
                                return
                        
                        if self.ai_transcript_buffer.strip():
                            ai_text = self.ai_transcript_buffer.strip()
                            print(f"🤖 AI(Live): {ai_text}")
                            log_transcript(f"[AI] {ai_text}", TRANSCRIPT_FILE_PATH)
                            self._add_to_history("AI", ai_text)
                            
                            # 発言が途中で切れているかチェック
                            is_incomplete = self._is_speech_incomplete(ai_text)
                            if is_incomplete:
                                print("   ⚠️ 発言が途中で切れた可能性あり")
                            
                            # 文字数をカウント
                            char_count = len(ai_text)
                            self.ai_char_count += char_count
                            remaining = self.MAX_AI_CHARS_BEFORE_RECONNECT - self.ai_char_count
                            print(f"   (累積: {self.ai_char_count}文字 / 残り: {remaining}文字)")
                            
                            self.ai_transcript_buffer = ""
                            
                            # 発言が途切れた場合は即座に再接続
                            if is_incomplete:
                                print("🔄 発言途切れのため即時再接続します...")
                                self.needs_reconnect = True
                            # 長い発話（80文字以上）をした場合、次で途切れるリスクが高いので再接続
                            elif char_count >= self.LONG_SPEECH_THRESHOLD:
                                print(f"🔄 長い発話({char_count}文字)のため次のターン前に再接続します。")
                                self.needs_reconnect = True
                            # 累積が上限に近づいた場合
                            elif self.ai_char_count >= self.MAX_AI_CHARS_BEFORE_RECONNECT:
                                print("🔄 累積制限に近づいています。再接続します。")
                                self.needs_reconnect = True
                        
                        print("--- ターン完了 ---")
                    
                    if hasattr(sc, 'generation_complete') and sc.generation_complete:
                        print("--- 生成完了 ---")
                
                # 割り込み検知
                if response.server_content and hasattr(response.server_content, 'interrupted'):
                    if response.server_content.interrupted:
                        print("🚨 割り込み検知！")
                        if self.ai_transcript_buffer.strip():
                            print(f"🤖 AI(中断): {self.ai_transcript_buffer.strip()}")
                            self.ai_transcript_buffer = ""
                        while not self.audio_queue_output.empty():
                            try:
                                self.audio_queue_output.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        continue
                
                # 入力トランスクリプション（バッファに蓄積、最終的にまとめて表示）
                if response.server_content and hasattr(response.server_content, 'input_transcription'):
                    if response.server_content.input_transcription:
                        user_text = response.server_content.input_transcription.text
                        if user_text:
                            self.user_transcript_buffer += user_text
                
                # 出力トランスクリプション
                if response.server_content and hasattr(response.server_content, 'output_transcription'):
                    if response.server_content.output_transcription:
                        ai_text = response.server_content.output_transcription.text
                        if ai_text:
                            self.ai_transcript_buffer += ai_text
                
                # 音声データを受信
                if response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if isinstance(part.inline_data.data, bytes):
                                self.audio_queue_output.put_nowait(part.inline_data.data)
    
    async def _handle_tool_call(self, tool_call, session):
        """Function Call（ツール呼び出し）を処理"""
        for fc in tool_call.function_calls:
            if fc.name == "request_explanation":
                topic = fc.args.get("topic", "") if fc.args else ""
                print(f"🔧 説明依頼検知: {topic}")
                
                # ツール結果を返す（説明だけ、次の質問はしない）
                instruction = f"「{topic}」について2〜3文で簡潔に説明してください。説明が終わったら「以上です」と言って止まってください。次の質問はしないでください。"
                
                try:
                    await session.send_tool_response(
                        function_responses=[
                            types.FunctionResponse(
                                name=fc.name,
                                id=fc.id,
                                response={
                                    "status": "ready",
                                    "instruction": instruction
                                }
                            )
                        ]
                    )
                    print("   ✅ ツール応答送信完了、Live APIの説明を待機中...")
                    # ※ ここでは再接続フラグを立てない！
                    # Live APIが説明を完了した後、通常のターン完了で累積制限により再接続される
                except Exception as e:
                    print(f"⚠️ ツール応答エラー: {e}")
    
    async def run(self):
        """メインループ（再接続対応）"""
        self.audio_queue_output = asyncio.Queue()
        self.audio_queue_mic = asyncio.Queue(maxsize=5)
        
        initialize_transcript(TRANSCRIPT_FILE_PATH, self.mode)
        
        mode_names = {
            'standard': 'スタンダードモード',
            'silent': 'サイレントモード',
            'interview': 'インタビューモード',
        }
        
        print("\n" + "=" * 60)
        print(f"🎙️  Gemini Live API - {mode_names.get(self.mode, self.mode)}")
        print("    （Live API統一方式 + 自動再接続）")
        print("=" * 60)
        print("⚠️  ヘッドセット推奨（エコー防止のため）")
        print("💡 全てLive APIで処理")
        print(f"💡 累積{self.MAX_AI_CHARS_BEFORE_RECONNECT}文字で自動再接続")
        print("💡 Ctrl+C で終了")
        print("=" * 60 + "\n")
        
        # マイク・スピーカーの初期化
        await self._init_audio_streams()
        
        try:
            while True:
                self.session_count += 1
                self.ai_char_count = 0  # 文字数カウントをリセット
                self.needs_reconnect = False
                
                # 再接続時は会話履歴を引き継ぐ
                context = None
                if self.session_count > 1:
                    context = self._get_context_summary()
                    print(f"\n🔄 セッション再接続 (#{self.session_count})")
                    if context:
                        print(f"   引き継ぎコンテキスト: {context[:80]}...")
                
                config = self._build_config(with_context=context)
                
                try:
                    async with self.client.aio.live.connect(
                        model=LIVE_API_MODEL,
                        config=config
                    ) as session:
                        if self.session_count == 1:
                            print("✅ Gemini Live API に接続しました。話しかけてください！\n")
                        else:
                            print("✅ 再接続完了。続けてください。\n")
                            # 再接続時は、まずテキストで挨拶を送って応答を促す
                            try:
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user", 
                                        parts=[types.Part(text="続きをお願いします")]
                                    ),
                                    turn_complete=True
                                )
                                print("   📤 再接続通知を送信しました")
                            except Exception as e:
                                print(f"   ⚠️ 再接続通知送信エラー: {e}")
                        
                        # セッションループ
                        await self._session_loop(session)
                        
                        # 再接続が必要な場合はループを継続
                        if not self.needs_reconnect:
                            break
                            
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"❌ セッションエラー: {e}")
                    
                    # 再接続可能なエラーかどうか判定
                    if any(keyword in error_msg for keyword in ["1011", "internal error", "disconnected", "closed", "websocket"]):
                        print("🔄 接続エラー。3秒後に再接続します...")
                        await asyncio.sleep(3)
                        self.needs_reconnect = True
                        continue
                    else:
                        raise
        
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    async def _init_audio_streams(self):
        """マイクとスピーカーのストリームを初期化"""
        device_info = self.p.get_device_info_by_index(self.input_device_index)
        print(f"🎤 入力: {device_info['name']}")
        
        self.mic_stream = await asyncio.to_thread(
            self.p.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        device_info = self.p.get_device_info_by_index(self.output_device_index)
        print(f"🔊 出力: {device_info['name']}")
        
        self.speaker_stream = await asyncio.to_thread(
            self.p.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device_index,
        )
    
    async def _session_loop(self, session):
        """1つのセッション内のメインループ"""
        
        async def listen_audio():
            """マイクから音声を取得してキューに入れる"""
            kwargs = {"exception_on_overflow": False}
            send_count = 0
            while not self.needs_reconnect:
                try:
                    data = await asyncio.to_thread(self.mic_stream.read, CHUNK_SIZE, **kwargs)
                    try:
                        self.audio_queue_mic.put_nowait({"data": data, "mime_type": "audio/pcm"})
                        send_count += 1
                        if send_count <= 3:
                            print(f"   🎤 音声送信中... ({send_count})")
                    except asyncio.QueueFull:
                        pass
                except Exception as e:
                    if self.needs_reconnect:
                        return
                    print(f"⚠️ マイク読み取りエラー: {e}")
                    self.needs_reconnect = True
                    return
        
        async def send_audio():
            """キューから音声を取得してLive APIに送信"""
            send_count = 0
            while not self.needs_reconnect:
                try:
                    msg = await asyncio.wait_for(
                        self.audio_queue_mic.get(),
                        timeout=0.1
                    )
                    await session.send_realtime_input(audio=msg)
                    send_count += 1
                    # 定期的に送信確認
                    if send_count % 500 == 0:
                        print(f"   📡 音声送信継続中... ({send_count})")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self.needs_reconnect:
                        return
                    print(f"⚠️ 送信エラー: {e}")
                    self.needs_reconnect = True
                    return
        
        async def play_audio():
            while not self.needs_reconnect:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_queue_output.get(),
                        timeout=0.1
                    )
                    await asyncio.to_thread(self.speaker_stream.write, audio_data)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self.needs_reconnect:
                        return
                    print(f"⚠️ 再生エラー: {e}")
                    return
        
        async def receive():
            try:
                await self.receive_audio(session)
            except Exception as e:
                if self.needs_reconnect:
                    return
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["1011", "1008", "internal error", "closed", "deadline", "policy"]):
                    if "deadline" in error_msg:
                        print(f"⏱️ サーバータイムアウト。再接続します...")
                    elif "1008" in error_msg or "policy" in error_msg:
                        print(f"⚠️ ポリシーエラー。再接続します...")
                    else:
                        print(f"⚠️ 受信エラー（再接続します）: {e}")
                    self.needs_reconnect = True
                else:
                    raise
        
        # キューをクリア（再接続時に古いデータが残らないように）
        while not self.audio_queue_mic.empty():
            try:
                self.audio_queue_mic.get_nowait()
            except:
                break
        while not self.audio_queue_output.empty():
            try:
                self.audio_queue_output.get_nowait()
            except:
                break
        
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_audio())
                tg.create_task(send_audio())
                tg.create_task(receive())
                tg.create_task(play_audio())
        except* Exception as eg:
            if not self.needs_reconnect:
                for e in eg.exceptions:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ["1011", "internal error", "closed", "websocket"]):
                        self.needs_reconnect = True
                    else:
                        print(f"タスクエラー: {e}")
    
    def _get_context_summary(self) -> str:
        """会話履歴の要約を取得"""
        if not self.conversation_history:
            return ""
        
        # 直近10ターンを取得
        recent = self.conversation_history[-10:]
        summary_parts = []
        
        for h in recent:
            role = h['role']
            text = h['text'][:150]  # 150文字まで
            summary_parts.append(f"{role}: {text}")
        
        summary = "\n".join(summary_parts)
        
        # 最後のAI発言が質問なら強調
        last_ai = None
        for h in reversed(self.conversation_history):
            if h['role'] == 'AI':
                last_ai = h['text']
                break
        
        if last_ai and ('?' in last_ai or '？' in last_ai or 'か?' in last_ai or 'か？' in last_ai):
            summary += f"\n\n【直前の質問（これに対する回答を待っています）】\n{last_ai[:200]}"
        
        return summary
    
    def cleanup(self):
        """リソース解放"""
        if self.mic_stream:
            self.mic_stream.close()
        if self.speaker_stream:
            self.speaker_stream.close()
        self.p.terminate()
        print("\n👋 接続を終了しました。")

# ============================================================
# メイン
# ============================================================

def find_device_index(p: pyaudio.PyAudio, device_name: str, is_input: bool) -> int:
    """デバイス名からインデックスを検索"""
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if device_name in info.get('name', ''):
            if is_input and info.get('maxInputChannels', 0) > 0:
                return i
            elif not is_input and info.get('maxOutputChannels', 0) > 0:
                return i
    return -1


def main():
    parser = argparse.ArgumentParser(description='Gemini Live API Meeting Assistant (Hybrid)')
    parser.add_argument('--silent', action='store_true', help='サイレントモード')
    parser.add_argument('--interview', action='store_true', help='インタビューモード')
    args = parser.parse_args()
    
    # モード決定
    if args.interview:
        mode = 'interview'
    elif args.silent:
        mode = 'silent'
    else:
        mode = 'standard'
    
    mode_names = {
        'standard': 'スタンダードモード (アクティブ介入)',
        'silent': 'サイレントモード (書記専念)',
        'interview': 'インタビューモード (進行支援)',
    }
    
    print(f"\n{'='*60}")
    print(f"★ 起動モード: {mode_names[mode]}")
    print(f"{'='*60}\n")
    
    # GCP Project設定
    if GCP_PROJECT_ID:
        os.environ['GOOGLE_CLOUD_PROJECT'] = GCP_PROJECT_ID
    
    # PyAudio初期化
    p = pyaudio.PyAudio()
    
    # デバイス検索
    input_device_index = find_device_index(p, INPUT_DEVICE_NAME, is_input=True)
    output_device_index = find_device_index(p, TTS_OUTPUT_DEVICE_NAME, is_input=False)
    
    if input_device_index == -1:
        print(f"❌ 入力デバイスが見つかりません: {INPUT_DEVICE_NAME}")
        print("\n利用可能なデバイス:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                print(f"  [{i}] {info['name']}")
        p.terminate()
        return
    
    if output_device_index == -1:
        print(f"❌ 出力デバイスが見つかりません: {TTS_OUTPUT_DEVICE_NAME}")
        print("\n利用可能なデバイス:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxOutputChannels', 0) > 0:
                print(f"  [{i}] {info['name']}")
        p.terminate()
        return
    
    print(f"✓ 入力デバイス: [{input_device_index}] {INPUT_DEVICE_NAME}")
    print(f"✓ 出力デバイス: [{output_device_index}] {TTS_OUTPUT_DEVICE_NAME}")
    
    p.terminate()
    
    # アプリ起動
    app = GeminiLiveApp(mode, input_device_index, output_device_index)
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n⏹️  ユーザーによる中断")


if __name__ == "__main__":
    main()
