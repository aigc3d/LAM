"""
A2E日本語音声テスト用: テスト音声ファイル生成スクリプト

EdgeTTSを使って日本語テスト音声を生成する。
OpenAvatarChatと同じ ja-JP-NanamiNeural voice を使用。

使い方:
    cd C:\Users\hamad\OpenAvatarChat
    conda activate oac
    python tests/a2e_japanese/generate_test_audio.py

出力:
    tests/a2e_japanese/audio_samples/ に WAV ファイルが生成される
"""

import asyncio
import os
import sys
import wave
import struct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_samples")

# テストケース: 日本語音声サンプル
# phoneme_test: 母音の網羅性テスト
# greeting: 日常的なフレーズ
# long_sentence: 長文での自然さテスト
# english_compare: 英語比較用
TEST_CASES = [
    {
        "id": "vowels_aiueo",
        "text": "あ、い、う、え、お",
        "lang": "ja",
        "description": "Japanese vowels (a, i, u, e, o) - basic lip shape test",
    },
    {
        "id": "greeting_konnichiwa",
        "text": "こんにちは、お元気ですか？今日はとても良い天気ですね。",
        "lang": "ja",
        "description": "Japanese greeting - natural conversation test",
    },
    {
        "id": "long_sentence",
        "text": "私はAIコンシェルジュです。何かお手伝いできることがあれば、お気軽にお声がけください。",
        "lang": "ja",
        "description": "Japanese service phrase - longer utterance test",
    },
    {
        "id": "mixed_phonemes",
        "text": "さしすせそ、たちつてと、なにぬねの、はひふへほ、まみむめも",
        "lang": "ja",
        "description": "Japanese consonant+vowel combinations - comprehensive phoneme coverage",
    },
    {
        "id": "numbers_and_names",
        "text": "東京タワーの高さは三百三十三メートルです。富士山は三千七百七十六メートルです。",
        "lang": "ja",
        "description": "Numbers and proper nouns - complex articulation test",
    },
    {
        "id": "english_compare",
        "text": "Hello, how are you? I'm doing great, thank you for asking.",
        "lang": "en",
        "description": "English comparison - to compare A2E output quality",
    },
    {
        "id": "chinese_compare",
        "text": "你好，我是AI助手，很高兴认识你。",
        "lang": "zh",
        "description": "Chinese comparison - original reference language",
    },
]

# EdgeTTS voice mapping
VOICE_MAP = {
    "ja": "ja-JP-NanamiNeural",
    "en": "en-US-JennyNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}


async def generate_with_edge_tts(text: str, voice: str, output_path: str):
    """EdgeTTSで音声を生成してWAVで保存"""
    try:
        import edge_tts
    except ImportError:
        print("ERROR: edge-tts not installed. Run: pip install edge-tts")
        sys.exit(1)

    mp3_path = output_path.replace(".wav", ".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(mp3_path)

    # MP3 → WAV 変換 (24kHz, mono, 16bit)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
        os.remove(mp3_path)
        return True
    except ImportError:
        # pydubがない場合はffmpegで変換
        import subprocess
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, "-ar", "24000", "-ac", "1",
                 "-sample_fmt", "s16", output_path],
                capture_output=True, check=True,
            )
            os.remove(mp3_path)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  WARNING: Could not convert to WAV. Keeping MP3: {mp3_path}")
            print("  Install pydub (pip install pydub) or ffmpeg for WAV conversion.")
            return False


def generate_sine_tone(output_path: str, freq: float = 440.0, duration: float = 1.0,
                       sample_rate: int = 24000):
    """サイン波テスト音声（無音声参照用）"""
    n_samples = int(sample_rate * duration)
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            value = int(16000 * __import__("math").sin(2 * __import__("math").pi * freq * t))
            wf.writeframes(struct.pack("<h", value))


def generate_silence(output_path: str, duration: float = 2.0, sample_rate: int = 24000):
    """無音テスト音声（ベースライン用）"""
    n_samples = int(sample_rate * duration)
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)


async def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("=" * 60)
    print("A2E Japanese Audio Test - Audio Sample Generator")
    print("=" * 60)

    # 1. EdgeTTS音声生成
    for case in TEST_CASES:
        voice = VOICE_MAP.get(case["lang"], VOICE_MAP["ja"])
        output_path = os.path.join(AUDIO_DIR, f"{case['id']}.wav")

        print(f"\n[{case['id']}] {case['description']}")
        print(f"  Text: {case['text'][:50]}...")
        print(f"  Voice: {voice}")

        if os.path.exists(output_path):
            print(f"  SKIP: Already exists at {output_path}")
            continue

        success = await generate_with_edge_tts(case["text"], voice, output_path)
        if success:
            # WAVファイル情報表示
            with wave.open(output_path, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
                print(f"  OK: {output_path} ({duration:.2f}s, {rate}Hz)")
        else:
            print(f"  WARN: MP3 only (WAV conversion failed)")

    # 2. ベースライン音声
    silence_path = os.path.join(AUDIO_DIR, "silence_baseline.wav")
    if not os.path.exists(silence_path):
        print(f"\n[silence_baseline] Generating 2s silence...")
        generate_silence(silence_path)
        print(f"  OK: {silence_path}")

    tone_path = os.path.join(AUDIO_DIR, "tone_440hz.wav")
    if not os.path.exists(tone_path):
        print(f"\n[tone_440hz] Generating 1s 440Hz sine tone...")
        generate_sine_tone(tone_path)
        print(f"  OK: {tone_path}")

    print("\n" + "=" * 60)
    print(f"Generated audio samples in: {AUDIO_DIR}")
    print("=" * 60)

    # 3. サマリー
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"\nTotal WAV files: {len(wav_files)}")
    for f in sorted(wav_files):
        fpath = os.path.join(AUDIO_DIR, f)
        try:
            with wave.open(fpath, "r") as wf:
                duration = wf.getnframes() / wf.getframerate()
                print(f"  {f}: {duration:.2f}s")
        except Exception:
            print(f"  {f}: (could not read)")


if __name__ == "__main__":
    asyncio.run(main())
