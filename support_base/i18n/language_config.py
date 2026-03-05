"""
言語マスター設定

gourmet-sp の CoreController.LANGUAGE_CODE_MAP に相当する設定を
プラットフォーム共通基盤として提供する。

[確認済み] concierge-controller.ts L526-546: ja/zh → 。分割、en/ko → . 分割
[確認済み] stt_stream.py L221: ja-JP-Wavenet-D
[推定] 他言語のTTS voice名は gourmet-sp リポジトリ確認後に正確な値で更新
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageProfile:
    """1言語の設定プロファイル"""
    code: str                    # "ja", "en", "ko", "zh"
    tts_language_code: str       # Google Cloud TTS の language_code
    tts_voice_name: str          # Google Cloud TTS の voice name
    live_api_language_code: str  # Gemini Live API の speech_config.language_code
    sentence_splitter: str       # "cjk" (。で分割) or "latin" (. で分割)
    display_name: str            # 表示用言語名


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "ja": LanguageProfile(
        code="ja",
        tts_language_code="ja-JP",
        tts_voice_name="ja-JP-Wavenet-D",
        live_api_language_code="ja-JP",
        sentence_splitter="cjk",
        display_name="日本語",
    ),
    "en": LanguageProfile(
        code="en",
        tts_language_code="en-US",
        tts_voice_name="en-US-Wavenet-D",
        live_api_language_code="en-US",
        sentence_splitter="latin",
        display_name="English",
    ),
    "ko": LanguageProfile(
        code="ko",
        tts_language_code="ko-KR",
        tts_voice_name="ko-KR-Wavenet-D",
        live_api_language_code="ko-KR",
        sentence_splitter="latin",
        display_name="한국어",
    ),
    "zh": LanguageProfile(
        code="zh",
        tts_language_code="cmn-CN",
        tts_voice_name="cmn-CN-Wavenet-D",
        live_api_language_code="cmn-CN",
        sentence_splitter="cjk",
        display_name="中文",
    ),
}


def get_language_profile(code: str) -> LanguageProfile:
    """言語プロファイルを取得（デフォルトは日本語）"""
    return LANGUAGE_PROFILES.get(code, LANGUAGE_PROFILES["ja"])
