"""
発話途切れ検知

stt_stream.py L501-529 から移植。
Live API の FLASH版で発話が途中で切れた場合を検知し、即時再接続を促す。
"""


class SpeechDetector:
    """発話途切れ検知（多言語対応）"""

    # 言語別ルール
    # [確認済み] stt_stream.py L509-527 の日本語ルールを移植
    # 他言語は段階的に追加
    RULES: dict[str, dict] = {
        "ja": {
            "normal_endings": [
                "。", "？", "?", "！", "!", "ます", "です",
                "ね", "よ", "した", "ください",
            ],
            "incomplete_patterns": [
                "、", "の", "を", "が", "は", "に", "で", "と", "も", "や",
            ],
            # ひらがな・カタカナのうち文末として不自然な文字
            "check_trailing_kana": True,
            "safe_trailing": "ねよかなわ",
        },
        "en": {
            "normal_endings": [".", "?", "!", "right", "okay"],
            "incomplete_patterns": [",", " and", " but", " or", " the", " a"],
            "check_trailing_kana": False,
        },
        "ko": {
            "normal_endings": [".", "?", "!", "요", "다", "죠"],
            "incomplete_patterns": [",", "는", "을", "를", "이", "가", "에"],
            "check_trailing_kana": False,
        },
        "zh": {
            "normal_endings": ["。", "？", "！", "了", "吗", "呢"],
            "incomplete_patterns": ["，", "的", "和", "在", "是"],
            "check_trailing_kana": False,
        },
    }

    @staticmethod
    def is_incomplete(text: str, language: str = "ja") -> bool:
        """
        発言が途中で切れているかチェック

        Returns:
            True: 途中で切れている可能性が高い → 再接続推奨
            False: 正常に終了している
        """
        if not text:
            return False

        text = text.strip()
        if not text:
            return False

        rules = SpeechDetector.RULES.get(language)
        if not rules:
            return False  # ルール未定義の言語は安全側(False)

        # 正常な終わり方チェック
        for ending in rules["normal_endings"]:
            if text.endswith(ending):
                return False

        # 途中切れパターンチェック
        for pattern in rules["incomplete_patterns"]:
            if text.endswith(pattern):
                return True

        # 日本語: ひらがな・カタカナの文末チェック
        if rules.get("check_trailing_kana"):
            last_char = text[-1]
            kana = (
                "あいうえおかきくけこさしすせそたちつてと"
                "なにぬねのはひふへほまみむめもやゆよ"
                "らりるれろわをん"
                "アイウエオカキクケコサシスセソタチツテト"
                "ナニヌネノハヒフヘホマミムメモヤユヨ"
                "ラリルレロワヲン"
            )
            if last_char in kana:
                safe = rules.get("safe_trailing", "")
                if last_char not in safe:
                    return True

        return False
