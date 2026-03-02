"""
短期記憶 (SessionMemory)

stt_stream.py L389, L490-495, L940-966 から移植。
セッション内の会話コンテキストを維持し、Live API再接続時のコンテキスト引き継ぎを担う。
"""

from datetime import datetime


class SessionMemory:
    """短期記憶 — セッション内インメモリ"""

    MAX_HISTORY = 20            # stt_stream.py L493-495: 直近20ターン保持
    CONTEXT_SUMMARY_TURNS = 10  # stt_stream.py L946: 要約は直近10ターン
    MAX_TEXT_IN_SUMMARY = 150   # stt_stream.py L951: 要約内テキスト上限

    def __init__(self):
        self.history: list[dict] = []

    def add(self, role: str, text: str) -> None:
        """
        会話ターンを追加

        stt_stream.py L490-495:
        conversation_history.append({"role": role, "text": text})
        直近20ターンを保持
        """
        self.history.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def get_context_summary(self) -> str:
        """
        再接続時のコンテキスト要約を生成

        stt_stream.py L940-966 を移植:
        - 直近10ターンを取得
        - 各ターンの先頭150文字を要約
        - 最後のAI発言が質問なら強調
        """
        if not self.history:
            return ""

        recent = self.history[-self.CONTEXT_SUMMARY_TURNS:]
        parts = [
            f"{h['role']}: {h['text'][:self.MAX_TEXT_IN_SUMMARY]}"
            for h in recent
        ]
        summary = "\n".join(parts)

        # 最後のAI発言が質問なら強調
        for h in reversed(self.history):
            if h["role"] == "AI":
                if any(q in h["text"] for q in ["?", "？"]):
                    summary += (
                        f"\n\n【直前の質問（これに対する回答を待っています）】\n"
                        f"{h['text'][:200]}"
                    )
                break

        return summary

    def get_history_string(self) -> str:
        """会話履歴を文字列で取得 (stt_stream.py L497-499)"""
        return "\n".join(
            f"{h['role']}: {h['text']}" for h in self.history
        )

    def get_last_user_message(self, max_len: int = 100) -> str:
        """直前のユーザー発言を取得 (stt_stream.py L417-420)"""
        for h in reversed(self.history):
            if h["role"] == "ユーザー":
                return h["text"][:max_len]
        return ""

    def clear(self) -> None:
        """履歴をクリア"""
        self.history.clear()
