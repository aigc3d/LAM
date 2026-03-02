"""
モードプラグイン基底クラス

各モード（グルメ、サポート、インタビュー等）はこの基底クラスを継承し、
モード固有のシステムプロンプト、ツール定義、記憶スキーマを提供する。
"""

from abc import ABC, abstractmethod


class BaseModePlugin(ABC):
    """モードプラグインの基底クラス"""

    @property
    @abstractmethod
    def name(self) -> str:
        """モード識別名 (例: "gourmet")"""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """表示用モード名 (例: "グルメコンシェルジュ")"""
        ...

    @property
    def default_dialogue_type(self) -> str:
        """デフォルトの対話方式 ("rest" | "live" | "hybrid")"""
        return "live"

    @abstractmethod
    def get_system_prompt(self, language: str = "ja", context: str | None = None) -> str:
        """
        システムプロンプトを生成

        Args:
            language: セッション言語
            context: 再接続時のコンテキスト要約 (None=初回接続)
        """
        ...

    def get_live_api_tools(self) -> list:
        """Live API用のFunction Callingツール定義"""
        return []

    def get_memory_schema(self) -> dict:
        """長期記憶のモード別スキーマ定義"""
        return {}

    def get_initial_greeting(self, language: str = "ja", user_profile: dict | None = None) -> str:
        """
        初回挨拶メッセージを生成

        Args:
            language: セッション言語
            user_profile: 長期記憶のユーザープロファイル (パーソナライズ用)
        """
        return ""
