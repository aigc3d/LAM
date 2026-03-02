"""
グルメコンシェルジュ モードプラグイン

GCS/ローカルから読み込んだプロンプトを使用。
Live API / REST 両方の経路で使用。

プロンプトソース:
  - GCS: gs://{PROMPTS_BUCKET_NAME}/prompts/concierge_{lang}.txt
  - ローカル: prompts/concierge_{lang}.txt
  - フォールバック: ハードコードされた最小プロンプト
"""

import logging

from support_base.modes.base_mode import BaseModePlugin

logger = logging.getLogger(__name__)

# GCS/ローカルから読み込んだプロンプトを取得
try:
    from support_base.core.support_core import SYSTEM_PROMPTS as LOADED_PROMPTS
    from support_base.core.support_core import INITIAL_GREETINGS as LOADED_GREETINGS
    _PROMPTS_LOADED = True
    logger.info("[GourmetPlugin] GCS/ローカルプロンプト読み込み成功")
except Exception as e:
    logger.warning(f"[GourmetPlugin] プロンプト読み込み失敗 (フォールバック使用): {e}")
    LOADED_PROMPTS = {}
    LOADED_GREETINGS = {}
    _PROMPTS_LOADED = False


class GourmetModePlugin(BaseModePlugin):
    """グルメコンシェルジュモード"""

    @property
    def name(self) -> str:
        return "gourmet"

    @property
    def display_name(self) -> str:
        return "グルメコンシェルジュ"

    @property
    def default_dialogue_type(self) -> str:
        return "live"

    def get_system_prompt(self, language: str = "ja", context: str | None = None) -> str:
        """
        GCS から読み込んだプロンプトを優先使用。
        読み込み失敗時はハードコードのフォールバック。
        """
        prompt = ""

        # GCS/ローカルから読み込んだプロンプトを使用
        if _PROMPTS_LOADED:
            # Live API では concierge プロンプトを使用
            concierge_prompts = LOADED_PROMPTS.get("concierge", {})
            prompt = concierge_prompts.get(language, concierge_prompts.get("ja", ""))

            # concierge プロンプトがなければ chat プロンプトを試す
            if not prompt:
                chat_prompts = LOADED_PROMPTS.get("chat", {})
                prompt = chat_prompts.get(language, chat_prompts.get("ja", ""))

        # フォールバック
        if not prompt:
            prompt = self._fallback_prompt(language)

        # 再接続コンテキスト追加
        if context:
            prompt += f"\n\n【これまでの会話の要約】\n{context}\n"
            prompt += (
                "\n【重要：必ず守ること】\n"
                "1. 直前の話者の発言に対して短い相槌を入れる\n"
                "2. 既に聞いた質問は絶対に繰り返さない\n"
                "3. 会話の流れを自然に引き継ぐ\n"
            )

        return prompt

    def _fallback_prompt(self, language: str) -> str:
        """ハードコードのフォールバック (GCS 読み込み失敗時)"""
        prompts = {
            "ja": (
                "あなたはグルメコンシェルジュAIです。\n"
                "ユーザーの食の好み・気分・シチュエーションをヒアリングし、最適なレストランを提案します。\n\n"
                "【対話スタイル】\n"
                "- 親しみやすく、でも丁寧な口調で話してください\n"
                "- 短く簡潔に応答してください（1-2文程度）\n"
                "- ユーザーの好みを引き出す質問を積極的にしてください\n"
                "- 一度に複数の質問をしないこと（1つずつ聞く）\n"
            ),
            "en": (
                "You are a Gourmet Concierge AI.\n"
                "Help users find the perfect restaurant by understanding their preferences, mood, and occasion.\n\n"
                "Keep responses short (1-2 sentences). Ask questions to understand preferences.\n"
            ),
            "ko": (
                "당신은 맛집 컨시어지 AI입니다.\n"
                "사용자의 음식 취향과 상황을 파악하여 최적의 레스토랑을 추천합니다.\n\n"
                "짧고 간결하게 응답하세요 (1-2문장).\n"
            ),
            "zh": (
                "你是一个美食顾问AI。\n"
                "了解用户的饮食偏好、心情和场合，推荐最合适的餐厅。\n\n"
                "简短回复（1-2句）。\n"
            ),
        }
        return prompts.get(language, prompts["ja"])

    def get_initial_greeting(self, language: str = "ja", user_profile: dict | None = None) -> str:
        """
        初回挨拶。GCS から読み込んだ INITIAL_GREETINGS を優先使用。
        """
        # GCS から読み込んだ挨拶を使用
        if _PROMPTS_LOADED and LOADED_GREETINGS:
            concierge_greetings = LOADED_GREETINGS.get("concierge", {})
            greeting = concierge_greetings.get(language)
            if greeting:
                return greeting

        # フォールバック
        greetings = {
            "ja": "いらっしゃいませ！今日はどんなお食事をお探しですか？",
            "en": "Welcome! What kind of dining experience are you looking for today?",
            "ko": "어서오세요! 오늘은 어떤 식사를 찾고 계신가요?",
            "zh": "欢迎！今天想找什么样的餐厅呢？",
        }
        return greetings.get(language, greetings["ja"])

    def get_memory_schema(self) -> dict:
        """グルメモード固有の長期記憶スキーマ"""
        return {
            "favorite_cuisines": [],
            "preferred_area": "",
            "budget_range": "",
            "dietary_restrictions": [],
            "past_searches": [],
        }
