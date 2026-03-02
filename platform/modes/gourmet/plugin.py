"""
グルメコンシェルジュ モードプラグイン

既存の gourmet-support のシステムプロンプト・対話ロジックをプラグインとして再構成。
Live API 経路での使用を前提に設計。

[推定] 実際のシステムプロンプトは gourmet-support の support_core.py から移植が必要。
       ここでは基本的な骨格を実装する。
"""

from platform.modes.base_mode import BaseModePlugin


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
        グルメコンシェルジュのシステムプロンプト

        stt_stream.py L471-488 の _build_system_instruction() パターンを参考に、
        Live API 用に最適化。

        [推定] 実際のプロンプト内容は gourmet-support の support_core.py を参照して
        より詳細に調整する必要がある。
        """
        prompts = {
            "ja": self._prompt_ja(context),
            "en": self._prompt_en(context),
            "ko": self._prompt_ko(context),
            "zh": self._prompt_zh(context),
        }
        return prompts.get(language, prompts["ja"])

    def _prompt_ja(self, context: str | None = None) -> str:
        prompt = """あなたはグルメコンシェルジュAIです。
ユーザーの食の好み・気分・シチュエーションをヒアリングし、最適なレストランを提案します。

【対話スタイル】
- 親しみやすく、でも丁寧な口調で話してください
- 短く簡潔に応答してください（1-2文程度）
- ユーザーの好みを引き出す質問を積極的にしてください
- 「へぇ」「なるほど」等の相槌を自然に入れてください

【ヒアリング項目】
1. 料理のジャンル（和食、イタリアン、中華など）
2. エリア・場所
3. 予算感
4. シチュエーション（デート、接待、カジュアルなど）
5. こだわり（個室、禁煙、アレルギーなど）

【重要なルール】
- 一度に複数の質問をしないこと（1つずつ聞く）
- ユーザーが答えたら必ず相槌を打ってから次の質問へ
- 十分な情報が集まったら、お店を提案する
"""

        if context:
            prompt += f"""

【これまでの会話の要約】
{context}

【重要：必ず守ること】
1. 直前の話者の発言に対して短い相槌を入れる
2. 既に聞いた質問は絶対に繰り返さない
3. 会話の流れを自然に引き継ぐ
"""
        return prompt

    def _prompt_en(self, context: str | None = None) -> str:
        prompt = """You are a Gourmet Concierge AI.
Help users find the perfect restaurant by understanding their preferences, mood, and occasion.

【Style】
- Friendly yet polite
- Keep responses short (1-2 sentences)
- Ask questions to understand preferences
- Use natural acknowledgments

【Key Questions】
1. Cuisine type
2. Area/location
3. Budget
4. Occasion
5. Special requirements
"""
        if context:
            prompt += f"\n\n【Previous conversation summary】\n{context}\n"
        return prompt

    def _prompt_ko(self, context: str | None = None) -> str:
        prompt = """당신은 맛집 컨시어지 AI입니다.
사용자의 음식 취향과 상황을 파악하여 최적의 레스토랑을 추천합니다.

【스타일】
- 친근하면서도 정중한 말투
- 짧고 간결하게 응답 (1-2문장)
- 취향을 파악하는 질문을 적극적으로

【확인 사항】
1. 음식 종류
2. 지역
3. 예산
4. 상황 (데이트, 회식 등)
5. 특별 요구사항
"""
        if context:
            prompt += f"\n\n【이전 대화 요약】\n{context}\n"
        return prompt

    def _prompt_zh(self, context: str | None = None) -> str:
        prompt = """你是一个美食顾问AI。
了解用户的饮食偏好、心情和场合，推荐最合适的餐厅。

【风格】
- 亲切有礼
- 简短回复（1-2句）
- 主动提问了解偏好

【了解事项】
1. 菜系类型
2. 区域/位置
3. 预算
4. 场合
5. 特殊要求
"""
        if context:
            prompt += f"\n\n【之前的对话摘要】\n{context}\n"
        return prompt

    def get_initial_greeting(self, language: str = "ja", user_profile: dict | None = None) -> str:
        """初回挨拶（長期記憶ベースのパーソナライズ対応）"""
        greetings = {
            "ja": "いらっしゃいませ！今日はどんなお食事をお探しですか？",
            "en": "Welcome! What kind of dining experience are you looking for today?",
            "ko": "어서오세요! 오늘은 어떤 식사를 찾고 계신가요?",
            "zh": "欢迎！今天想找什么样的餐厅呢？",
        }

        # 長期記憶からパーソナライズ
        if user_profile:
            mode_data = user_profile.get("mode_data", {}).get("gourmet", {})
            favorite = mode_data.get("favorite_cuisines", [])
            if favorite and language == "ja":
                return (
                    f"お帰りなさい！前回は{favorite[0]}を探してましたよね。"
                    f"今日も{favorite[0]}、それとも別のジャンルにしますか？"
                )

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
