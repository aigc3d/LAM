# -*- coding: utf-8 -*-
"""
長期記憶管理モジュール（新設計版）
- user_id をPRIMARY KEYとして使用
- サマリーベースの記憶管理
- LLMによる会話サマリー生成
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ========================================
# Supabaseクライアント初期化
# ========================================

_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """Supabaseクライアントを取得（シングルトン）"""
    global _supabase_client

    if _supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("[LTM] SUPABASE_URL または SUPABASE_KEY が設定されていません")
            raise ValueError("Supabase credentials not configured")

        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info("[LTM] Supabaseクライアント初期化完了")

    return _supabase_client


# ========================================
# ユーザープロファイル管理（user_idベース）
# ========================================

class LongTermMemory:
    """長期記憶管理クラス（新設計版）"""

    def __init__(self):
        self.client = get_supabase_client()
        self._cache = {}  # プロファイルキャッシュ

    # ----------------------------------------
    # プロファイル操作（user_idベース）
    # ----------------------------------------

    def get_profile_basic(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        軽量プロファイル取得（名前のみ）
        - 初期起動の高速化用
        - サマリーは含まない
        """
        if not user_id:
            logger.warning("[LTM] get_profile_basic: user_id が空です")
            return None

        # キャッシュ確認
        if user_id in self._cache:
            logger.info(f"[LTM] キャッシュからプロファイル取得: {user_id}")
            return self._cache[user_id]

        try:
            response = self.client.table('user_profiles').select(
                'user_id, preferred_name, name_honorific, visit_count'
            ).eq('user_id', user_id).execute()

            if response.data and len(response.data) > 0:
                profile = response.data[0]
                self._cache[user_id] = profile  # キャッシュに保存
                logger.info(f"[LTM] 軽量プロファイル取得成功: {user_id}")
                return profile
            else:
                logger.info(f"[LTM] プロファイル未登録: {user_id}")
                return None
        except Exception as e:
            logger.error(f"[LTM] 軽量プロファイル取得エラー: {e}")
            return None

    def get_profile(self, user_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """プロファイル取得（全カラム）"""
        if not user_id:
            logger.warning("[LTM] get_profile: user_id が空です")
            return None

        try:
            response = self.client.table('user_profiles').select('*').eq('user_id', user_id).execute()

            if response.data and len(response.data) > 0:
                profile = response.data[0]
                self._cache[user_id] = profile  # キャッシュ更新
                logger.info(f"[LTM] プロファイル取得成功: {user_id}")
                return profile
            else:
                logger.info(f"[LTM] プロファイル未登録: {user_id}")
                return None
        except Exception as e:
            logger.error(f"[LTM] プロファイル取得エラー: {e}")
            return None

    def get_summary(self, user_id: str) -> Optional[str]:
        """サマリーのみ取得（遅延読み込み用）"""
        if not user_id:
            return None

        try:
            response = self.client.table('user_profiles').select(
                'conversation_summary'
            ).eq('user_id', user_id).execute()

            if response.data and len(response.data) > 0:
                return response.data[0].get('conversation_summary')
            return None
        except Exception as e:
            logger.error(f"[LTM] サマリー取得エラー: {e}")
            return None

    def create_profile(self, user_id: str, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """新規プロファイル作成"""
        if not user_id:
            logger.error("[LTM] create_profile: user_id が空です")
            return None

        try:
            now = datetime.now().isoformat()
            profile_data = {
                'user_id': user_id,
                'preferred_name': data.get('preferred_name') if data else None,
                'name_honorific': data.get('name_honorific', '') if data else '',
                'conversation_summary': data.get('conversation_summary') if data else None,
                'default_language': data.get('language', 'ja') if data else 'ja',
                'preferred_mode': data.get('mode', 'chat') if data else 'chat',
                'first_visit_at': now,
                'last_visit_at': now,
                'visit_count': 1,
                'created_at': now,
                'updated_at': now
            }

            response = self.client.table('user_profiles').insert(profile_data).execute()

            if response.data and len(response.data) > 0:
                logger.info(f"[LTM] プロファイル作成成功: {user_id}")
                return response.data[0]
            else:
                logger.error(f"[LTM] プロファイル作成失敗: {user_id}")
                return None
        except Exception as e:
            logger.error(f"[LTM] プロファイル作成エラー: {e}")
            return None

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        プロファイル更新（UPSERT動作）
        - レコードが存在すれば更新
        - レコードがなければ新規作成
        """
        if not user_id:
            logger.error("[LTM] update_profile: user_id が空です")
            return False

        try:
            now = datetime.now().isoformat()

            # upsert用データを準備
            upsert_data = {
                'user_id': user_id,
                'updated_at': now,
                'last_visit_at': now,
                **updates
            }

            # 新規の場合のデフォルト値
            if 'visit_count' not in upsert_data:
                upsert_data['visit_count'] = 1
            if 'created_at' not in upsert_data:
                upsert_data['created_at'] = now
            if 'first_visit_at' not in upsert_data:
                upsert_data['first_visit_at'] = now

            # Supabase upsert (on_conflict で user_id を指定)
            response = self.client.table('user_profiles').upsert(
                upsert_data,
                on_conflict='user_id'
            ).execute()

            if response.data:
                # キャッシュを更新
                self._cache[user_id] = response.data[0]
                logger.info(f"[LTM] プロファイルupsert成功: {user_id}")
                return True
            else:
                logger.error(f"[LTM] プロファイルupsert失敗: {user_id}")
                return False

        except Exception as e:
            logger.error(f"[LTM] プロファイル更新エラー: {e}")
            return False

    def increment_visit_count(self, user_id: str, current_count: int = None) -> bool:
        """
        訪問回数をインクリメント
        - current_count が渡されればそれを使用（DB照会を省略）
        """
        if not user_id:
            return False

        try:
            if current_count is None:
                # キャッシュから取得を試みる
                cached = self._cache.get(user_id)
                if cached:
                    current_count = cached.get('visit_count', 0)
                else:
                    # キャッシュにない場合のみDB照会
                    profile = self.get_profile_basic(user_id)
                    current_count = profile.get('visit_count', 0) if profile else 0

            new_count = current_count + 1

            # 直接UPDATE（upsertではなく）
            response = self.client.table('user_profiles').update({
                'visit_count': new_count,
                'last_visit_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }).eq('user_id', user_id).execute()

            if response.data:
                # キャッシュ更新
                if user_id in self._cache:
                    self._cache[user_id]['visit_count'] = new_count
                logger.info(f"[LTM] 訪問回数更新成功: {user_id} -> {new_count}")
                return True
            return False
        except Exception as e:
            logger.error(f"[LTM] 訪問回数更新エラー: {e}")
            return False

    def is_first_visit(self, user_id: str) -> bool:
        """
        初回訪問かどうか判定（軽量版）
        - DBにレコードがなければ初回
        - レコードがあれば2回目以降
        """
        if not user_id:
            return True

        profile = self.get_profile_basic(user_id)
        return profile is None

    def append_conversation_summary(self, user_id: str, new_summary: str) -> bool:
        """
        会話サマリーを追記（マージ）
        - 既存のサマリーがあれば、新しいサマリーを追記
        - なければ新規として保存
        """
        if not user_id or not new_summary:
            return False

        try:
            profile = self.get_profile(user_id)
            if not profile:
                logger.warning(f"[LTM] append_conversation_summary: プロファイルが見つかりません user_id={user_id}")
                return False

            existing_summary = profile.get('conversation_summary', '') or ''

            # 既存サマリーがあればマージ（改行で区切る）
            if existing_summary:
                merged_summary = f"{existing_summary}\n\n---\n\n{new_summary}"
            else:
                merged_summary = new_summary

            return self.update_profile(user_id, {'conversation_summary': merged_summary})

        except Exception as e:
            logger.error(f"[LTM] サマリー追記エラー: {e}")
            return False

    # ----------------------------------------
    # システムプロンプト用コンテキスト生成
    # ----------------------------------------

    def generate_system_prompt_context(self, user_id: str, language: str = 'ja') -> str:
        """システムプロンプトに注入するコンテキストを生成"""
        if not user_id:
            return ""

        profile = self.get_profile(user_id)
        if not profile:
            return ""

        # 言語別のテンプレート
        if language == 'ja':
            return self._generate_context_ja(profile)
        elif language == 'en':
            return self._generate_context_en(profile)
        elif language == 'zh':
            return self._generate_context_zh(profile)
        elif language == 'ko':
            return self._generate_context_ko(profile)
        else:
            return self._generate_context_ja(profile)

    def _generate_context_ja(self, profile: Dict) -> str:
        """日本語コンテキスト生成"""
        context_parts = []

        # ユーザー情報
        context_parts.append("【ユーザー情報】")
        preferred_name = profile.get('preferred_name', '')
        name_honorific = profile.get('name_honorific', '')
        if preferred_name:
            context_parts.append(f"- 呼び方: {preferred_name}{name_honorific}")
        context_parts.append(f"- 訪問回数: {profile.get('visit_count', 1)}回目")

        # 会話サマリー（存在する場合）
        conversation_summary = profile.get('conversation_summary', '')
        if conversation_summary:
            context_parts.append("\n【過去の会話記録】")
            context_parts.append(conversation_summary)

        return "\n".join(context_parts)

    def _generate_context_en(self, profile: Dict) -> str:
        """英語コンテキスト生成"""
        context_parts = []

        context_parts.append("【User Information】")
        preferred_name = profile.get('preferred_name', '')
        name_honorific = profile.get('name_honorific', '')
        if preferred_name:
            context_parts.append(f"- Address as: {preferred_name}{name_honorific}")
        context_parts.append(f"- Visit count: {profile.get('visit_count', 1)} visit(s)")

        conversation_summary = profile.get('conversation_summary', '')
        if conversation_summary:
            context_parts.append("\n【Past Conversation Records】")
            context_parts.append(conversation_summary)

        return "\n".join(context_parts)

    def _generate_context_zh(self, profile: Dict) -> str:
        """中国語コンテキスト生成"""
        context_parts = []
        context_parts.append("【用户信息】")
        preferred_name = profile.get('preferred_name', '')
        name_honorific = profile.get('name_honorific', '')
        if preferred_name:
            context_parts.append(f"- 称呼: {preferred_name}{name_honorific}")
        context_parts.append(f"- 访问次数: 第{profile.get('visit_count', 1)}次")

        conversation_summary = profile.get('conversation_summary', '')
        if conversation_summary:
            context_parts.append("\n【过去的对话记录】")
            context_parts.append(conversation_summary)

        return "\n".join(context_parts)

    def _generate_context_ko(self, profile: Dict) -> str:
        """韓国語コンテキスト生成"""
        context_parts = []
        context_parts.append("【사용자 정보】")
        preferred_name = profile.get('preferred_name', '')
        name_honorific = profile.get('name_honorific', '')
        if preferred_name:
            context_parts.append(f"- 호칭: {preferred_name}{name_honorific}")
        context_parts.append(f"- 방문 횟수: {profile.get('visit_count', 1)}회")

        conversation_summary = profile.get('conversation_summary', '')
        if conversation_summary:
            context_parts.append("\n【과거 대화 기록】")
            context_parts.append(conversation_summary)

        return "\n".join(context_parts)


# ========================================
# 後方互換性のためのダミークラス・関数
# ========================================

class PreferenceExtractor:
    """
    後方互換性のためのダミークラス
    新設計ではLLMがサマリーを生成するため、正規表現ベースの抽出は廃止
    """

    @staticmethod
    def extract_from_text(text: str, language: str = 'ja') -> List[Dict[str, Any]]:
        """ダミー: 常に空リストを返す"""
        return []

    @staticmethod
    def extract_and_save(session_id: str, text: str, language: str = 'ja') -> int:
        """ダミー: 何もしない（0を返す）"""
        return 0


def extract_name_from_text(text: str) -> Optional[str]:
    """
    テキストから名前を抽出（後方互換性のため残す）
    新設計ではLLMが名前を抽出してactionで返すため、この関数は使用されない
    """
    import re

    # パターン1: 「〜と呼んで」
    match = re.search(r'([^\s、。]+)(?:と|って)(?:呼んで|呼ばれ)', text)
    if match:
        return match.group(1)

    # パターン2: 「名前は〜」
    match = re.search(r'名前は([^\s、。]+)', text)
    if match:
        return match.group(1)

    # パターン3: 単独の名前らしき文字列（ひらがな・カタカナ2-10文字）
    match = re.search(r'^([ぁ-んァ-ヶー]{2,10})$', text.strip())
    if match:
        return match.group(1)

    return None
