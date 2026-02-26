# プラットフォーム設計書

> **文書ID**: ARCH-PLATFORM-001
> **作成日**: 2026-02-26
> **ステータス**: Draft
> **根拠文書**: `docs/DESIGN_REQUEST.md`, `docs/PLATFORM_REQUIREMENTS.md`
> **前提**: 本文書は `LAM_gpro` リポジトリ内の実コードを読解した上で作成。gourmet-sp / gourmet-support は別リポジトリのため、パッチファイルとドキュメントからの推定箇所は明記。

---

## 目次

1. [設計方針](#1-設計方針)
2. [全体アーキテクチャ](#2-全体アーキテクチャ)
3. [データフロー](#3-データフロー)
4. [バックエンド設計](#4-バックエンド設計)
5. [Live API 統合設計](#5-live-api-統合設計)
6. [記憶機能の統一設計](#6-記憶機能の統一設計)
7. [フロントエンド設計](#7-フロントエンド設計)
8. [API設計](#8-api設計)
9. [既存サービスとの共存戦略](#9-既存サービスとの共存戦略)
10. [iPhone SE 対応戦略](#10-iphone-se-対応戦略)
11. [開発ロードマップ](#11-開発ロードマップ)

---

## 1. 設計方針

### 1.1 基本原則

| # | 原則 | 根拠 |
|---|------|------|
| P1 | **既存を壊さない** | α版テスト中のグルメサポートAIを中断しない（DESIGN_REQUEST §0.3 優先度1） |
| P2 | **事実ベースで設計する** | 推測部分は「未確認」と明記。gourmet-sp/gourmet-support の内部構造は推定として扱う |
| P3 | **最小限の抽象化** | 動くものを作ってから抽象化する。過剰設計を避ける |
| P4 | **段階的移行** | 既存と新プラットフォームを並行稼働させ、検証しながら段階的に移行する |

### 1.2 確認済み / 未確認の分類基準

本文書では以下の表記を使用する:

- **[確認済み]** — このリポジトリ内の実コードを読解して確認した事実
- **[推定]** — パッチファイル・ドキュメントから推定した内容（gourmet-sp/gourmet-support の内部構造等）
- **[設計判断]** — 本設計書で新たに行う設計上の判断。根拠を併記
- **[未検証]** — 技術的実現可能性が未検証の項目

---

## 2. 全体アーキテクチャ

### 2.1 現状構成（確認済み）

```
┌─────────────────────────────────────────────────────────────────┐
│                        現状の3コンポーネント                      │
├─────────────────┬──────────────────┬────────────────────────────┤
│  gourmet-sp     │ gourmet-support  │ audio2exp-service          │
│  (フロントエンド)│ (バックエンド)    │ (A2Eマイクロサービス)       │
│  Astro+TS       │ Flask+Socket.IO  │ Flask                      │
│  Vercel         │ Cloud Run        │ Cloud Run                  │
│  [別リポジトリ]  │ [別リポジトリ]    │ [確認済み]                  │
├─────────────────┴──────────────────┴────────────────────────────┤
│  AI_Meeting_App/stt_stream.py                                   │
│  (デスクトップ専用 Live API アプリ — Web統合なし)  [確認済み]       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 プラットフォーム化後の構成（設計判断）

```
┌──────────────────────────────────────────────────────────────────────┐
│                            ブラウザ                                   │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              Platform Frontend (Astro + TypeScript)           │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │    │
│  │  │ ModeRouter   │  │ AvatarRenderer│  │ AudioIO          │  │    │
│  │  │ (モード切替)  │  │ (3D描画)      │  │ (WebAudio入出力) │  │    │
│  │  └──────┬───────┘  └───────┬───────┘  └──────┬───────────┘  │    │
│  │         │                  │                  │              │    │
│  │  ┌──────┴──────────────────┴──────────────────┴──────────┐   │    │
│  │  │              DialogueManager                          │   │    │
│  │  │  (REST対話 / Live API対話 の共通インターフェース)        │   │    │
│  │  └──────────────────────┬─────────────────────────────────┘   │    │
│  └─────────────────────────┼────────────────────────────────────┘    │
│                            │ WebSocket / REST                        │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────────────┐
│                    Platform Backend (Python)                          │
│  ┌─────────────────────────┴────────────────────────────────────┐    │
│  │                    Gateway Layer                              │    │
│  │  /api/v2/*  (新プラットフォーム)                                │    │
│  │  /api/*     (既存 gourmet-support 互換 — Phase 1ではプロキシ)  │    │
│  └──────┬──────────────┬──────────────┬────────────────────┬────┘    │
│         │              │              │                    │         │
│  ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴─────┐  ┌──────────┴──────┐  │
│  │ SessionMgr  │ │ LiveRelay  │ │ MemoryMgr│  │  ModeRegistry   │  │
│  │ (セッション) │ │ (Live API  │ │ (短期+   │  │  (モード        │  │
│  │             │ │  WS中継)   │ │  長期記憶)│  │   プラグイン)   │  │
│  └─────────────┘ └────────────┘ └──────────┘  └─────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                     Service Connectors                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │    │
│  │  │ Gemini   │  │ GCP TTS  │  │ A2E      │  │ External   │  │    │
│  │  │ REST/Live│  │          │  │ Client   │  │ APIs       │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │   audio2exp-service      │
                          │   Cloud Run (既存)        │
                          │   [確認済み・変更なし]     │
                          └─────────────────────────┘
```

**設計判断の根拠**:

1. **Gateway Layer で新旧エンドポイントを共存** — 既存の `/api/*` を温存しつつ、新プラットフォームは `/api/v2/*` で提供。Phase 1 では既存バックエンドへのプロキシも可能
2. **LiveRelay をサーバーサイドに配置** — Gemini API キーをクライアントに露出しないため（TC-06）。ブラウザ ↔ サーバー は WebSocket、サーバー ↔ Gemini は Live API WebSocket
3. **ModeRegistry でプラグイン管理** — モード追加時は設定 + プラグインモジュールの追加のみ
4. **audio2exp-service は変更しない** — 既にデプロイ済みで動作確認済み。A2E Client 経由で呼び出す

---

## 3. データフロー

### 3.1 REST API 経路（既存方式の拡張）

```
[ユーザー発話]
    │
    ▼ (1) ブラウザ AudioIO: WebAudio → PCM 16kHz
    │
    ▼ (2) Socket.IO streaming → バックエンド STT
    │
    ▼ (3) テキスト → Gemini REST API (gemini-2.5-flash)
    │
    ▼ (4) LLM応答テキスト
    │
    ├──▶ (5a) GCP TTS → 音声(MP3 base64)
    │        │
    │        ├──▶ (5b) audio2exp-service → 52次元 ARKit @30fps
    │        │
    │        └──▶ (5c) 音声 + Expression を同梱してクライアントに返却
    │
    ▼ (6) ブラウザ: 音声再生 + Expression フレーム同期 → アバター描画

    レイテンシ見積: (2)~500ms + (3)~2s + (5a)~1s + (5b)~2s = 約5-6秒
    [確認済み: この経路は gourmet-support で実装済み、パッチファイルで確認]
```

### 3.2 Live API 経路（新規）

```
[ユーザー発話]
    │
    ▼ (1) ブラウザ AudioIO: WebAudio → PCM 16kHz
    │
    ▼ (2) WebSocket → バックエンド LiveRelay
    │
    ▼ (3) LiveRelay → Gemini Live API (gemini-2.5-flash-native-audio-preview)
    │         │
    │         ├── 入力: PCM 16kHz (リアルタイムストリーミング)
    │         ├── 出力: PCM 24kHz (AI音声) + transcription
    │         └── VAD: Gemini側で発話区間検出
    │
    ▼ (4) AI音声(PCM 24kHz) → クライアントに中継
    │         │
    │         └──▶ (4b) 並行: AI音声 → audio2exp-service → Expression
    │                   [設計判断: チャンク単位で非同期送信]
    │
    ▼ (5) ブラウザ: PCM再生 + Expression フレーム同期 → アバター描画

    レイテンシ見積: (2)~50ms + (3)~300ms + (4)~50ms = 約400ms
    ※ A2Eは並行処理のため音声再生を遅延させない
    [未検証: WebSocket中継のレイテンシは実測が必要]
```

### 3.3 Live API + A2E 同期の詳細設計

Live API 経路では、音声出力とA2E推論を並行して行う必要がある:

```
時間軸 →

LiveRelay:
  AI音声チャンク受信 ──┬── チャンク1(1秒分PCM) ── チャンク2 ── チャンク3 ──
                       │
クライアント送信:       ├── 音声チャンク1送信 ──── 音声チャンク2 ──── ...
                       │   (即時、遅延なし)
                       │
A2Eリクエスト:         └── A2E(チャンク1) ──── A2E(チャンク2) ──── ...
                            │                    │
                            ▼                    ▼
Expression送信:         Exp1をWS送信 ──── Exp2をWS送信 ──── ...
```

**設計判断**: A2E推論（~2秒/文）は音声再生と並行で行い、Expression はできた分からクライアントに送信する。音声チャンクは即時転送し、Expression は少し遅れて到着する。クライアント側でフレームバッファリングして再生時刻に同期する。

**根拠**:
- [確認済み] `LAMAvatar.astro` L581-637 にフレームバッファ + `ttsPlayer.currentTime` 同期の仕組みが既に実装されている
- [確認済み] `concierge-controller.ts` L392-465 で Expression をフレームバッファに投入する `applyExpressionFromTts()` が実装済み

---

## 4. バックエンド設計

### 4.1 ディレクトリ構成

```
platform/
├── server.py                  # エントリーポイント (FastAPI or Flask)
├── config/
│   ├── __init__.py
│   └── settings.py            # 環境変数・設定
│
├── gateway/
│   ├── __init__.py
│   ├── router.py              # /api/v2/* ルーティング
│   └── legacy_proxy.py        # /api/* 既存互換プロキシ (Phase 1)
│
├── session/
│   ├── __init__.py
│   └── manager.py             # セッション管理 (SessionManager)
│
├── live/
│   ├── __init__.py
│   ├── relay.py               # Live API WebSocket中継 (LiveRelay)
│   ├── reconnect.py           # 累積制限回避・自動再接続ロジック
│   └── speech_detector.py     # 発話途切れ検知
│
├── memory/
│   ├── __init__.py
│   ├── session_memory.py      # 短期記憶 (SessionMemory)
│   └── long_term_memory.py    # 長期記憶 (LongTermMemory)
│
├── dialogue/
│   ├── __init__.py
│   ├── rest_dialogue.py       # REST API対話
│   └── hybrid_dialogue.py     # Live/REST ハイブリッド切替
│
├── services/
│   ├── __init__.py
│   ├── gemini_client.py       # Gemini REST/Live API クライアント
│   ├── tts_client.py          # Google Cloud TTS クライアント
│   └── a2e_client.py          # audio2exp-service クライアント
│
└── modes/
    ├── __init__.py
    ├── registry.py            # ModeRegistry (モードプラグイン管理)
    ├── base_mode.py           # BaseModePlugin (基底クラス)
    ├── gourmet/
    │   ├── __init__.py
    │   └── plugin.py          # グルメコンシェルジュモード
    ├── support/
    │   ├── __init__.py
    │   └── plugin.py          # カスタマーサポートモード
    └── interview/
        ├── __init__.py
        └── plugin.py          # インタビューモード
```

### 4.2 コア設計: SessionManager

**責務**: セッションのライフサイクル管理。モードの割り当て、短期記憶の保持、接続状態の追跡。

```python
# platform/session/manager.py

class Session:
    """1つの対話セッションを表現"""
    session_id: str
    user_id: str | None
    mode: str                          # "gourmet", "support", "interview"
    dialogue_type: str                 # "rest", "live", "hybrid"
    created_at: datetime
    memory: SessionMemory              # 短期記憶
    live_state: LiveSessionState | None  # Live API 接続状態

class SessionManager:
    """セッション管理"""
    _sessions: dict[str, Session]      # session_id → Session

    def create_session(self, mode: str, user_id: str = None) -> Session: ...
    def get_session(self, session_id: str) -> Session | None: ...
    def end_session(self, session_id: str) -> None: ...
```

### 4.3 コア設計: ModeRegistry

**責務**: モードプラグインの登録・取得。新モード追加時はプラグインファイルと設定の追加のみ。

```python
# platform/modes/registry.py

class ModeRegistry:
    """モードプラグインのレジストリ"""
    _modes: dict[str, BaseModePlugin]

    def register(self, name: str, plugin: BaseModePlugin) -> None: ...
    def get(self, name: str) -> BaseModePlugin: ...
    def list_modes(self) -> list[str]: ...

# platform/modes/base_mode.py

class BaseModePlugin:
    """モードプラグインの基底クラス"""
    name: str
    display_name: str
    dialogue_type: str                 # "rest" | "live" | "hybrid"

    def get_system_prompt(self, context: dict = None) -> str: ...
    def get_live_api_config(self) -> dict: ...
    def get_tools(self) -> list: ...              # Function Calling 定義
    def get_memory_schema(self) -> dict: ...       # 長期記憶のモード別スキーマ
    def on_session_start(self, session: Session) -> str | None: ...  # 初回挨拶
    def on_session_end(self, session: Session) -> None: ...
```

**設計判断の根拠**:
- [確認済み] `stt_stream.py` L471-488 で `_build_system_instruction()` がモード別プロンプトを切り替えている。これをプラグインに抽出
- [確認済み] `stt_stream.py` L440-468 で `_build_config()` がモード別 Live API 設定を構築している。これもプラグインに抽出
- [推定] gourmet-support の `support_core.py` にグルメ固有の対話ロジックがあるが、内部構造は未確認

### 4.4 コア設計: A2E Client

**責務**: audio2exp-service への HTTP リクエスト。REST 経路と Live API 経路の両方で使用。

```python
# platform/services/a2e_client.py

class A2EClient:
    """audio2exp-service クライアント"""

    def __init__(self, base_url: str = None):
        # [確認済み] エンドポイント: POST /api/audio2expression
        self.base_url = base_url or os.getenv("A2E_SERVICE_URL")

    async def process_audio(
        self,
        audio_base64: str,
        session_id: str,
        audio_format: str = "mp3"
    ) -> dict:
        """
        音声 → 52次元 ARKit ブレンドシェイプ

        Returns:
            {
                "names": list[str],     # 52個のARKit名
                "frames": list[list[float]],  # N×52
                "frame_rate": 30
            }

        [確認済み] a2e_engine.py L381-401: process() の入出力仕様
        """
        ...

    async def health_check(self) -> dict:
        """[確認済み] GET /health"""
        ...
```

### 4.5 Web フレームワーク選定

**設計判断**: **FastAPI** を採用する。

**根拠**:
- Live API の WebSocket 中継には非同期 I/O が必須。FastAPI は ASGI ベースで WebSocket をネイティブサポート
- [確認済み] `stt_stream.py` は `asyncio` ベースで書かれており、FastAPI への移植が自然
- [推定] 既存の gourmet-support は Flask（同期）だが、新プラットフォームは別プロセスで稼働させるため、フレームワークの不一致は問題にならない
- Flask + Socket.IO で WebSocket を扱うことも可能だが、Live API の非同期ストリーミングには asyncio ネイティブの方が扱いやすい

---

## 5. Live API 統合設計

### 5.1 移植対象（stt_stream.py からの機能抽出）

[確認済み] `stt_stream.py` から移植すべき機能とその対応先:

| stt_stream.py の機能 | コード箇所 | 移植先モジュール | 移植の難易度 |
|---------------------|-----------|----------------|------------|
| Live API 接続 | `run()` L714-796 | `live/relay.py` | 中（PyAudio→WebSocket） |
| 累積文字数制限回避 | L372-373, L624-643 | `live/reconnect.py` | 低（ロジックそのまま） |
| 自動再接続ループ | `run()` L741-796 | `live/relay.py` | 低（ロジックそのまま） |
| コンテキスト引き継ぎ | `_get_context_summary()` L940-966 | `memory/session_memory.py` | 低（ロジックそのまま） |
| 再接続時config構築 | `_build_config()` L410-468 | `live/relay.py` + `modes/` | 低 |
| 発話途切れ検知 | `_is_speech_incomplete()` L501-529 | `live/speech_detector.py` | 低（ロジックそのまま） |
| 音声送受信 | `listen_audio()`, `send_audio()`, `receive_audio()`, `play_audio()` | `live/relay.py` | 高（PyAudio→WebSocket中継） |
| 割り込み処理 | L650-662 | `live/relay.py` | 中 |
| VAD設定 | L448-456 | `live/relay.py` config | 低 |
| REST API ハイブリッド | `RestAPIHandler` L307-362 | `dialogue/hybrid_dialogue.py` | 中 |
| 会話履歴管理 | L389, L490-499 | `memory/session_memory.py` | 低 |
| スライディングウィンドウ | L457-461 | `live/relay.py` config | 低（設定値のみ） |

### 5.2 LiveRelay 設計

**責務**: ブラウザとGemini Live APIの間でWebSocket中継を行う。累積文字数制限の回避、自動再接続、コンテキスト引き継ぎを管理。

```python
# platform/live/relay.py

class LiveRelay:
    """Gemini Live API WebSocket 中継"""

    def __init__(self, session: Session, mode_plugin: BaseModePlugin):
        self.session = session
        self.mode_plugin = mode_plugin
        self.reconnect_mgr = ReconnectManager(
            max_chars=800,           # [確認済み] L372
            long_speech_threshold=500  # [確認済み] L373
        )

    async def handle_client_ws(self, websocket: WebSocket):
        """
        クライアント WebSocket ハンドラ

        プロトコル:
          クライアント → サーバー:
            { "type": "audio", "data": "<base64 PCM 16kHz>" }
            { "type": "text",  "data": "テキスト入力" }

          サーバー → クライアント:
            { "type": "audio",        "data": "<base64 PCM 24kHz>" }
            { "type": "transcription", "role": "user"|"ai", "text": "..." }
            { "type": "interrupted" }
            { "type": "reconnecting",  "reason": "char_limit"|"incomplete"|"error" }
            { "type": "expression",    "data": { names, frames, frame_rate } }
        """
        while True:
            try:
                await self._run_session(websocket)
                if not self.reconnect_mgr.needs_reconnect:
                    break
            except Exception as e:
                if self.reconnect_mgr.is_retriable_error(e):
                    await asyncio.sleep(3)  # [確認済み] L791
                    continue
                raise

    async def _run_session(self, client_ws: WebSocket):
        """1つのGeminiセッションを実行"""
        # コンテキスト引き継ぎ
        context = None
        if self.reconnect_mgr.session_count > 0:
            context = self.session.memory.get_context_summary()

        config = self._build_live_config(context)
        self.reconnect_mgr.reset_for_new_session()

        async with gemini_client.live.connect(
            model="gemini-2.5-flash-native-audio-preview",
            config=config
        ) as gemini_session:

            if self.reconnect_mgr.session_count > 0:
                # [確認済み] L766-776: 再接続時に「続きをお願いします」送信
                await gemini_session.send_client_content(
                    turns=Content(role="user", parts=[Part(text="続きをお願いします")]),
                    turn_complete=True
                )

            # 3つの非同期タスクを並行実行
            # [確認済み] L908-930: asyncio.TaskGroup で並行処理
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._relay_client_to_gemini(client_ws, gemini_session))
                tg.create_task(self._relay_gemini_to_client(gemini_session, client_ws))
                tg.create_task(self._process_a2e(client_ws))
```

### 5.3 ReconnectManager 設計

```python
# platform/live/reconnect.py

class ReconnectManager:
    """累積文字数制限の回避ロジック"""

    # [確認済み] stt_stream.py L372-373
    def __init__(self, max_chars: int = 800, long_speech_threshold: int = 500):
        self.max_chars = max_chars
        self.long_speech_threshold = long_speech_threshold
        self.ai_char_count = 0
        self.needs_reconnect = False
        self.session_count = 0

    def on_ai_speech_complete(self, text: str) -> None:
        """
        AI発話完了時に呼び出す。再接続判定を行う。

        [確認済み] stt_stream.py L624-643 のロジックを移植:
        1. 累積文字数を加算
        2. 発話途切れ → 即時再接続
        3. 長文発話(500文字超) → 次ターン前に再接続
        4. 累積800文字超 → 再接続
        """
        char_count = len(text)
        self.ai_char_count += char_count

        if SpeechDetector.is_incomplete(text):
            self.needs_reconnect = True
            self.reconnect_reason = "incomplete"
        elif char_count >= self.long_speech_threshold:
            self.needs_reconnect = True
            self.reconnect_reason = "long_speech"
        elif self.ai_char_count >= self.max_chars:
            self.needs_reconnect = True
            self.reconnect_reason = "char_limit"

    def reset_for_new_session(self) -> None:
        self.ai_char_count = 0
        self.needs_reconnect = False
        self.session_count += 1

    @staticmethod
    def is_retriable_error(error: Exception) -> bool:
        """[確認済み] stt_stream.py L786-796"""
        msg = str(error).lower()
        return any(kw in msg for kw in ["1011", "internal error", "disconnected", "closed", "websocket"])
```

### 5.4 SpeechDetector 設計

```python
# platform/live/speech_detector.py

class SpeechDetector:
    """発話途切れ検知"""

    # [確認済み] stt_stream.py L501-529 をそのまま移植
    NORMAL_ENDINGS = ['。', '？', '?', '！', '!', 'か?', 'か？', 'ます', 'です', 'ね', 'よ', 'した', 'ください']
    INCOMPLETE_PATTERNS = ['、', 'の', 'を', 'が', 'は', 'に', 'で', 'と', 'も', 'や']

    @staticmethod
    def is_incomplete(text: str) -> bool:
        """発言が途中で切れているかチェック"""
        if not text:
            return False
        text = text.strip()

        for ending in SpeechDetector.NORMAL_ENDINGS:
            if text.endswith(ending):
                return False

        for pattern in SpeechDetector.INCOMPLETE_PATTERNS:
            if text.endswith(pattern):
                return True

        # 最後の文字がひらがな・カタカナで文末パターンでない場合
        last_char = text[-1]
        if last_char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん':
            if last_char not in 'ねよかなわ':
                return True

        return False
```

### 5.5 Live API 経路のブラウザ音声入出力

**設計判断**: ブラウザ側で WebAudio API を使用し、PCM 16kHz でマイク入力を取得、PCM 24kHz で音声出力を再生する。

```
ブラウザ側の音声処理:

[マイク入力]
  │
  ▼ navigator.mediaDevices.getUserMedia()
  │
  ▼ AudioWorklet: 48kHz → 16kHz ダウンサンプリング
  │   [推定] 既存の AudioManager が 48kHz→16kHz 変換を実装済み
  │
  ▼ PCM 16kHz chunks (100ms単位) → WebSocket 送信

[音声出力]
  │
  ▼ WebSocket 受信: PCM 24kHz chunks
  │
  ▼ AudioWorklet: PCM → AudioBuffer → AudioContext.destination
  │
  ▼ スピーカー出力
```

**注意**: [推定] 既存の gourmet-sp の `AudioManager` は Socket.IO 経由で STT バックエンドに音声を送信している（`concierge-controller.ts` L17 `new AudioManager(8000)` から推定）。新プラットフォームでは WebSocket 送信先を LiveRelay に切り替える。

---

## 6. 記憶機能の統一設計

### 6.1 短期記憶 (SessionMemory)

**責務**: セッション内の会話コンテキスト維持。Live API 再接続時のコンテキスト引き継ぎ。

```python
# platform/memory/session_memory.py

class SessionMemory:
    """短期記憶 — セッション内インメモリ"""

    MAX_HISTORY = 20           # [確認済み] stt_stream.py L493-495
    CONTEXT_SUMMARY_TURNS = 10 # [確認済み] stt_stream.py L946

    def __init__(self):
        self.history: list[dict] = []  # [{role: str, text: str, timestamp: datetime}]

    def add(self, role: str, text: str) -> None:
        """会話ターンを追加（直近20ターン保持）"""
        # [確認済み] stt_stream.py L490-495
        self.history.append({"role": role, "text": text, "timestamp": datetime.now()})
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def get_context_summary(self) -> str:
        """
        再接続時のコンテキスト要約を生成

        [確認済み] stt_stream.py L940-966 を移植:
        - 直近10ターンを取得
        - 各ターンの先頭150文字を要約
        - 最後のAI発言が質問なら強調
        """
        recent = self.history[-self.CONTEXT_SUMMARY_TURNS:]
        parts = [f"{h['role']}: {h['text'][:150]}" for h in recent]
        summary = "\n".join(parts)

        # 最後のAI発言が質問なら強調
        for h in reversed(self.history):
            if h['role'] == 'AI':
                if any(q in h['text'] for q in ['?', '？']):
                    summary += f"\n\n【直前の質問】\n{h['text'][:200]}"
                break

        return summary

    def get_history_for_prompt(self) -> str:
        """LLMプロンプト注入用の履歴文字列"""
        return "\n".join([f"{h['role']}: {h['text']}" for h in self.history])
```

### 6.2 長期記憶 (LongTermMemory)

**設計判断**: モード非依存のコアスキーマ + モード別拡張スキーマ。

```python
# platform/memory/long_term_memory.py

class LongTermMemory:
    """長期記憶 — セッションを超えたユーザー情報の永続化"""

    def __init__(self, storage_backend):
        """
        storage_backend: FirestoreBackend or SupabaseBackend
        [未確認] gourmet-support の実装が Firestore か Supabase かは要確認
        """
        self.storage = storage_backend

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """ユーザープロファイル取得"""
        ...

    async def update_user_profile(self, user_id: str, updates: dict) -> None:
        """ユーザープロファイル更新"""
        ...

    async def get_mode_data(self, user_id: str, mode: str) -> dict:
        """モード固有データ取得"""
        ...

    async def update_mode_data(self, user_id: str, mode: str, data: dict) -> None:
        """モード固有データ更新"""
        ...


class UserProfile:
    """ユーザープロファイル（モード非依存）"""
    user_id: str
    display_name: str | None
    language: str                  # "ja", "en", "ko", "zh"
    created_at: datetime
    last_session_at: datetime
    session_count: int
    mode_data: dict[str, dict]     # mode名 → モード固有データ
```

**データスキーマ**:

```json
{
  "user_id": "u_abc123",
  "display_name": null,
  "language": "ja",
  "created_at": "2026-02-20T10:00:00Z",
  "last_session_at": "2026-02-26T14:30:00Z",
  "session_count": 5,
  "mode_data": {
    "gourmet": {
      "favorite_cuisines": ["イタリアン", "和食"],
      "preferred_area": "渋谷",
      "budget_range": "3000-5000",
      "past_searches": [...]
    },
    "support": {
      "issue_history": [...],
      "satisfaction_scores": [...]
    },
    "interview": {
      "completed_topics": [...],
      "preferences": {}
    }
  }
}
```

**設計判断の根拠**:
- [推定] gourmet-support の `long_term_memory.py` はグルメ固有スキーマ（好きな料理、エリア等）を持つ
- [確認済み] `concierge-controller.ts` L191 で `data.initial_message` としてバックエンドから長期記憶ベースの挨拶を受け取っている
- モード非依存の `UserProfile` + モード別 `mode_data` で拡張性を確保

### 6.3 ストレージバックエンド

**未確認事項**: gourmet-support の長期記憶ストレージが Firestore か Supabase かは文書間で矛盾がある:
- `docs/SYSTEM_ARCHITECTURE.md` → Firestore
- 前回の `docs/PLATFORM_DESIGN.md` → Supabase

**設計判断**: ストレージバックエンドをインターフェースで抽象化し、実装を差し替え可能にする。初期実装では Firestore を使用（Cloud Run との親和性が高い）。

```python
class StorageBackend(ABC):
    @abstractmethod
    async def get(self, collection: str, doc_id: str) -> dict | None: ...
    @abstractmethod
    async def set(self, collection: str, doc_id: str, data: dict) -> None: ...
    @abstractmethod
    async def update(self, collection: str, doc_id: str, updates: dict) -> None: ...

class FirestoreBackend(StorageBackend): ...
class SupabaseBackend(StorageBackend): ...  # 必要に応じて実装
```

---

## 7. フロントエンド設計

### 7.1 現状の構成（確認済み + 推定）

[確認済み] パッチファイルから判明しているクラス階層:

```
CoreController (基底クラス) [推定: gourmet-sp に存在]
  ├── ConciergeController [確認済み: concierge-controller.ts]
  │     ├── A2E統合 (applyExpressionFromTts)
  │     ├── TTS並行処理 (speakResponseInChunks)
  │     ├── 即答処理 (handleStreamingSTTComplete)
  │     └── リップシンク診断 (runLipSyncDiagnostic)
  └── ChatController [推定: gourmet-sp に存在]

LAMAvatarController [確認済み: LAMAvatar.astro]
  ├── Gaussian Splatting レンダラー管理
  ├── フレームバッファ + TTS同期再生
  ├── WebSocket Manager (OpenAvatarChat連携)
  └── Expression data のサニタイズ + SDK汚染対策

ExpressionManager [確認済み: vrm-expression-manager.ts]
  └── 52次元ARKit → mouthOpenness 変換 (GVRMボーンシステム用)
```

### 7.2 新プラットフォームのフロントエンド構成

**設計判断**: 既存の `CoreController` → `ConciergeController` 階層を参考に、モード非依存の共通層を抽出する。

```
platform-frontend/
├── src/
│   ├── components/
│   │   ├── PlatformApp.astro          # メインアプリコンポーネント
│   │   ├── AvatarRenderer.astro       # LAMAvatar.astro をベースに汎用化
│   │   └── ChatPanel.astro            # チャットUI
│   │
│   ├── scripts/
│   │   ├── platform/
│   │   │   ├── platform-controller.ts # CoreController 相当の共通基盤
│   │   │   ├── dialogue-manager.ts    # REST/Live/Hybrid 切替
│   │   │   ├── audio-io.ts           # WebAudio 入出力 (マイク + スピーカー)
│   │   │   └── live-ws-client.ts     # LiveRelay WebSocket クライアント
│   │   │
│   │   ├── avatar/
│   │   │   ├── avatar-controller.ts   # LAMAvatarController ベース
│   │   │   ├── expression-sync.ts     # フレームバッファ + TTS同期
│   │   │   └── expression-manager.ts  # ExpressionManager ベース
│   │   │
│   │   └── modes/
│   │       ├── gourmet-mode.ts        # グルメモード固有UI/ロジック
│   │       ├── support-mode.ts        # サポートモード固有UI/ロジック
│   │       └── interview-mode.ts      # インタビューモード固有UI/ロジック
│   │
│   └── pages/
│       ├── index.astro                # メインページ
│       └── [...mode].astro            # モード別ルーティング
```

### 7.3 DialogueManager 設計

**責務**: REST対話とLive API対話の共通インターフェース。モードに応じて対話方式を切り替える。

```typescript
// src/scripts/platform/dialogue-manager.ts

interface DialogueManager {
  /** 対話を開始 */
  startSession(mode: string, userId?: string): Promise<SessionInfo>;

  /** テキスト送信（REST経路） */
  sendText(text: string): Promise<DialogueResponse>;

  /** 音声ストリーミング開始（Live API経路） */
  startLiveStream(): Promise<void>;

  /** 音声ストリーミング停止 */
  stopLiveStream(): Promise<void>;

  /** セッション終了 */
  endSession(): Promise<void>;

  /** イベントリスナー */
  on(event: 'ai_audio', handler: (data: ArrayBuffer) => void): void;
  on(event: 'ai_text', handler: (text: string) => void): void;
  on(event: 'expression', handler: (data: ExpressionData) => void): void;
  on(event: 'interrupted', handler: () => void): void;
  on(event: 'reconnecting', handler: (reason: string) => void): void;
}
```

### 7.4 ExpressionSync 設計

[確認済み] 既存の `LAMAvatar.astro` L581-676 のフレームバッファ同期ロジックを抽出・汎用化する。

```typescript
// src/scripts/avatar/expression-sync.ts

class ExpressionSync {
  private frameBuffer: ExpressionData[] = [];
  private frameRate: number = 30;
  private audioElement: HTMLAudioElement | null = null;

  // [確認済み] LAMAvatar.astro L698-720
  queueFrames(frames: ExpressionData[], frameRate: number): void { ... }
  clearBuffer(): void { ... }

  // [確認済み] LAMAvatar.astro L552-676
  // SDK呼び出し(~60fps)で現在の再生時刻に対応するフレームを返す
  getFrameAtTime(currentTime: number): ExpressionData { ... }

  // [確認済み] LAMAvatar.astro L592-596: fade-in
  // [確認済み] LAMAvatar.astro L642-666: fade-out
  private applyFade(data: ExpressionData, alpha: number): ExpressionData { ... }
}
```

### 7.5 既存パッチとの関係

[確認済み] `services/frontend-patches/` のファイルは、新プラットフォームの以下のモジュールのベースとなる:

| パッチファイル | 新プラットフォームでの位置 | 変更点 |
|--------------|------------------------|--------|
| `concierge-controller.ts` | `modes/gourmet-mode.ts` + `platform/platform-controller.ts` | モード固有部分とプラットフォーム共通部分を分離 |
| `vrm-expression-manager.ts` | `avatar/expression-manager.ts` | そのまま使用可能 |
| `LAMAvatar.astro` | `components/AvatarRenderer.astro` + `avatar/avatar-controller.ts` | WebSocket Manager 部分を DialogueManager に統合 |

---

## 8. API設計

### 8.1 新プラットフォーム API (`/api/v2/`)

#### セッション管理

```
POST /api/v2/session/start
Request:
  {
    "mode": "gourmet" | "support" | "interview",
    "user_id": "u_abc123",        // optional
    "language": "ja",
    "dialogue_type": "rest" | "live" | "hybrid"  // optional, default from mode config
  }
Response:
  {
    "session_id": "sess_xyz789",
    "mode": "gourmet",
    "dialogue_type": "live",
    "initial_message": "いらっしゃいませ！前回はイタリアンを...",  // 長期記憶ベース
    "ws_url": "wss://host/api/v2/live/sess_xyz789"               // Live API用WebSocket URL
  }

POST /api/v2/session/end
Request:  { "session_id": "sess_xyz789" }
Response: { "success": true }
```

#### REST 対話

```
POST /api/v2/chat
Request:
  {
    "session_id": "sess_xyz789",
    "message": "渋谷でイタリアンを探して",
    "language": "ja"
  }
Response:
  {
    "response": "渋谷エリアのイタリアンですね...",
    "audio": "<base64 MP3>",              // TTS音声
    "expression": {                        // A2E結果（同梱）
      "names": ["eyeBlinkLeft", ...],
      "frames": [[0.1, ...], ...],
      "frame_rate": 30
    },
    "shops": [...],                        // モード固有データ（グルメの場合）
    "metadata": {
      "tts_duration_ms": 1200,
      "a2e_duration_ms": 800,
      "llm_duration_ms": 1500
    }
  }
```

#### Live API WebSocket

```
WebSocket /api/v2/live/{session_id}

# クライアント → サーバー
{ "type": "audio", "data": "<base64 PCM 16kHz>" }
{ "type": "text",  "data": "テキスト入力" }
{ "type": "stop" }

# サーバー → クライアント
{ "type": "audio",         "data": "<base64 PCM 24kHz>" }
{ "type": "transcription", "role": "user" | "ai", "text": "..." }
{ "type": "expression",    "data": { "names": [...], "frames": [...], "frame_rate": 30 } }
{ "type": "interrupted" }
{ "type": "reconnecting",  "reason": "char_limit" | "incomplete" | "error" }
{ "type": "reconnected",   "session_count": 2 }
{ "type": "tool_result",   "data": { ... } }  // Function Calling結果
```

### 8.2 既存互換 API (`/api/`)

Phase 1 では、既存の gourmet-support エンドポイントをそのまま維持する。

[推定] 既存エンドポイント（パッチファイルの API 呼び出しから推定）:

```
POST /api/session/start     → 既存 gourmet-support にプロキシ
POST /api/session/end       → 既存 gourmet-support にプロキシ
POST /api/chat              → 既存 gourmet-support にプロキシ
POST /api/tts/synthesize    → 既存 gourmet-support にプロキシ（A2E同梱返却）
GET  /api/health            → 既存 gourmet-support にプロキシ
```

**注意**: 実際のエンドポイント一覧は gourmet-support リポジトリのコード確認が必要。上記はパッチファイルから推定したもの。

---

## 9. 既存サービスとの共存戦略

### 9.1 Phase 1: 並行稼働

```
                    ┌─────────────────────────┐
                    │     ロードバランサー       │
                    └──────┬──────────┬────────┘
                           │          │
              /api/*       │          │  /api/v2/*
              (既存)       │          │  (新プラットフォーム)
                           ▼          ▼
                    ┌──────────┐ ┌──────────┐
                    │gourmet-  │ │platform- │
                    │support   │ │backend   │
                    │(既存)    │ │(新規)    │
                    └──────────┘ └──────────┘
                           │          │
                           └────┬─────┘
                                │
                         ┌──────┴──────┐
                         │audio2exp-   │
                         │service(共用)│
                         └─────────────┘
```

**ポイント**:
- 既存の gourmet-support は**一切変更しない**
- 新プラットフォームバックエンドは別プロセス・別 Cloud Run サービスとしてデプロイ
- audio2exp-service は両方から呼び出される（変更なし）
- フロントエンドは URL パスまたはサブドメインで切り分け

### 9.2 Phase 2: 段階的移行

```
1. グルメモードを新プラットフォーム上で再現
   → 既存 gourmet-support と同等の動作を確認
2. 新プラットフォーム側にトラフィックを徐々に切り替え
   → A/B テスト、既存が問題なく動作することを常時確認
3. 全トラフィック移行完了後、既存 gourmet-support を退役
```

### 9.3 フロントエンドの共存

**設計判断**: 既存の gourmet-sp と新フロントエンドは別デプロイとする。

| 項目 | 既存 (gourmet-sp) | 新プラットフォーム |
|------|-------------------|-------------------|
| URL | `gourmet.example.com` | `platform.example.com` |
| デプロイ先 | Vercel | Vercel (別プロジェクト) |
| バックエンド | gourmet-support | platform-backend |
| 変更 | **なし** | 新規開発 |

---

## 10. iPhone SE 対応戦略

### 10.1 レンダリング方式の選択肢

| 方式 | 技術 | 品質 | iPhone SE 動作 | 状態 |
|------|------|------|----------------|------|
| A | LAM WebGL SDK (Gaussian Splatting) | 最高（論文品質） | **[未検証]** 81,424 Gaussians | 現在の実装 |
| B | Three.js + GLB メッシュ + ブレンドシェイプ | 中〜高 | 動作実績あり（TalkingHead等） | 未実装 |
| C | 2Dアニメーション + 口パク | 低〜中 | 軽量 | フォールバック |

### 10.2 判断基準

**設計判断**: 方式A を第一候補とし、iPhone SE 実機テストの結果で判断する。

```
iPhone SE 実機テスト結果
    │
    ├── 30fps 以上 → 方式A を採用
    │
    ├── 15-30fps → Gaussian 数を削減 (LOD) して再テスト
    │     │
    │     ├── 30fps 以上 → 方式A (LOD版) を採用
    │     │
    │     └── 15fps 未満 → 方式B にフォールバック
    │
    └── 15fps 未満 → 方式B を採用
```

**根拠**:
- [確認済み] LAM プロジェクトページによると iPhone 16 で 35FPS の実績あり
- iPhone SE (A13/A15) は iPhone 16 (A18) より GPU性能が低い
- [確認済み] `LAMAvatar.astro` L337-339 にフォールバック画像表示の仕組みが既にある

### 10.3 方式B の実装指針（フォールバック）

方式B を採用する場合:
1. LAM で生成したアバターの FLAME メッシュを GLB 形式でエクスポート
2. Three.js で GLB をロードし、52次元ブレンドシェイプを直接適用
3. [確認済み] `vrm-expression-manager.ts` の `ExpressionData` インターフェースはそのまま使用可能
4. GPU 負荷が Gaussian Splatting の1/10以下になるため、iPhone SE でも問題なく動作する見込み

**注意**: GLB メッシュの品質は Gaussian Splatting より劣る。トレードオフの判断はオーナーが実機で確認して決定する。

---

## 11. 開発ロードマップ

### Phase 0: 技術検証（前提条件の確認）

| タスク | 成果物 | 検証基準 | 見積もり |
|--------|--------|---------|---------|
| iPhone SE 実機テスト | FPSベンチマーク結果 | 30fps 以上で方式A採用決定 | 見積もり不可（実機入手・テスト環境構築に依存） |
| gourmet-support リポジトリ精査 | 実コード読解結果 | 長期記憶スキーマ・エンドポイント一覧の確定 | 見積もり不可（リポジトリの規模に依存） |
| gourmet-sp リポジトリ精査 | 実コード読解結果 | CoreController の完全なインターフェース確定 | 同上 |
| Live API WebSocket 中継レイテンシ計測 | プロトタイプ + 計測結果 | 中継込み往復 500ms 以内 | 見積もり不可 |

**重要**: Phase 0 の結果により、Phase 1 以降の設計が変更される可能性がある。特に:
- iPhone SE テスト結果 → レンダリング方式の決定（方式A or B）
- gourmet-support 精査結果 → 記憶機能の詳細設計、既存互換APIの完全な仕様

### Phase 1: プラットフォーム基盤 + Live API MVP

| タスク | 成果物 | 検証基準 |
|--------|--------|---------|
| バックエンド骨格構築 | `platform/` ディレクトリ、FastAPI サーバー | ヘルスチェック通過 |
| SessionManager 実装 | セッション CRUD | 単体テスト通過 |
| ModeRegistry + グルメモードプラグイン | モード切替動作 | グルメモードでセッション開始・終了 |
| REST 対話経路の実装 | `/api/v2/chat` | テキスト送信→LLM応答→TTS+A2E同梱返却 |
| LiveRelay 実装 | WebSocket 中継 | ブラウザから Live API 音声対話が動作 |
| ReconnectManager 実装 | 累積制限回避 | 800文字超で自動再接続、コンテキスト引き継ぎ |
| SessionMemory 実装 | 短期記憶 | 20ターン保持、再接続時コンテキスト要約 |
| フロントエンド MVP | 最小限の対話UI | テキスト入力 + 音声入力 でAI応答 |

### Phase 2: アバター統合 + 記憶機能

| タスク | 成果物 | 検証基準 |
|--------|--------|---------|
| AvatarRenderer 統合 | LAMAvatar ベースの汎用コンポーネント | 3D アバターが表情アニメーション付きで動作 |
| ExpressionSync 実装 | TTS同期フレーム再生 | 音声と口パクが同期 |
| Live API 経路の A2E 統合 | 音声チャンク→A2E→Expression WS送信 | Live API 対話中にリップシンク動作 |
| LongTermMemory 実装 | Firestore 永続化 | セッション跨ぎでユーザー情報が保持される |
| グルメモード完全移植 | 既存 gourmet-support 同等の動作 | E2E テスト通過 |

### Phase 3: マルチモード + 最適化

| タスク | 成果物 | 検証基準 |
|--------|--------|---------|
| カスタマーサポートモード | support プラグイン | モード切替で異なる対話体験 |
| インタビューモード | interview プラグイン | スクリプト進行管理が動作 |
| iPhone SE 最適化 | LOD / フォールバック実装 | iPhone SE で 30fps |
| 既存 gourmet-support からの移行 | トラフィック切り替え | 全ユーザーが新プラットフォーム経由 |

---

## 付録A: 未確認事項一覧

| # | 項目 | 影響範囲 | 確認方法 | 確認優先度 |
|---|------|---------|---------|-----------|
| 1 | iPhone SE で LAM WebGL SDK が 30fps 出るか | レンダリング方式決定 | 実機テスト | **最高** |
| 2 | gourmet-support の長期記憶ストレージ (Firestore or Supabase) | 記憶機能設計 | gourmet-support リポジトリ確認 | 高 |
| 3 | gourmet-support の `long_term_memory.py` データスキーマ | 長期記憶の統一スキーマ設計 | gourmet-support リポジトリ確認 | 高 |
| 4 | gourmet-sp の `CoreController` 完全インターフェース | フロントエンド共通基盤設計 | gourmet-sp リポジトリ確認 | 高 |
| 5 | WebSocket 中継 (ブラウザ→サーバー→Gemini) のレイテンシ | Live API 経路の UX 品質 | プロトタイプ計測 | 高 |
| 6 | REST API ハイブリッド方式の実効性 | Live/REST 切替設計 | stt_stream.py でのツール定義復活・検証 | 中 |
| 7 | gourmet-support の完全なエンドポイント一覧 | 既存互換 API 設計 | gourmet-support リポジトリ確認 | 中 |
| 8 | AudioManager の内部実装（48kHz→16kHz変換の詳細） | Live API の音声入力設計 | gourmet-sp リポジトリ確認 | 中 |

## 付録B: 設計判断ログ

| # | 判断 | 選択肢 | 選択理由 |
|---|------|--------|---------|
| D1 | Web フレームワーク: FastAPI | Flask, FastAPI, Starlette | Live API の WebSocket 中継に非同期 I/O が必須。stt_stream.py が asyncio ベース |
| D2 | 新旧 API 共存: URL プレフィックス分離 (/api/ vs /api/v2/) | サブドメイン分離, ヘッダ分離, URL分離 | 最もシンプルでロードバランサー設定が容易 |
| D3 | Live API 中継: サーバーサイド | クライアント直接接続, サーバー中継 | API キー保護が必須。中継レイテンシは要計測だが、セキュリティ優先 |
| D4 | 記憶ストレージ: Firestore (初期) | Firestore, Supabase, Redis | Cloud Run との親和性。インターフェース抽象化で後から変更可能 |
| D5 | フロントエンド: 新規デプロイ (既存と別) | 既存改修, 新規デプロイ | 既存を壊さない原則 (P1) に従う |
| D6 | レンダリング: 方式A優先 + 方式Bフォールバック | A固定, B固定, A+Bハイブリッド | 論文超え品質が最上位ゴール。未検証リスクをフォールバックで担保 |
