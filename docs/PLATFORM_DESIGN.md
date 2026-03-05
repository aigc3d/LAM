# プラットフォーム設計書 — Live API 統合 & マルチモード化

> **作成日**: 2026-02-25
> **対象**: gourmet-support バックエンド / gourmet-sp フロントエンド / audio2exp-service / AI_Meeting_App
> **目的**: 現在のグルメサポートAIを汎用プラットフォームに進化させ、複数のAIアプリケーションを単一基盤で運用する

---

## 目次

1. [背景と目的](#1-背景と目的)
2. [現状のシステム構成](#2-現状のシステム構成)
3. [プラットフォーム全体設計](#3-プラットフォーム全体設計)
4. [共通基盤 vs モード固有の仕分け](#4-共通基盤-vs-モード固有の仕分け)
5. [Live API 統合設計](#5-live-api-統合設計)
6. [バックエンド設計](#6-バックエンド設計)
7. [フロントエンド設計](#7-フロントエンド設計)
8. [モード別仕様](#8-モード別仕様)
9. [開発ロードマップ](#9-開発ロードマップ)
10. [移行戦略と既存エンドポイント温存方針](#10-移行戦略と既存エンドポイント温存方針)

---

## 1. 背景と目的

### 1.1 現在の課題

- フロントエンド（gourmet-sp）とバックエンド（gourmet-support）がグルメサポート専用に密結合
- モード追加のたびにハードコードが必要（ページ、コントローラ、ルート）
- `stt_stream.py`（インタビューモード）がデスクトップ専用のスタンドアロンアプリで、Webプラットフォームと統合されていない
- Live API（Gemini Native Audio）の能力がプラットフォームに組み込まれていない

### 1.2 ゴール

1. **プラットフォーム化**: フロントエンド・バックエンドを汎用基盤として再構成し、モード追加を容易にする
2. **Live API 統合**: Gemini Live API をプラットフォームの標準機能として組み込む
3. **既存エンドポイント温存**: α版テスト中のグルメサポートAIを中断しない
4. **段階的移植**: グルメコンシェルジュ → カスタマーサポート → インタビューの順で展開

---

## 2. 現状のシステム構成

### 2.1 gourmet-support バックエンド（4モジュール構成）

```
gourmet-support/
├── app_customer_support.py    # Flask routes, TTS, STT, A2E統合, Socket.IO
├── support_core.py            # SupportSession, SupportAssistant (Gemini LLM)
├── api_integrations.py        # HotPepper, Places, TripAdvisor API
├── long_term_memory.py        # Supabase ユーザープロフィール・長期記憶
└── prompts/                   # モード別×言語別プロンプト
```

**既存エンドポイント（温存対象）**:

| エンドポイント | メソッド | 機能 |
|---------------|---------|------|
| `/api/session/start` | POST | セッション開始、mode: chat/concierge |
| `/api/session/end` | POST | セッション終了 |
| `/api/chat` | POST | LLM応答（Gemini 2.0 Flash） |
| `/api/tts/synthesize` | POST | TTS + A2E バンドル応答 |
| `/api/stt/transcribe` | POST | STT（単発） |
| `audio_chunk` (WS) | Socket.IO | ストリーミングSTT |
| `/api/finalize` | POST | セッション最終化 |
| `/health` | GET | ヘルスチェック |

### 2.2 gourmet-sp フロントエンド

```
CoreController (base-controller.ts)
├── ConciergeController (concierge-controller.ts)
│   └── LAMAvatarController (LAMAvatar.astro) — 3D Gaussian Splatting
└── ChatController (chat-controller.ts) — テキストのみ
```

- **モード切替**: ページ遷移（`/concierge` ↔ `/`）、ハードコード
- **プラグイン機構**: なし

### 2.3 AI_Meeting_App/stt_stream.py（スタンドアロン）

```
GeminiLiveApp
├── Live API 接続管理（自動再接続、累積文字数制限）
├── 会話履歴管理（直近20ターン保持）
├── 発話途切れ検知（_is_speech_incomplete）
├── REST API フォールバック（長文処理）
├── TTS Player（GCP TTS + PyAudio直接再生）
└── 3モード: standard / silent / interview
```

**stt_stream.py から移植可能な機能**:

| 機能 | stt_stream.py 実装 | プラットフォーム移植先 |
|------|-------------------|---------------------|
| Live API 接続・再接続 | `GeminiLiveApp.run()` | `platform/live_api_proxy.py` |
| 会話履歴管理 | `conversation_history` | `platform/base_session.py` |
| 自動再接続（累積文字数） | `MAX_AI_CHARS_BEFORE_RECONNECT=800` | `platform/live_api_proxy.py` |
| 発話途切れ検知 | `_is_speech_incomplete()` | `platform/live_api_proxy.py` |
| コンテキスト要約 | `_get_context_summary()` | `platform/base_session.py` |
| モード別システムプロンプト | `_build_system_instruction()` | `modes/*/config.py` |
| スクリプト進行管理 | `_get_next_question_from_script()` | `modes/interview/script_manager.py` |
| 議事録 | `log_transcript()` | `modes/interview/transcript.py` |
| REST API ハイブリッド | `RestAPIHandler` | `platform/base_assistant.py` |

---

## 3. プラットフォーム全体設計

```
┌──────────────────────────────────────────────────────────────────┐
│                    Platform Core（共通基盤）                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Frontend (Astro + TypeScript)                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CoreController: セッション, TTS再生, マイク, UI共通       │   │
│  │ LiveAPIClient: WebSocket経由のLive API通信               │   │
│  │ LAMAvatarController: 3Dアバター + A2E blendshape         │   │
│  │ AudioManager: マイク入力パイプライン (16kHz PCM, VAD)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Backend (Flask + Socket.IO)                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ModeRegistry: モード登録・取得・一覧                      │   │
│  │ BaseSession: セッション管理, 会話履歴                     │   │
│  │ BaseAssistant: Gemini REST API ラッパー                   │   │
│  │ TTSService: Google Cloud TTS 合成                        │   │
│  │ STTService: Google Cloud STT (Chirp2)                    │   │
│  │ LiveAPIProxy: Gemini Live API WebSocket 中継              │   │
│  │ MemoryService: Supabase 長期記憶                         │   │
│  │ A2EClient: audio2exp-service 連携                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  External Services                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ audio2exp-service (Cloud Run): Wav2Vec2 → ARKit 52ch     │   │
│  │ Google Cloud TTS / STT                                    │   │
│  │ Gemini 2.0/2.5 Flash (REST + Live API)                   │   │
│  │ Supabase (長期記憶)                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                Mode Plugins（モード固有ロジック）                   │
├────────────┬────────────┬──────────────┬─────────────────────────┤
│  Gourmet   │  Customer  │  Interview   │  将来の新モード          │
│ Concierge  │  Support   │              │                         │
├────────────┼────────────┼──────────────┼─────────────────────────┤
│ system     │ system     │ system       │ system prompt           │
│ prompt     │ prompt     │ prompt       │                         │
│ (グルメ    │ (CS対応    │ (インタビュー │                         │
│  コンシェ   │  FAQ回答)  │  進行)       │                         │
│  ルジュ)   │            │              │                         │
├────────────┼────────────┼──────────────┼─────────────────────────┤
│ HotPepper  │ FAQ DB     │ script       │ 外部API / データソース   │
│ Places     │ Ticket Sys │ PDF参照      │                         │
│ TripAdvisor│            │              │                         │
├────────────┼────────────┼──────────────┼─────────────────────────┤
│ ショップ   │ チケット   │ 録音         │ モード固有UI             │
│ カードUI   │ UI         │ 議事録UI     │ コンポーネント           │
├────────────┼────────────┼──────────────┼─────────────────────────┤
│ 対話: Live │ 対話: Live │ 全部: Live   │ Live / REST 配分        │
│ 説明: REST │ 回答: REST │ 長文のみREST │                         │
└────────────┴────────────┴──────────────┴─────────────────────────┘
```

---

## 4. 共通基盤 vs モード固有の仕分け

| レイヤー | 共通（Platform Core） | モード固有（Plugin） |
|---------|----------------------|---------------------|
| **セッション** | session管理, 会話履歴, タイムアウト | 初期化パラメータ, greeting |
| **LLM** | Gemini接続, REST/Live切替, retry | system prompt, tools, function calling |
| **TTS** | GCP TTS合成, A2E連携, 音声配信 | voice設定（声種, 速度, ピッチ） |
| **STT** | GCP STT, WebSocket streaming | 言語設定, 認識パラメータ |
| **A2E** | audio2exp連携, frame配信 | （共通、モード依存なし） |
| **アバター** | GVRM renderer, blendshape適用 | （共通、モード依存なし） |
| **記憶** | Supabase CRUD, session storage | collection名, schema, 保存項目 |
| **Live API** | WebSocket proxy, 再接続, VAD | context window設定, speech config |
| **外部API** | HTTP client共通 | HotPepper, FAQ DB, スクリプトファイル |
| **UI** | チャットUI, マイクボタン, TTS再生 | ショップカード, チケットUI, 議事録UI |

---

## 5. Live API 統合設計

### 5.1 アーキテクチャ

```
Browser                      Backend                           Gemini
┌────────┐   WebSocket    ┌──────────────┐   WebSocket      ┌──────────┐
│ Audio  │ ──────────────►│ LiveAPI      │ ────────────────►│ Gemini   │
│ Manager│ PCM 16kHz      │ Proxy        │ audio/pcm        │ Live API │
│        │◄──────────────│              │◄────────────────│          │
│        │ audio chunks   │              │ audio + text     │          │
└────────┘                │              │                  └──────────┘
                          │              │
                          │   ┌──────────┤
                          │   │ Session  │ ← 会話履歴, 再接続コンテキスト
                          │   │ Manager  │
                          │   └──────────┤
                          │   │ A2E      │ ← TTS音声 → 表情データ生成
                          │   │ Client   │
                          │   └──────────┘
                          └──────────────┘
```

### 5.2 Live API Proxy の責務

`stt_stream.py` の `GeminiLiveApp` をWebサーバー向けに再設計:

| stt_stream.py（デスクトップ） | live_api_proxy.py（Web） |
|------------------------------|-------------------------|
| PyAudio 直接入力 | WebSocket 経由で受信 |
| PyAudio 直接出力 | WebSocket 経由で配信 |
| asyncio.Queue | Socket.IO rooms |
| 単一ユーザー | マルチセッション対応 |
| ローカルファイル議事録 | DB/メモリ内議事録 |
| 環境変数でモード切替 | ModeRegistry から動的取得 |

### 5.3 Live API 設定（モード共通）

```python
# stt_stream.py から移植する設定
LIVE_API_CONFIG = {
    "response_modalities": ["AUDIO"],
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
```

### 5.4 モード別の Live/REST 使い分け

| モード | Live API（低遅延対話） | REST API（長文生成） |
|--------|----------------------|---------------------|
| **グルメコンシェルジュ** | 好みヒアリング、相槌、確認 | ショップカード説明、詳細レビュー |
| **カスタマーサポート** | 状況ヒアリング、共感、確認 | FAQ回答、手順説明、チケット作成 |
| **インタビュー** | 質問、相槌、進行（メイン） | 資料参照の長文説明時のみ |

### 5.5 自動再接続メカニズム（stt_stream.py から移植）

```
セッション開始
    │
    ▼
Live API 接続
    │
    ├── 音声送受信ループ
    │   │
    │   ├── AI発話 → 累積文字数カウント
    │   │   └── 800文字超過 → 再接続フラグ ON
    │   │
    │   ├── 発話途切れ検知 → 即時再接続フラグ ON
    │   │
    │   └── 長い発話(500文字超) → 再接続フラグ ON
    │
    ▼
再接続フラグ ON
    │
    ├── 会話履歴からコンテキスト要約を生成
    ├── system_instruction に要約を追加
    ├── Live API 再接続
    └── 「続きをお願いします」で再開
```

---

## 6. バックエンド設計

### 6.1 ディレクトリ構成

```
gourmet-support/
│
├── app_customer_support.py      # ★既存（温存、一切変更しない）
├── support_core.py              # ★既存（温存）
├── api_integrations.py          # ★既存（温存）
├── long_term_memory.py          # ★既存（温存）
├── prompts/                     # ★既存（温存）
│
├── app_platform.py              # 新規: プラットフォーム用 Flask app
│
├── platform/                    # 新規: 共通基盤モジュール
│   ├── __init__.py
│   ├── mode_registry.py         # モード登録・取得・一覧
│   ├── base_session.py          # 共通セッション管理
│   ├── base_assistant.py        # 共通LLMラッパー (REST API)
│   ├── tts_service.py           # TTS共通化
│   ├── stt_service.py           # STT共通化
│   ├── live_api_proxy.py        # Live API WebSocket proxy
│   ├── memory_service.py        # 長期記憶共通化
│   └── a2e_client.py            # A2E連携クライアント
│
└── modes/                       # 新規: モードプラグイン
    ├── __init__.py
    ├── base_mode.py             # モード基底クラス
    ├── gourmet/
    │   ├── __init__.py
    │   ├── config.py            # プロンプト, voice設定, API keys
    │   ├── assistant.py         # グルメ固有ロジック (support_core.pyベース)
    │   └── apis.py              # HotPepper, Places (api_integrations.pyベース)
    ├── customer_support/
    │   ├── __init__.py
    │   ├── config.py
    │   └── assistant.py
    └── interview/
        ├── __init__.py
        ├── config.py            # stt_stream.py のプロンプト移植
        ├── assistant.py
        ├── script_manager.py    # stt_stream.py のスクリプト管理移植
        └── transcript.py        # 議事録管理
```

### 6.2 ModeRegistry

```python
class ModeConfig:
    """モード設定の定義"""
    mode_id: str                    # "gourmet", "support", "interview"
    display_name: str               # "グルメコンシェルジュ"
    system_prompt: str              # LLM用システムプロンプト
    live_api_instruction: str       # Live API用システムインストラクション
    voice_config: dict              # {"language_code": "ja-JP", "name": "ja-JP-Neural2-B", ...}
    live_api_enabled: bool          # Live APIを使うか
    live_rest_split: str            # "live_primary" | "rest_primary" | "live_only"
    tools: list                     # Function Calling ツール定義
    external_apis: list             # 外部API設定
    memory_collection: str          # Supabase collection名
    ui_components: list             # フロントエンドで表示するUIコンポーネント
    greeting: str                   # 初回挨拶

class ModeRegistry:
    """モード登録・管理"""
    _modes: dict[str, ModeConfig]

    def register(mode_config: ModeConfig) -> None
    def get(mode_id: str) -> ModeConfig
    def list_modes() -> list[ModeConfig]
    def get_default() -> ModeConfig
```

### 6.3 プラットフォーム用エンドポイント

```python
# app_platform.py — 既存エンドポイントとは別ポート or URL prefix で共存

# === モード管理 ===
GET  /api/v2/modes                    # 利用可能モード一覧
GET  /api/v2/modes/{mode_id}/config   # モード設定取得

# === セッション ===
POST /api/v2/session/start            # { mode: "gourmet" | "support" | "interview" }
POST /api/v2/session/end

# === REST API チャット ===
POST /api/v2/chat                     # 既存互換 + モード自動切替

# === TTS ===
POST /api/v2/tts/synthesize           # TTS + A2E バンドル

# === STT ===
POST /api/v2/stt/transcribe
WS   /api/v2/stt/stream               # ストリーミングSTT（Socket.IO）

# === Live API（新規）===
WS   /api/v2/live/connect             # Live API WebSocket proxy
WS   /api/v2/live/audio               # 音声ストリーム（上り/下り）

# === ヘルスチェック ===
GET  /health
```

### 6.4 Live API Proxy 詳細設計

```python
class LiveAPIProxy:
    """
    Gemini Live API の WebSocket 中継サーバー
    stt_stream.py の GeminiLiveApp をWebサーバー向けに再設計
    """

    # セッションごとの状態
    sessions: dict[str, LiveSession]

    class LiveSession:
        session_id: str
        mode_config: ModeConfig
        gemini_session: object          # Gemini Live API session
        conversation_history: list      # 会話履歴（直近20ターン）
        ai_char_count: int              # 累積文字数（再接続判定用）
        session_count: int              # 再接続回数
        transcript_buffer: dict         # user/ai のトランスクリプトバッファ

    async def connect(session_id: str, mode_id: str) -> None
        """Live APIセッション開始"""

    async def send_audio(session_id: str, audio_chunk: bytes) -> None
        """クライアント→Live APIに音声送信"""

    async def receive_loop(session_id: str) -> AsyncGenerator
        """Live API→クライアントに音声・テキスト配信"""

    async def reconnect(session_id: str) -> None
        """コンテキスト引き継ぎで再接続（stt_stream.py方式）"""

    def is_speech_incomplete(text: str) -> bool
        """発話途切れ検知（stt_stream.py から移植）"""

    def get_context_summary(session_id: str) -> str
        """会話履歴の要約（stt_stream.py から移植）"""
```

---

## 7. フロントエンド設計

### 7.1 クラス階層の変更

**現状（ハードコード）**:
```
CoreController
├── ConciergeController (ページ固定)
└── ChatController (ページ固定)
```

**プラットフォーム化後（動的モード）**:
```
PlatformController (旧CoreController拡張)
├── ModeManager         ← モード動的切替
├── LiveAPIClient       ← Live API WebSocket通信（新規）
├── RESTClient          ← REST API通信（既存）
├── TTSPlayer           ← TTS再生（既存）
├── AudioManager        ← マイク入力（既存）
└── LAMAvatarController ← 3Dアバター（既存）

ModePlugin (interface)
├── GourmetMode         ← ショップカード、店舗検索UI
├── CustomerSupportMode ← チケットUI、FAQ表示
└── InterviewMode       ← スクリプト表示、録音、議事録
```

### 7.2 ModePlugin インターフェース

```typescript
interface ModePlugin {
    modeId: string;
    displayName: string;

    // UI
    renderCustomUI(container: HTMLElement): void;
    destroyCustomUI(): void;

    // メッセージ処理
    onAssistantMessage(message: AssistantMessage): void;
    onUserMessage(message: string): void;

    // Live API イベント
    onLiveAudioReceived?(audioData: ArrayBuffer): void;
    onLiveTranscript?(text: string, role: 'user' | 'ai'): void;

    // ライフサイクル
    onActivate(): void;
    onDeactivate(): void;
}
```

### 7.3 LiveAPIClient（新規）

```typescript
class LiveAPIClient {
    private ws: WebSocket;
    private sessionId: string;

    // 接続
    connect(sessionId: string, modeId: string): Promise<void>;
    disconnect(): void;

    // 音声送受信
    sendAudio(pcmData: ArrayBuffer): void;
    onAudioReceived(callback: (data: ArrayBuffer) => void): void;

    // トランスクリプト
    onTranscript(callback: (text: string, role: string) => void): void;

    // 状態
    onReconnecting(callback: () => void): void;
    onReconnected(callback: () => void): void;
}
```

---

## 8. モード別仕様

### 8.1 グルメコンシェルジュ

| 項目 | 仕様 |
|------|------|
| **mode_id** | `gourmet` |
| **対話方式** | Live API（ヒアリング） + REST（店舗説明） |
| **外部API** | HotPepper, Google Places, TripAdvisor |
| **固有UI** | ショップカード、マップ、レビュー表示 |
| **記憶** | Supabase（ユーザー好み、過去の検索履歴） |
| **アバター** | 3D Gaussian Splatting + A2E blendshape |
| **voice** | ja-JP-Neural2-B（女性、明るい） |

### 8.2 カスタマーサポート

| 項目 | 仕様 |
|------|------|
| **mode_id** | `support` |
| **対話方式** | Live API（状況ヒアリング） + REST（FAQ回答、手順説明） |
| **外部API** | FAQ DB, チケットシステム |
| **固有UI** | FAQ検索結果、チケット作成フォーム、ステータス表示 |
| **記憶** | Supabase（問い合わせ履歴、顧客情報） |
| **アバター** | 3D Gaussian Splatting + A2E blendshape |
| **voice** | ja-JP-Neural2-C（女性、落ち着いた） |

### 8.3 インタビュー

| 項目 | 仕様 |
|------|------|
| **mode_id** | `interview` |
| **対話方式** | Live API主体（全対話）、長文説明時のみREST |
| **外部API** | なし（スクリプトファイル、参照PDF） |
| **固有UI** | スクリプト表示、録音コントロール、議事録ダウンロード |
| **記憶** | セッション内のみ（議事録はファイル保存） |
| **アバター** | 3D Gaussian Splatting + A2E blendshape |
| **voice** | ja-JP-Neural2-D（男性、落ち着いた） |
| **stt_stream.py からの移植** | Live API接続管理、自動再接続、発話途切れ検知、スクリプト進行、議事録 |

---

## 9. 開発ロードマップ

### Phase 0: 設計・準備

- [x] プラットフォーム設計書（本ドキュメント）
- [ ] gourmet-support の既存コード精査・依存関係整理
- [ ] ModeConfig / ModeRegistry のインターフェース確定

### Phase 1: バックエンド Platform Core

```
目標: platform/ + modes/gourmet/ + app_platform.py を構築し、
      既存 gourmet-support と並行稼働できることを確認
```

**1-1. platform/ 共通基盤**
- [ ] `mode_registry.py` — モード登録・取得
- [ ] `base_session.py` — 共通セッション管理（support_core.py からリファクタ）
- [ ] `base_assistant.py` — 共通LLMラッパー（support_core.py からリファクタ）
- [ ] `tts_service.py` — TTS共通化（app_customer_support.py から抽出）
- [ ] `stt_service.py` — STT共通化（app_customer_support.py から抽出）
- [ ] `memory_service.py` — 長期記憶共通化（long_term_memory.py からリファクタ）
- [ ] `a2e_client.py` — A2E連携（app_customer_support.py から抽出）

**1-2. live_api_proxy.py（stt_stream.py 移植）**
- [ ] WebSocket中継サーバー（マルチセッション対応）
- [ ] 自動再接続メカニズム（累積文字数 + 発話途切れ）
- [ ] コンテキスト要約・引き継ぎ
- [ ] Live API 設定のモード別カスタマイズ

**1-3. modes/gourmet/**
- [ ] `config.py` — プロンプト、voice設定、API設定
- [ ] `assistant.py` — グルメ固有ロジック
- [ ] `apis.py` — HotPepper, Places連携

**1-4. app_platform.py**
- [ ] /api/v2/ エンドポイント実装
- [ ] Socket.IO Live API ルーム
- [ ] 既存 app_customer_support.py との共存確認

**テスト**: platform経由でグルメコンシェルジュが動作すること

### Phase 2: フロントエンド Platform Core

**2-1. PlatformController**
- [ ] CoreController を PlatformController に拡張
- [ ] ModeManager — 動的モード切替（ページ遷移→SPA内切替）
- [ ] ModePlugin インターフェース定義

**2-2. LiveAPIClient**
- [ ] WebSocket 接続管理
- [ ] 音声送受信パイプライン
- [ ] 再接続UI（インジケータ表示）

**2-3. GourmetMode プラグイン**
- [ ] 既存 ConciergeController のロジックを ModePlugin に移行
- [ ] ショップカードUI
- [ ] Live API ↔ REST 自動切替

**テスト**: 新フロントエンド → app_platform.py で動作確認

### Phase 3: カスタマーサポートモード

- [ ] `modes/customer_support/config.py` — CS用プロンプト
- [ ] `modes/customer_support/assistant.py` — FAQ検索、チケット作成
- [ ] フロントエンド `CustomerSupportMode` プラグイン
- [ ] FAQ DB 連携（Supabase or 外部API）

### Phase 4: インタビューモード移植

- [ ] `modes/interview/config.py` — stt_stream.py のプロンプト移植
- [ ] `modes/interview/script_manager.py` — スクリプト進行管理
- [ ] `modes/interview/transcript.py` — 議事録管理
- [ ] フロントエンド `InterviewMode` プラグイン
- [ ] スクリプト表示UI、録音コントロール、議事録ダウンロード

---

## 10. 移行戦略と既存エンドポイント温存方針

### 10.1 共存アーキテクチャ

```
                     ┌─────────────────────────┐
                     │      Cloud Run          │
                     │                         │
α版フロントエンド ──►│  app_customer_support.py │◄── 既存 :5001
(gourmet-sp)         │  /api/session/start     │
                     │  /api/chat              │
                     │  /api/tts/synthesize    │
                     │                         │
新フロントエンド  ──►│  app_platform.py        │◄── 新規 :5002
(platform-sp)        │  /api/v2/session/start  │
                     │  /api/v2/chat           │
                     │  /api/v2/live/connect   │
                     │  /api/v2/modes          │
                     └─────────────────────────┘
```

### 10.2 温存ルール

1. **`app_customer_support.py` は一切変更しない** — α版テストに影響を与えない
2. **`support_core.py`, `api_integrations.py`, `long_term_memory.py` も変更しない**
3. `platform/` は既存コードからの **コピー＆リファクタ**（import依存にしない）
4. 将来、α版テスト完了後に既存エンドポイントを deprecate → platform に一本化

### 10.3 デプロイ戦略

| フェーズ | app_customer_support.py | app_platform.py |
|---------|------------------------|----------------|
| Phase 1 | Cloud Run（現行） | ローカル開発 |
| Phase 2 | Cloud Run（現行） | Cloud Run（別サービス or 別ポート） |
| Phase 3 | Cloud Run（現行） | Cloud Run（本番） |
| 統合後 | 廃止 | Cloud Run（全モード統合） |
