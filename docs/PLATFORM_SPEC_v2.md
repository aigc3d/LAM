# グルメポートAI プラットフォーム仕様書 v2

> **文書ID**: SPEC-PLATFORM-002
> **作成日**: 2026-03-02
> **ステータス**: Draft
> **前提**: `support_base/` および `docs/` 既存ドキュメントの実コード読解に基づき作成

---

## 目次

1. [目的と背景](#1-目的と背景)
2. [プラットフォーム目標](#2-プラットフォーム目標)
3. [システム構成](#3-システム構成)
4. [LiveAPI + REST API ハイブリッド仕様](#4-liveapi--rest-api-ハイブリッド仕様)
5. [プロンプト外部管理 (GCS)](#5-プロンプト外部管理-gcs)
6. [記憶機能 (短期・長期)](#6-記憶機能-短期長期)
7. [LLM 検索機能 SDK 対応](#7-llm-検索機能-sdk-対応)
8. [多言語対応](#8-多言語対応)
9. [マルチデバイス対応](#9-マルチデバイス対応)
10. [リップシンク実写アバター](#10-リップシンク実写アバター)
11. [API 仕様](#11-api-仕様)
12. [データフロー](#12-データフロー)
13. [実装ステータスと残作業](#13-実装ステータスと残作業)

---

## 1. 目的と背景

### 1.1 目的

**グルメポートAIアプリのコンシェルジュモードとグルメモードの両方を LiveAPI 仕様に変更する。**

現行の gourmet-support は REST API ベースのテキスト対話（ユーザー入力 → LLM応答 → TTS読み上げ → アバター口パク）であり、以下の体験上の問題がある:

| 課題 | 詳細 |
|------|------|
| **往復レイテンシ** | STT → REST API → TTS の直列処理で **5〜10秒のラグ** |
| **割り込み不可** | 「話す → 待つ → 聞く」の交互方式。AIの回答中にユーザーが割り込めない |
| **相槌なし** | 「へぇ」「なるほど」等のリアルタイム応答ができない |

Gemini Live API はこれらを根本的に解決する:

- **超低遅延**: 音声入力 → 音声出力が数百ms
- **割り込み対応**: ユーザーの発話を検知してAIが応答を中断
- **ネイティブ音声生成**: TTS不要、AIが直接音声を生成

### 1.2 対象モード

| モード | 現行方式 | 変更後 |
|--------|---------|--------|
| **コンシェルジュモード** (concierge) | REST API + TTS | **Live API** (ヒアリング・相槌) + REST API (店舗説明・詳細レビュー) |
| **グルメモード** (chat) | REST API + TTS | **Live API** (対話) + REST API (検索結果説明) |

---

## 2. プラットフォーム目標

本アプリを様々なAIアプリに流用するためのプラットフォーム化を行う。

### 2.1 7つの柱

| # | 柱 | 概要 |
|---|-----|------|
| 1 | **LiveAPI + REST API ハイブリッド** | 短文・相槌は Live API、長文・検索結果は REST API + TTS |
| 2 | **プロンプト GCS 外部保存** | 流用・メンテ・チューニングの利便性 |
| 3 | **短期記憶 + 長期記憶** | LLMの弱点を補完。セッション内コンテキスト + ユーザーパーソナライゼーション |
| 4 | **LLM 検索機能 SDK 対応** | Google Search Grounding による最新情報取得 |
| 5 | **多言語対応** | 多言語ファイルによる ja/en/ko/zh 対応 |
| 6 | **マルチデバイス** | スマホ (iPhone/Android) + Web アプリ |
| 7 | **リップシンク実写アバター** | スマホ単体で軽量に動く実写アバター |

---

## 3. システム構成

### 3.1 リポジトリ構成

| レイヤー | リポジトリ | 技術スタック | デプロイ先 | 状態 |
|---------|-----------|-------------|-----------|------|
| **フロントエンド** | [mirai-gpro/gourmet-sp2](https://github.com/mirai-gpro/gourmet-sp2) | Astro + TypeScript | Vercel (**未連携**) | アバターリップシンク実装・テスト済 |
| **バックエンド** | [mirai-gpro/support-base](https://github.com/mirai-gpro/support-base) | FastAPI + Python | Cloud Run | LiveAPI対応・プラットフォーム化済 |
| **A2Eサービス** | LAM_gpro/services/audio2exp-service | Flask + PyTorch | Cloud Run | デプロイ済・ヘルスチェックOK |

> **注意**: `gourmet-sp` (旧リポジトリ) はアバター対応なし。アバター付きリップシンクの実装・テストは `gourmet-sp2` で実施済み。今回のプラットフォーム化は `gourmet-sp2` をベースとする。

### 3.2 バックエンド (support-base) モジュール構成 [確認済み]

```
support_base/
├── server.py                    # FastAPI エントリーポイント (uvicorn)
├── config/
│   └── settings.py              # 環境変数ベース設定
├── core/
│   ├── support_core.py          # ビジネスロジック (SupportSession, SupportAssistant)
│   │                            # GCS/ローカルプロンプト読み込み
│   │                            # Gemini REST API (gemini-2.5-flash) 対話
│   │                            # Google Search Grounding 対応
│   ├── api_integrations.py      # HotPepper / Google Places / TripAdvisor API
│   └── long_term_memory.py      # 長期記憶 (Supabase user_profiles テーブル)
├── live/
│   ├── relay.py                 # Live API WebSocket 中継 (LiveRelay)
│   ├── reconnect.py             # 累積文字数制限 回避ロジック (ReconnectManager)
│   └── speech_detector.py       # 発話途切れ検知 (多言語対応)
├── memory/
│   └── session_memory.py        # 短期記憶 (SessionMemory, 直近20ターン)
├── session/
│   └── manager.py               # セッション管理 (Session, SessionManager)
├── modes/
│   ├── base_mode.py             # モードプラグイン基底クラス (BaseModePlugin)
│   ├── registry.py              # モードレジストリ (ModeRegistry)
│   └── gourmet/
│       └── plugin.py            # グルメコンシェルジュ プラグイン
├── rest/
│   └── router.py                # REST API ルーター (gourmet-support 互換)
├── services/
│   └── a2e_client.py            # audio2exp-service HTTP クライアント
└── i18n/
    └── language_config.py       # 言語マスター設定 (LanguageProfile)
```

### 3.3 フロントエンド (gourmet-sp2) モジュール構成 [確認済み]

```
gourmet-sp2/
├── astro.config.mjs                # Astro設定 (SSG, PWA, COOP/COEP, WASM対応)
├── package.json                    # 依存: three, gaussian-splat-renderer-for-lam, onnxruntime-web
├── gs.ts                           # Gaussian Splatting ビューアー (Three.js, LBS skinning)
├── gvrm.ts                         # GVRM アバターシステム (bone texture, lip-sync API)
├── src/
│   ├── pages/
│   │   ├── index.astro             # グルメモード (Chat) ページ
│   │   ├── concierge.astro         # コンシェルジュモード (3Dアバター) ページ
│   │   └── 404.astro
│   ├── components/
│   │   ├── GourmetChat.astro       # チャットUIコンポーネント
│   │   ├── Concierge.astro         # コンシェルジュUIコンポーネント (アバターステージ+チャット)
│   │   ├── LAMAvatar.astro         # LAM 3Dアバター (Gaussian Splatting WebGL レンダリング)
│   │   ├── ShopCardList.astro      # 店舗カードリスト
│   │   ├── ReservationModal.astro  # 予約モーダル
│   │   ├── ProposalCard.astro      # 提案カード
│   │   └── InstallPrompt.astro     # PWA インストールプロンプト (iOS/Android対応)
│   ├── scripts/
│   │   ├── chat/
│   │   │   ├── core-controller.ts          # 基底コントローラー (セッション, TTS, STT, 多言語)
│   │   │   ├── chat-controller.ts          # グルメモード (テキスト/音声チャット)
│   │   │   ├── concierge-controller.ts     # コンシェルジュモード (アバター+A2E+並行TTS)
│   │   │   └── audio-manager.ts            # マイク入力 (iOS/Android/PC分岐, AudioWorklet, VAD)
│   │   ├── lam/
│   │   │   ├── lam-websocket-manager.ts    # OpenAvatarChat WebSocket (JBIN形式受信)
│   │   │   └── audio-sync-player.ts        # 音声再生 精密タイミング同期
│   │   └── avatar/
│   │       └── concierge-interface.ts      # アバターインターフェース
│   ├── constants/
│   │   └── i18n.ts                 # 多言語定義 (ja/en/zh/ko) + LANGUAGE_CODE_MAP
│   ├── styles/
│   └── layouts/
│       └── Layout.astro
├── public/
│   ├── avatar/
│   │   └── concierge/              # LAMアバターアセット (skin.glb, offset.ply)
│   ├── ort-wasm/                   # ONNX Runtime WASM
│   └── manifest.webmanifest        # PWA マニフェスト
└── docs/
    └── backend-integration.md      # API統合ドキュメント
```

### 3.4 コントローラー階層 [確認済み]

```
CoreController (core-controller.ts)
├── セッション管理, TTS再生, Socket.IO STT, 多言語切替, ショップカード表示
│
├── ChatController (chat-controller.ts)
│   └── グルメモード。テキスト/音声チャット。モード切替トグル
│
└── ConciergeController (concierge-controller.ts)
    ├── LAMAvatar との TTS プレーヤー連携
    ├── applyExpressionFromTts() — 52次元ブレンドシェイプのキュー投入
    ├── speakResponseInChunks() — 文分割→並行TTS合成→順次再生
    ├── __testLipSync() — リップシンク診断テスト (ブラウザコンソール)
    └── 無音検出タイムアウト: 8000ms (chatモードは4500ms)
```

### 3.5 全体アーキテクチャ図

```
┌──────────────────────────────────────────────────────────────────────┐
│                      フロントエンド (gourmet-sp2)                       │
│                    Astro + TypeScript + Vercel (予定)                 │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ ModeRouter     │  │ AudioIO      │  │ AvatarRenderer           │  │
│  │ (モード切替)    │  │ (WebAudio)   │  │ (LAM WebGL / Three.js)   │  │
│  └──────┬─────────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                   │                      │                  │
│  ┌──────┴───────────────────┴──────────────────────┴───────────────┐  │
│  │                    DialogueManager                              │  │
│  │  ┌─── LiveAPIClient (WebSocket) ───┐  ┌── RESTClient (HTTP) ──┐│  │
│  │  │ PCM 16kHz 送信 / 24kHz 受信    │  │ /api/v2/rest/*        ││  │
│  │  └────────────────────────────────┘  └────────────────────────┘│  │
│  └──────────────────────┬─────────────────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────────────────┘
                          │ WebSocket / HTTPS
┌─────────────────────────┼────────────────────────────────────────────┐
│               バックエンド (support-base) Cloud Run                    │
│  ┌──────────────────────┴────────────────────────────────────────┐   │
│  │                    FastAPI Gateway                             │   │
│  │  WS /api/v2/live/{session_id}  │  POST /api/v2/rest/*        │   │
│  └──────┬────────────────────────────────────────┬───────────────┘   │
│         │                                        │                   │
│  ┌──────┴──────┐  ┌─────────────┐  ┌────────────┴──────┐           │
│  │ LiveRelay   │  │ SessionMgr  │  │ REST Router       │           │
│  │ (WS中継)    │  │ + Memory    │  │ (gourmet互換)     │           │
│  └──────┬──────┘  └──────┬──────┘  └────────┬──────────┘           │
│         │                │                   │                      │
│  ┌──────┴────────────────┴───────────────────┴──────────────────┐   │
│  │              共通サービス層                                     │   │
│  │  ModeRegistry │ SessionMemory │ LongTermMemory │ A2EClient   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ Gemini       │    │ Google Cloud │    │ audio2exp-service│
│ Live API     │    │ TTS / STT   │    │ (Cloud Run)      │
│ (WS)        │    │              │    │ Wav2Vec2→ARKit52 │
└──────────────┘    └──────────────┘    └──────────────────┘
                           │
                    ┌──────┴──────┐
                    │ GCS         │
                    │ (プロンプト)  │
                    └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ Supabase    │
                    │ (長期記憶)   │
                    └─────────────┘
```

---

## 4. LiveAPI + REST API ハイブリッド仕様

### 4.1 対話経路の使い分け

| 経路 | 用途 | LLMモデル | 音声生成 |
|------|------|----------|---------|
| **Live API** | 短文対話 (ヒアリング・相槌・確認) | `gemini-2.5-flash-native-audio-preview` | AIネイティブ音声 (TTS不要) |
| **REST API** | 長文生成 (検索結果・詳細説明・FAQ回答) | `gemini-2.5-flash` | Google Cloud TTS (Wavenet/Chirp3) |

### 4.2 モード別対話方式

| モード | Live API | REST API |
|--------|---------|----------|
| **コンシェルジュ** (concierge) | 好みヒアリング、シーン確認、相槌 | ショップカード説明、詳細レビュー、エリア案内 |
| **グルメ** (chat) | 対話、質問応答、相槌 | 検索結果一覧、店舗詳細、おすすめ理由 |

### 4.3 Live API 接続アーキテクチャ [確認済み]

```
ブラウザ                    バックエンド                    Gemini
┌────────┐   WebSocket   ┌──────────────┐   WebSocket   ┌──────────┐
│ Audio  │──────────────►│ LiveRelay    │──────────────►│ Gemini   │
│ IO     │ PCM 16kHz     │              │ PCM 16kHz     │ Live API │
│        │◄──────────────│              │◄──────────────│          │
│        │ PCM 24kHz     │              │ PCM 24kHz     │          │
└────────┘  + transcript  │  ┌─────────┐│  + transcript └──────────┘
                          │  │Reconnect││
                          │  │Manager  ││ ← 累積文字数制限回避
                          │  └─────────┘│
                          │  ┌─────────┐│
                          │  │Session  ││ ← 短期記憶 (20ターン)
                          │  │Memory   ││
                          │  └─────────┘│
                          │  ┌─────────┐│
                          │  │A2E      ││ ← 表情データ生成
                          │  │Client   ││
                          │  └─────────┘│
                          └──────────────┘
```

### 4.4 WebSocket プロトコル [確認済み]

**クライアント → サーバー:**

| type | data | 説明 |
|------|------|------|
| `audio` | base64 PCM 16kHz | マイク音声チャンク |
| `text` | テキスト文字列 | テキスト入力 |
| `stop` | — | セッション終了 |

**サーバー → クライアント:**

| type | フィールド | 説明 |
|------|-----------|------|
| `audio` | `data`: base64 PCM 24kHz | AI音声チャンク |
| `transcription` | `role`: user/ai, `text`, `is_partial` | 音声テキスト化 |
| `expression` | `data`: {names, frames, frame_rate} | 52次元ARKitブレンドシェイプ |
| `interrupted` | — | ユーザー割り込み検知 |
| `reconnecting` | `reason` | 再接続開始通知 |
| `reconnected` | `session_count` | 再接続完了通知 |
| `error` | `message` | エラー通知 |

### 4.5 FLASH版 累積文字数制限の回避 [確認済み]

Gemini FLASH版 (`gemini-2.5-flash-native-audio-preview`) にはセッション内の累積トークン制限がある。`support_base/live/reconnect.py` の `ReconnectManager` で以下のロジックを実装済み:

```
AI発話のたびに文字数を累積カウント (ai_char_count)
  │
  ├── 累積 800文字超過 → 再接続フラグ ON
  │     (MAX_AI_CHARS_BEFORE_RECONNECT = 800)
  │
  ├── 1回の発話が 500文字超 → 次のターン前に再接続
  │     (LONG_SPEECH_THRESHOLD = 500)
  │
  ├── 発話が途中で切れた → 即時再接続
  │     (SpeechDetector.is_incomplete(): 多言語対応)
  │
  └── API側エラー (1011/1008) → 3秒後に自動再接続

再接続時の処理 (LiveRelay._run_gemini_session):
  1. SessionMemory から直近10ターンの要約を生成
  2. 新セッションの system_instruction に要約を注入
  3. 「続きをお願いします」を送信して再開
  4. ai_char_count をリセット
```

### 4.6 Live API 設定 [確認済み]

```python
# support_base/live/relay.py _build_live_config()
config = {
    "response_modalities": ["AUDIO"],
    "system_instruction": <モードプラグインから取得>,
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    "speech_config": {
        "language_code": <LanguageProfile.live_api_language_code>,
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
        "sliding_window": { "target_tokens": 32000 }
    },
}
```

---

## 5. プロンプト外部管理 (GCS)

### 5.1 GCS ストレージ構成 [確認済み]

```
gs://{PROMPTS_BUCKET_NAME}/
└── prompts/
    ├── support_system_ja.txt     # グルメ(chat)モード 日本語
    ├── support_system_en.txt     # グルメ(chat)モード 英語
    ├── support_system_zh.txt     # グルメ(chat)モード 中国語
    ├── support_system_ko.txt     # グルメ(chat)モード 韓国語
    ├── concierge_ja.txt          # コンシェルジュモード 日本語
    ├── concierge_en.txt          # コンシェルジュモード 英語
    ├── concierge_zh.txt          # コンシェルジュモード 中国語
    └── concierge_ko.txt          # コンシェルジュモード 韓国語
```

### 5.2 読み込み優先度 [確認済み]

```
1. GCS (PROMPTS_BUCKET_NAME が設定されている場合)
   ↓ 失敗時
2. ローカルファイル (prompts/ ディレクトリ)
   ↓ 失敗時
3. ハードコードのフォールバック (GourmetModePlugin._fallback_prompt())
```

**実装箇所**: `support_base/core/support_core.py` の `load_system_prompts()`

### 5.3 プロンプト利用経路

| 経路 | プロンプト取得元 | 利用モジュール |
|------|----------------|---------------|
| **Live API** | `GourmetModePlugin.get_system_prompt()` → `LOADED_PROMPTS` (GCS/ローカル) | `LiveRelay._build_live_config()` |
| **REST API** | `SupportAssistant.__init__()` → `SYSTEM_PROMPTS` (GCS/ローカル) | `rest/router.py` の各エンドポイント |

### 5.4 プラットフォーム化での拡張方針

新モード追加時は以下のみ:

1. GCS に `prompts/{mode_name}_{lang}.txt` を追加
2. `modes/{mode_name}/plugin.py` を作成し、`BaseModePlugin` を継承
3. `server.py` で `mode_registry.register()` に登録

---

## 6. 記憶機能 (短期・長期)

### 6.1 短期記憶 [確認済み]

**クラス**: `support_base/memory/session_memory.py` の `SessionMemory`

| 項目 | 仕様 |
|------|------|
| **ストレージ** | インメモリ (セッション単位) |
| **保持ターン数** | 直近 20 ターン |
| **データ構造** | `[{role: str, text: str, timestamp: str}]` |
| **コンテキスト要約** | 直近 10 ターン、各150文字上限 |
| **用途** | Live API 再接続時のコンテキスト引き継ぎ |

**主要メソッド:**

| メソッド | 説明 |
|---------|------|
| `add(role, text)` | 会話ターンを追加 (20ターン超は古い方を削除) |
| `get_context_summary()` | 再接続用の会話要約を生成 |
| `get_last_user_message()` | 直前のユーザー発言を取得 |

### 6.2 長期記憶 [確認済み]

**クラス**: `support_base/core/long_term_memory.py` の `LongTermMemory`

| 項目 | 仕様 |
|------|------|
| **ストレージ** | **Supabase** (`user_profiles` テーブル) |
| **キー** | `user_id` (PRIMARY KEY) |
| **スキーマ** | preferred_name, name_honorific, visit_count, conversation_summary, default_language, preferred_mode, first_visit_at, last_visit_at |
| **用途** | ユーザーパーソナライゼーション (名前での呼びかけ、訪問回数、過去の会話記録) |

**主要メソッド:**

| メソッド | 説明 |
|---------|------|
| `get_profile_basic(user_id)` | 軽量プロファイル取得 (名前・訪問回数) |
| `update_profile(user_id, updates)` | UPSERT (存在すれば更新、なければ新規作成) |
| `increment_visit_count(user_id)` | 訪問回数インクリメント |
| `append_conversation_summary(user_id, summary)` | 会話サマリー追記 |
| `generate_system_prompt_context(user_id, language)` | システムプロンプト注入用コンテキスト生成 |

### 6.3 記憶の連携フロー

```
セッション開始
  │
  ├── [長期記憶] user_id でプロファイル取得
  │   ├── リピーター → 名前で呼びかけ + 訪問回数表示
  │   └── 新規 → 名前を聞く
  │
  ├── [短期記憶] SessionMemory 初期化
  │
  ├── Live API 対話中
  │   ├── [短期記憶] 各ターンを add()
  │   ├── 再接続時 → get_context_summary() で文脈引き継ぎ
  │   └── LLM action → [長期記憶] update_profile()
  │
  └── セッション終了
      └── [長期記憶] append_conversation_summary()
```

---

## 7. LLM 検索機能 SDK 対応

### 7.1 Google Search Grounding [確認済み]

REST API 経路では Gemini の Google Search Grounding を有効化。

**実装箇所**: `support_base/core/support_core.py` の `SupportAssistant.process_user_message()`

```python
# フォローアップ質問でない場合、Google検索を有効化
tools = [types.Tool(google_search=types.GoogleSearch())]

config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    tools=tools,
)

response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=history,
    config=config,
)
```

### 7.2 検索利用シーン

| シーン | 検索利用 | 理由 |
|--------|---------|------|
| 初回質問 (店舗検索) | **有効** | 最新の店舗情報・レビューを取得 |
| フォローアップ質問 | **無効** | 既に提案済みの店舗情報を参照して回答 |
| Live API 経路 | **将来対応** | Live API の Function Calling で対応予定 |

### 7.3 Live API でのツール定義

`BaseModePlugin.get_live_api_tools()` で各モードがツール定義を返す。現状は空リスト（将来拡張ポイント）。

---

## 8. 多言語対応

### 8.1 言語マスター設定 [確認済み]

**ファイル**: `support_base/i18n/language_config.py`

| 言語 | コード | TTS | Live API | 文分割 |
|------|--------|-----|----------|--------|
| 日本語 | `ja` | `ja-JP` / `ja-JP-Wavenet-D` | `ja-JP` | CJK (`。`) |
| 英語 | `en` | `en-US` / `en-US-Wavenet-D` | `en-US` | Latin (`. `) |
| 韓国語 | `ko` | `ko-KR` / `ko-KR-Wavenet-D` | `ko-KR` | Latin (`. `) |
| 中国語 | `zh` | `cmn-CN` / `cmn-CN-Wavenet-D` | `cmn-CN` | CJK (`。`) |

### 8.2 多言語対応箇所

| 対応箇所 | 実装状態 | ファイル |
|---------|---------|---------|
| プロンプト (GCS) | [確認済み] | `core/support_core.py` |
| 初回挨拶 | [確認済み] | `modes/gourmet/plugin.py`, `core/support_core.py` |
| 発話途切れ検知 | [確認済み] ja/en/ko/zh | `live/speech_detector.py` |
| TTS 言語・音声選択 | [確認済み] | `i18n/language_config.py` |
| Live API speech_config | [確認済み] | `live/relay.py` |
| REST API レスポンスメッセージ | [確認済み] | `rest/router.py` |
| 長期記憶コンテキスト生成 | [確認済み] ja/en/ko/zh | `core/long_term_memory.py` |
| 会話要約テンプレート | [確認済み] ja/en/ko/zh | `core/support_core.py` |

### 8.3 新言語追加手順

1. `i18n/language_config.py` の `LANGUAGE_PROFILES` に `LanguageProfile` を追加
2. `live/speech_detector.py` の `RULES` に発話途切れパターンを追加
3. GCS に `prompts/{mode}_{lang}.txt` を追加
4. `core/support_core.py` の各テンプレート辞書に言語エントリを追加

---

## 9. マルチデバイス対応

### 9.1 対象デバイス

| デバイス | OS | ブラウザ | 備考 |
|---------|-----|---------|------|
| iPhone (SE以降) | iOS | Safari | **最小動作基準** |
| Android スマホ | Android 10+ | Chrome | |
| PC | Windows/Mac | Chrome / Safari / Firefox | |

### 9.2 フロントエンド技術スタック [確認済み — gourmet-sp2]

| 技術 | 用途 | 状態 |
|------|------|------|
| **Astro 4.0** | SSG フレームワーク (`output: 'static'`) | 実装済 |
| **TypeScript** | アプリロジック | 実装済 |
| **Three.js** (v0.182) | 3D レンダリング | 実装済 |
| **gaussian-splat-renderer-for-lam** (v0.0.9-alpha) | LAM アバター SDK | 実装済 |
| **onnxruntime-web** (v1.23) | ニューラルネット推論 (DINOv2等) | 実装済 |
| **Socket.IO** | STT ストリーミング | 実装済 |
| **WebSocket API** | Live API 中継接続 | **未実装 (今回追加)** |
| **Web Audio API** | マイク入力 (AudioWorklet → PCM 16kHz) + 音声再生 | 実装済 |
| **WebGL 2.0** | Gaussian Splatting レンダリング | 実装済 |
| **PWA** | ホーム画面追加 (iOS手動ガイド / Android自動プロンプト) | 実装済 |

**主要依存パッケージ (package.json):**
```json
{
  "@huggingface/transformers": "^3.8.1",
  "@mkkellogg/gaussian-splats-3d": "^0.4.7",
  "gaussian-splat-renderer-for-lam": "^0.0.9-alpha.1",
  "gsplat": "^1.2.9",
  "onnxruntime-web": "^1.23.2",
  "three": "^0.182.0",
  "@vite-pwa/astro": "^1.2.0",
  "astro": "^4.0.0"
}
```

### 9.3 Vercel デプロイ設定 [要実施]

`gourmet-sp2` は現在ローカルホストでテスト中。スマホテストのため Vercel 連携を新規設定する。

**必要な設定:**

| 項目 | 設定内容 |
|------|---------|
| **Framework** | Astro |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist/` |
| **Node.js** | 18.x 以上 |
| **環境変数** | `PUBLIC_API_URL` = Cloud Run バックエンド URL |
| **カスタムヘッダー** | `Cross-Origin-Embedder-Policy: require-corp`<br>`Cross-Origin-Opener-Policy: same-origin`<br>(WebAssembly ONNX Runtime 用, astro.config.mjs で設定済み) |

**Vercel 連携手順:**

```
1. Vercel ダッシュボード → New Project → Import Git Repository
   https://github.com/mirai-gpro/gourmet-sp2

2. Framework Preset: Astro を選択

3. 環境変数を設定:
   PUBLIC_API_URL = https://<support-base-service>.run.app

4. Deploy
```

### 9.4 モバイル固有の考慮事項 [確認済み — gourmet-sp2 で対応済]

| 項目 | 対策 | 実装状態 |
|------|------|---------|
| **マイク権限** | ユーザー操作起点で `getUserMedia()` を呼ぶ (iOS Safari 制約) | 実装済 |
| **AudioContext** | ユーザー操作後に `resume()` (iOS Safari autoplay policy) | 実装済 |
| **iOS AudioWorklet** | iOS専用パス: 8192バッファ, 500msフラッシュ, サーバー待機最大500ms | 実装済 |
| **Android AudioWorklet** | デフォルトパス: 16000バッファ, サーバー待機最大700ms, VAD チェック100ms毎 | 実装済 |
| **バックグラウンド復帰** | 120秒以上バックグラウンド → ソフトリセット | 実装済 |
| **メモリ制約** | iPhone SE (3-4GB RAM) でのアバターレンダリング最適化 | **要実機テスト** |
| **PWA インストール** | iOS: 3ステップ手動ガイド, Android: `beforeinstallprompt` 自動プロンプト | 実装済 |

---

## 10. リップシンク実写アバター

### 10.1 実装状態 [確認済み — gourmet-sp2 でテスト完了]

**gourmet-sp2 のコンシェルジュモード (`/concierge`) でリップシンクアバターは実装・テスト済み。**

### 10.2 パイプライン概要

```
1枚の顔写真
    ↓ [オフライン] LAM (Large Avatar Model) — HF Spaces / ModelScope
3D Gaussian Splatting アバター (skin.glb + offset.ply + animation.glb)
    ↓ [アプリ起動時] GaussianSplatRenderer でロード
    ↓
AI音声出力 (TTS / Live API)
    ↓ [サーバー] audio2exp-service (Cloud Run, 4Gi)
    │   Wav2Vec2 (95M params) → A2E Decoder
    ↓
52次元 ARKit ブレンドシェイプ @30fps
    ↓ [クライアント] LAMAvatar.astro
    │   frameBuffer にキュー → ttsPlayer.currentTime 同期
    │   30fps → 60fps フレーム補間
    ↓
Gaussian Splatting WebGL レンダリング
    ↓ カメラ: pos(0, 1.72, 0.55), FOV 38°, target(0, 1.66, 0)
リアルタイムリップシンク + 表情アニメーション
```

### 10.3 LAMAvatar コンポーネント [確認済み — gourmet-sp2]

**ファイル**: `gourmet-sp2/src/components/LAMAvatar.astro`

| 機能 | 詳細 |
|------|------|
| **SDK** | `gaussian-splat-renderer-for-lam` (v0.0.9-alpha) |
| **フレームバッファ** | Expression frames を時系列でキュー管理 |
| **TTS同期** | `ttsPlayer.currentTime` (ms) → `frameBuffer[frameIndex]` |
| **フレーム補間** | 30fps A2E → 60fps レンダリング (スムーズ化) |
| **フェードイン/アウト** | 200ms スムーズトランジション |
| **ブレンドシェイプ増幅** | 口元の動きをスケーリングして視認性を向上 |
| **FLAME LBS制約** | 値を 0.7 でクランプ (数値安定性) |
| **フォールバック** | SDK ロード失敗時 → 静止画表示 |

### 10.4 A2E サービス仕様 [確認済み]

**デプロイ**: Cloud Run (us-central1), CPU, メモリ 4Gi

```
POST /api/audio2expression
Request:  { audio_base64: string, session_id: string, audio_format: "mp3"|"wav"|"pcm" }
Response: { names: string[52], frames: number[N][52], frame_rate: 30 }

GET /health
Response: { status: "healthy", engine_ready: bool, mode: "infer"|"fallback", device: "cpu" }
```

### 10.5 表情データの伝送経路

| 経路 | トリガー | A2E入力 | Expression送信先 |
|------|---------|---------|-----------------|
| **Live API** | AIターン完了時 | PCM 24kHz (ai_audio_buffer) | WebSocket `{"type": "expression"}` |
| **REST API** | TTS合成時 | MP3 base64 | REST レスポンス `{expression: {...}}` |

### 10.6 TTS + A2E 同期フロー [確認済み — gourmet-sp2]

```
ConciergeController.speakResponseInChunks(response)
    ↓ 文分割 (。 or .)
    ↓ 各文を並行 TTS 合成
POST /api/tts/synthesize { text, language_code, voice_name, session_id }
    ↓ バックエンド
    ├── Google Cloud TTS → MP3 base64
    └── audio2exp-service → { names[52], frames[N][52], frame_rate: 30 }
    ↓ レスポンス
ConciergeController.applyExpressionFromTts(expression)
    ↓ フレーム変換: {names, frames[{weights}]} → {name: weight}[]
    ↓ lamController.clearFrameBuffer()
    ↓ lamController.queueExpressionFrames(frames, 30)
    ↓
ttsPlayer.play() → 'play' イベント発火
    ↓ LAMAvatar 内部
getExpressionData() [16ms間隔, ~60fps]
    ↓ frameIndex = ttsPlayer.currentTime × frame_rate
    ↓ frameBuffer[frameIndex] から 52次元係数を読出
    ↓ Gaussian Splatting レンダラーに適用
    ↓
ttsPlayer.ended → フェードアウト (200ms) → アイドル状態
```

### 10.7 GS レンダリング (gs.ts / gvrm.ts) [確認済み — gourmet-sp2]

**gs.ts — Gaussian Splatting ビューアー:**
- PLY アバターメッシュのロード
- 頂点シェーダー: LBS (Linear Blend Skinning) + ボーンマトリクス
- Jaw アニメーション: 口の開閉制御
- インスタンスドレンダリング: パフォーマンス最適化
- Sigmoid 活性化: 不透明度のリアルな合成

**gvrm.ts — GVRM アバターシステム:**
- Three.js シーン管理
- ボーンテクスチャ (64マトリクス × 4×4)
- `updateLipSync(level)` — jawOpen の直接制御
- `setPose(matrices)` — 全スケルトンポーズの適用

### 10.8 リップシンク診断テスト [確認済み — gourmet-sp2]

ブラウザコンソールから実行可能:

```javascript
// コンシェルジュページ (/concierge) で:
__testLipSync()

// 5つの日本語母音 (あいうえお) を順次合成
// 各母音の既知ブレンドシェイプパターンで検証:
//   あ: jawOpen高, mouthSmile低
//   い: jawOpen低, mouthSmile高
//   う: mouthFunnel高, mouthPucker高
//   え: jawOpen中, mouthSmile中
//   お: jawOpen高, mouthFunnel高
```

### 10.9 スマホ軽量化戦略

| 項目 | 方針 | 実装状態 |
|------|------|---------|
| **A2E推論** | サーバー側 (CPU) で実行。52次元係数のみ送信 (~10KB/sec) | 実装済 |
| **レンダリング** | クライアント側 WebGL。LAM Gaussian Splatting SDK | 実装済 |
| **フォールバック** | SDK ロード失敗時 → 静止画表示 | 実装済 |
| **動作基準** | iPhone SE (A13/A15, 3-4GB RAM) で 30fps | **要実機テスト** |

---

## 11. API 仕様

### 11.1 エンドポイント一覧 [確認済み]

#### Live API 系

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/v2/session/start` | セッション開始 (Live API用) |
| POST | `/api/v2/session/end` | セッション終了 |
| WS | `/api/v2/live/{session_id}` | Live API WebSocket 中継 |
| GET | `/api/v2/modes` | 利用可能モード一覧 |
| GET | `/api/v2/health` | ヘルスチェック |

#### REST API 系 (gourmet-support 互換)

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/v2/rest/session/start` | REST セッション開始 |
| POST | `/api/v2/rest/chat` | チャット処理 (Gemini + Google Search) |
| POST | `/api/v2/rest/finalize` | セッション完了 (最終要約生成) |
| POST | `/api/v2/rest/cancel` | 処理中止 |
| POST | `/api/v2/rest/tts/synthesize` | TTS合成 + A2E表情データ |
| POST | `/api/v2/rest/stt/transcribe` | 音声認識 (単発) |
| POST | `/api/v2/rest/stt/stream` | 音声認識 (ストリーミング) |
| GET | `/api/v2/rest/session/{id}` | セッション情報取得 |

### 11.2 セッション開始 API

**POST `/api/v2/session/start`**

```json
// Request
{
    "mode": "gourmet",           // "gourmet" | "concierge" | (将来の新モード)
    "language": "ja",            // "ja" | "en" | "ko" | "zh"
    "dialogue_type": "live",     // "live" | "rest" | "hybrid"
    "user_id": "uuid-string"     // 長期記憶用 (オプション)
}

// Response
{
    "session_id": "sess_xxxxxxxxxxxx",
    "mode": "gourmet",
    "language": "ja",
    "dialogue_type": "live",
    "greeting": "いらっしゃいませ！今日はどんなお食事をお探しですか？",
    "ws_url": "/api/v2/live/sess_xxxxxxxxxxxx"
}
```

### 11.3 REST チャット API

**POST `/api/v2/rest/chat`**

```json
// Request
{
    "session_id": "sess_xxxxxxxxxxxx",
    "message": "新宿で美味しいイタリアン",
    "stage": "conversation",
    "language": "ja",
    "mode": "chat"
}

// Response
{
    "response": "ご希望に合うお店を3件ご紹介します。...",
    "summary": "3軒のお店を提案しました。",
    "shops": [
        {
            "name": "トラットリア XX",
            "area": "新宿",
            "description": "..."
        }
    ],
    "should_confirm": true,
    "is_followup": false
}
```

### 11.4 TTS + A2E API

**POST `/api/v2/rest/tts/synthesize`**

```json
// Request
{
    "text": "こんにちは、お元気ですか？",
    "language_code": "ja-JP",
    "voice_name": "ja-JP-Chirp3-HD-Leda",
    "speaking_rate": 1.0,
    "pitch": 0.0,
    "session_id": "sess_xxxxxxxxxxxx"
}

// Response
{
    "success": true,
    "audio": "<base64 MP3>",
    "expression": {
        "names": ["eyeBlinkLeft", ..., "tongueOut"],  // 52個
        "frames": [[0.0, ...], [0.1, ...], ...],      // N×52
        "frame_rate": 30
    }
}
```

---

## 12. データフロー

### 12.1 Live API 経路 (コンシェルジュ/グルメ — ヒアリング・対話)

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: マイク入力 → Live API                                     │
├──────────────────────────────────────────────────────────────────┤
│  マイクタップ → getUserMedia()                                     │
│      ↓ AudioWorkletProcessor                                     │
│  48kHz/44.1kHz → 16kHz Int16 PCM                                │
│      ↓ base64                                                    │
│  WebSocket send: {"type": "audio", "data": "<base64>"}          │
│      ↓ LiveRelay                                                 │
│  Gemini Live API に PCM 16kHz 転送                                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: Gemini AI 応答 → ブラウザ                                  │
├──────────────────────────────────────────────────────────────────┤
│  Gemini → PCM 24kHz 音声 + transcription                         │
│      ↓ LiveRelay                                                 │
│  WebSocket send: {"type": "audio"} + {"type": "transcription"}  │
│      ↓ ブラウザ                                                   │
│  AudioContext で PCM 24kHz を再生                                  │
│  transcription をチャットUIに表示                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 3: A2E → アバターアニメーション                                │
├──────────────────────────────────────────────────────────────────┤
│  AI音声バッファ (ai_audio_buffer)                                  │
│      ↓ ターン完了時                                                │
│  A2EClient.process_audio() → audio2exp-service                   │
│      ↓ 52次元 ARKit ブレンドシェイプ @30fps                        │
│  WebSocket send: {"type": "expression"}                          │
│      ↓ ブラウザ                                                   │
│  LAMAvatarController.queueExpressionFrames()                     │
│      ↓ 音声再生と同期                                              │
│  WebGL アバター リップシンク + 表情アニメーション                     │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2 REST API 経路 (店舗説明・詳細レビュー)

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: テキスト入力 → LLM 応答                                    │
├──────────────────────────────────────────────────────────────────┤
│  POST /api/v2/rest/chat                                          │
│      ↓ SupportAssistant.process_user_message()                   │
│  Gemini 2.5 Flash + Google Search Grounding                      │
│      ↓ JSON パース (message, shops, action)                       │
│  shops → HotPepper / Google Places でエンリッチ                    │
│      ↓                                                           │
│  Response: {response, shops, summary}                            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: TTS + A2E → アバター応答                                   │
├──────────────────────────────────────────────────────────────────┤
│  文分割 (。 or .) → 並行 TTS 合成                                  │
│  POST /api/v2/rest/tts/synthesize (文ごと)                       │
│      ↓ Google Cloud TTS → MP3                                    │
│      ↓ audio2exp-service → 52次元ブレンドシェイプ                   │
│  Response: {audio: base64, expression: {names, frames}}          │
│      ↓ ブラウザ                                                   │
│  順次再生: 文1 → 文2 → ... + Expression 同期適用                   │
│      ↓                                                           │
│  WebGL アバター リップシンク + 表情アニメーション                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 13. 実装ステータスと残作業

### 13.1 バックエンド (support-base) 実装ステータス

| 機能 | ステータス | ファイル |
|------|-----------|---------|
| FastAPI サーバー | **実装済み** | `server.py` |
| Live API WebSocket 中継 | **実装済み** | `live/relay.py` |
| 累積文字数制限 回避 | **実装済み** | `live/reconnect.py` |
| 発話途切れ検知 (多言語) | **実装済み** | `live/speech_detector.py` |
| 短期記憶 (SessionMemory) | **実装済み** | `memory/session_memory.py` |
| 長期記憶 (Supabase) | **実装済み** | `core/long_term_memory.py` |
| GCS プロンプト読み込み | **実装済み** | `core/support_core.py` |
| REST API ルーター | **実装済み** | `rest/router.py` |
| モードプラグインアーキテクチャ | **実装済み** | `modes/base_mode.py`, `modes/registry.py` |
| グルメモード プラグイン | **実装済み** | `modes/gourmet/plugin.py` |
| A2E クライアント | **実装済み** | `services/a2e_client.py` |
| 言語マスター設定 | **実装済み** | `i18n/language_config.py` |
| Google Search Grounding | **実装済み** | `core/support_core.py` |
| A2E Live API 連携 (Expression WS送信) | **実装済み** | `live/relay.py` |
| セッション管理 | **実装済み** | `session/manager.py` |

### 13.2 フロントエンド (gourmet-sp2) 実装ステータス

| 機能 | ステータス | ファイル |
|------|-----------|---------|
| Astro SSG + PWA | **実装済み** | `astro.config.mjs`, `manifest.webmanifest` |
| CoreController (基底) | **実装済み** | `src/scripts/chat/core-controller.ts` |
| ChatController (グルメモード) | **実装済み** | `src/scripts/chat/chat-controller.ts` |
| ConciergeController (コンシェルジュ) | **実装済み** | `src/scripts/chat/concierge-controller.ts` |
| AudioManager (iOS/Android/PC対応) | **実装済み** | `src/scripts/chat/audio-manager.ts` |
| LAMAvatar (Gaussian Splatting) | **実装済み** | `src/components/LAMAvatar.astro` |
| GS レンダラー (Three.js + LBS) | **実装済み** | `gs.ts`, `gvrm.ts` |
| A2E 統合 (TTS同期リップシンク) | **実装済み** | `concierge-controller.ts` |
| リップシンク診断テスト | **実装済み** | `__testLipSync()` |
| 多言語 UI (ja/en/zh/ko) | **実装済み** | `src/constants/i18n.ts` |
| Socket.IO STT ストリーミング | **実装済み** | `audio-manager.ts` |
| ショップカード + 予約モーダル | **実装済み** | `ShopCardList.astro`, `ReservationModal.astro` |
| PWA インストールプロンプト | **実装済み** | `InstallPrompt.astro` |
| REST API 連携 (/api/chat, /api/tts) | **実装済み** | `core-controller.ts`, `concierge-controller.ts` |

### 13.3 フロントエンド (gourmet-sp2) 残作業

| タスク | 優先度 | 説明 |
|--------|--------|------|
| **Vercel 連携** | **P1** | gourmet-sp2 を Vercel に接続。スマホテストに必須 |
| **LiveAPIClient 実装** | **P1** | WebSocket 接続 (`/api/v2/live/{session_id}`)、PCM 16kHz 送信 / 24kHz 受信 |
| **Live API 音声再生** | **P1** | PCM 24kHz → AudioContext で再生 (現行は MP3 TTS 再生のみ) |
| **DialogueManager 実装** | **P1** | Live API / REST API のモード別切替管理 |
| **Expression WS 受信** | **P2** | `{"type": "expression"}` → LAMAvatar.frameBuffer にキュー |
| **Transcription 表示** | **P2** | `{"type": "transcription"}` → チャット UI にリアルタイム表示 |
| **割り込み (barge-in) UI** | **P2** | `{"type": "interrupted"}` → TTS停止 + アバター即停止 |
| **再接続 UI** | **P3** | `reconnecting` / `reconnected` のインジケータ表示 |
| **API エンドポイント切替** | **P1** | 既存 `/api/*` → 新 `/api/v2/*` (support-base) への接続先変更 |

### 13.4 インフラ・デプロイ残作業

| タスク | 優先度 | 説明 |
|--------|--------|------|
| **gourmet-sp2 Vercel デプロイ** | **P1** | GitHub 連携 + 環境変数 (PUBLIC_API_URL) 設定 |
| support-base Cloud Run デプロイ | **P1** | FastAPI + uvicorn、WebSocket 対応 |
| GCS プロンプトバケット設定 | **P1** | 4言語 × 2モードのプロンプトファイル配置 |
| Supabase user_profiles テーブル | **P1** | スキーマ作成、RLS ポリシー設定 |
| 環境変数設定 | **P1** | GEMINI_API_KEY, PROMPTS_BUCKET_NAME, SUPABASE_URL/KEY 等 |
| iPhone SE 実機テスト | **P2** | LAM WebGL SDK の 30fps 動作検証 (Vercel デプロイ後) |

### 13.5 未確認事項・リスク

| # | 項目 | 影響度 | ステータス |
|---|------|--------|-----------|
| 1 | iPhone SE で Gaussian Splatting が 30fps 出るか | **致命的** | **未検証** |
| 2 | WebSocket 中継 (ブラウザ→サーバー→Gemini) のレイテンシ | **高** | **未検証** |
| 3 | Live API の Function Calling (店舗検索等) | **中** | **未実装** |
| 4 | ハイブリッド方式での Live→REST 切替トリガー | **中** | **未設計** |
| 5 | PWA 対応 | **低** | **実装済み** (gourmet-sp2 で PWA 対応済) |

---

## 付録A: 環境変数一覧

| 変数名 | 必須 | 説明 | デフォルト |
|--------|------|------|-----------|
| `GEMINI_API_KEY` | Yes | Gemini API キー | — |
| `PROMPTS_BUCKET_NAME` | No | GCS プロンプトバケット名 | — (ローカルファイル使用) |
| `A2E_SERVICE_URL` | No | audio2exp-service URL | — (Expression 無効) |
| `GCP_PROJECT_ID` | No | GCP プロジェクトID (TTS/STT用) | — |
| `SUPABASE_URL` | No | Supabase プロジェクト URL | — (長期記憶無効) |
| `SUPABASE_KEY` | No | Supabase Anon Key | — |
| `HOTPEPPER_API_KEY` | No | HotPepper API キー | — |
| `GOOGLE_PLACES_API_KEY` | No | Google Places API キー | — |
| `TRIPADVISOR_API_KEY` | No | TripAdvisor API キー | — |
| `HOST` | No | サーバーホスト | `0.0.0.0` |
| `PORT` | No | サーバーポート | `8080` |
| `CORS_ORIGINS` | No | CORS 許可オリジン (カンマ区切り) | `*` |
| `LEGACY_BACKEND_URL` | No | 既存 gourmet-support URL (プロキシ用) | — |

**フロントエンド (gourmet-sp2) 環境変数:**

| 変数名 | 必須 | 説明 | デフォルト |
|--------|------|------|-----------|
| `PUBLIC_API_URL` | Yes | バックエンド (support-base) の URL | — |

## 付録B: モードプラグイン追加手順

新しいモード（例: カスタマーサポート）を追加する場合:

```
1. GCS にプロンプトを追加
   gs://{BUCKET}/prompts/customer_support_ja.txt
   gs://{BUCKET}/prompts/customer_support_en.txt
   ...

2. モードプラグインを作成
   support_base/modes/customer_support/
   ├── __init__.py
   └── plugin.py    # BaseModePlugin を継承

3. server.py に登録
   from support_base.modes.customer_support.plugin import CustomerSupportPlugin
   mode_registry.register(CustomerSupportPlugin())

4. (オプション) フロントエンドにモード固有UIを追加
   gourmet-sp2/src/pages/customer-support.astro
```

以上の手順で、コア基盤のコードを変更することなく新モードを追加できる。
