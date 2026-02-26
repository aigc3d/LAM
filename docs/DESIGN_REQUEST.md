# プラットフォーム設計書 作成依頼

> **作成日**: 2026-02-26
> **目的**: 別のAI（またはエンジニア）に対して、プラットフォーム設計書を一から作り直してもらうための指示書
> **背景**: 前回AIが作成した設計書（`docs/PLATFORM_DESIGN.md`）は、コードの推測補完・未検証の設計判断が混在しており信頼性に問題がある。事実ベースで再設計が必要。

---

## 注意事項（設計担当者へ）

1. **推測で設計するな。** 実際のコードを読んでから設計すること。特に gourmet-support / gourmet-sp は別リポジトリにあり、このリポジトリには含まれていない。パッチファイルのみ存在する。
2. **未確認事項は「未確認」と明記せよ。** わからないことを埋めて書くな。
3. **前回の設計書 `docs/PLATFORM_DESIGN.md` はたたき台としてのみ参照。** 内容をそのまま信頼しないこと。後述の信頼性評価を参照。

---

## 0. なぜプラットフォーム化するのか — 目的と直近のゴール

### 0.1 背景

現在、3つの独立したコンポーネントが存在する:

1. **gourmet-sp / gourmet-support** — グルメコンシェルジュ専用のWebアプリ（Astro + Flask）。REST API経由のテキスト対話 + TTS + 3Dアバター
2. **AI_Meeting_App/stt_stream.py** — デスクトップ専用の会議アシスタント。Gemini Live API によるリアルタイム音声対話。PyAudio前提
3. **audio2exp-service** — 音声→表情変換マイクロサービス。Cloud Runデプロイ済み

これらは**それぞれ別々に開発された**ため、以下の問題がある:
- グルメサポートは「グルメ」にハードコードされており、他の用途（カスタマーサポート、インタビュー等）に流用できない
- Live API（低遅延音声対話）はデスクトップ版にしか実装されておらず、Webアプリでは使えない
- 記憶機能（短期・長期）が別々の場所に別々の方式で実装されている（後述 §0.4）

### 0.2 プラットフォーム化の目的

**「3Dアバター × AI対話」の共通基盤を作り、モード（用途）を差し替えるだけで異なるAIアプリを素早く立ち上げられるようにする。**

具体的には:
- **共通基盤（Platform Core）**: セッション管理、LLM対話（REST + Live API）、TTS/STT、A2E連携、記憶管理、アバターレンダリング
- **モードプラグイン**: グルメコンシェルジュ、カスタマーサポート、インタビュー等。システムプロンプト・外部API・UI部品だけを差し替える

### 0.3 直近のゴール（α版）

| 優先度 | ゴール | 具体的な完了条件 |
|--------|--------|----------------|
| **1** | グルメコンシェルジュのα版を壊さず動かし続ける | 既存エンドポイント一切変更なし |
| **2** | Live API をWebプラットフォームに統合する | ブラウザからリアルタイム音声対話ができる |
| **3** | モード追加が容易な構造にする | 新モード追加時に変更するファイルが最小限 |
| **4** | 記憶機能を統一仕様にする | 短期記憶・長期記憶を共通サービスとして提供 |

### 0.4 記憶機能の現状と統一仕様化の必要性

**LLMの弱点である記憶の問題を補完する2つのロジック**が、別々に開発されたため、別々の場所・方式で存在する。プラットフォーム化でこれらを**統一仕様**にすることが重要な要件。

#### 短期記憶（会話コンテキストの維持）

**課題**: LLMはセッション内でもコンテキストウィンドウを超えると過去の会話を忘れる。特にLive APIはFLASH版（`gemini-2.5-flash-native-audio-preview`）を使用しており、累積トークン制限がある。

**現在の実装** (`AI_Meeting_App/stt_stream.py`):
- `conversation_history` — 直近20ターンのリスト保持（`_add_to_history()`: L490-495）
- `_get_context_summary()` — 再接続時に直近10ターンを要約（L940-966）
- `context_window_compression.sliding_window` — Gemini API側のスライディングウィンドウ設定（target_tokens: 32000）（L457-461）
- 累積800文字で自動再接続時にsystem_instructionに会話要約を注入（`_build_config(with_context=...)`: L410-438）

**問題**: この短期記憶ロジックは `stt_stream.py` のPyAudioデスクトップアプリ内にベタ書きされており、Web版では使えない。

#### 長期記憶（ユーザーへのパーソナライゼーション）

**課題**: ユーザーの好み・過去のやりとりをセッションを超えて記憶し、パーソナライズされた応対を行う。

**現在の実装** (`gourmet-support` リポジトリ):
- `long_term_memory.py` (~200行) — Firestore に長期記憶を永続化
- `/api/session/start` — セッション開始時に長期記憶から挨拶文を生成（「前回はイタリアンを探してましたよね」等）
- `support_core.py` の対話フロー内で長期記憶更新（ユーザーの好み・過去のやりとり）
- フロントエンド (`concierge-controller.ts` L191, L480-489) で長期記憶対応済み挨拶を表示・保持

**問題**: この長期記憶ロジックは `gourmet-support` のグルメコンシェルジュ専用コードに埋め込まれており、他のモードでは使えない。また、データスキーマもグルメ固有（「好きな料理」「エリア」等）。

#### 統一仕様化の要件

| 項目 | 短期記憶 | 長期記憶 |
|------|---------|---------|
| **現在の所在** | `stt_stream.py` (デスクトップ) | `long_term_memory.py` (gourmet-support) |
| **ストレージ** | インメモリ (セッション内) | Firestore/Supabase (永続) |
| **データ構造** | `[{role, text}]` リスト | グルメ固有スキーマ（要確認） |
| **統一後のあるべき姿** | 共通 `SessionMemory` サービスとして、モード非依存で提供 | 共通 `LongTermMemory` サービスとして、モード別スキーマを拡張可能に |
| **プラットフォームでの要件** | Live API再接続時のコンテキスト引き継ぎをWebでも動くようにする | 任意のモードからユーザープロファイルを読み書きできるようにする |

---

## 1. 現状の構成と課題

### 1.1 確定事実（コードで確認済み）

#### サービス構成（3サービス）

| サービス | デプロイ先 | ソースコード所在 |
|---------|-----------|----------------|
| **gourmet-sp** (フロントエンド) | Vercel | **別リポジトリ** (このリポにはパッチファイルのみ) |
| **gourmet-support** (バックエンド) | Cloud Run (us-central1) | **別リポジトリ** (このリポにはなし) |
| **audio2exp-service** | Cloud Run (us-central1) | `services/audio2exp-service/` |

**重要**: gourmet-sp と gourmet-support の実ソースはこのリポジトリにない。設計書作成時にはそれらのリポジトリも必ず参照すること。

#### audio2exp-service（このリポジトリにある、確認済み）

- `services/audio2exp-service/app.py` — Flask API (port 8081)
- `services/audio2exp-service/a2e_engine.py` — 推論エンジン
- エンドポイント: `POST /api/audio2expression`, `GET /health`
- パイプライン: 音声(base64) → Wav2Vec2(768dim) → A2Eデコーダー → 52次元ARKitブレンドシェイプ @30fps
- フォールバック: A2Eデコーダー未ロード時はエネルギーベースの近似生成
- デプロイ状態: ヘルスチェックOK (status: healthy, engine_ready: True, device: cpu, mode: fallback)

#### フロントエンドパッチ（このリポジトリにある、確認済み）

`services/frontend-patches/` にあるファイル:
- `concierge-controller.ts` — A2E統合済みのコントローラー（gourmet-sp にパッチとして適用する前提）
- `vrm-expression-manager.ts` — 52次元ARKit → mouthOpenness変換
- `LAMAvatar.astro` — LAMアバター統合コンポーネント（OpenAvatarChat連携、WebSocket通信）
- `FRONTEND_INTEGRATION.md` — 統合ガイド

#### AI_Meeting_App（このリポジトリにある、確認済み）

- `AI_Meeting_App/stt_stream.py` — デスクトップ専用のスタンドアロンアプリ
- Gemini Live API (`gemini-2.5-flash-native-audio-preview-12-2025`) を使用
- PyAudio直接入出力（ブラウザ非対応）
- Google Cloud TTS (`TTSPlayer` クラス, ja-JP-Wavenet-D)
- 3モード: standard / silent / interview
- 自動再接続: 累積800文字で再接続、会話履歴20ターン保持
- Voicemeeterデバイス連携（Windows環境前提）

#### gourmet-support バックエンドの構成（パッチファイル・ドキュメントからの推定）

以下は `docs/SYSTEM_ARCHITECTURE.md` と `services/frontend-patches/concierge-controller.ts` から推定した情報。**実コードは別リポジトリにあり、直接確認していない。**

- `app_customer_support.py` — Flask + Socket.IO、APIエンドポイント提供
- `support_core.py` — Gemini LLM 対話ロジック
- `api_integrations.py` — HotPepper API等
- `long_term_memory.py` — 長期記憶（Firestore or Supabase）
- エンドポイント: `/api/session/start`, `/api/chat`, `/api/tts/synthesize`, `/api/stt/transcribe` 等
- TTS + A2E 統合: `/api/tts/synthesize` でTTS合成後に audio2exp-service を呼び、音声＋表情データを同梱返却

#### gourmet-sp フロントエンドの構成（パッチファイルからの推定）

- Astro + TypeScript
- クラス階層: `CoreController` → `ConciergeController` / `ChatController`
- `AudioManager` — マイク入力 (48kHz→16kHz, Socket.IO streaming)
- GVRM — Gaussian Splatting 3Dアバターレンダラー
- リップシンク: FFTベース（デフォルト）、A2Eブレンドシェイプ（パッチ適用時）
- `window.lamAvatarController` を介した LAMAvatar 連携

### 1.2 現状の課題

1. **グルメサポート専用の密結合**: フロントエンド・バックエンドがグルメコンシェルジュ専用にハードコードされている
2. **Live API がプラットフォームに統合されていない**: `stt_stream.py` はデスクトップ専用のスタンドアロンアプリ。PyAudio前提でブラウザから使えない
3. **モード追加が困難**: 新モード（カスタマーサポート、インタビュー等）を追加するには、ページ・コントローラー・ルートをハードコードで追加する必要がある
4. **A2Eパッチが未適用**: `services/frontend-patches/` のファイルは作成済みだが、gourmet-sp への適用・結合テストが未実施

### 1.3 前回設計書の信頼性評価

前回の `docs/PLATFORM_DESIGN.md` の内容を、事実/推測で分類:

| セクション | 信頼性 | 理由 |
|-----------|--------|------|
| 2. 現状のシステム構成 | **高** | 実コードのパッチファイル・ドキュメントと整合 |
| 3. プラットフォーム全体設計 (図) | **中** | 構想としては妥当だが、実装可能性の検証なし |
| 4. 共通基盤 vs モード固有の仕分け | **中** | 分類は合理的だが、実コードの依存関係を精査していない |
| 5. Live API 統合設計 | **中〜低** | stt_stream.py の読解は正確だが、Web移植の設計は未検証の構想 |
| 6. バックエンド設計 (ディレクトリ/クラス設計) | **低** | gourmet-support の実コードを読まずに設計している。依存関係の分解が実現可能か不明 |
| 7. フロントエンド設計 | **低** | gourmet-sp の実コードを読まずに設計している |
| 8. モード別仕様 | **中** | 要件定義としては参考になるが、voice設定等は推測 |
| 9. 開発ロードマップ | **低** | 工数・難易度の見積もりなし。実現可能性が未検証 |
| 10. 移行戦略 | **中** | 方針は妥当だが、実装詳細は未検証 |

---

## 2. 新しいプラットフォーム化の要件・要望

### 2.1 オーナーの最上位ゴール

**論文超えクオリティの3D対話アバターを、バックエンドGPUなしで、iPhone SE単体で軽く動かす。即実用のアルファ版。**

| # | 要件 | 詳細 |
|---|------|------|
| 1 | **論文超えの自然さ** | 口元だけでなく、表情・頭の動き・セリフとの連動が自然。低遅延 |
| 2 | **スマホ単体完結** | バックエンドGPU一切不要。推論もレンダリングも全てオンデバイス |
| 3 | **iPhone SEで軽く動く** | 最も制約の厳しいデバイスが動作基準 |
| 4 | **技術スタックに固執しない** | 動くものを即テスト→見極め→次へ。理論より実証 |

### 2.2 プラットフォーム化の要件

1. **マルチモード対応**: 単一基盤で複数のAIアプリケーション（グルメコンシェルジュ、カスタマーサポート、インタビュー等）を運用できること
2. **Live API 統合**: Gemini Live API（ネイティブオーディオ）をWebプラットフォームの標準機能として組み込む。現在 `stt_stream.py` にあるデスクトップ版の機能をWeb版に移植する
3. **既存サービス温存**: α版テスト中のグルメサポートAI（gourmet-sp + gourmet-support）を中断しない。既存エンドポイントは一切変更しない
4. **段階的移行**: 既存と新プラットフォームを並行稼働させ、段階的に移行する
5. **モード追加の容易さ**: 新モード追加時にハードコードの変更を最小限にする。プラグイン的なアーキテクチャ

### 2.3 Live API 導入の理由と背景

#### なぜ Live API を導入するのか

既存の gourmet-support は **REST API ベースのテキスト対話**（ユーザー入力→LLM応答→TTS読み上げ→アバター口パク）であり、以下の体験上の問題がある:

1. **往復レイテンシ**: ユーザーが話す → STTで文字起こし → REST APIでLLM応答 → TTSで合成 → 再生。**5〜10秒のラグ**が発生
2. **割り込み不可**: LLMが回答中にユーザーが割り込めない（従来の「話す→待つ→聞く」の交互方式）
3. **相槌なし**: 「へぇ」「なるほど」等のリアルタイムな応答ができない

**Gemini Live API** は音声→音声のストリーミング対話を提供し、上記を根本的に解決する:
- **超低遅延**: 音声入力→音声出力が数百ms
- **割り込み対応**: ユーザーの発話を検知してAIが自動的に応答を中断
- **ネイティブ音声生成**: TTS不要、AIが直接音声を生成

#### FLASH版の制約と累積文字数制限の回避ロジック

**重要な留意点**: 現在使用しているLive APIモデルは**FLASH版（`gemini-2.5-flash-native-audio-preview-12-2025`）**であり、以下の制約がある:

1. **累積トークン制限**: FLASH版にはセッション内の累積入出力トークンに制限がある。長時間対話するとAPIエラー（1011/1008）が発生しセッションが切断される
2. **音声出力の途切れ**: 累積制限に近づくとAIの発話が途中で切れる現象が発生する

**この制約への回避ロジック** が `stt_stream.py` の `GeminiLiveApp` クラスに実装済み:

```
[回避ロジックの全体像]

AI発話のたびに文字数を累積カウント (ai_char_count)
      │
      ├── 累積800文字超過 → 再接続フラグ ON
      │     (MAX_AI_CHARS_BEFORE_RECONNECT = 800)
      │
      ├── 1回の発話が500文字超 → 次のターン前に再接続
      │     (LONG_SPEECH_THRESHOLD = 500)
      │
      ├── 発話が途中で切れた → 即時再接続
      │     (_is_speech_incomplete(): 「が」「で」「を」等で終わる)
      │
      └── API側エラー (1011/1008) → 3秒後に自動再接続

再接続時の処理:
  1. conversation_history から直近10ターンを取得
  2. _get_context_summary() で要約を生成
  3. 新セッションの system_instruction に要約を注入
  4. 「続きをお願いします」を送信して再開
  5. ai_char_count をリセット
```

**コード上の対応箇所** (`AI_Meeting_App/stt_stream.py`):
- L372: `MAX_AI_CHARS_BEFORE_RECONNECT = 800` — 累積文字数閾値
- L373: `LONG_SPEECH_THRESHOLD = 500` — 長文発話閾値
- L392-394: `ai_char_count`, `needs_reconnect`, `session_count` — 再接続管理変数
- L410-468: `_build_config(with_context=)` — 再接続時のコンテキスト注入 + `context_window_compression.sliding_window`
- L501-529: `_is_speech_incomplete()` — 日本語の文末パターンで発話途切れ検知
- L624-643: 累積カウント＋再接続判定ロジック
- L714-796: `run()` — メインの再接続ループ
- L940-966: `_get_context_summary()` — 会話履歴要約生成

**プラットフォーム化での要件**: この回避ロジック全体をWeb版でも動作するように移植する。将来FLASH版の制限が緩和されても、安全策として保持する設計にすること。

#### Live API + REST API ハイブリッド方式

`stt_stream.py` は **Live API と REST API を使い分けるハイブリッド方式** を採用している:

- **Live API**: 短い応答（相槌、確認、質問）→ 低遅延、ネイティブ音声
- **REST API + TTS**: 長い応答（要約、検索結果、資料説明）→ 正確な長文生成 + Google Cloud TTS (Wavenet)

この切り替えは `RestAPIHandler` クラス（L307-362）と Function Calling で実現。REST API側は `gemini-2.5-flash`（非Live版、通常のテキストAPI）を使用。

### 2.3.1 Live API 統合の要件（stt_stream.py から移植すべき機能）

`AI_Meeting_App/stt_stream.py` に実装済みの以下の機能をWeb版に移植する:

| 機能 | stt_stream.py での実装箇所 | 備考 |
|------|---------------------------|------|
| Live API 接続・音声送受信 | `GeminiLiveApp.run()` (L714-796) | PyAudio→WebSocket経由に変更が必要 |
| **累積文字数制限の回避** | `MAX_AI_CHARS_BEFORE_RECONNECT` (L372), 再接続ループ (L714-796) | **FLASH版の最重要制約。必ず移植** |
| 自動再接続（累積800文字） | `ai_char_count` (L624-643) | Geminiのコンテキストウィンドウ制限への対処 |
| **コンテキスト引き継ぎ（短期記憶）** | `_get_context_summary()` (L940-966), `_build_config(with_context=)` (L410-438) | **再接続時に会話の文脈を失わないためのロジック** |
| 発話途切れ検知 | `_is_speech_incomplete()` (L501-529) | 文末の「が」「で」「けど」等を検出 |
| REST API ハイブリッド | `RestAPIHandler` (L307-362) | 長文はREST API + TTS、短文はLive API |
| 会話履歴管理 | `conversation_history` (L389, L490-499) | 直近20ターン保持、再接続時のコンテキスト源 |
| **スライディングウィンドウ圧縮** | `context_window_compression` (L457-461) | Gemini API側のコンテキスト圧縮設定 |
| モード別システムプロンプト | `_build_system_instruction()` (L471-488) | standard/silent/interview で切替 |
| スクリプト進行管理 | `_get_next_question_from_script()` (L531-577) | interview モード用 |
| 議事録保存 | `log_transcript()` (L203-207) | Markdown形式 |

### 2.4 対話方式の要件

| モード | Live API（低遅延対話） | REST API（長文生成） |
|--------|----------------------|---------------------|
| グルメコンシェルジュ | 好みヒアリング、相槌、確認 | ショップカード説明、詳細レビュー |
| カスタマーサポート | 状況ヒアリング、共感、確認 | FAQ回答、手順説明 |
| インタビュー | 質問、相槌、進行（メイン） | 資料参照の長文説明時のみ |

### 2.5 技術的制約

- **A2E推論はサーバー側（CPUで動く）**: Wav2Vec2 (95Mパラメータ) はサーバーで推論。結果の52次元係数(~10KB/sec)をクライアントに送る
- **レンダリングはクライアント側**: LAM WebGL SDK (Gaussian Splatting) または Three.js + GLBメッシュ
- **iPhone SEが動作基準**: A13/A15チップ、3-4GB RAM。81,424 Gaussianが30FPSで回るかは**未検証**（最重要の技術リスク）
- **gourmet-sp / gourmet-support は別リポジトリ**: プラットフォーム化の際、既存コードの改修範囲を正確に把握するには両リポジトリの精査が必要

---

## 3. 参考にすべきリポジトリ・リソース

### 3.1 このリポジトリ内の参考ファイル

| ファイル | 内容 | 信頼性 |
|---------|------|--------|
| `AI_Meeting_App/stt_stream.py` | Live API 実装の実コード（デスクトップ版） | **高** — 実動作するコード |
| `services/audio2exp-service/` | A2Eマイクロサービス一式 | **高** — デプロイ済み・ヘルスチェックOK |
| `services/frontend-patches/` | フロントエンドパッチ（A2E統合） | **高** — 実コード。ただし未適用・未テスト |
| `docs/SYSTEM_ARCHITECTURE.md` | 現状システムの全体設計書 | **中** — 構成は正確だが、別リポジトリの内容はドキュメントベース |
| `docs/SESSION_HANDOFF.md` | 引き継ぎドキュメント | **中** — 経緯と判断の記録として有用 |
| `docs/PLATFORM_DESIGN.md` | 前回のプラットフォーム設計書 | **低〜中** — **たたき台としてのみ参照**。推測部分あり |
| `tests/a2e_japanese/` | A2E日本語テストスイート | **高** — 実コード。ただし未実行 |

### 3.2 外部リポジトリ（設計時に必ず参照すべき）

| リポジトリ | URL | 参照すべき内容 |
|-----------|-----|---------------|
| **LAM公式** | https://github.com/aigc3d/LAM | アバター生成パイプライン、FLAME モデル、論文の実装 |
| **LAM_Audio2Expression** | https://github.com/aigc3d/LAM_Audio2Expression | A2Eモデルのアーキテクチャ、推論コード |
| **LAM_WebRender** | https://github.com/aigc3d/LAM_WebRender | WebGL SDK の API、npmパッケージ `gaussian-splat-renderer-for-lam` |
| **OpenAvatarChat** | https://github.com/HumanAIGC-Engineering/OpenAvatarChat | LLM + ASR + TTS + Avatar 対話SDK。統合の参考アーキテクチャ |
| **gourmet-sp** | (オーナーに確認) | フロントエンド実コード。Astro + TypeScript |
| **gourmet-support** | (オーナーに確認) | バックエンド実コード。Flask + Socket.IO |

### 3.3 論文・技術資料

| 資料 | URL | 参照すべき内容 |
|------|-----|---------------|
| LAM論文 | https://arxiv.org/abs/2502.17796 | SIGGRAPH 2025。アバター生成・アニメーション・レンダリングの技術詳細 |
| PanoLAM論文 | https://arxiv.org/abs/2509.07552 | LAMの拡張。coarse-to-fine、合成データ訓練 |
| LAMプロジェクトページ | https://aigc3d.github.io/projects/LAM/ | デモ、ベンチマーク (iPhone 16で35FPS等) |
| ModelScope Space | https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model | アバターZIP生成（実際に生成可能） |

### 3.4 参考OSS（アーキテクチャの参考）

| プロジェクト | URL | 参考になる点 |
|------------|-----|-------------|
| **TalkingHead** | https://github.com/met4citizen/TalkingHead | ブラウザで動く対話アバター。Three.js + ブレンドシェイプ。iPhone SEでも動く軽量アプローチ |
| **NVIDIA Audio2Face-3D** | https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark | NVIDIA の A2E モデル。品質の参考 |

---

## 4. 設計書に含めるべき内容

### 必須セクション

1. **現状分析**: 各リポジトリの実コードを読んだ上での正確な現状把握
2. **アーキテクチャ設計**: マルチモード対応のバックエンド・フロントエンド設計
3. **Live API 統合設計**: stt_stream.py の機能をWeb版に移植する具体設計。特に**FLASH版の累積文字数制限の回避ロジック**（§2.3参照）の移植方法を明示すること
4. **記憶機能の統一設計**: 短期記憶（セッション内コンテキスト）と長期記憶（ユーザーパーソナライゼーション）を**モード非依存の共通サービス**として設計（§0.4参照）。現在は別々の場所に別々の方式で実装されている問題を解決する
5. **既存サービスとの共存戦略**: α版を壊さずに新プラットフォームを構築する方法
6. **データフロー**: 音声入力→STT→LLM→TTS→A2E→アバターレンダリングの全体フロー（Live API経路とREST API経路の両方を明示）
7. **API設計**: 新エンドポイントの仕様
8. **iPhone SE対応戦略**: レンダリング方式の選択（LAM WebGL vs Three.js vs ハイブリッド）と判断基準
9. **開発ロードマップ**: フェーズ分け、各フェーズの成果物と検証基準

### 各セクションで守るべきルール

- **「確認済み」と「未確認・推定」を必ず区別** して記載すること
- 実コードを読んでいないモジュールの内部設計は「要確認」と書くこと
- 設計判断には**根拠（なぜその選択か）**を必ず付けること
- 工数・スケジュールの見積もりは、根拠がなければ「見積もり不可」と書くこと

---

## 5. 前回の設計書で参考にしてよい部分

以下は前回の `docs/PLATFORM_DESIGN.md` で、方向性として妥当と判断できる部分:

- **マルチモード・プラグインアーキテクチャの基本方針** (セクション3, 4): モード固有ロジックを分離する方針自体は妥当
- **Live API の Live/REST ハイブリッド方式** (セクション5.4): stt_stream.py の実装に基づいており合理的
- **既存エンドポイント温存方針** (セクション10): α版を壊さない方針は正しい
- **stt_stream.py からの移植対象一覧** (セクション2.3の表): 実コードの読解に基づいており正確

**ただし、これらも実コードとの突合なしに信頼しないこと。**
