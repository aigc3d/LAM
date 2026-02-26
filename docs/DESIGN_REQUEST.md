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

### 2.3 Live API 統合の要件（stt_stream.py から移植すべき機能）

`AI_Meeting_App/stt_stream.py` に実装済みの以下の機能をWeb版に移植する:

| 機能 | stt_stream.py での実装箇所 | 備考 |
|------|---------------------------|------|
| Live API 接続・音声送受信 | `GeminiLiveApp.run()` | PyAudio→WebSocket経由に変更が必要 |
| 自動再接続（累積文字数800文字） | `MAX_AI_CHARS_BEFORE_RECONNECT` | Geminiのコンテキストウィンドウ制限への対処 |
| 発話途切れ検知 | `_is_speech_incomplete()` | 文末の「が」「で」「けど」等を検出 |
| REST API ハイブリッド | `RestAPIHandler` | 長文はREST API + TTS、短文はLive API |
| 会話履歴管理 | `conversation_history` (直近20ターン) | 再接続時のコンテキスト引き継ぎ |
| コンテキスト要約 | `_get_context_summary()` | 再接続時にsystem_instructionに追加 |
| モード別システムプロンプト | `_build_system_instruction()` | standard/silent/interview で切替 |
| スクリプト進行管理 | `_get_next_question_from_script()` | interview モード用 |
| 議事録保存 | `log_transcript()` | Markdown形式 |

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
3. **Live API 統合設計**: stt_stream.py の機能をWeb版に移植する具体設計
4. **既存サービスとの共存戦略**: α版を壊さずに新プラットフォームを構築する方法
5. **データフロー**: 音声入力→STT→LLM→TTS→A2E→アバターレンダリングの全体フロー
6. **API設計**: 新エンドポイントの仕様
7. **iPhone SE対応戦略**: レンダリング方式の選択（LAM WebGL vs Three.js vs ハイブリッド）と判断基準
8. **開発ロードマップ**: フェーズ分け、各フェーズの成果物と検証基準

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
