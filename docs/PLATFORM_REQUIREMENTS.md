# プラットフォーム化 要件定義書

> **文書ID**: REQ-PLATFORM-001
> **作成日**: 2026-02-26
> **ステータス**: Draft
> **根拠文書**: `docs/DESIGN_REQUEST.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/SESSION_HANDOFF.md`

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現状分析（確認済み事実）](#2-現状分析確認済み事実)
3. [ビジネス要件](#3-ビジネス要件)
4. [機能要件](#4-機能要件)
5. [非機能要件](#5-非機能要件)
6. [技術的制約](#6-技術的制約)
7. [未確認事項・リスク](#7-未確認事項リスク)
8. [用語集](#8-用語集)

---

## 1. プロジェクト概要

### 1.1 背景

現在、3つの独立したコンポーネントが存在する:

| コンポーネント | 概要 | 所在 |
|--------------|------|------|
| **gourmet-sp / gourmet-support** | グルメコンシェルジュ専用Webアプリ（Astro + Flask）。REST API経由のテキスト対話 + TTS + 3Dアバター | **別リポジトリ**（LAM_gpro にはパッチファイルのみ） |
| **AI_Meeting_App/stt_stream.py** | デスクトップ専用の会議アシスタント。Gemini Live API によるリアルタイム音声対話。PyAudio前提 | LAM_gpro リポジトリ内 |
| **audio2exp-service** | 音声→表情変換マイクロサービス。Cloud Run デプロイ済み | LAM_gpro リポジトリ内 |

これらが別々に開発されたため、以下の問題がある:

- グルメサポートは「グルメ」にハードコードされており、他の用途に流用できない
- Live API（低遅延音声対話）はデスクトップ版にしか実装されておらず、Webアプリでは使えない
- 記憶機能（短期・長期）が別々の場所に別々の方式で実装されている

### 1.2 目的

**「3Dアバター × AI対話」の共通基盤を作り、モード（用途）を差し替えるだけで異なるAIアプリを素早く立ち上げられるようにする。**

### 1.3 スコープ

| 対象 | スコープ内/外 |
|------|-------------|
| バックエンド共通基盤（セッション管理、LLM対話、TTS/STT、A2E連携、記憶管理） | **スコープ内** |
| Live API のWeb統合 | **スコープ内** |
| モードプラグインアーキテクチャ | **スコープ内** |
| フロントエンド共通基盤（アバターレンダリング、音声入出力） | **スコープ内** |
| 記憶機能の統一（短期・長期） | **スコープ内** |
| gourmet-sp / gourmet-support の実コード改修 | **スコープ外**（別リポジトリ。設計・インターフェース定義のみ） |
| LAMモデルの再学習・ファインチューニング | **スコープ外** |
| iPhone SE での WebGL パフォーマンス検証 | **スコープ外**（別途技術検証タスク） |

---

## 2. 現状分析（確認済み事実）

### 2.1 audio2exp-service（確認済み — コード読解完了）

**ソース**: `services/audio2exp-service/app.py`, `services/audio2exp-service/a2e_engine.py`

- Flask API（port 8081）、CORS有効
- エンドポイント: `POST /api/audio2expression`, `GET /health`
- パイプライン: 音声(base64) → PCM 16kHz → Wav2Vec2(768dim) → A2Eデコーダー → 52次元ARKitブレンドシェイプ @30fps
- 2段階ロード: INFER パイプライン（LAM_Audio2Expression モジュール使用）を優先、未インストール時は Wav2Vec2 エネルギーベースのフォールバック
- デプロイ状態: Cloud Run（us-central1）、ヘルスチェック OK（status: healthy, engine_ready: True, device: cpu, mode: fallback）
- メモリ要件: 4Gi（2Gi では不足で3回失敗した実績あり）

**API仕様（確認済み）**:
```
POST /api/audio2expression
Request:  { audio_base64: string, session_id: string, audio_format: "mp3"|"wav"|"pcm" }
Response: { names: string[52], frames: number[N][52], frame_rate: 30 }

GET /health
Response: { status: "healthy"|"loading", engine_ready: bool, mode: "infer"|"fallback", device: string }
```

### 2.2 AI_Meeting_App/stt_stream.py（確認済み — コード読解完了）

**ソース**: `AI_Meeting_App/stt_stream.py`（約1,063行）

| 機能 | 実装箇所 | 詳細 |
|------|---------|------|
| Live API 接続 | `GeminiLiveApp.run()` L714-796 | `gemini-2.5-flash-native-audio-preview-12-2025` モデル使用 |
| 音声送受信 | `listen_audio()`, `send_audio()`, `receive_audio()`, `play_audio()` | PCM 16kHz入力 / 24kHz出力、asyncio TaskGroup で4タスク並行 |
| 累積文字数制限回避 | `MAX_AI_CHARS_BEFORE_RECONNECT=800`, `LONG_SPEECH_THRESHOLD=500` | FLASH版の累積トークン制限への対処 |
| 自動再接続 | `run()` L714-796 | 累積800文字 / 長文500文字 / 発話途切れ / APIエラー(1011/1008)で再接続 |
| コンテキスト引き継ぎ | `_get_context_summary()` L940-966, `_build_config(with_context=)` L410-438 | 直近10ターンの要約を新セッションの system_instruction に注入 |
| 発話途切れ検知 | `_is_speech_incomplete()` L501-529 | 文末パターン（「が」「で」「を」等）で日本語の途切れを検出 |
| 会話履歴 | `conversation_history` L389, L490-499 | 直近20ターン保持、`[{role, text}]` リスト |
| スライディングウィンドウ | `context_window_compression` L457-461 | target_tokens: 32000 |
| REST APIハイブリッド | `RestAPIHandler` L307-362 | `gemini-2.5-flash` + Google Search。ただし**現在未使用**（ツール定義が空リスト） |
| モード切替 | standard / silent / interview | モード別 system_instruction |
| VAD設定 | `automatic_activity_detection` | start/end sensitivity: HIGH, silence_duration: 500ms |
| 割り込み処理 | `response.server_content.interrupted` | 出力キューをフラッシュし再生停止 |
| TTS | `TTSPlayer` L213-301 | Google Cloud TTS `ja-JP-Wavenet-D`、200文字チャンク分割 |
| 議事録保存 | `log_transcript()` L203-207 | Markdown形式でタイムスタンプ付き記録 |

**重要な制約**: PyAudio 直接入出力のため、ブラウザ非対応。Voicemeeter デバイス連携（Windows環境前提）。

### 2.3 フロントエンドパッチ（確認済み — コード読解完了）

**ソース**: `services/frontend-patches/`

| ファイル | 内容 | 状態 |
|---------|------|------|
| `concierge-controller.ts` | `CoreController` を継承。A2E統合済みコントローラー。`window.lamAvatarController` との連携、TTS プレーヤーリンク、リップシンク診断テスト | 作成済み・未適用 |
| `vrm-expression-manager.ts` | 52次元ARKit → mouthOpenness 変換。フレームバッファリング、30fps 再生タイマー | 作成済み・未適用 |
| `LAMAvatar.astro` | LAMアバター統合 Astro コンポーネント。WebSocket 通信、OpenAvatarChat 連携 | 作成済み・未適用 |
| `FRONTEND_INTEGRATION.md` | 統合ガイド | 作成済み |

**重要**: これらは gourmet-sp に適用するパッチとして作成されたが、**結合テスト未実施**。

### 2.4 gourmet-support バックエンド（推定 — 別リポジトリ、直接確認不可）

以下は `docs/SYSTEM_ARCHITECTURE.md` とフロントエンドパッチの API 呼び出しから推定した構成:

| ファイル | 推定内容 |
|---------|---------|
| `app_customer_support.py` | Flask + Socket.IO、APIエンドポイント提供 |
| `support_core.py` | Gemini LLM 対話ロジック |
| `api_integrations.py` | HotPepper API 等の外部連携 |
| `long_term_memory.py` | 長期記憶（Firestore に永続化） |

推定エンドポイント: `/api/session/start`, `/api/session/end`, `/api/chat`, `/api/tts/synthesize`, `/api/health`

> **注意**: 上記は推定であり、実コードの確認が必要。特に長期記憶のストレージ（Firestore vs Supabase）は文書間で矛盾がある（`SYSTEM_ARCHITECTURE.md` は Firestore、前回の `PLATFORM_DESIGN.md` は Supabase と記載）。

### 2.5 gourmet-sp フロントエンド（推定 — 別リポジトリ、直接確認不可）

パッチファイルの import 文と `SYSTEM_ARCHITECTURE.md` から推定:

- Astro + TypeScript
- クラス階層: `CoreController` → `ConciergeController` / `ChatController`
- `AudioManager` — マイク入力（48kHz→16kHz、Socket.IO streaming）
- GVRM — Gaussian Splatting 3Dアバターレンダラー
- リップシンク: FFTベース（デフォルト）、A2Eブレンドシェイプ（パッチ適用時）

---

## 3. ビジネス要件

### 3.1 最上位ゴール

**論文超えクオリティの3D対話アバターを、バックエンドGPUなしで、iPhone SE単体で軽く動かす。即実用のアルファ版。**

| # | 要件 | 詳細 |
|---|------|------|
| BR-01 | **論文超えの自然さ** | 口元だけでなく、表情・頭の動き・セリフとの連動が自然。低遅延 |
| BR-02 | **スマホ単体完結** | バックエンドGPU一切不要。推論もレンダリングも全てオンデバイス（A2Eを除く） |
| BR-03 | **iPhone SEで軽く動く** | 最も制約の厳しいデバイスが動作基準 |
| BR-04 | **技術スタックに固執しない** | 動くものを即テスト→見極め→次へ。理論より実証 |

### 3.2 直近のゴール（α版）

| 優先度 | ゴール | 完了条件 |
|--------|--------|---------|
| **P1** | グルメコンシェルジュのα版を壊さず動かし続ける | 既存エンドポイント一切変更なし |
| **P2** | Live API をWebプラットフォームに統合する | ブラウザからリアルタイム音声対話ができる |
| **P3** | モード追加が容易な構造にする | 新モード追加時に変更するファイルが最小限 |
| **P4** | 記憶機能を統一仕様にする | 短期記憶・長期記憶を共通サービスとして提供 |

---

## 4. 機能要件

### 4.1 マルチモード対応

| ID | 要件 | 優先度 | 備考 |
|----|------|--------|------|
| FR-MODE-01 | 単一基盤で複数のAIアプリケーション（グルメコンシェルジュ、カスタマーサポート、インタビュー等）を運用できる | P3 | プラグインアーキテクチャ |
| FR-MODE-02 | モードごとにシステムプロンプト、外部API連携、UI部品を差し替え可能 | P3 | |
| FR-MODE-03 | 新モード追加時にハードコード変更を最小限にする | P3 | 設定ファイルとモードプラグインの追加のみ |
| FR-MODE-04 | 既存のグルメコンシェルジュモードが新基盤上で動作する | P1 | 既存エンドポイント温存 |

### 4.2 Live API 統合

| ID | 要件 | 優先度 | 移植元 |
|----|------|--------|--------|
| FR-LIVE-01 | ブラウザからGemini Live APIによるリアルタイム音声対話ができる | P2 | `stt_stream.py` 全体 |
| FR-LIVE-02 | 累積文字数制限の回避（自動再接続） | P2 | `MAX_AI_CHARS_BEFORE_RECONNECT=800`, L624-643 |
| FR-LIVE-03 | 再接続時のコンテキスト引き継ぎ（短期記憶） | P2 | `_get_context_summary()` L940-966, `_build_config()` L410-438 |
| FR-LIVE-04 | 発話途切れ検知と即時再接続 | P2 | `_is_speech_incomplete()` L501-529 |
| FR-LIVE-05 | ユーザー割り込み（barge-in）対応 | P2 | `response.server_content.interrupted` |
| FR-LIVE-06 | スライディングウィンドウによるコンテキスト圧縮 | P2 | `context_window_compression` L457-461 |
| FR-LIVE-07 | VAD（音声区間検出）の設定可能化 | P3 | `automatic_activity_detection` 設定 |
| FR-LIVE-08 | Live API + REST API のハイブリッド対話（短文はLive、長文はREST+TTS） | P3 | `RestAPIHandler` L307-362 |

### 4.3 対話方式

| ID | 要件 | 優先度 |
|----|------|--------|
| FR-DIAL-01 | REST APIベースのテキスト対話（既存方式の温存） | P1 |
| FR-DIAL-02 | Live APIベースのリアルタイム音声対話 | P2 |
| FR-DIAL-03 | モードごとにLive/REST/ハイブリッドを選択可能 | P3 |

**モード別対話方式マトリクス**:

| モード | Live API（低遅延対話） | REST API（長文生成） |
|--------|----------------------|---------------------|
| グルメコンシェルジュ | 好みヒアリング、相槌、確認 | ショップカード説明、詳細レビュー |
| カスタマーサポート | 状況ヒアリング、共感、確認 | FAQ回答、手順説明 |
| インタビュー | 質問、相槌、進行（メイン） | 資料参照の長文説明時のみ |

### 4.4 記憶機能

| ID | 要件 | 優先度 | 現状の実装箇所 |
|----|------|--------|--------------|
| FR-MEM-01 | 短期記憶: セッション内の会話コンテキストを維持 | P2 | `stt_stream.py` の `conversation_history`（直近20ターン） |
| FR-MEM-02 | 短期記憶: Live API再接続時にコンテキストを引き継ぎ | P2 | `_get_context_summary()` |
| FR-MEM-03 | 長期記憶: セッションを超えたユーザー好みの永続化 | P4 | `gourmet-support` の `long_term_memory.py`（Firestore） |
| FR-MEM-04 | 長期記憶: モード非依存のデータスキーマ（モード別拡張可能） | P4 | 現状はグルメ固有スキーマ |
| FR-MEM-05 | 短期記憶を共通 `SessionMemory` サービスとして提供 | P4 | 現状は `stt_stream.py` にベタ書き |
| FR-MEM-06 | 長期記憶を共通 `LongTermMemory` サービスとして提供 | P4 | 現状は `gourmet-support` に埋め込み |

### 4.5 音声処理

| ID | 要件 | 優先度 |
|----|------|--------|
| FR-AUD-01 | TTS: テキスト→音声合成（Google Cloud TTS または Live APIネイティブ音声） | P1 |
| FR-AUD-02 | STT: ブラウザマイク入力の音声認識（Live API transcription または専用STT） | P2 |
| FR-AUD-03 | A2E: 音声→52次元ARKitブレンドシェイプ変換（audio2exp-service 経由） | P1 |
| FR-AUD-04 | A2E結果のクライアントへのストリーミング配信（~10KB/sec） | P2 |

### 4.6 アバターレンダリング

| ID | 要件 | 優先度 |
|----|------|--------|
| FR-AVT-01 | 52次元ARKitブレンドシェイプによる表情アニメーション | P1 |
| FR-AVT-02 | 30fps でのスムーズなアニメーション再生 | P1 |
| FR-AVT-03 | LAM WebGL SDK（Gaussian Splatting）によるレンダリング | P2 |
| FR-AVT-04 | Three.js + GLBメッシュによる軽量レンダリング（フォールバック） | P3 |
| FR-AVT-05 | レンダリング方式の動的切替（デバイス性能に応じて） | P3 |

### 4.7 既存サービス共存

| ID | 要件 | 優先度 |
|----|------|--------|
| FR-COMPAT-01 | 既存の gourmet-support エンドポイントを一切変更しない | P1 |
| FR-COMPAT-02 | 既存と新プラットフォームを並行稼働させる | P1 |
| FR-COMPAT-03 | 段階的に既存モジュールを新基盤に移行する | P3 |

---

## 5. 非機能要件

### 5.1 パフォーマンス

| ID | 要件 | 目標値 | 備考 |
|----|------|--------|------|
| NFR-PERF-01 | Live API 応答遅延 | < 500ms（音声入力→音声出力） | Gemini側の仕様に依存 |
| NFR-PERF-02 | REST API 応答遅延 | < 3秒（テキスト入力→テキスト出力） | |
| NFR-PERF-03 | TTS 合成遅延 | < 1秒/文 | Google Cloud TTS |
| NFR-PERF-04 | A2E 推論遅延 | < 2秒/文 | CPU推論 |
| NFR-PERF-05 | エンドツーエンド遅延（REST経路） | < 6秒（発話→アバター応答開始） | TTS + A2E + レンダリング |
| NFR-PERF-06 | アバターレンダリング FPS | >= 30fps（iPhone SE） | **未検証 — 最重要技術リスク** |

### 5.2 可用性

| ID | 要件 | 目標値 |
|----|------|--------|
| NFR-AVAIL-01 | Live API セッション中断からの自動復帰 | 再接続成功率 > 95% |
| NFR-AVAIL-02 | audio2exp-service のヘルスチェック応答 | 常時200応答 |
| NFR-AVAIL-03 | A2Eフォールバック（INFERパイプライン障害時） | エネルギーベース近似に自動切替 |

### 5.3 拡張性

| ID | 要件 |
|----|------|
| NFR-EXT-01 | 新モード追加が設定ファイル + プラグインモジュール追加のみで完了する |
| NFR-EXT-02 | 長期記憶のデータスキーマがモード別に拡張可能 |
| NFR-EXT-03 | 対話方式（Live/REST/ハイブリッド）がモード単位で選択可能 |

### 5.4 セキュリティ

| ID | 要件 |
|----|------|
| NFR-SEC-01 | APIキー（Gemini等）をサーバーサイドで管理し、クライアントに露出しない |
| NFR-SEC-02 | セッションIDによるアクセス制御 |
| NFR-SEC-03 | 長期記憶データの暗号化（個人情報を含む可能性） |

### 5.5 デバイス対応

| ID | 要件 | 備考 |
|----|------|------|
| NFR-DEV-01 | iPhone SE（A13/A15, 3-4GB RAM）で30fps描画 | **未検証** |
| NFR-DEV-02 | モダンブラウザ対応（Chrome, Safari, Firefox の最新2バージョン） | |
| NFR-DEV-03 | WebGL 2.0 または WebGPU 対応 | LAM WebGL SDK の要件 |

---

## 6. 技術的制約

### 6.1 確定制約

| # | 制約 | 根拠 |
|---|------|------|
| TC-01 | A2E推論はサーバー側（CPU）で実行 | Wav2Vec2（95Mパラメータ）はモバイルでは重すぎる。audio2exp-service として Cloud Run デプロイ済み |
| TC-02 | レンダリングはクライアント側 | ユーザー体験（遅延）とサーバーコスト（GPU不要）の両面 |
| TC-03 | Live API は FLASH版（`gemini-2.5-flash-native-audio-preview`）を使用 | 累積トークン制限あり。回避ロジック必須 |
| TC-04 | gourmet-sp / gourmet-support は別リポジトリ | プラットフォーム側でインターフェースを定義し、既存コードの改修は最小限にする |
| TC-05 | audio2exp-service は 4Gi メモリが必要 | 2Gi では OOM で3回失敗した実績 |
| TC-06 | Live API の WebSocket 接続は**サーバーサイド**で管理 | APIキー保護のため。ブラウザ → サーバー → Gemini の中継が必要 |

### 6.2 技術選定の制約

| # | 制約 | 理由 |
|---|------|------|
| TC-07 | バックエンドは Python（Flask or FastAPI） | 既存の gourmet-support が Flask、stt_stream.py が Python。統合コストを最小化 |
| TC-08 | フロントエンドは Astro + TypeScript | 既存の gourmet-sp の技術スタック |
| TC-09 | LLM は Gemini ファミリー | Live API が Gemini 固有機能 |
| TC-10 | TTS は Google Cloud TTS（REST経路）または Live API ネイティブ音声 | 既存実装との整合性 |

---

## 7. 未確認事項・リスク

### 7.1 未確認事項

| # | 項目 | 影響 | 確認方法 |
|---|------|------|---------|
| UNC-01 | iPhone SE で LAM WebGL SDK（81,424 Gaussians）が 30fps で動作するか | **致命的** — レンダリング方式の根本的判断に影響 | 実機テスト |
| UNC-02 | gourmet-support の長期記憶ストレージが Firestore か Supabase か | 記憶機能統一設計に影響 | gourmet-support リポジトリの実コード確認 |
| UNC-03 | gourmet-support の `long_term_memory.py` のデータスキーマ詳細 | モード非依存化の設計に影響 | gourmet-support リポジトリの実コード確認 |
| UNC-04 | gourmet-sp の `CoreController` の完全なインターフェース | フロントエンド共通基盤の設計に影響 | gourmet-sp リポジトリの実コード確認 |
| UNC-05 | REST APIハイブリッド方式の実効性（stt_stream.py では現在未使用） | Live/REST切替設計に影響 | stt_stream.py でのツール定義復活と動作検証 |
| UNC-06 | Live API を WebSocket 経由でブラウザに中継する際のレイテンシ | UX品質に影響 | プロトタイプでの計測 |

### 7.2 リスク

| # | リスク | 発生確率 | 影響度 | 対策 |
|---|--------|---------|--------|------|
| RISK-01 | iPhone SE で Gaussian Splatting が 30fps 出ない | **高** | **致命的** | Three.js + GLB メッシュへのフォールバック経路を設計段階で用意 |
| RISK-02 | Live API の FLASH版制限が将来変更される | 中 | 中 | 回避ロジックを抽象化し、設定で ON/OFF 可能にする |
| RISK-03 | gourmet-support の実コード構造が推定と異なる | 中 | 高 | プラットフォームのインターフェースを先に定義し、既存コードへの依存を最小化 |
| RISK-04 | WebSocket 中継（ブラウザ→サーバー→Gemini）のレイテンシが許容範囲を超える | 低〜中 | 高 | 早期プロトタイプで計測。不可の場合はクライアント直接接続 + トークン制限方式を検討 |
| RISK-05 | 記憶機能の統一により既存グルメモードのパーソナライゼーション品質が低下 | 低 | 中 | 既存の長期記憶ロジックは温存し、新基盤では並行して新スキーマを構築 |

---

## 8. 用語集

| 用語 | 定義 |
|------|------|
| **A2E (Audio2Expression)** | 音声から52次元ARKitブレンドシェイプ係数を生成する技術・サービス |
| **ARKit ブレンドシェイプ** | Apple ARKit 準拠の52次元顔表情パラメータ。jawOpen, eyeBlinkLeft 等 |
| **Live API** | Google Gemini の Native Audio 対話 API。音声→音声のストリーミング対話 |
| **FLASH版** | `gemini-2.5-flash-native-audio-preview` モデル。累積トークン制限あり |
| **REST API** | 従来のテキストベース LLM API（`gemini-2.5-flash`） |
| **LAM (Large Avatar Model)** | SIGGRAPH 2025 論文。FLAME ベースの Gaussian Splatting アバター生成・アニメーション技術 |
| **GVRM** | Gaussian Splatting ベースのアバターレンダラー |
| **INFER パイプライン** | LAM_Audio2Expression モジュールによる完全な A2E 推論パイプライン |
| **フォールバック** | INFER パイプライン未使用時の、Wav2Vec2 エネルギーベース近似生成 |
| **短期記憶** | セッション内の会話コンテキスト維持。インメモリ |
| **長期記憶** | セッションを超えたユーザー情報の永続化。Firestore/Supabase |
| **モード** | プラットフォーム上の用途別アプリケーション（グルメ、サポート、インタビュー等） |
| **モードプラグイン** | モード固有のロジック・設定をカプセル化したモジュール |
