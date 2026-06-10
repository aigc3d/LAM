# セッション引き継ぎドキュメント

> **作成日**: 2026-02-22
> **対象セッション**: claude/test-a2e-japanese-audio-j9VBT
> **作成経緯**: 20+セッションでの作業蓄積を次セッションに引き継ぐため

---

## 0. オーナーの真のゴール（最重要 — 必ず最初に読め）

**論文超えクオリティの3D対話アバターを、バックエンドGPUなしで、iPhone SE単体で軽く動かす。即実用のアルファ版。**

| # | 要件 | 詳細 |
|---|------|------|
| 1 | **論文超えの自然さ** | 口元だけでなく、表情・頭の動き・セリフとの連動が自然。低遅延 |
| 2 | **スマホ単体完結** | バックエンドGPU一切不要。推論もレンダリングも全てオンデバイス |
| 3 | **iPhone SEで軽く動く** | 最も制約の厳しいデバイスが動作基準 |
| 4 | **技術スタックに固執しない** | 動くものを即テスト→見極め→次へ。理論より実証 |

### 過去セッションの反省（次のAIへの警告）

- **論文を読め。上辺の字面を舐めて古い知識で推論するな。** LAMの論文(arXiv:2502.17796, SIGGRAPH 2025)とWebGL SDKは2025年5月以降の最新技術。Claudeの学習データにない内容が多い。
- **「検証」や「調査」をゴールにするな。** オーナーのゴールは動くプロダクト。検証はゴールへの通過点に過ぎない。
- **冗長な説明をするな。** オーナーは技術に精通している。わかりきったことの長い説明は不要。
- **推測で回答するな。** 知らないなら「知らない、今から調べる」と言え。

---

## 1. LAM とは何か（公式情報ベース）

**LAM (Large Avatar Model)** — SIGGRAPH 2025, Alibaba Tongyi Lab

> "Build 3D Interactive Chatting Avatar with One Image in Seconds!"

### 1.1 公式エコシステム

| コンポーネント | 説明 | リポジトリ |
|--------------|------|-----------|
| **LAM本体** | 写真1枚 → 81,424個の3D Gaussian Head Avatar (1.4秒) | [aigc3d/LAM](https://github.com/aigc3d/LAM) |
| **LAM-A2E** | 音声 → 52次元ARKitブレンドシェイプ (リアルタイム) | [aigc3d/LAM_Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) |
| **LAM_WebRender** | WebGL 2.0 Gaussian Splatting レンダラー (npmパッケージ) | [aigc3d/LAM_WebRender](https://github.com/aigc3d/LAM_WebRender) |
| **OpenAvatarChat** | LLM + ASR + TTS + Avatar 対話SDK | [HumanAIGC-Engineering/OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat) |
| **PanoLAM** | LAMの拡張 (coarse-to-fine, synthetic training data) | arXiv:2509.07552 |

### 1.2 論文の核心技術

**アバター生成 (サーバー側1回のみ)**:
- 入力: 顔写真1枚
- FlameTracking → DINOv2マルチスケール特徴 → Transformer → canonical Gaussian属性生成
- FLAME canonical点(5,023頂点 → 2回サブディバイド → 81,424 Gaussian)をクエリとして使用
- 出力: position, opacity, rotation, scale, SH色係数

**アニメーション (クライアント側、毎フレーム)**:
- **ニューラルネットワーク不要** — 純粋な行列演算
- `T_G(θ,φ) = G_bar + B_P(θ;P) + B_E(φ;E)`
- `Animated_G = S(T_G, J_bar, θ, W)` (標準Linear Blend Skinning)
- 52次元ARKitブレンドシェイプ係数で表情駆動
- FLAME準拠のpose blendshapes + expression blendshapes + LBS

**WebGLレンダリング (クライアント側)**:
- **Pass 1**: Transform Feedback — ブレンドシェイプ係数+LBSウェイトをGPUテクスチャに格納、頂点シェーダーで全Gaussianを変形
- **Pass 2**: Gaussian Splatting — 変形済みGaussianをスクリーンに投影、α合成
- npmパッケージ `gaussian-splat-renderer-for-lam` (クローズドソース)

**公式ベンチマーク**:

| デバイス | FPS |
|---------|-----|
| A100 (サーバー) | 280.96 |
| MacBook M1 Pro | 120 |
| iPhone 16 | 35 |
| Xiaomi 14 | 26 |

### 1.3 重要な認識ギャップ

過去セッションで誤認していた点:
- ❌ 「LAMはサーバーGPU前提」→ ⭕ **アバター生成だけがGPU。アニメーション+レンダリングはWebGL SDKでスマホ完結**
- ❌ 「Gaussian SplattingはiPhoneで動かない」→ ⭕ **iPhone 16で35FPS実証済み** (iPhone SEは未検証)
- ❌ 「A2EはWav2Vec2(95M)がサーバー前提」→ ⭕ A2E推論はサーバー側だが、**結果の52次元係数(~10KB/sec)をクライアントに送るだけ**。レンダリング自体はオンデバイス

**未解決の技術的問題**: iPhone SE (A13/A15, 3-4GB RAM) で81,424 Gaussianのソートと描画が30FPSで回るか。iPhone 16 (A18)で35FPSなので、SE世代ではさらに厳しい可能性がある。

---

## 2. リポジトリ構成

### 2.1 ブランチ

| ブランチ | 説明 |
|---------|------|
| `master` | LAM公式コード + 初期カスタマイズ |
| `claude/test-a2e-japanese-audio-j9VBT` | **現在のメインブランチ** — A2Eサービス、フロントエンドパッチ、テストスイート |
| `claude/gradio-concierge-ui-4gev2` | Modal/HF Spacesデプロイ (Gradio UI) |
| `claude/test-concierge-modal-rewGs` | Modal GPU上でのアバター生成テスト |

### 2.2 ディレクトリ構成（カスタム部分のみ）

```
LAM_gpro/
├── services/
│   ├── audio2exp-service/          # A2Eマイクロサービス (Flask)
│   │   ├── app.py                  # APIサーバー (port 8081)
│   │   ├── a2e_engine.py           # 推論エンジン (Wav2Vec2 + A2Eデコーダー)
│   │   ├── Dockerfile
│   │   ├── LAM_Audio2Expression/   # 公式A2Eモジュール (git clone)
│   │   └── models/                 # モデルファイル (gitignore)
│   ├── frontend-patches/           # gourmet-sp フロントエンドパッチ
│   │   ├── concierge-controller.ts # A2E統合済みコントローラー
│   │   ├── vrm-expression-manager.ts # 52dim→ボーンマッピング
│   │   └── FRONTEND_INTEGRATION.md
│   └── DEPLOYMENT_GUIDE.md
├── tests/
│   └── a2e_japanese/               # 日本語A2Eテストスイート
│       ├── generate_test_audio.py
│       ├── test_a2e_cpu.py
│       ├── analyze_blendshapes.py
│       ├── patch_*.py              # OpenAvatarChat バグ修正パッチ群
│       ├── chat_with_lam_jp.yaml   # 日本語設定
│       └── TEST_PROCEDURE.md
├── docs/
│   ├── SYSTEM_ARCHITECTURE.md      # 全体設計書 (詳細)
│   └── SESSION_HANDOFF.md          # ← このファイル
└── (LAM公式コード一式)
```

---

## 3. 現在のシステム構成（クラウド版 — 動作する版）

```
┌──────────────────┐  REST   ┌────────────────────┐  REST   ┌──────────────────┐
│ gourmet-sp       │◄──────►│ gourmet-support     │◄──────►│ audio2exp-service│
│ (Astro + TS)     │        │ (Flask + SocketIO)  │        │ (Flask)          │
│ Vercel           │        │ Cloud Run           │        │ Cloud Run        │
│                  │        │                     │        │ 2vCPU, 2GB RAM   │
│ ・3D avatar      │        │ ・Gemini 2.0 Flash  │        │                  │
│ ・FFT lipsync    │        │ ・Google Cloud TTS  │        │ Wav2Vec2 (360MB) │
│ ・A2E lipsync    │        │ ・Google Cloud STT  │        │ + A2E Dec (50MB) │
│   (パッチ適用時) │        │ ・HotPepper API    │        │ → 52dim @30fps   │
│                  │        │ ・Firestore         │        │                  │
└──────────────────┘        └────────────────────┘        └──────────────────┘
```

### 3.1 外部サービス依存

| サービス | 用途 | 代替不可 |
|---------|------|---------|
| Google Cloud TTS | 音声合成 (ja-JP) | TTSは必須、ベンダーは変更可 |
| Google Cloud STT (Chirp2) | 音声認識 | STTは必須、ベンダーは変更可 |
| Gemini 2.0 Flash | LLM対話 | LLMは必須、モデルは変更可 |
| HotPepper API | グルメ検索 | ドメイン固有 |
| Firestore | 長期記憶 | 任意のKVSで代替可 |

### 3.2 gourmet-sp / gourmet-support は別リポジトリ

**重要**: gourmet-sp (フロントエンド) と gourmet-support (バックエンド) のソースコードはこのリポジトリにはない。`services/frontend-patches/` にあるのはパッチファイルのみ。本体は別のGitリポジトリ。

---

## 4. 完了済みの作業

### 4.1 audio2exp-service (完成・Cloud Runデプロイ可能)

- Flask REST API (`/api/audio2expression`, `/health`)
- Wav2Vec2 + LAM A2Eデコーダーの推論パイプライン
- INFER パイプライン (公式LAM_Audio2Expression使用) 優先、エネルギーフォールバック
- Docker化、Cloud Runデプロイ設定
- 1秒チャンクのストリーミング推論、コンテキスト引き継ぎ

### 4.2 フロントエンドパッチ (完成・未適用)

- `concierge-controller.ts`: TTS応答に同梱されたA2Eデータを使ったリップシンク
- `vrm-expression-manager.ts`: 52次元ARKit → 1次元mouthOpenness変換
- 2つの統合方式: ExpressionManager方式 (GVRM直接) / LAMAvatar方式 (外部コントローラー)
- FFTフォールバック機能

### 4.3 日本語テストスイート (完成・未実行)

- EdgeTTSでの日本語テスト音声生成 (母音、会話、長文、英語/中国語比較)
- A2E CPU推論テスト
- ブレンドシェイプ分析・可視化
- OpenAvatarChatバグ修正パッチ群 (ASR言語、VAD dtype、LLM Gemini対応)
- 日本語OpenAvatarChat設定ファイル

### 4.4 Modal/HF Spacesデプロイ (別ブランチ、多数のバグ修正)

- `claude/gradio-concierge-ui-4gev2`: Gradio UI + GPU推論
- bird monsterバグ（vertex_order.json上書き問題）の修正
- nvdiffrast JITプリコンパイル
- xformersバージョン整合

### 4.5 バグ修正履歴 (主要なもの)

| コミット | 問題 | 修正 |
|---------|------|------|
| `a58395b` | ASR 2回目推論が24倍遅延 → システムフリーズ | パフォーマンスパッチ |
| `2e16f78` | テキスト入力時にTTS再生されない | concierge-controller修正 |
| `4332c8f` | autoplay deadlock → STT停止 | play-and-waitパターン修正 |
| `e1b8d30` | Flask dotenv自動読み込みでエンコーディングエラー | 自動ロード無効化 |
| `8f99c70` | INFER パイプライン起動エラー | DDP環境変数設定 |

---

## 5. 未完了・未検証の作業

### 5.1 最重要（ゴール直結）

| 項目 | 状態 | 詳細 |
|------|------|------|
| **iPhone SEでのWebGLレンダリング検証** | 未着手 | 81,424 Gaussianが30FPSで回るか。`gaussian-splat-renderer-for-lam` npmパッケージで検証 |
| **A2Eのオンデバイス化** | 未着手 | 現在はサーバー側Wav2Vec2(95M)。MFCC + 軽量モデル or ONNX量子化 |
| **表情・頭の動きの自然さ向上** | 未着手 | 現在A2Eは口元のみ。頭の動き、瞬き、眉の動きはプロシージャル生成が必要 |
| **エンドツーエンド統合テスト** | 未実行 | gourmet-sp + gourmet-support + audio2exp-service の結合テスト |

### 5.2 テスト未実行

| テスト | 理由 |
|--------|------|
| 日本語A2Eテストスイート | ローカルWindows環境(C:\Users\hamad\OpenAvatarChat)で実行する前提。Claude Codeからは実行不可 |
| OpenAvatarChat統合テスト | 同上 |
| Cloud Runデプロイ | GCPプロジェクトへのアクセスが必要 |

### 5.3 アーキテクチャ未決定

オーナーのゴール「iPhone SE単体、バックエンドGPU不要」に対して、以下のアプローチが候補:

**A. LAM WebGL SDK + サーバーA2E**
- 現在のアーキテクチャの延長
- レンダリングはWebGL SDK (クライアント)、A2E推論はサーバー
- A2Eサーバーは**CPUで動く** (GPU不要) — 2vCPU Cloud Runで2秒/文
- 課題: iPhone SEでGaussian Splattingが30FPS出るか

**B. Three.js + GLBメッシュ + 軽量オーディオ分析**
- Gaussian Splattingを捨てて、通常のメッシュ(20-50kポリゴン) + 52 ARKitブレンドシェイプ
- MFCC + 軽量CNN (1-5Mパラメータ、CoreML/ONNX) でオンデバイスA2E
- Three.jsで60FPS確実
- 参考: [TalkingHead](https://github.com/met4citizen/TalkingHead) (ブラウザで動くOSS)
- 課題: LAMの超リアルなGaussian品質を失う

**C. ネイティブiOSアプリ (SceneKit/RealityKit)**
- GLBメッシュ + CoreMLで完全オンデバイス
- A15 Neural Engine: 15.8 TOPS → 小型モデルなら余裕
- 課題: Web版が不要になる、開発コスト

**D. ハイブリッド: LAM WebGL + TTS事前生成A2E**
- アバター生成: サーバー (1回のみ)
- A2E推論: TTS合成時にサーバーで事前計算、結果(~10KB/sec)をクライアントに送信
- レンダリング: LAM WebGL SDK (クライアント)
- iPhone SEで動くかがボトルネック

---

## 6. 重要なファイルパス

### 6.1 このリポジトリ

| ファイル | 説明 |
|---------|------|
| `docs/SYSTEM_ARCHITECTURE.md` | 全体設計書（最も詳細） |
| `services/audio2exp-service/a2e_engine.py` | A2E推論エンジン |
| `services/audio2exp-service/app.py` | A2E Flask API |
| `services/frontend-patches/concierge-controller.ts` | A2E統合フロントエンド |
| `services/frontend-patches/vrm-expression-manager.ts` | ブレンドシェイプ変換 |
| `services/DEPLOYMENT_GUIDE.md` | デプロイ手順 |
| `tests/a2e_japanese/TEST_PROCEDURE.md` | 日本語テスト手順 |
| `tests/a2e_japanese/test_a2e_cpu.py` | A2Eテスト本体 |
| `tests/a2e_japanese/analyze_blendshapes.py` | 出力分析 |
| `lam/models/rendering/flame_model/` | FLAMEモデル実装 |
| `lam/models/rendering/gs_renderer.py` | Gaussian Splattingレンダラー (Python/CUDA) |
| `tools/generateARKITGLBWithBlender.py` | ZIP生成パイプライン |

### 6.2 外部リポジトリ (参照のみ)

| リポジトリ | URL |
|-----------|-----|
| LAM公式 | https://github.com/aigc3d/LAM |
| LAM_Audio2Expression | https://github.com/aigc3d/LAM_Audio2Expression |
| LAM_WebRender | https://github.com/aigc3d/LAM_WebRender |
| OpenAvatarChat | https://github.com/HumanAIGC-Engineering/OpenAvatarChat |
| TalkingHead (参考OSS) | https://github.com/met4citizen/TalkingHead |

### 6.3 外部リソース

| リソース | URL |
|---------|-----|
| LAM論文 | https://arxiv.org/abs/2502.17796 |
| PanoLAM論文 | https://arxiv.org/abs/2509.07552 |
| LAMプロジェクトページ | https://aigc3d.github.io/projects/LAM/ |
| ModelScope Space (ZIP生成可) | https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model |
| npm WebGLレンダラー | gaussian-splat-renderer-for-lam (クローズドソース) |
| NVIDIA Audio2Face-3D | https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark |

---

## 7. WebGLレンダリングの技術詳細

### 7.1 LAM_WebRender SDK の使い方

```typescript
import { GaussianAvatar } from './gaussianAvatar';

// アバターZIP (skin.glb + offset.ply + animation.glb) を指定
const avatar = new GaussianAvatar(containerDiv, './asset/arkit/avatar.zip');
avatar.start();
```

SDK API:
```typescript
GaussianSplatRenderer.getInstance(container, assetPath, {
  getChatState: () => "Idle" | "Listening" | "Thinking" | "Responding",
  getExpressionData: () => ({ jawOpen: 0.5, mouthFunnel: 0.2, ... }),  // 毎フレーム呼ばれる
  backgroundColor: "0xff0000",
  alpha: 0.2
});
```

### 7.2 A2E → レンダラーのデータフロー

```
A2Eサーバー応答:
{
  names: ["browDownLeft", ..., "tongueOut"],  // 52個
  frames: [[0.0, 0.1, ...], ...],            // 各フレーム52次元
  frame_rate: 30
}

↓ フロントエンドで変換

getExpressionData() が毎フレーム返す:
{
  "jawOpen": 0.45,
  "mouthFunnel": 0.12,
  "mouthPucker": 0.08,
  "eyeBlinkLeft": 0.0,
  ...
}

↓ WebGLレンダラー内部

GPUテクスチャにパック → 頂点シェーダーでLBS計算 → Transform Feedback → Gaussian Splatting描画
```

---

## 8. 次のセッションでやるべきこと

### 最優先: iPhone SEでの実機検証

1. `gaussian-splat-renderer-for-lam` をnpm installしてミニマルHTML作成
2. ModelScope SpaceでアバターZIP生成
3. iPhone SE実機 (Safari) でFPS計測
4. → 30FPS出るなら Approach A (LAM WebGL SDK)
5. → 出ないなら Approach B (Three.js + GLBメッシュ) に切り替え

### 並行: 日本語A2Eテスト実行

オーナーのローカル環境 (`C:\Users\hamad\OpenAvatarChat`) で:
```powershell
conda activate oac
python tests/a2e_japanese/run_all_tests.py
```

### その後: 技術スタック決定 → アルファ版実装

ゴールは「動くもの」。調査や検証で止まるな。

---

## 9. コミット履歴サマリー (113コミット)

| フェーズ | コミット範囲 | 内容 |
|---------|-------------|------|
| LAM公式 | `5c204d4`〜`f8187a7` | 公式リリース、README更新、PanoLAMレポート |
| Modal/GPU格闘 | `f7cc25f`〜`006213f` | Modal L4/A10G GPU、bird monsterバグ、VHAP timeout、ZIP生成 |
| OpenAvatarChat日本語化 | `3003c1b`〜`a58395b` | パッチ群、テストスイート、ASR性能修正 |
| A2Eサービス構築 | `0875af7`〜`8f99c70` | マイクロサービス、INFER パイプライン、Docker |
| フロントエンド統合 | `cde7c54`〜`2e16f78` | A2Eリップシンク統合、TTS修正、データ形式修正 |
