# LAM_gpro システム全体設計書

> **最終更新**: 2026-02-21
> **対象**: gourmet-support バックエンド / gourmet-sp フロントエンド / audio2exp-service / LAM公式ツール

---

## 目次

1. [全体アーキテクチャ](#1-全体アーキテクチャ)
2. [バックエンド (gourmet-support)](#2-バックエンド-gourmet-support)
3. [フロントエンド (gourmet-sp)](#3-フロントエンド-gourmet-sp)
4. [Audio2Expression サービス](#4-audio2expression-サービス)
5. [A2E フロントエンド統合パッチ](#5-a2e-フロントエンド統合パッチ)
6. [公式HF SpacesでカスタムZIPを生成する手順](#6-公式hf-spacesでカスタムzipを生成する手順)
7. [テストスイート (tests/a2e_japanese)](#7-テストスイート-testsa2e_japanese)
8. [デプロイ構成](#8-デプロイ構成)
9. [データフロー全体図](#9-データフロー全体図)

---

## 1. 全体アーキテクチャ

```
┌─────────────────────┐     REST      ┌─────────────────────────┐     REST      ┌─────────────────────┐
│  gourmet-sp          │ ◄──────────► │  gourmet-support         │ ◄──────────► │  audio2exp-service   │
│  (Astro + TS)        │              │  (Flask + SocketIO)      │              │  (Flask)             │
│  Vercel              │              │  Cloud Run               │              │  Cloud Run           │
├──────────────────────┤              ├──────────────────────────┤              ├──────────────────────┤
│ concierge-controller │              │ app_customer_support.py  │              │ app.py               │
│ core-controller      │              │ support_core.py          │              │ a2e_engine.py        │
│ audio-manager        │              │ api_integrations.py      │              │  ├ Wav2Vec2          │
│ gvrm (3D avatar)     │              │ long_term_memory.py      │              │  └ A2E Decoder       │
│ lipsync              │              │                          │              │                      │
└──────────────────────┘              └──────────────────────────┘              └──────────────────────┘
                                              │
                                              ├── Google Cloud TTS
                                              ├── Google Cloud STT (Chirp2)
                                              ├── Gemini 2.0 Flash (LLM)
                                              ├── HotPepper API
                                              └── Firestore (長期記憶)
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 公式LAMツールチェーン (別系統 — アバター生成用)                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [HF Spaces / ModelScope / ローカルGradio]                                │
│       app_hf_space.py / app_lam.py                                       │
│           ↓                                                              │
│  1枚の顔画像 → FlameTracking → LAM-20K推論 → 3Dアバター生成               │
│           ↓                                                              │
│  「Export ZIP for Chatting Avatar」チェックボックス                         │
│           ↓                                                              │
│  ZIP出力: skin.glb + offset.ply + animation.glb                          │
│           ↓                                                              │
│  OpenAvatarChat / gourmet-sp で使用可能                                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. バックエンド (gourmet-support)

### 2.1 ファイル構成

| ファイル | 行数 | 役割 |
|----------|------|------|
| `app_customer_support.py` | ~450行 | Flaskアプリ本体、全APIエンドポイント |
| `support_core.py` | ~350行 | Gemini LLM対話ロジック、プロンプト管理 |
| `api_integrations.py` | ~250行 | HotPepper API、場所検索 |
| `long_term_memory.py` | ~200行 | Firestore長期記憶 |

### 2.2 APIエンドポイント一覧

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/api/session/start` | POST | セッション開始。長期記憶から挨拶文を生成 |
| `/api/session/end` | POST | セッション終了 |
| `/api/chat` | POST | LLMチャット。Gemini 2.0 Flashで応答生成 |
| `/api/tts/synthesize` | POST | Google Cloud TTS + A2E表情データ生成 |
| `/health` | GET | ヘルスチェック |

### 2.3 TTS + A2E 統合フロー (`app_customer_support.py`)

```python
@app.route('/api/tts/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']
    language_code = request.json['language_code']
    voice_name = request.json['voice_name']
    session_id = request.json.get('session_id')

    # 1. Google Cloud TTS で MP3 生成
    audio_base64 = synthesize_with_gcp(text, language_code, voice_name)

    # 2. A2E表情データ生成 (AUDIO2EXP_SERVICE_URL が設定されている場合)
    expression = None
    if AUDIO2EXP_SERVICE_URL and audio_base64:
        expression = get_expression_frames(audio_base64, session_id)

    # 3. 音声 + 表情データを同梱して返却
    return jsonify({
        'success': True,
        'audio': audio_base64,
        'expression': expression  # {names, frames, frame_rate} or None
    })
```

`get_expression_frames()` は内部で `audio2exp-service` の `/api/audio2expression` を呼ぶ。
タイムアウト10秒。失敗時は `expression=None` でフォールバック。

### 2.4 LLM対話フロー (`support_core.py`)

```
ユーザー入力
    ↓
support_core.process_message(session_id, message, stage, language, mode)
    ↓
1. Gemini 2.0 Flash に送信 (system prompt + 会話履歴 + ユーザー入力)
    ↓
2. レスポンス解析:
   - shops データあり → HotPepper URL付きで返却
   - shops なし → テキストのみ返却
    ↓
3. 長期記憶更新 (ユーザーの好み・過去のやりとり)
```

### 2.5 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCPプロジェクトID |
| `GEMINI_API_KEY` | Yes | Gemini API キー |
| `HOTPEPPER_API_KEY` | Yes | HotPepper APIキー |
| `AUDIO2EXP_SERVICE_URL` | No | A2Eサービスの URL (未設定時はFFTフォールバック) |
| `FIRESTORE_COLLECTION` | No | 長期記憶のコレクション名 |

---

## 3. フロントエンド (gourmet-sp)

### 3.1 ファイル構成

| ファイル | 行数 | 役割 |
|----------|------|------|
| `core-controller.ts` | ~1040行 | 基底コントローラー。セッション管理、TTS再生、STT、UI |
| `concierge-controller.ts` | ~812行 | コンシェルジュモード。GVRM 3Dアバター + リップシンク |
| `chat-controller.ts` | ~45行 | チャットモード。テキストのみ |
| `audio-manager.ts` | ~733行 | マイク入力、AudioWorklet、VAD |
| `gvrm.ts` | ~353行 | Gaussian Splatting 3Dアバターレンダラー |
| `lipsync.ts` | ~61行 | FFTベースリップシンク解析 |
| `concierge.astro` | ~559行 | コンシェルジュモードのページ |
| `index.astro` | ~572行 | チャットモードのページ |
| `Concierge.astro` | ~329行 | コンシェルジュUIコンポーネント |

### 3.2 クラス継承

```
CoreController (core-controller.ts)
├── ConciergeController (concierge-controller.ts)
│   └── GVRM 3Dアバター + リップシンク
└── ChatController (chat-controller.ts)
    └── テキストのみ
```

### 3.3 CoreController 主要メソッド

| メソッド | 説明 |
|----------|------|
| `init()` | 初期化。イベントバインド、Socket.IO、セッション開始 |
| `initializeSession()` | `/api/session/start` → 挨拶音声 + ACK事前生成 |
| `toggleRecording()` | マイク ON/OFF |
| `handleStreamingSTTComplete()` | STT完了 → エコー判定 → ACK再生 → `sendMessage()` |
| `sendMessage()` | `/api/chat` → レスポンス表示 + TTS再生 |
| `speakTextGCP()` | `/api/tts/synthesize` → `ttsPlayer` で再生 |
| `extractShopsFromResponse()` | Markdownレスポンスからショップ情報を抽出 |

### 3.4 ConciergeController 追加機能

| メソッド | 説明 |
|----------|------|
| `setupAudioAnalysis()` | FFT解析用 AudioContext + AnalyserNode 作成 |
| `startLipSyncLoop()` | requestAnimationFrame で FFT → `gvrm.updateLipSync(level)` |
| `stopAvatarAnimation()` | 口を閉じる + animationFrame キャンセル |
| `speakResponseInChunks()` | 文単位で分割 → 並行TTS合成 → 順次再生 |

### 3.5 現在のリップシンク方式 (FFTベース)

```
ttsPlayer (HTMLAudioElement)
    ↓ MediaElementAudioSource
AnalyserNode (fftSize=256)
    ↓ getByteFrequencyData()
全周波数ビンの平均値
    ↓ Math.min(1.0, (average/255) * 2.5)
gvrm.updateLipSync(0.0 ~ 1.0)
    ↓ VRMManager.setLipSync(level)
Jaw/Mouthボーン回転
```

- 更新レート: ~60Hz (requestAnimationFrame)
- ノイズゲート: average < 0.02 → 0
- 感度: ×2.5 で増幅、1.0でクリップ
- 制限: 音量ベースのため母音の区別不可

### 3.6 AudioManager 音声入力パイプライン

```
マイク → MediaStream (48kHz/44.1kHz)
    ↓ AudioWorkletProcessor
ダウンサンプリング → 16kHz Int16 PCM
    ↓ base64エンコード
Socket.IO emit('audio_chunk')
    ↓
サーバー: Google Cloud STT (Chirp2)
    ↓ transcript イベント
handleStreamingSTTComplete()
```

| 設定 | Chat | Concierge |
|------|------|-----------|
| 無音検出タイムアウト | 4500ms | 8000ms |
| 無音閾値 | 35 (dB相当) | 35 |
| 最小録音時間 | 3秒 | 3秒 |
| 最大録音時間 | 60秒 | 60秒 |
| バッファ上限 | 48チャンク (3秒) | 48チャンク (3秒) |

### 3.7 GVRM レンダリングパイプライン (`gvrm.ts`)

```
loadAssets():
  PLYLoader → 頂点位置データ
  TemplateDecoder → 変形テンプレート
  ImageEncoder (DINOv2) → ID特徴量抽出
  vertex_mapping.json → PLY↔テンプレート対応
  GSViewer → Gaussian Splatting レンダラー

animate() (毎フレーム):
  VRM.update() → ボーンポーズ更新
  8回のLatentタイルパス (32ch / 4×2グリッド)
    → 256×256 RenderTarget
    → Float32Array 読み出し
  NeuralRefiner.process(coarseFm, idEmbedding)
    → 512×512 RGB 生成
  WebGLDisplay.display(refinedRgb)
    → Canvas表示
```

---

## 4. Audio2Expression サービス

### 4.1 ファイル構成

```
services/audio2exp-service/
├── app.py              # Flask API サーバー (port 8081)
├── a2e_engine.py       # 推論エンジン本体
├── requirements.txt    # Python依存関係
├── Dockerfile          # コンテナビルド
├── start.sh            # 起動スクリプト
└── models/             # モデルファイル (gitignore)
    ├── wav2vec2-base-960h/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── ...
    └── LAM_audio2exp_streaming.tar
```

### 4.2 推論パイプライン (`a2e_engine.py`)

```
音声 (base64 MP3/WAV)
    ↓ pydub デコード
PCM float32 @ 16kHz
    ↓
Wav2Vec2 (facebook/wav2vec2-base-960h)
    ↓ 音響特徴量 (1, T, 768)
    ↓
A2Eデコーダー (3DAIGC/LAM_audio2exp)  ← 存在する場合
    ↓ 52次元 ARKit ブレンドシェイプ (T', 52)
    ↓
リサンプリング → 30fps
    ↓
{names: [52 strings], frames: [[52 floats], ...], frame_rate: 30}
```

### 4.3 フォールバック (A2Eデコーダーなし)

A2Eデコーダーが見つからない場合、Wav2Vec2の768次元特徴量から
エネルギーベースでブレンドシェイプを近似生成:

```
features (T, 768)
├── 低周波帯 [0:256]   → jawOpen (母音の開き)
├── 中周波帯 [256:512] → mouthFunnel/Pucker (う/お)
└── 高周波帯 [512:768] → mouthSmile (い/え)
    ↓
スムージング (3フレーム移動平均)
    ↓
無音マスク (speech_activity < 0.1 → ×0.1)
```

### 4.4 52次元ARKitブレンドシェイプ

```
Index  Name                    リップシンクへの影響
─────  ──────────────────────  ──────────────────
 17    jawOpen                 ★★★ メイン (口の開閉)
 18    mouthClose              ★★  jawOpenの逆
 19    mouthFunnel             ★★  「う」「お」
 20    mouthPucker             ★   「う」すぼめ
 23    mouthSmileLeft          ★★  「い」「え」横開き
 24    mouthSmileRight         ★★  「い」「え」横開き
 37    mouthLowerDownLeft      ★   下唇の下がり
 38    mouthLowerDownRight     ★   下唇の下がり
 39    mouthUpperUpLeft        ★   上唇の上がり
 40    mouthUpperUpRight       ★   上唇の上がり
```

### 4.5 APIリファレンス

#### POST `/api/audio2expression`

**Request:**
```json
{
    "audio_base64": "<base64 encoded MP3/WAV>",
    "session_id": "uuid-string",
    "audio_format": "mp3"
}
```

**Response:**
```json
{
    "names": ["eyeBlinkLeft", "eyeLookDownLeft", ..., "tongueOut"],
    "frames": [
        {"weights": [0.0, 0.0, ..., 0.0]},
        {"weights": [0.1, 0.0, ..., 0.0]}
    ],
    "frame_rate": 30
}
```

#### GET `/health`

```json
{
    "status": "healthy",
    "engine_ready": true,
    "device": "cpu",
    "model_dir": "/app/models"
}
```

### 4.6 モデルダウンロード

```bash
# Wav2Vec2 (~360MB)
git lfs install
git clone https://huggingface.co/facebook/wav2vec2-base-960h models/wav2vec2-base-960h

# LAM A2E Decoder (~50MB)
wget -O models/LAM_audio2exp_streaming.tar \
  https://huggingface.co/3DAIGC/LAM_audio2exp/resolve/main/LAM_audio2exp_streaming.tar
```

---

## 5. A2E フロントエンド統合パッチ

### 5.1 パッチファイル一覧

```
services/frontend-patches/
├── FRONTEND_INTEGRATION.md       # 統合ガイド
├── vrm-expression-manager.ts     # A2Eブレンドシェイプ→ボーン変換
└── concierge-controller.ts       # パッチ適用済みコントローラー
```

### 5.2 ExpressionManager (`vrm-expression-manager.ts`)

A2Eの52次元ARKitブレンドシェイプをGVRMのボーンシステムにマッピングするクラス。

```typescript
class ExpressionManager {
  constructor(renderer: GVRM);

  // A2Eフレームデータを音声に同期して再生
  playExpressionFrames(expression: ExpressionData, audioElement: HTMLAudioElement): void;

  // 停止
  stop(): void;

  // バリデーション
  static isValid(expression: any): expression is ExpressionData;
}
```

**マッピングロジック:**
```
jawOpen × 0.6
+ (mouthLowerDownL + mouthLowerDownR) / 2 × 0.2
+ (mouthUpperUpL + mouthUpperUpR) / 2 × 0.1
+ mouthFunnel × 0.05
+ mouthPucker × 0.05
= mouthOpenness (0.0 ~ 1.0)
→ gvrm.updateLipSync(mouthOpenness)
```

### 5.3 パッチ版 concierge-controller.ts の主な変更点

現在のgourmet-spの `concierge-controller.ts` との差分:

| 項目 | 現行 (gourmet-sp) | パッチ版 |
|------|-------------------|----------|
| リップシンク | FFT音量ベース | A2E 52次元ブレンドシェイプ |
| 3Dアバター | GVRM直接制御 | `window.lamAvatarController` 経由 |
| TTS応答処理 | `setupAudioAnalysis()` + FFTループ | `applyExpressionFromTts()` でバッファ投入 |
| ACK処理 | スマートACK選択 | 「はい」のみに簡略化 |
| 挨拶文 | 固定テキスト | バックエンドからの長期記憶対応挨拶 |
| 並行処理 | 文分割 + 並行TTS | 同様 + Expression同梱処理 |

**`applyExpressionFromTts()` の動作:**
```typescript
private applyExpressionFromTts(expression: any): void {
  const lamController = (window as any).lamAvatarController;
  if (!lamController) return;

  // バッファクリア (前セグメントの残りフレーム防止)
  lamController.clearFrameBuffer();

  // フレーム変換: {names, frames[{weights}]} → {name: weight} の配列
  const frames = expression.frames.map(f => {
    const frame = {};
    expression.names.forEach((name, i) => { frame[name] = f.weights[i]; });
    return frame;
  });

  // LAMAvatarのキューにフレームを投入
  lamController.queueExpressionFrames(frames, expression.frame_rate || 30);
}
```

### 5.4 2つの統合方式

**方式A: ExpressionManager方式 (GVRM直接)**
- `FRONTEND_INTEGRATION.md` に記載
- `ExpressionManager` が `gvrm.updateLipSync(level)` を直接呼ぶ
- 現行のGVRMレンダラーを維持

**方式B: LAMAvatar方式 (外部コントローラー)**
- パッチ版 `concierge-controller.ts` で実装
- `window.lamAvatarController` にフレームをキュー投入
- LAMAvatarが独自にレンダリング

---

## 6. 公式HF SpacesでカスタムZIPを生成する手順

### 6.1 概要

LAM公式が提供するGradio UIを使い、1枚の顔画像から
OpenAvatarChat互換のアバターZIPファイルを生成する手順。

生成されたZIPは以下で利用可能:
- OpenAvatarChat (公式チャットSDK)
- gourmet-sp (当プロジェクトのフロントエンド)

### 6.2 方法一覧

| 方法 | URL / コマンド | ZIP出力 | GPU必要 |
|------|---------------|---------|---------|
| **ModelScope Space** | https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model | Yes (2025/5/10〜対応) | 不要 (クラウドGPU) |
| **HuggingFace Space** | https://huggingface.co/spaces/3DAIGC/LAM | 動画のみ (ZIP非対応) | 不要 (ZeroGPU) |
| **ローカルGradio** | `python app_lam.py --blender_path ...` | Yes | 必要 (CUDA) |

### 6.3 方法A: ModelScope Space (推奨 — 環境構築不要)

> **[2025/5/10更新]** ModelScope DemoがOpenAvatarChat用ZIPの直接エクスポートに対応。

1. ブラウザで以下を開く:
   https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model

2. **Input Image** に正面顔画像をアップロード
   - 正面向きが最良の結果を得る
   - 解像度: 特に制限なし（内部で自動リサイズ）

3. **Input Video** にドライビング動画を選択
   - サンプル動画が複数用意されている
   - 音声付き動画の場合、音声もアバターに適用される

4. **「Export ZIP file for Chatting Avatar」** チェックボックスを **ON**

5. **Generate** をクリック

6. 処理完了後、**Export ZIP File Path** にZIPファイルのパスが表示される

7. ZIPをダウンロード

### 6.4 方法B: ローカルGradio (GPU環境がある場合)

#### 前提条件

```
- Python 3.10
- CUDA 12.1 or 11.8
- Blender >= 4.0.0
- Python FBX SDK 2020.2+
- VRAM: 8GB以上推奨
```

#### Step 1: 環境セットアップ

```bash
git clone https://github.com/aigc3d/LAM.git
cd LAM

# CUDA 12.1の場合
sh ./scripts/install/install_cu121.sh

# モデルウェイトのダウンロード
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
huggingface-cli download 3DAIGC/LAM-20K \
  --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

#### Step 2: FBX SDK + Blender インストール

```bash
# FBX SDK (Linux)
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl
pip install fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl
pip install pathlib patool

# Blender (Linux)
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xvf blender-4.0.2-linux-x64.tar.xz -C ~/software/
```

#### Step 3: テンプレートファイルのダウンロード

```bash
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar
tar -xf sample_oac.tar -C assets/
```

#### Step 4: Gradio起動

```bash
python app_lam.py --blender_path ~/software/blender-4.0.2-linux-x64/blender
```

ブラウザで `http://localhost:7860` を開き:
1. **Input Image** に正面顔画像をアップロード
2. **Input Video** にドライビング動画を選択
3. **「Export ZIP file for Chatting Avatar」** チェック ON
4. **Generate** をクリック
5. `output/open_avatar_chat/<image_name>.zip` にZIPが生成される

### 6.5 ZIP の中身

```
<image_name>/
├── skin.glb          # スキンメッシュ (GLBフォーマット、Blenderで生成)
├── offset.ply        # 頂点オフセット (Gaussian Splatting用)
└── animation.glb     # アニメーションデータ (テンプレートからコピー)
```

#### 各ファイルの役割

| ファイル | 説明 | 生成元 |
|----------|------|--------|
| `skin.glb` | ARKit互換のスキンメッシュ。FLAMEパラメトリックモデルから生成したヘッドメッシュを、テンプレートFBXのボーン構造にバインドしたもの | `tools/generateARKITGLBWithBlender.py` |
| `offset.ply` | canonical空間でのGaussian Splatting頂点オフセット。`rgb2sh=False, offset2xyz=True` で保存 | `lam.renderer.flame_model` → `cano_gs_lst[0].save_ply()` |
| `animation.glb` | 汎用アニメーションデータ。全アバター共通 | `assets/sample_oac/animation.glb` からコピー |

#### ZIP生成の内部処理 (`app_lam.py` L304-344)

```python
# 1. FLAMEモデルからシェイプメッシュを保存
saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
    shape_param.unsqueeze(0).cuda(), fd=oac_dir
)

# 2. Gaussian Splatting オフセットを保存
res['cano_gs_lst'][0].save_ply(
    os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True
)

# 3. BlenderでGLBを生成
generate_glb(
    input_mesh=Path(saved_head_path),
    template_fbx=Path("./assets/sample_oac/template_file.fbx"),
    output_glb=Path(os.path.join(oac_dir, "skin.glb")),
    blender_exec=Path(cfg.blender_path)
)

# 4. アニメーションファイルをコピー
shutil.copy(src='./assets/sample_oac/animation.glb',
            dst=os.path.join(oac_dir, 'animation.glb'))

# 5. ZIPアーカイブ作成
patoolib.create_archive(archive=output_zip_path, filenames=[base_iid_dir])
```

### 6.6 h5_render_data.zip (旧形式 — 参考)

`app_lam.py` / `app_hf_space.py` には `h5_rendering=True` 時に
別形式のZIPを生成する `create_zip_archive()` 関数もある:

```
h5_render_data/
├── lbs_weight_20k.json   # Linear Blend Skinning ウェイト
├── offset.ply            # 頂点オフセット
├── skin.glb              # スキンメッシュ
├── vertex_order.json     # 頂点順序マッピング
├── bone_tree.json        # ボーンツリー構造
└── flame_params.json     # FLAMEパラメータ
```

現在は `h5_rendering = False` がデフォルトのため、
こちらの形式は通常使われない。

### 6.7 生成したZIPの使い方

#### OpenAvatarChatで使う場合

```bash
# ZIPを展開して所定のディレクトリに配置
unzip <image_name>.zip -d /path/to/OpenAvatarChat/assets/avatar/

# 設定ファイルでアバターパスを指定
# config/chat_with_lam.yaml 内の avatar_path を更新
```

#### gourmet-sp で使う場合

ZIPから `skin.glb` と `offset.ply` を取り出し、
gourmet-sp の `public/assets/` に配置。
`gvrm.ts` の `loadAssets()` でパスを指定する。

---

## 7. テストスイート (tests/a2e_japanese)

### 7.1 目的

A2Eが日本語音声で十分なリップシンクを生成するか検証する。
もし生成できるなら、公式HF SpacesのZIP（英語/中国語で作成）を
日本語コンシェルジュでもそのまま使える。

### 7.2 テストファイル

```
tests/a2e_japanese/
├── generate_test_audio.py     # EdgeTTSでテスト音声生成
├── test_a2e_cpu.py            # A2E推論テスト (CPU)
├── save_a2e_output.py         # A2E出力をNPYで保存
├── analyze_blendshapes.py     # ブレンドシェイプ分析・可視化
├── run_all_tests.py           # 全テスト一括実行
├── setup_oac_env.py           # 環境チェック・修正
├── patch_asr_language.py      # ASR日本語強制パッチ
├── patch_vad_handler.py       # VAD numpy dtype修正パッチ
├── patch_llm_handler.py       # Gemini dict content修正パッチ
├── patch_config_japanese.py   # 設定ファイル日本語化パッチ
├── patch_asr_perf_fix.py      # ASRパフォーマンス修正パッチ
├── chat_with_lam_jp.yaml      # OpenAvatarChat日本語設定
├── diagnose_onnx_error.py     # ONNX問題診断
└── TEST_PROCEDURE.md          # テスト手順書
```

### 7.3 テスト音声

| ファイル | 内容 | 目的 |
|----------|------|------|
| `vowels_aiueo.wav` | あ、い、う、え、お | 母音のリップシェイプ |
| `greeting_konnichiwa.wav` | こんにちは、お元気ですか？ | 自然な会話 |
| `long_sentence.wav` | AIコンシェルジュの定型文 | 長文テスト |
| `mixed_phonemes.wav` | さしすせそ、たちつてと | 子音+母音 |
| `english_compare.wav` | Hello, how are you? | 英語比較 |
| `chinese_compare.wav` | 你好，我是AI助手 | 中国語比較 |
| `silence_baseline.wav` | 無音 2秒 | ベースライン |

### 7.4 判定基準

**A2Eが日本語で十分な場合 (ZIPそのまま使える):**
- jawOpen が発話時に適切に変動
- mouthFunnel/Pucker が「う」「お」で活性化
- mouthSmile系が「い」「え」で活性化
- 無音時にリップが閉じる
- 英語テストとの品質差が小さい

**A2Eが日本語で不十分な場合 (別途対応が必要):**
- リップが発話に追従しない
- 母音の区別ができない
- 英語と比べて明らかに品質が低い

### 7.5 重要な技術的知見

Wav2Vec2 (`facebook/wav2vec2-base-960h`) は英語960時間で訓練されているが、
**音響レベルで動作し、言語パラメータはゼロ**。
理論上、どの言語の音声でもブレンドシェイプを生成可能。
A2Eデコーダーも音響特徴量→表情の変換であり、
言語依存ではなく音響依存のため、日本語でも機能する見込み。

---

## 8. デプロイ構成

### 8.1 サービス一覧

| サービス | デプロイ先 | 環境 |
|----------|-----------|------|
| gourmet-support | Cloud Run (us-central1) | Python 3.11, 2vCPU, 2GB RAM |
| audio2exp-service | Cloud Run (us-central1) | Python 3.10, 2vCPU, 2GB RAM, min-instances=1 |
| gourmet-sp | Vercel | Astro SSG |

### 8.2 パフォーマンス目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| TTS合成 | < 1秒 | Google Cloud TTS |
| A2E推論 | < 2秒/文 | CPU, 2vCPU |
| TTS + A2E合計 | < 3秒 | 直列 (TTS→A2E) |
| LLMレスポンス | < 3秒 | Gemini 2.0 Flash |
| エンドツーエンド | < 6秒 | 音声入力→アバター応答 |

### 8.3 フォールバック動作

`AUDIO2EXP_SERVICE_URL` が未設定/サービスダウン時:

1. バックエンド: `expression` フィールドなしでレスポンス返却
2. フロントエンド: 従来のFFTベースリップシンクで動作
3. ユーザー体験への影響: リップシンクの精度が下がるのみ、音声再生は正常

---

## 9. データフロー全体図

### 9.1 音声入力 → アバター応答 (コンシェルジュモード)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 1: ユーザー音声入力                                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🎤 タップ → toggleRecording()                                       │
│      ↓                                                               │
│  AudioWorkletProcessor (48kHz → 16kHz Int16 PCM)                     │
│      ↓ base64チャンク                                                 │
│  Socket.IO emit('audio_chunk')                                       │
│      ↓                                                               │
│  Google Cloud STT (Chirp2, ja-JP)                                    │
│      ↓ transcript                                                    │
│  handleStreamingSTTComplete(text)                                    │
│      ↓                                                               │
│  エコー判定 → ACK「はい」再生 → sendMessage()                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 2: LLM応答生成                                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  POST /api/chat { session_id, message, stage, language, mode }       │
│      ↓                                                               │
│  Gemini 2.0 Flash (system prompt + 会話履歴)                          │
│      ↓                                                               │
│  { response: "...", shops?: [...], summary?: "..." }                 │
│      ↓                                                               │
│  addMessage('assistant', response) → UIチャットバブル表示              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 3: TTS合成 + A2E表情生成                                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  speakResponseInChunks(response)                                     │
│      ↓ 文分割 (。で区切り)                                             │
│  ┌─ 文1: POST /api/tts/synthesize ─────────────────────────────┐    │
│  │   ↓ Google Cloud TTS → MP3 base64                           │    │
│  │   ↓ audio2exp-service → 52次元ブレンドシェイプ              │    │
│  │   ↓ { audio, expression: {names, frames, frame_rate} }      │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  ┌─ 文2: POST /api/tts/synthesize (並行開始) ──────────────────┐    │
│  │   ↓ 同上                                                    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 4: 音声再生 + アバターアニメーション                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ■ A2Eデータあり (expression != null):                                │
│    applyExpressionFromTts(expression)                                │
│      ↓ lamController.queueExpressionFrames(frames, fps)             │
│      ↓ audioElement.currentTime に同期してフレーム選択                 │
│      ↓ jawOpen等 → mouthOpenness算出 → updateLipSync(level)         │
│                                                                      │
│  ■ A2Eデータなし (フォールバック):                                     │
│    setupAudioAnalysis() → AnalyserNode (fftSize=256)                │
│      ↓ startLipSyncLoop() [requestAnimationFrame]                   │
│      ↓ getByteFrequencyData → 平均値 → updateLipSync(level)         │
│                                                                      │
│  共通: gvrm.updateLipSync(0.0 ~ 1.0)                                │
│      ↓ VRMManager.setLipSync(level)                                 │
│      ↓ Jaw/Mouthボーン回転                                           │
│      ↓ GaussianSplatting レンダリング → Canvas表示                    │
│                                                                      │
│  文1再生完了 → 文2再生 → ... → stopAvatarAnimation()                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 公式ZIP生成フロー

```
┌──────────────────────────────────────────────────────────────────────┐
│ HF Spaces / ModelScope / ローカルGradio (app_lam.py)                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  顔画像 (1枚)                                                        │
│      ↓                                                               │
│  FlameTracking (FaceBoxesV2 → VGGHead → FLAME最適化)                 │
│      ↓ FLAME shape/expression パラメータ                              │
│      ↓ セグメンテーションマスク                                       │
│                                                                      │
│  LAM-20K 推論 (DINOv2 + Gaussian Splatting)                          │
│      ↓ 3D Gaussian Head Avatar                                       │
│      ↓ canonical GS + shape param                                    │
│                                                                      │
│  [Export ZIP for Chatting Avatar] チェック ON の場合:                   │
│      ↓                                                               │
│  1. save_shaped_mesh() → FLAME メッシュ (.obj)                       │
│  2. save_ply(offset2xyz=True) → offset.ply                           │
│  3. Blender → generateARKITGLBWithBlender.py → skin.glb             │
│  4. animation.glb をコピー                                            │
│  5. patoolib.create_archive() → <name>.zip                          │
│                                                                      │
│  出力: output/open_avatar_chat/<name>.zip                             │
│      ├── skin.glb                                                    │
│      ├── offset.ply                                                  │
│      └── animation.glb                                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```
