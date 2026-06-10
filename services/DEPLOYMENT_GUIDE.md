# A2E (Audio2Expression) 統合デプロイメントガイド

## アーキテクチャ

```
[ブラウザ (gourmet-sp)]
    ↕ REST API
[gourmet-support (Cloud Run)]
    ├── /api/tts/synthesize → Google Cloud TTS → MP3
    │                           ↓ (MP3 base64)
    │                      [audio2exp-service (Cloud Run)]
    │                           ↓ Wav2Vec2 → A2E Decoder
    │                           ↓ 52-dim ARKit blendshapes
    │                           ↓
    └── JSON Response: { audio: "mp3...", expression: {names, frames, frame_rate} }
```

## サービス構成

| サービス | 説明 | デプロイ先 |
|----------|------|-----------|
| gourmet-support | メインバックエンド | Cloud Run (既存) |
| audio2exp-service | A2E推論マイクロサービス | Cloud Run (新規) |
| gourmet-sp | フロントエンド | Vercel (既存) |

## デプロイ手順

### 1. audio2exp-service のデプロイ

#### 1a. モデルの準備

```bash
# LAM_audio2exp Streaming モデル (HuggingFace)
# ※ Non-Streaming フルモデルは未公開。Streaming + バッチ推論 + フルポストプロセッシングで運用。
mkdir -p models
wget -O models/LAM_audio2exp_streaming.tar \
  https://huggingface.co/3DAIGC/LAM_audio2exp/resolve/main/LAM_audio2exp_streaming.tar

# Wav2Vec2 モデル
git lfs install
git clone https://huggingface.co/facebook/wav2vec2-base-960h models/wav2vec2-base-960h
```

対応するディレクトリ構造（どちらでもOK）:
```
models/
├── LAM_audio2exp_streaming.tar          ← フラット配置（推奨）
└── wav2vec2-base-960h/

# または
models/
├── pretrained_models/
│   └── lam_audio2exp_streaming.tar      ← 展開済み配置
└── wav2vec2-base-960h/
```

**品質向上ポイント**: Streaming モデルでも、バッチ推論（全音声一括入力）+
ポストプロセッシング強化（movement_smooth, brow_movement）により高品質な出力を実現。

#### 1b. ローカルテスト

```bash
cd services/audio2exp-service

# 依存関係インストール
pip install -r requirements.txt

# 起動
MODEL_DIR=./models python app.py

# ヘルスチェック
curl http://localhost:8081/health
```

#### 1c. Docker ビルド & Cloud Run デプロイ

```bash
# Cloud Run デプロイ（--source 方式、推奨）
# ※ modelsディレクトリに Streaming モデルを配置してから実行
gcloud run deploy audio2exp-service \
  --source . \
  --project hp-support-477512 \
  --region us-central1 \
  --memory 4Gi \
  --cpu 4 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 3 \
  --cpu-boost \
  --set-env-vars "MODEL_DIR=/app/models,DEVICE=cpu,WARMUP_TIMEOUT=0,ENGINE_LOAD_TIMEOUT=1500"
```

**成功パラメータの根拠:**
- `--memory 4Gi`: torch + transformers + LAM Streaming モデルの同時ロードに必要
- `--cpu 4`: ロード高速化
- `--cpu-boost`: 起動時のCPUブースト
- `ENGINE_LOAD_TIMEOUT=1500`: CPUでのモデルロードに約19分→25分の猶予
- `WARMUP_TIMEOUT=0`: warmup（ダミー推論）をスキップ（CPUタイムアウト回避）
- `--min-instances 1`: コールドスタートを排除

### 2. gourmet-support の設定

```bash
# 環境変数に audio2exp-service のURLを設定
gcloud run services update gourmet-support \
  --set-env-vars "AUDIO2EXP_SERVICE_URL=https://audio2exp-service-xxxxx.run.app"
```

`app_customer_support.py` は既に `AUDIO2EXP_SERVICE_URL` を参照済み。

### 3. フロントエンド (gourmet-sp) の更新

1. `services/frontend-patches/vrm-expression-manager.ts` を
   `gourmet-sp/src/scripts/avatar/` にコピー

2. `FRONTEND_INTEGRATION.md` に従って
   `concierge-controller.ts` を修正

3. Vercel にデプロイ

## モデルサイズ

| モデル | サイズ | 用途 |
|--------|--------|------|
| wav2vec2-base-960h | ~360MB | 音響特徴量抽出 |
| LAM_audio2exp_streaming | ~373MB (圧縮) | 表情デコーダー (Streaming, 12 identity) |
| Total | ~733MB | |

## API リファレンス

### POST /api/audio2expression

**Request:**
```json
{
    "audio_base64": "<base64 encoded MP3/WAV>",
    "session_id": "uuid-string",
    "is_start": true,
    "is_final": true,
    "audio_format": "mp3"
}
```

**Response (成功):**
```json
{
    "names": [
        "eyeBlinkLeft", "eyeLookDownLeft", ..., "tongueOut"
    ],
    "frames": [
        [0.0, 0.0, ..., 0.0],
        [0.1, 0.0, ..., 0.0],
        ...
    ],
    "frame_rate": 30
}
```

**Response (エラー):**
```json
{
    "error": "Error message"
}
```

### GET /health

**Response:**
```json
{
    "status": "healthy",
    "engine_ready": true,
    "device": "cpu",
    "model_dir": "/app/models"
}
```

## パフォーマンス目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| 推論レイテンシ | < 3秒 (1文あたり) | CPU, 4vCPU, バッチ推論 |
| TTS + A2E合計 | < 5秒 | 並列化不可 (TTS→A2E) |
| メモリ使用量 | < 3GB | Streaming モデル + ポストプロセッシング |
| 同時リクエスト | 3 | max-instances=3 |

## フォールバック動作

`AUDIO2EXP_SERVICE_URL` が未設定、またはサービスがダウンしている場合:

1. バックエンドは `expression` フィールドなしでレスポンスを返す
2. フロントエンドは従来のFFTベースリップシンクで動作（劣化なし）
3. ヘルスチェックで `audio2exp: "not configured"` が表示される

## トラブルシューティング

### A2Eサービスが応答しない
```bash
# ログ確認
gcloud run services logs read audio2exp-service --limit 50

# ヘルスチェック
curl https://audio2exp-service-xxxxx.run.app/health
```

### expressionデータが空
- `AUDIO2EXP_SERVICE_URL` が正しく設定されているか確認
- gourmet-support のログで `[Audio2Exp]` を検索
- タイムアウト（10秒）を超えていないか確認

### リップシンクがFFTと変わらない
- フロントエンドに `vrm-expression-manager.ts` が追加されているか
- `concierge-controller.ts` で `session_id` を送信しているか
- ブラウザのdevtoolsで `/api/tts/synthesize` のレスポンスに `expression` があるか
