# Audio2Expression Service

gourmet-sp の TTS 音声から表情データを生成するマイクロサービス。

## アーキテクチャ

```
┌─ gourmet-sp (既存) ─────────────────────────────┐
│  ユーザー → LLM → GCP TTS → 音声(base64)        │
└────────────────────────┬────────────────────────┘
                         │ POST /api/audio2expression
                         ▼
┌─ Audio2Expression Service (このサービス) ───────┐
│  音声 → Audio2Expression → 表情データ (52ch)    │
└────────────────────────┬────────────────────────┘
                         │ WebSocket
                         ▼
┌─ ブラウザ ──────────────────────────────────────┐
│  LAMAvatar (WebGL) ← 表情データで口パク         │
└─────────────────────────────────────────────────┘
```

## セットアップ

### 1. 依存関係のインストール

```bash
cd audio2exp-service
pip install -r requirements.txt
```

### 2. モデルのダウンロード（オプション）

Audio2Expression モデルを使用する場合:

```bash
# HuggingFaceからダウンロード
huggingface-cli download 3DAIGC/LAM_audio2exp --local-dir ./models
```

モデルがない場合は**モックモード**で動作します（音声振幅に基づく簡易的な口パク）。

### 3. サービス起動

```bash
# 基本起動（モックモード）
python app.py

# モデル指定
python app.py --model-path ./models/pretrained_models/lam_audio2exp_streaming.tar

# ポート指定
python app.py --port 8283
```

## API

### REST API

#### POST /api/audio2expression

TTS音声を表情データに変換。

**Request:**
```json
{
  "audio_base64": "base64エンコードされた音声 (PCM 16-bit, 16kHz)",
  "session_id": "セッションID",
  "is_final": false
}
```

**Response:**
```json
{
  "session_id": "セッションID",
  "channels": ["browDownLeft", "browDownRight", ...],
  "weights": [[0.0, 0.1, ...]],
  "timestamp": 1234567890.123
}
```

### WebSocket

#### WS /ws/{session_id}

リアルタイム表情データストリーミング。

**接続:**
```javascript
const ws = new WebSocket('ws://localhost:8283/ws/my-session');
```

**受信データ:**
```json
{
  "type": "expression",
  "session_id": "my-session",
  "channels": ["browDownLeft", ...],
  "weights": [[0.0, 0.1, ...]],
  "is_final": false,
  "timestamp": 1234567890.123
}
```

## gourmet-sp との連携

### バックエンド側（最小変更）

TTS音声取得後に、このサービスにも送信:

```python
# 既存のTTS処理後に追加
async def send_to_audio2expression(audio_base64: str, session_id: str):
    async with aiohttp.ClientSession() as session:
        await session.post(
            "http://localhost:8283/api/audio2expression",
            json={
                "audio_base64": audio_base64,
                "session_id": session_id,
                "is_final": False
            }
        )
```

### フロントエンド側（LAMAvatar）

WebSocket接続:

```javascript
const controller = window.lamAvatarController;
await controller.connectWebSocket('ws://localhost:8283/ws/' + sessionId);
```

## GCP Cloud Run デプロイ

### 方法1: ソースから直接デプロイ

```bash
cd audio2exp-service

gcloud run deploy audio2exp-service \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300
```

### 方法2: Cloud Build を使用

```bash
cd audio2exp-service

# Cloud Build でビルド＆デプロイ
gcloud builds submit --config cloudbuild.yaml
```

### デプロイ後の確認

```bash
# サービスURLを取得
gcloud run services describe audio2exp-service \
  --region asia-northeast1 \
  --format 'value(status.url)'

# ヘルスチェック
curl https://audio2exp-service-xxxxx-an.a.run.app/health
```

## gourmet-support との連携設定

### 1. 環境変数の設定

gourmet-support の Cloud Run に環境変数を追加:

```bash
gcloud run services update gourmet-support \
  --region asia-northeast1 \
  --set-env-vars "AUDIO2EXP_SERVICE_URL=https://audio2exp-service-xxxxx-an.a.run.app"
```

### 2. バックエンドコードの追加

`integration/gourmet_support_integration.py` を参照して、
TTS処理後に audio2exp-service へ音声を転送するコードを追加。

## 動作モード

| モード | 条件 | 精度 |
|--------|------|------|
| 推論モード | Audio2Expressionモデルあり | 高精度 |
| モックモード | モデルなし | 音声振幅ベース（簡易） |

モックモードでも口パクの動作確認は可能です。
