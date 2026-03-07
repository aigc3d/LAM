# LAM_Audio2Expression 解析・実装依頼

## 依頼の背景

Audio2ExpressionサービスをGoogle Cloud Runにデプロイしようと48時間以上、40回以上試行したが、モデルが「mock」モードのままで正しく初期化されない。対症療法的な修正を繰り返しても解決できないため、根本的なアプローチの見直しが必要。

## 前任AIの反省点

**重要**: 前任AI（Claude）は以下の問題を抱えていた：

1. **古い知識ベースからの推論に依存**
   - 一般的な「Cloud Runデプロイ」パターンを適用しようとした
   - LAM_Audio2Expression固有の設計思想を理解できていなかった

2. **表面的なコード理解**
   - コードを読んだが、なぜそのように設計されているかを理解していなかった
   - 元々どのような環境・ユースケースを想定したコードなのかを考慮しなかった

3. **対症療法の繰り返し**
   - ログからエラーを見つけ→修正→デプロイ→また別のエラー、の無限ループ
   - 根本原因を特定せず、見えている症状だけを修正し続けた

4. **思い込み**
   - 「モデルの読み込みや初期化がうまくいっていない」と決めつけていた
   - 問題はそこではなく、もっと根本的なアプローチの誤りである可能性がある

**この解析を行う際は、上記の落とし穴にハマらないよう注意してください。**

## 解析対象コード

### 主要ファイル

**1. audio2exp-service/app.py** (現在のサービス実装)
- FastAPI を使用したWebサービス
- `/health`, `/debug`, `/api/audio2expression`, `/ws/{session_id}` エンドポイント
- `Audio2ExpressionEngine` クラスでモデル管理

**2. LAM_Audio2Expression/engines/infer.py**
- `InferBase` クラス: モデル構築の基底クラス
- `Audio2ExpressionInfer` クラス: 音声→表情推論
- `infer_streaming_audio()`: リアルタイムストリーミング推論

**3. LAM_Audio2Expression/models/network.py**
- `Audio2Expression` クラス: PyTorchニューラルネットワーク
- wav2vec2 エンコーダー + Identity Encoder + Decoder構成

**4. LAM_Audio2Expression/engines/defaults.py**
- `default_config_parser()`: 設定ファイル読み込み
- `default_setup()`: バッチサイズ等の設定計算
- `create_ddp_model()`: 分散データ並列ラッパー

## 具体的な解析依頼

### Q1: モデル初期化が完了しない根本原因

```python
# app.py での初期化
self.infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
self.infer.model.eval()
```

この処理がCloud Run環境で正常に完了しない理由を特定してください。

考えられる原因:
- [ ] メモリ不足 (8GiBで足りない?)
- [ ] CPU環境での動作制限
- [ ] 分散処理設定が単一インスタンスで問題を起こす
- [ ] ファイルシステムの書き込み権限
- [ ] タイムアウト (コールドスタート時間)
- [ ] その他

### Q2: default_setup() の問題

```python
# defaults.py
def default_setup(cfg):
    world_size = comm.get_world_size()  # Cloud Runでは1
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0  # 失敗する可能性?
```

推論時にこの設定が問題を起こしていないか確認してください。

### Q3: ロガー設定の問題

```python
# infer.py
self.logger = get_root_logger(
    log_file=os.path.join(cfg.save_path, "infer.log"),
    file_mode="a" if cfg.resume else "w",
)
```

Cloud Runのファイルシステムでログファイル作成が失敗する可能性を確認してください。

### Q4: wav2vec2 モデル読み込み

```python
# network.py
if os.path.exists(pretrained_encoder_path):
    self.audio_encoder = Wav2Vec2Model.from_pretrained(pretrained_encoder_path)
else:
    config = Wav2Vec2Config.from_pretrained(wav2vec2_config_path)
    self.audio_encoder = Wav2Vec2Model(config)  # ランダム重み!
```

- wav2vec2-base-960h フォルダの構成は正しいか?
- HuggingFaceからのダウンロードが必要なファイルはないか?

### Q5: 適切なデプロイ方法

Cloud Runが不適切な場合、以下の代替案を検討:
- Google Compute Engine (GPU インスタンス)
- Cloud Run Jobs (バッチ処理)
- Vertex AI Endpoints
- Kubernetes Engine

## 期待する成果

### 1. 分析結果
- 根本原因の特定
- なぜ40回以上の試行で解決できなかったかの説明

### 2. 修正されたコード
```
audio2exp-service/
├── app.py         # 修正版
├── Dockerfile     # 必要なら修正
└── cloudbuild.yaml # 必要なら修正
```

### 3. 動作確認方法
```bash
# ヘルスチェック
curl https://<service-url>/health
# 期待する応答: {"model_initialized": true, "mode": "inference", ...}

# 推論テスト
curl -X POST https://<service-url>/api/audio2expression \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "...", "session_id": "test"}'
```

## 技術スペック

### モデル仕様
| 項目 | 値 |
|------|-----|
| 入力サンプルレート | 24kHz (API) / 16kHz (内部) |
| 出力フレームレート | 30 fps |
| 出力次元 | 52 (ARKit blendshape) |
| モデルファイルサイズ | ~500MB (LAM) + ~400MB (wav2vec2) |

### デプロイ環境
| 項目 | 値 |
|------|-----|
| プラットフォーム | Cloud Run Gen 2 |
| リージョン | asia-northeast1 |
| メモリ | 8GiB |
| CPU | 4 |
| max-instances | 4 |

### 依存関係 (requirements.txt)
```
torch==2.0.1
torchaudio==2.0.2
transformers==4.30.2
librosa==0.10.0
fastapi==0.100.0
uvicorn==0.23.0
numpy==1.24.3
scipy==1.11.1
pydantic==2.0.3
```

## ファイルの場所

```bash
# プロジェクトルート
cd /home/user/LAM_gpro

# メインサービス
cat audio2exp-service/app.py

# 推論エンジン
cat audio2exp-service/LAM_Audio2Expression/engines/infer.py

# ニューラルネットワーク
cat audio2exp-service/LAM_Audio2Expression/models/network.py

# 設定
cat audio2exp-service/LAM_Audio2Expression/engines/defaults.py
cat audio2exp-service/LAM_Audio2Expression/configs/lam_audio2exp_config_streaming.py
```

---

以上、よろしくお願いいたします。
