# LAM_Audio2Expression 引継ぎ・解析依頼文

## 1. プロジェクト概要

### 目的
Audio2Expressionサービスを Google Cloud Run にデプロイし、音声からARKit 52 blendshape係数をリアルタイムで生成するAPIを提供する。

### リポジトリ構成
```
/home/user/LAM_gpro/
├── audio2exp-service/
│   ├── app.py                 # FastAPI サービス本体
│   ├── Dockerfile             # Dockerイメージ定義
│   ├── cloudbuild.yaml        # Cloud Build設定
│   ├── requirements.txt       # Python依存関係
│   ├── start.sh               # 起動スクリプト
│   ├── models/                # モデルファイル格納
│   │   ├── LAM_audio2exp_streaming.tar  # LAMモデル重み
│   │   └── wav2vec2-base-960h/          # wav2vec2事前学習モデル
│   └── LAM_Audio2Expression/  # LAMモデルソースコード
│       ├── configs/
│       │   └── lam_audio2exp_config_streaming.py
│       ├── engines/
│       │   ├── defaults.py    # 設定パーサー・セットアップ
│       │   └── infer.py       # 推論エンジン (Audio2ExpressionInfer)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── builder.py     # モデルビルダー
│       │   ├── default.py     # DefaultEstimator
│       │   ├── network.py     # Audio2Expression ニューラルネットワーク
│       │   └── utils.py       # 後処理ユーティリティ
│       └── utils/
│           ├── comm.py        # 分散処理ユーティリティ
│           ├── config.py      # 設定管理
│           ├── env.py         # 環境設定
│           └── logger.py      # ロギング
```

## 2. コア技術アーキテクチャ

### Audio2Expression モデル (network.py)

```python
# 入力 → 出力フロー
input_audio_array (24kHz or 16kHz)
    → wav2vec2 audio_encoder (768次元特徴)
    → feature_projection (512次元)
    → identity_encoder (話者特徴 + GRU)
    → decoder (Conv1D + LayerNorm + ReLU)
    → output_proj (52次元)
    → sigmoid
    → ARKit 52 blendshape coefficients (0-1)
```

### 重要なパラメータ
- **内部サンプルレート**: 16kHz
- **出力フレームレート**: 30 fps
- **出力次元**: 52 (ARKit blendshape)
- **identity classes**: 12 (話者ID用)

### wav2vec2の読み込みロジック (network.py:40-44)
```python
if os.path.exists(pretrained_encoder_path):
    self.audio_encoder = Wav2Vec2Model.from_pretrained(pretrained_encoder_path)
else:
    # 警告: この場合、ランダム重みで初期化される
    config = Wav2Vec2Config.from_pretrained(wav2vec2_config_path)
    self.audio_encoder = Wav2Vec2Model(config)
```

### ストリーミング推論 (infer.py)

`infer_streaming_audio()` メソッド:
1. コンテキスト管理 (`previous_audio`, `previous_expression`, `previous_volume`)
2. 64フレーム最大長でバッファリング
3. 16kHzへリサンプリング
4. 後処理パイプライン:
   - `smooth_mouth_movements()` - 無音時の口動き抑制
   - `apply_frame_blending()` - フレーム間ブレンディング
   - `apply_savitzky_golay_smoothing()` - 平滑化フィルタ
   - `symmetrize_blendshapes()` - 左右対称化
   - `apply_random_eye_blinks_context()` - 瞬き追加

## 3. 現在の問題

### 症状
- Cloud Runへのデプロイは成功する
- ヘルスチェック応答:
  ```json
  {
    "model_initialized": false,
    "mode": "mock",
    "init_step": "...",
    "init_error": "..."
  }
  ```
- 48時間以上、40回以上のデプロイ試行で解決できていない

### 試行した解決策（全て失敗）
1. gsutil でモデルダウンロード
2. Python GCSクライアントでモデルダウンロード
3. Cloud Storage FUSE でマウント
4. Dockerイメージにモデルを焼き込み
5. max-instances を 10 → 5 → 4 に削減（quota対策）
6. ステップ別エラー追跡を追加

### 重要な指摘
ユーザーからの指摘:
> 「キミは、モデルの読み込みや、初期化が上手く行ってないと、思い込んでるでしょ？そうじゃなく、根本的にやり方が間違ってるんだよ！」
> 「LAM_Audio2Expressionのロジックを本質的に理解できてないでしょ？」

つまり、問題は単なる「ファイルが見つからない」「初期化エラー」ではなく、**アプローチ自体が根本的に間違っている**可能性がある。

## 4. 解析依頼事項

### 4.1 根本原因の特定
1. **LAM_Audio2Expressionの設計思想**
   - このモデルは元々どのような環境で動作することを想定しているか？
   - GPU必須か？CPU動作可能か？
   - リアルタイムストリーミング vs バッチ処理の制約は？

2. **Cloud Run適合性**
   - コールドスタート時間の問題はないか？
   - メモリ8GiBで十分か？
   - CPUのみで実用的な速度が出るか？

3. **初期化プロセス**
   - `default_setup(cfg)` のバッチサイズ計算が問題を起こしていないか？
   - `create_ddp_model()` がシングルプロセス環境で正しく動作するか？
   - ロガー設定がCloud Run環境で問題を起こしていないか？

### 4.2 app.py の問題点
現在の `app.py` の初期化フローを確認:
```python
# lifespan内で非同期初期化
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, engine.initialize)
```

- この初期化方法は正しいか？
- エラーが正しくキャッチ・伝播されているか？

### 4.3 設定ファイルの問題
`lam_audio2exp_config_streaming.py`:
```python
num_worker = 16    # Cloud Runで問題になる？
batch_size = 16    # 推論時も必要？
```

## 5. 期待する成果物

1. **根本原因の分析レポート**
   - なぜ現在のアプローチが機能しないのか
   - Cloud Runでこのモデルを動作させることは可能か

2. **正しい実装方針**
   - 必要な場合、代替デプロイメント方法の提案
   - app.py の正しい実装

3. **動作する実装コード**
   - モデル初期化が成功する
   - `/health` エンドポイントで `model_initialized: true` を返す
   - `/api/audio2expression` でリアルタイム推論が機能する

## 6. 関連ファイル一覧

### 必読ファイル
| ファイル | 説明 |
|---------|------|
| `audio2exp-service/app.py` | FastAPIサービス本体 |
| `LAM_Audio2Expression/engines/infer.py` | 推論エンジン |
| `LAM_Audio2Expression/models/network.py` | ニューラルネットワーク定義 |
| `LAM_Audio2Expression/engines/defaults.py` | 設定パーサー |
| `LAM_Audio2Expression/configs/lam_audio2exp_config_streaming.py` | ストリーミング設定 |

### 補助ファイル
| ファイル | 説明 |
|---------|------|
| `LAM_Audio2Expression/models/utils.py` | 後処理ユーティリティ |
| `LAM_Audio2Expression/utils/comm.py` | 分散処理ユーティリティ |
| `LAM_Audio2Expression/models/builder.py` | モデルビルダー |

## 7. デプロイ環境

- **Cloud Run Gen 2**
- **メモリ**: 8GiB
- **CPU**: 4
- **max-instances**: 4
- **コンテナポート**: 8080
- **リージョン**: asia-northeast1

## 8. Git情報

- **ブランチ**: `claude/implementation-testing-w2xCb`
- **最新コミット**: `4ba662c Simplify deployment: bake models into Docker image`

---

作成日: 2026-02-07
