# LAM ZIPモデルファイル生成 -- 技術ドキュメント

> 本ドキュメントは、2026-02-26のChatGPT技術相談ログ(12,422行)の分析結果と、
> それに基づく実装内容を纏めたものである。

---

## 1. 背景と目的

### 1.1 LAM (Large Avatar Model) とは

- **論文**: Alibaba 3DAIGC Lab, SIGGRAPH 2025 (2025年11月公開)
- **機能**: 1枚の顔画像から、アニメーション可能な3D Gaussianアバターを生成
- **公式**: https://github.com/aigc3d/LAM

### 1.2 プロジェクトの目的

HuggingFace Spaces の公式デモと同等のZIPモデルファイル生成を、
Modal (サーバーレスGPU) 上で自前で確実に行う。

### 1.3 ZIPモデルファイルの中身

```
avatar.zip
├── skin.glb          # Blender経由で生成した3Dアバターメッシュ (GLB形式)
├── offset.ply        # Gaussian Splatting用オフセットデータ (PLY形式)
├── animation.glb     # アニメーションテンプレート (サンプルからコピー)
└── vertex_order.json # 頂点順序マッピング
```

- `skin.glb`: `generateARKITGLBWithBlender.generate_glb()` で生成
- `offset.ply`: `cano_gs_lst[0].save_ply(rgb2sh=False, offset2xyz=True)` で保存
- `animation.glb`: `./model_zoo/sample_oac/animation.glb` からの直接コピー

---

## 2. ChatGPTログから得られた知見

### 2.1 旧設計の問題点

| 問題 | 原因 | 影響 |
|------|------|------|
| Modalクレジット大量消費 | Gradio常駐GPU + `keep_warm=1` | コスト肥大化 |
| 依存関係地獄 | gradio 4.x / huggingface_hub / diffusers の三すくみ | ビルド失敗 |
| 「鳥のばけもの」 | FLAME tracking破綻 + torch.compile推論破壊 | メッシュ崩壊 |
| nvdiffrast コンパイルエラー | clangの`c++11-narrowing` | ビルド失敗 |
| Identity drift | FLAME shape_paramの平均顔への正則化 | 顔の同一性ずれ |

### 2.2 合意されたLLM役割分担

| LLM | 役割 | 注意点 |
|-----|------|--------|
| **ClaudeCode** | 実装ロボット | 確定仕様の逐語的実装、GitHub横断の機械作業。理論解釈はさせない |
| **ChatGPT** | 研究監査役 | 論文と実装結果の矛盾整理、実証データ読み解き |
| **Gemini** | 発想ブレスト要員 | 別アプローチ検討のみ。現行実装の正当化には使わない |

### 2.3 「鳥のばけもの」(Bird Monster) の原因と対策

**原因チェーン:**

```
torch.compile有効 → DINOv2推論結果が静かに破損
                  ↓
FLAME tracking入力が不正 → shape_param が異常値 (NaN, abs>5.0)
                         ↓
メッシュ頂点が爆発 → 「鳥のばけもの」出現
```

**対策 (全て実装済み):**

1. `TORCHDYNAMO_DISABLE=1` 環境変数
2. `torch._dynamo.config.disable = True` 明示設定
3. `_shape_guard()` 関数で shape_param の NaN / abs > 5.0 を検出

### 2.4 Identity Drift (顔の同一性ずれ) への対策

- **現象**: 日本人女性 → 中国人女性風になる
- **原因**: FLAME trackingの正則化が強く、shape_paramが平均顔に収束
- **対策**: `shape_scale` パラメータで個性を強調

```json
{"shape_scale": 1.15}
```

`shape_param = shape_param * alpha` (alpha=1.0〜1.2) でスケーリング。

### 2.5 flame_param の本質

- **mp4動画はデータとして一切読まれていない** -- フォルダ名を決める「名前札」に過ぎない
- 実際の動きは `flame_param/*.npy` (時系列FLAMEパラメータ) が100%決定
- 各フレームのパラメータ構成:
  - `global_orient` (3): 頭全体の回転
  - `transl` (3): 頭の移動
  - `jaw_pose` (3): 顎の回転
  - `expression` (~50): 表情ブレンドシェイプ
  - `neck_pose` (3): 首の回転
  - `leye_pose` / `reye_pose` (3+3): 眼球の回転

---

## 3. アーキテクチャ

### 3.1 三層分離設計

| レイヤー | 方針 | 詳細 |
|----------|------|------|
| **重みファイル** | Modal Imageに焼き込み固定 | `_download_missing_models()` でHFから自動DL |
| **コード** | 必要なものだけローカルからマウント | `tools`, `lam`, `configs`, `vhap`, `external` |
| **入力データ** | 実行時にバイト列として渡す | `add_local_dir` は使わず、bytes/dictで転送 |

### 3.2 推論パイプライン

```
入力: 顔画像 (PNG/JPG) + パラメータJSON
                ↓
  [Modal GPU コンテナ (L4)]
  ┌───────────────────────────────────┐
  │                                    │
  │  Step 1: FLAME Tracking            │
  │    画像 → shape_param推定          │
  │    → _shape_guard() で異常値検出   │
  │                                    │
  │  Step 2: Motion準備                │
  │    flame_param/*.npy 読込          │
  │    → motion_seq 生成               │
  │                                    │
  │  Step 3: 前処理                    │
  │    画像セグメンテーション          │
  │    shape_scale適用 (個性強調)      │
  │                                    │
  │  Step 4: LAM推論                   │
  │    lam.infer_single_view()         │
  │    → Gaussian splatting結果        │
  │                                    │
  │  Step 5: エクスポート              │
  │    Blender GLB生成                 │
  │    offset.ply 保存                 │
  │    → avatar.zip 作成               │
  │                                    │
  └───────────────────────────────────┘
                ↓
出力: avatar.zip + preview.png + compare.png + result_meta.json
```

### 3.3 2つの実行モード

| モード | ファイル | 用途 | GPU常駐 |
|--------|----------|------|---------|
| **Web UI** | `concierge_modal.py` | Gradio経由のインタラクティブ生成 | min_containers=0 |
| **バッチ** | `lam_avatar_batch.py` | CLI経由の単発実験・パラメータスイープ | なし (ワンショット) |

---

## 4. 実装詳細

### 4.1 concierge_modal.py の修正内容

#### 修正1: コンパイラ -- clang → gcc

```python
# 変更前
"clang", "llvm", "libclang-dev",
# 変更後
"gcc", "g++",
```

**理由**: nvdiffrastのJITコンパイルでclangが`c++11-narrowing`エラーを出す。
gccでは警告のみで正常にコンパイルされる。

#### 修正2: GPU arch -- 8.6 → 8.9

```python
# 変更前
"TORCH_CUDA_ARCH_LIST": "8.6",
# 変更後
"TORCH_CUDA_ARCH_LIST": "8.9",
```

**理由**: Modal L4 GPU = NVIDIA L4 (Ada Lovelace) = sm_89。

#### 修正3: nvdiffrast -- ShenhanQianフォーク → NVlabs公式

```python
# 変更前
"pip install git+https://github.com/ShenhanQian/nvdiffrast.git@backface-culling --no-build-isolation",
# 変更後
"pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation",
```

**理由**: backface-cullingブランチが不安定。公式リポジトリが安定。

#### 修正4: TorchDynamo無効化

```python
"TORCHDYNAMO_DISABLE": "1",
```

**理由**: `@torch.compile` デコレータ (`Dinov2FusionWrapper.forward`,
`ModelLAM.forward_latent_points`) が推論結果を静かに破壊する。

#### 修正5: nvdiffrastプリコンパイル追加

```python
def _precompile_nvdiffrast():
    import torch
    import nvdiffrast.torch as dr
    print("nvdiffrast pre-compiled OK")

image = image.run_function(_precompile_nvdiffrast)
```

**理由**: コールドスタート時のJIT再コンパイル (10〜30分) を回避。

#### 修正6: c++11-narrowingモンキーパッチ削除

gcc移行により不要になったため、`Generator.setup()` 内の
`-Wno-c++11-narrowing` パッチを完全削除。

#### 修正7: Modal 1.0移行

```python
# 変更前
@app.cls(..., scaledown_window=10)
# 変更後
@app.cls(..., scaledown_window=300, min_containers=0, max_containers=1)
```

**理由**: `keep_warm` は非推奨。`min_containers` / `max_containers` が正式API。

### 4.2 lam_avatar_batch.py の設計

#### コアコンセプト

- `concierge_modal.py` の **image定義をそのまま再利用** (依存関係を一切変更しない)
- UIなし / keep_warmなし → **Modalクレジット消費は実行時間のみ**
- 入力はバイト列 (image_bytes + params dict) → Modalリモート関数に直接転送
- 結果はModal Volume (`lam-batch-output`) に永続保存

#### _shape_guard() -- 鳥のばけもの検出

```python
def _shape_guard(shape_param):
    arr = shape_param.detach().cpu().numpy()
    if np.isnan(arr).any():
        raise RuntimeError("shape_param contains NaN")
    if np.abs(arr).max() > 5.0:
        raise RuntimeError(f"shape_param exploded (max abs = {max_abs:.2f})")
```

正常範囲: `[-3.0, +3.0]`。異常: NaN または abs > 5.0。

#### パラメータJSON

```json
{
  "shape_scale": 1.15,
  "motion_name": "talk"
}
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `shape_scale` | float | 1.0 | shape_paramスケーリング (1.0〜1.2で個性強調) |
| `motion_name` | str | "talk" | モーションフォルダ名 (talk, laugh, sing, nod等) |

#### 出力ファイル (Modal Volume: `lam-batch-output`)

| ファイル | 内容 |
|----------|------|
| `avatar.zip` | ZIPモデルファイル (skin.glb + offset.ply + animation.glb) |
| `preview.png` | 推論結果の最初のフレーム |
| `compare.png` | 入力画像と出力の左右比較 (512x256) |
| `preprocessed_input.png` | LAMに入力された前処理済み画像 |
| `result_meta.json` | パラメータ・shape_param範囲・ZIPサイズ |

---

## 5. 使い方

### 5.1 バッチ実行 (lam_avatar_batch.py)

```bash
# デフォルトパラメータで実行
modal run lam_avatar_batch.py --image-path ./input/input.png

# パラメータJSON指定
modal run lam_avatar_batch.py \
  --image-path ./input/input.png \
  --param-json-path ./input/params.json
```

### 5.2 Web UI (concierge_modal.py)

```bash
# 開発モード
modal serve concierge_modal.py

# デプロイ
modal deploy concierge_modal.py
```

### 5.3 実験フロー (推奨)

ChatGPTログで合意された方法論:

1. **静止画 (PNG) のみで比較** -- 動画チェックは捨てる
2. **JSONでパラメータ管理** -- `shape_scale` 等を変えながらスイープ
3. **compare.png で一覧比較** -- 入力と出力を左右並びで視覚的に評価
4. **勝ちパラメータだけ動画生成** -- 無駄なGPU時間を削減

---

## 6. 依存関係

### 6.1 Modal Imageの構成

| コンポーネント | バージョン |
|----------------|-----------|
| ベースイメージ | `nvidia/cuda:11.8.0-devel-ubuntu22.04` |
| Python | 3.10 |
| PyTorch | 2.3.0 + CUDA 11.8 |
| xformers | 0.0.26.post1 |
| Blender | 4.2 LTS |
| GPU | L4 (sm_89, TORCH_CUDA_ARCH_LIST=8.9) |
| コンパイラ | gcc/g++ |

### 6.2 主要Pythonパッケージ (バージョン固定)

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| torch | 2.3.0 | 推論エンジン |
| torchvision | 0.18.0 | 画像変換 |
| xformers | 0.0.26.post1 | DINOv2 attention精度 |
| numpy | 1.23.5 | 数値計算 |
| gradio | 4.44.0 | Web UI (conciergeのみ) |
| transformers | 4.44.2 | DINOv2エンコーダ |
| diffusers | 0.30.3 | SD3条件付きTransformer |
| accelerate | 0.34.2 | モデルロード最適化 |
| omegaconf | 2.3.0 | 設定管理 |
| safetensors | (latest) | チェックポイントロード |
| trimesh | (latest) | メッシュ処理 |
| mediapipe | 0.10.21 | 顔検出補助 |

### 6.3 CUDAソースビルド拡張

| 拡張 | ソース |
|------|--------|
| pytorch3d | `github.com/facebookresearch/pytorch3d` |
| diff-gaussian-rasterization | `github.com/ashawkey/diff-gaussian-rasterization` |
| nvdiffrast | `github.com/NVlabs/nvdiffrast` (公式) |
| FBX SDK | `fbx-2020.3.4-cp310` (Alibaba OSS) |
| cpu_nms | LAM内部 Cythonビルド |

### 6.4 モデルダウンロード

| アセット | ソース | サイズ |
|----------|--------|--------|
| LAM-20K | HuggingFace `3DAIGC/LAM-20K` | ~2 GB |
| FLAME tracking | HuggingFace `3DAIGC/LAM-assets` / `thirdparty_models.tar` | ~500 MB |
| FLAME parametric | HuggingFace `3DAIGC/LAM-assets` / `LAM_human_model.tar` | ~100 MB |
| Sample motions | HuggingFace `3DAIGC/LAM-assets` / `LAM_assets.tar` | ~200 MB |
| DINOv2 | `dl.fbaipublicfiles.com` | ~1.1 GB |
| Sample OAC | Alibaba OSS | ~50 MB |

---

## 7. ファイル構成

```
LAM_gpro/
├── concierge_modal.py        # Web UI版 (Gradio + Modal GPU)    [修正済み]
├── lam_avatar_batch.py       # バッチ版 (CLI + Modal GPU)       [新規作成]
├── app_lam.py                # LAM公式Gradioアプリ (参照用)
├── app_concierge.py          # Modal-free Docker版
├── docs/
│   └── LAM_ZIP_MODEL_GENERATION.md  # 本ドキュメント
├── lam/                      # LAMコアモジュール (229ファイル)
│   ├── models/               #   ModelLAM, DINOv2, FLAME, Gaussian
│   ├── runners/infer/        #   推論パイプライン
│   ├── datasets/             #   データセットローダー
│   └── utils/                #   前処理、ビデオ、ロギング
├── vhap/                     # VHAP顔アニメーショントラッキング
├── tools/
│   ├── flame_tracking_single_image.py   # FLAME単一画像トラッキング
│   ├── generateARKITGLBWithBlender.py   # Blender GLB生成 (公式)
│   └── generateGLBWithBlender_v2.py     # Blender GLB生成 (v2)
├── configs/
│   └── inference/lam-20k-8gpu.yaml      # 推論設定
├── external/                 # 外部依存 (顔検出, マッティング)
└── audio2exp-service/        # Audio2Expression マイクロサービス
```

---

## 8. 既知の課題と今後の方針

### 8.1 残存課題

| 課題 | 状態 | 対策案 |
|------|------|--------|
| `app_lam.py` に独立した `generate()` 関数がない | 未着手 | `demo_lam()` 内の推論ロジックを `lam_core.py` に切り出し |
| 依存関係のバージョンレンジ (`huggingface_hub>=0.24.0`) | 要検討 | `pip-compile` で完全固定を検討 |
| `shape_scale` の最適値はデータセット依存 | 実験中 | パラメータスイープで経験的に決定 |

### 8.2 今後の実験方針

1. `shape_scale` を 1.0〜1.2 の範囲でスイープし、最適値を特定
2. 複数モーション (talk, laugh, sing, nod) での品質比較
3. 異なるエスニシティの入力画像でのIdentity drift評価
4. OpenAvatarChat との統合テスト

---

## 9. 参考: ChatGPTログの構成 (12,422行)

| 区間 | 内容 |
|------|------|
| 1-2000行 | LLM役割分担、FLAME/shape_paramの本質、依存関係地獄の分析 |
| 2000-4000行 | concierge_modal.py → lam_avatar_batch.py への設計転換 |
| 4000-8000行 | 依存関係「モグラ叩き」、Volume マウント問題、反復修正 |
| 8000-12422行 | gcc移行、NVlabs nvdiffrast、Modal 1.0移行、最終コード確定 |

**最大の教訓**: 成功済み環境 (`concierge_modal.py` のimage定義) を流用し、
依存関係を一切変更しないアプローチが最も確実。
