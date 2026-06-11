# 引継ぎ文: ModelScope実証テスト

## 目的
公式ModelScopeのコードをベースに、新規ModelScopeアカウント（GitHub連携）上でOAC ZIP生成の実証テストを行う。

---

## 1. 公式コードの所在

GitHubリポジトリ `mirai-gpro/LAM_gpro` の `origin/lam-large-upload` ブランチに、ModelScope公式デモの全ソースが格納済み:

```
git fetch origin lam-large-upload
git show origin/lam-large-upload:LAM_Large_Avatar_Model/app.py
```

`LAM_Large_Avatar_Model/` 配下が公式コード一式。

---

## 2. OAC ZIP生成パイプライン（公式app.py 415-445行目）

```
画像入力
  → FLAME tracking (preprocess → optimize → export)
  → preprocess_image (セグメンテーション + shape_param抽出)
  → prepare_motion_seqs (モーションシーケンス準備)
  → lam.infer_single_view (LAM推論)
  → save_shaped_mesh → nature.obj 出力
  → generate_glb (4ステップ):
      Step 1: update_flame_shape (OBJ頂点をFBXテンプレートに注入)
      Step 2: convert_ascii_to_binary (FBX ASCII→Binary, FBX SDK必須)
      Step 3: convert_with_blender (FBX→GLB, Blender必須)
      Step 4: gen_vertex_order_with_blender (vertex_order.json生成)
  → save_ply → offset.ply 出力
  → animation.glb コピー (assets/sample_oac/から)
  → ZIP化
```

### ZIPの中身（4ファイル）
| ファイル | 生成方法 |
|---------|---------|
| `offset.ply` | `res['cano_gs_lst'][0].save_ply(rgb2sh=False, offset2xyz=True)` |
| `skin.glb` | `generate_glb` (FBXテンプレート経由) |
| `vertex_order.json` | `generate_glb` 内のStep 4で自動生成 |
| `animation.glb` | `assets/sample_oac/animation.glb` からコピー |

---

## 3. 公式コードの依存関係

| 依存 | 用途 |
|------|------|
| FBX SDK (Python) | `convert_ascii_to_binary` で使用。`import fbx` |
| Blender 4.x | `convertFBX2GLB.py` と `generateVertexIndices.py` をバックグラウンド実行 |
| template_file.fbx | `assets/sample_oac/` に配置。FLAME骨格構造入りFBXテンプレート |
| animation.glb | `assets/sample_oac/` に配置。テンプレートアニメーション |

---

## 4. 我々のリポジトリと公式コードの差分（本セッションで精査済み）

### 完全一致のファイル
- `flame_tracking_single_image.py`
- `generateARKITGLBWithBlender.py`
- `convertFBX2GLB.py`
- `lam/` 配下61ファイル

### 公式にのみ存在（我々のリポジトリに不在）
- `generateVertexIndices.py` — vertex_order.json生成用Blenderスクリプト
- `generateGLBWithBlender_v2.py` — v2版（現在のパイプラインでは未使用）
- `lam/models/encoders/` の8エンコーダラッパー — 推論時は `dinov2_fusion` のみ使用のため不要
- `lam/runners/train/` の3ファイル — トレーニング用、推論不要

### 意図的な差分
- パス変更: `pretrained_models/human_model_files` → `model_zoo/human_parametric_models`
- `@torch.compile` 無効化（3ファイル）: bird-monster対策
- `flame.py`: `save_bone_tree()`, `save_h5_info()`, `save_shaped_mesh()` 追加（OAC ZIP生成に `save_shaped_mesh` が必要）
- `modeling_lam.py`: エンコーダファクトリを `dinov2_fusion` のみに簡略化

### concierge_modal.py の現状
- `generate_glb` を既に使用（639行目）
- ただし `app_concierge.py` は自作 `convert_and_order.py` スクリプトを使用（536行目）

---

## 5. ModelScope環境の確認事項

- xGPUサービス: 無料でGPU付き創空間をホスティング可能
- 公式LAMデモが創空間で実際に稼働中（動作実績あり）
- Blender 4.x のパス: 公式app.pyでは `./blender-4.0.2-linux-x64/blender`（cfg.blender_path）
- FBX SDK: 公式環境にインストール済み

---

## 6. 未解決の問題

- 「鳥の化け物」問題の根本原因は未特定
- ModelScope公式デモでは正常動作が確認されているため、同一環境での再現テストが切り分けに有効
