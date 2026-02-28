# 修正計画書 v2: Modal版をModelScope公式app.pyに準拠させる
**作成日**: 2026-02-28（v2更新）
**作成者**: ClaudeCode（ユーザー・他AI確認用）
**対象ブランチ**: `claude/update-lam-modelscope-UQKxj`

---

## 0. 前提の訂正（v2で追加）

### v1の誤り

v1では「現在のconcierge_modal.pyは既に公式のgenerate_glbを使っており問題ない」と書いたが、
これはClaudeの過信による誤りだった。

ユーザーから**実際に完走したバージョン**として `app_concierge.py` が提示された。
このファイルはリポジトリに現存する（コミット `5a1e8bd`）。

### 信頼できるベースライン

| ファイル | 状態 | 説明 |
|---------|------|------|
| `LAM_Large_Avatar_Model/app.py` | 正解 | ModelScope公式。正常動作確認済み |
| `app_concierge.py` | 完走実績あり | HF Spaces/Docker版。ZIPまで生成完了 |
| `concierge_modal.py` | **要検証** | Claudeが複数回改変＆revertを繰り返したもの。信頼性低 |
| `lam_avatar_batch.py` | **要検証** | Claudeが作成。concierge_modal.pyに依存 |

### 現在のconcierge_modal.pyの来歴

Claudeがconcierge_modal.pyに対して行った操作の履歴:
```
35abf87 Delete concierge_now.zip
bd8ca68 fix: disable torch.compile + Gaussian diagnostics
d5a62e2 fix: format string error
bd54b56 Replace manual weight loading with load_state_dict
90828ed Add encoder feature sanity check
13453e8 Add --smoke-test
b8bf7b1 Fix bird monster: add xformers
9c14d1f Add PyTorch/xformers version to DIAGNOSTICS
17afde2 Update concierge_modal.py
ff0cd19 Finalize fixes for GLB export
3d0e991 Replace inline Blender script with official generate_glb pipeline  ← ★ここで方式変更
73aa5ae Fix stale output
33718c5 Redesign Gradio UI
8becf54 Finalize concierge_modal.py with fixes and optimizations
...（さらにbird-monster fix、CUDA migration、revert等が続く）...
391e477 revert: restore to last working version  ← 1回目のrevert
da39749 refactor: align with official app.py pipeline  ← またClaude改変
fb784b7 revert: restore to pre-session state (da39749)  ← 2回目のrevert
```

**revertが2回行われており、正しく戻せた保証がない。**

---

## 1. 修正目的

ModelScope公式の `LAM_Large_Avatar_Model/app.py`（正常動作確認済み）のOAC ZIP生成ロジックを、
Modal実行環境の `concierge_modal.py` と `lam_avatar_batch.py` に正確に移植する。

---

## 2. 3つのGLB生成パイプラインの比較

### 2.1 比較表

現在、3つの異なるGLB生成方法が存在する。

#### (A) 公式 `app.py`（ModelScope）— 正解

```python
# app.py 402-427行目
from generateARKITGLBWithBlender import generate_glb

generate_glb(
    input_mesh=Path(saved_head_path),
    template_fbx=Path("./assets/sample_oac/template_file.fbx"),
    output_glb=Path(os.path.join(oac_dir, "skin.glb")),
    blender_exec=Path(cfg.blender_path)
)
```

`generate_glb` 内部の処理:
1. `update_flame_shape()` — OBJの頂点をFBXテンプレートに注入 → ASCII FBX
2. `convert_ascii_to_binary()` — FBX SDK で ASCII → Binary 変換
3. `convert_with_blender()` — `convertFBX2GLB.py` 経由でFBX→GLB（マテリアル除去なし）
4. `gen_vertex_order_with_blender()` — `generateVertexIndices.py` 経由でOBJインポート→90°回転→Z座標ソート

ZIPに入るファイル: `offset.ply`, `skin.glb`, `vertex_order.json`, `animation.glb`

#### (B) `app_concierge.py`（完走バージョン）

```python
# app_concierge.py
from tools.generateARKITGLBWithBlender import update_flame_shape, convert_ascii_to_binary

# Step 1-2は公式と同じ
update_flame_shape(Path(saved_head_path), temp_ascii, template_fbx)
convert_ascii_to_binary(temp_ascii, temp_binary)

# Step 3-4は自作Blenderスクリプト（convert_and_order.py）で一括処理
cmd = [str(blender_exec), "--background", "--python", str(convert_script),
       "--", str(temp_binary), str(skin_glb_path), str(vertex_order_path)]
subprocess.run(cmd, ...)
```

自作スクリプトの特徴:
- `strip_materials()` でマテリアルを完全除去
- `export_materials='NONE'` 指定
- `world_matrix @ v.co` でワールド座標変換してからvertex_order生成
- FBX→GLB変換とvertex_order生成を **1回のBlenderセッション** で実行
- `export_morph_normal=False` 追加（公式にはない）

ZIPに入るファイル: `offset.ply`, `skin.glb`, `vertex_order.json`, `animation.glb`

#### (C) 現在の `concierge_modal.py`（Claude改変版）

```python
# concierge_modal.py 684行目
from tools.generateARKITGLBWithBlender import generate_glb

generate_glb(
    input_mesh=Path(saved_head_path),
    template_fbx=Path("./model_zoo/sample_oac/template_file.fbx"),
    output_glb=Path(os.path.join(oac_dir, "skin.glb")),
    blender_exec=Path("/usr/local/bin/blender")
)
```

公式の `generate_glb` をそのまま呼び出し。方式(A)と同一。

### 2.2 方式間の差異

| 項目 | (A) 公式 app.py | (B) app_concierge.py（完走） | (C) concierge_modal.py（現在） |
|------|----------------|---------------------------|------------------------------|
| skin.glb生成 | convertFBX2GLB.py | 自作スクリプト | convertFBX2GLB.py |
| vertex_order生成 | generateVertexIndices.py | 自作スクリプト（同一セッション） | generateVertexIndices.py |
| マテリアル除去 | なし | あり（strip_materials） | なし |
| Blender起動回数 | 2回 | 1回 | 2回 |
| 90°回転適用 | あり（vertex_order時のみ） | なし（world_matrix使用） | あり（vertex_order時のみ） |
| CWD依存 | `convertFBX2GLB.py`パス | 明示パス指定 | パスは`tools/`内 |

### 2.3 判断が必要な点

**(C)は(A)と方式は同一だが、「完走実績」がない。**

完走実績があるのは(B)のみ。(A)はModelScopeで正常動作確認済みだが、
Modal環境では未検証。(C)はClaude改変を経ており信頼性が低い。

**選択肢**:
1. **(B)に戻す**: 完走実績を優先。GLB方式はapp_concierge.pyの自作スクリプトに戻す
2. **(A)に合わせる**: 公式準拠を優先。generate_glb を使い続けるが、Modal環境差異を検証
3. **ハイブリッド**: OAC生成部分のみ(B)から移植し、それ以外は(A)に合わせる

→ **ユーザー/Geminiに判断を委ねる**

---

## 3. 公式app.pyと各版の差異一覧（行単位比較）

### 3.1 `prepare_motion_seqs` の `enlarge_ratio` パラメータ

| 項目 | 公式 app.py (382行) | concierge_modal.py (753行) | app_concierge.py |
|------|---------------------|---------------------------|-----------------|
| enlarge_ratio | `[1.0, 1, 0]` | `[1.0, 1.0]` | `[1.0, 1.0]` |

**分析**:
- 公式の `[1.0, 1, 0]` はPython構文上 **3要素のリスト** `[1.0, 1, 0]`
- `1.0` を `1, 0`（カンマ前後にスペースなし）と書き損じた**タイプミスの可能性が高い**
- `prepare_motion_seqs` の内部実装で `enlarge_ratio[:2]` 等のスライスで使っている場合、
  3要素でも2要素でも同じ結果になる可能性あり

→ **要確認: `prepare_motion_seqs`の内部実装を確認してから判断**

### 3.2 `prepare_motion_seqs` の `max_squen_length` パラメータ

| 項目 | 公式 app.py (386行) | concierge_modal.py | app_concierge.py |
|------|---------------------|---------------------|-----------------|
| max_squen_length | `300` | **指定なし** | **指定なし** |

→ **修正**: 両方に `max_squen_length=300` を追加する

### 3.3 offset.ply の保存順序

| 項目 | 公式 app.py (411行) | concierge_modal.py (794行) | app_concierge.py |
|------|---------------------|---------------------------|-----------------|
| 順序 | mesh → **ply → glb** → anim | mesh → **glb → ply** → anim | mesh → **glb → ply** → anim |

→ **修正**: 公式と同じ **mesh → ply → glb → anim** の順序に変更

### 3.4 `base_iid` の命名

| 項目 | 公式 app.py (318行) | concierge_modal.py (687行) | app_concierge.py |
|------|---------------------|---------------------------|-----------------|
| base_iid | `'chatting_avatar_' + datetime(YYYYMMDDHHMMSS)` | `"concierge"` (固定) | `"concierge"` (固定) |

→ **修正**: 公式に合わせて `chatting_avatar_YYYYMMDDHHMMSS` にする

### 3.5 ZIP作成方法

| 項目 | 公式 app.py (427行) | concierge_modal.py (809行) | app_concierge.py |
|------|---------------------|---------------------------|-----------------|
| ZIP作成 | `os.system('zip -r ...')` | Python `zipfile.ZipFile` | Python `zipfile.ZipFile` |

→ **要確認**: Chatting AvatarのZIPパーサーが構造に厳密か

### 3.6 モデルロード方法

| 項目 | 公式 app.py (641-648行) | concierge_modal.py (401-481行) | app_concierge.py |
|------|------------------------|-------------------------------|-----------------|
| ロード方法 | `wrap_model_hub().from_pretrained()` | `ModelLAM(**cfg.model)` + 手動safetensors | `ModelLAM(**cfg.model)` + `load_state_dict` |

**分析**:
- 公式は `from_pretrained` を使用
- app_concierge.py（完走版）は `ModelLAM(**cfg.model)` + `load_state_dict(strict=False)`
- concierge_modal.py は `ModelLAM(**cfg.model)` + 手動key-by-keyコピー（より冗長）

→ **app_concierge.py方式（load_state_dict）で統一すれば、完走実績と合致する**

### 3.7 `add_audio_to_video` の10秒制限

| 項目 | 公式 app.py (236行) | concierge_modal.py | app_concierge.py |
|------|---------------------|---------------------|-----------------|
| 10秒制限 | あり | なし | なし |

→ **低優先度**: 音声クリップの長さであり、ZIP品質には無関係

### 3.8 `NUMBA_THREADING_LAYER`

| 項目 | 公式 app.py (658行) | concierge_modal.py (398行) | app_concierge.py |
|------|---------------------|---------------------------|-----------------|
| 値 | `forseq` | `forseq` ✓ | `omp` |

→ concierge_modal.py は既に正しい

### 3.9 convertFBX2GLB.py vs 自作スクリプトの差異

公式 `convertFBX2GLB.py`:
```python
bpy.ops.export_scene.gltf(
    filepath=str(output_glb),
    export_format='GLB',
    export_skins=True,
    export_texcoords=False,
    export_normals=False,
    export_colors=False,
)
```

app_concierge.py 自作スクリプト:
```python
strip_materials()  # ← 公式にはない
bpy.ops.export_scene.gltf(
    filepath=str(output_glb),
    export_format='GLB',
    export_skins=True,
    export_materials='NONE',     # ← 公式にはない
    export_normals=False,
    export_texcoords=False,
    export_morph_normal=False,   # ← 公式にはない
)
```

**差異**: `strip_materials`, `export_materials='NONE'`, `export_morph_normal=False`

→ **要確認**: この差異がChatting Avatarの表示に影響するか

### 3.10 generateVertexIndices.py vs 自作vertex_order生成の差異

公式 `generateVertexIndices.py`:
```python
import_obj(str(input_mesh))        # OBJインポート
apply_rotation(base_obj)           # 90°回転（X軸）を適用
vertices = [(i, v.co.z) for ...]   # ローカル座標のZ値
sorted_vertices = sorted(vertices, key=lambda x: x[1])
```

app_concierge.py 自作:
```python
bpy.ops.import_scene.fbx(...)      # FBXインポート（OBJではない）
world_matrix = mesh_obj.matrix_world
vertices = [(i, (world_matrix @ v.co).z) for ...]  # ワールド座標のZ値
sorted_vertices = sorted(vertices, key=lambda x: x[1])
```

**差異**:
1. 入力形式: OBJ vs FBX（同じ頂点データだが、インポーターが異なる）
2. 座標系: ローカル座標+90°回転 vs ワールド座標変換
3. 結果: **頂点のインデックス順序が異なる可能性あり**

→ **これは重大な差異。vertex_orderが異なると、offset.plyとskin.glbの頂点対応がずれ、
   アニメーション適用時に「鳥のばけもの」が発生する可能性がある**

---

## 4. 修正方針の選択肢（ユーザー/Gemini判断必要）

### 選択肢A: 公式app.py完全準拠

concierge_modal.py を公式 `generate_glb` に完全準拠させる。

**メリット**:
- 公式と完全に同じ出力が期待できる
- ModelScopeで正常動作確認済みのコードパスを使う

**リスク**:
- Modal環境では未検証（完走実績なし）
- CWDやBlenderパスの違いで`convertFBX2GLB.py`が見つからない等の環境問題の可能性

**修正内容**:
1. 現在のgenerate_glb呼び出しは維持（既に公式準拠）
2. offset.plyの保存順序を修正
3. max_squen_length=300を追加
4. base_iidの命名を修正
5. モデルロード方法は現状維持（手動ロード）

### 選択肢B: 完走バージョン(app_concierge.py)のロジックに戻す

concierge_modal.pyのOAC生成部分を app_concierge.py の自作スクリプト方式に戻す。

**メリット**:
- 完走実績がある
- 環境問題が少ない（Blenderスクリプトを動的生成するため、CWD依存がない）

**リスク**:
- 自作スクリプトのGLB出力が公式と異なる可能性
- Geminiが「自作スクリプトが鳥の化け物の原因」と指摘している

**修正内容**:
1. GLB生成部分をapp_concierge.pyのロジックに差し替え
2. 公式との差異（max_squen_length等）は追加で修正

### 選択肢C: ハイブリッド（推奨検討案）

公式の `generate_glb` を使いつつ、Modal環境での動作を保証する。

**修正内容**:
1. `generate_glb` を使い続けるが、内部で呼ぶ `convertFBX2GLB.py` と
   `generateVertexIndices.py` のパスをModal環境に合わせて修正
2. offset.plyの保存順序を公式に合わせる
3. max_squen_length=300を追加
4. base_iidの命名を公式に合わせる
5. 動作確認後、app_concierge.pyの自作スクリプトは廃止

---

## 5. 全修正項目一覧（方針決定後に実施）

### 確定修正（方針に関わらず実施）

| # | 対象ファイル | 修正内容 | 根拠 |
|---|-------------|---------|------|
| F1 | concierge_modal.py | `prepare_motion_seqs` に `max_squen_length=300` を追加 | 3.2 |
| F2 | concierge_modal.py | offset.ply の保存順序を公式に合わせる（ply → glb の順） | 3.3 |
| F3 | concierge_modal.py | `base_iid` を `chatting_avatar_YYYYMMDDHHMMSS` に変更 | 3.4 |

### 方針依存の修正

| # | 対象ファイル | 修正内容 | 依存 |
|---|-------------|---------|------|
| D1 | concierge_modal.py | GLB生成方式の選択 (generate_glb vs 自作スクリプト) | 選択肢A/B/C |
| D2 | concierge_modal.py | モデルロード方法の変更（手動key-by-key → load_state_dict） | 選択肢B/C |
| D3 | concierge_modal.py | ZIP作成方法の変更 | 選択肢A |

### 低優先度

| # | 対象ファイル | 修正内容 |
|---|-------------|---------|
| L1 | concierge_modal.py | `add_audio_to_video` に10秒制限追加 |
| L2 | concierge_modal.py | ビデオ保存後のフレーム数検証追加 |

---

## 6. 判断を委ねる事項（ユーザー/Gemini確認必要）

1. **GLB生成方式**: 選択肢A/B/Cのどれを採用するか？
   - 公式generate_glb (A) vs 完走実績のある自作スクリプト (B) vs ハイブリッド (C)

2. **`enlarge_ratio` の値**: 公式の `[1.0, 1, 0]` はタイプミスか意図的か？
   - `prepare_motion_seqs` の内部実装を確認してから判断

3. **vertex_orderの差異**: 公式(OBJ+90°回転+ローカルZ) vs 完走版(FBX+ワールドZ)
   - どちらの vertex_order が Chatting Avatar で正しく動作するか？

4. **マテリアル除去**: 完走版の `strip_materials()` は必要か不要か？
   - Chatting Avatar がマテリアル情報を読むなら不要（むしろ有害）
   - 読まないなら無害

---

## 7. 修正しない項目（理由付き）

| 項目 | 理由 |
|------|------|
| app_lam.py の NUMBA_THREADING_LAYER | 今回のスコープ外（Modal版のみ対象） |
| torch.compile 無効化処理 | Modal環境固有の対策であり公式には不要 |
| Gradio UIの差異 | UI層は移植対象外 |
| _shape_guard 関数 | 公式にはないが安全装置として有用（削除不要） |

---

## 8. 次のステップ

1. **ユーザー**: この計画書をGemini/ChatGPTに共有
2. **Gemini/ChatGPT**: セクション6の判断事項について方針提示
3. **ユーザー**: 選択肢A/B/Cを決定
4. **ClaudeCode**: 決定に従い、確定修正 (F1-F3) + 方針依存修正 (D1-D3) を実装
5. **ユーザー**: Modal環境で動作確認
