# セッション引継文
**作成日**: 2026-02-28
**対象ブランチ**: `claude/update-lam-modelscope-UQKxj`
**目的**: 前セッションのClaudeの失敗と発見を正確に記録し、次セッションで同じ過ちを繰り返さないようにする

---

## 0. 最重要: Claudeへの注意事項

**前セッションのClaudeは以下の致命的な失敗を繰り返した:**

1. **ローカルの`app_lam.py`を公式app.pyだと勘違いした** — 公式は`lam-large-upload`ブランチの`LAM_Large_Avatar_Model/app.py`であり、ローカルの`app_lam.py`はそのローカルコピー（改変あり・不正確）
2. **コードを読まずに「既に正しい」と判断する癖** — 実際のファイルを読まず、過去の自分の分析を信じて安易に判断した
3. **決定済み事項を蒸し返す癖** — Geminiが決定した方針について何度も「これでいいですか？」と聞き返した
4. **会話ログを読まない** — ユーザーが順を追って教えようとしているのに、文脈を無視して自分の推論で突っ走った

**次セッションのClaudeは:**
- まず公式app.pyを読め（取得方法は下記）
- ユーザーの発言を最優先で読め
- 推測するな、コードを読め
- 決定済み事項を蒸し返すな

---

## 1. 公式app.pyの取得方法

```bash
git show origin/lam-large-upload:LAM_Large_Avatar_Model/app.py
```

このファイルが**唯一の正解リファレンス**。677行。
ローカルの`app_lam.py`（553行）はローカルコピーであり、公式とは差異がある。

---

## 2. 公式app.pyの核心部分（正解）

### _build_model (641-648行)
```python
from lam.models import model_dict
from lam.utils.hf_hub import wrap_model_hub
hf_model_cls = wrap_model_hub(model_dict["lam"])
model = hf_model_cls.from_pretrained(cfg.model_name)
```
**`from_pretrained`方式。手動copy_でもload_state_dictでもない。**

### base_iid (318行)
```python
base_iid = 'chatting_avatar_' + datetime.now().strftime("%Y%m%d%H%M%S")
```

### OAC生成順序 (409-422行)
```python
oac_dir = os.path.join('./', base_iid)
saved_head_path = save_shaped_mesh(...)           # 1. OBJ
save_ply(os.path.join(oac_dir, "offset.ply"))     # 2. offset.ply
generate_glb(input_mesh, template_fbx, output_glb, blender_exec)  # 3. skin.glb + vertex_order.json
shutil.copy(animation.glb)                        # 4. animation.glb
os.remove(saved_head_path)                        # 5. OBJ削除
```
順序: **mesh → ply → glb → anim**

### generate_glbのインポート (404行)
```python
from generateARKITGLBWithBlender import generate_glb
```
※ `tools/`プレフィックスなし（公式はsys.pathが異なるため）
※ Modal環境では`from tools.generateARKITGLBWithBlender import generate_glb`が正しい

### ZIP作成 (427行)
```python
os.system('zip -r {} {}'.format(output_zip_path, oac_dir))
```
`os.system('zip -r ...')`方式。`zipfile.ZipFile`ではない。

### prepare_motion_seqs (381-386行)
```python
prepare_motion_seqs(..., enlarge_ratio=[1.0, 1, 0], ..., max_squen_length=300)
```
- `enlarge_ratio=[1.0, 1, 0]` — タイプミス（Gemini確認済み、`[1.0, 1.0]`が正しい）
- `max_squen_length=300` — あり

### preprocess_image (365-370行)
```python
preprocess_image(..., enlarge_ratio=[1.0, 1.0], ...)
```
こちらは`[1.0, 1.0]`（正しい値）

### NUMBA_THREADING_LAYER (658行)
```python
'NUMBA_THREADING_LAYER': 'forseq'
```

### add_audio_to_video (225-245行)
- fps引数あり
- 10秒制限あり: `if audio_clip.duration > 10: audio_clip = audio_clip.subclip(0, 10)`

### モデルロード後のdtype
```python
lam.to('cuda')
```
**`torch.float32`への明示的キャストなし。** `from_pretrained`が内部で処理している可能性。

---

## 3. 公式app.py vs app_lam.py の差異

| 項目 | 公式app.py | app_lam.py |
|------|-----------|-----------|
| `_build_model` | `from_pretrained` | 手動`state_dict[k].copy_(v)` |
| `base_iid` | `chatting_avatar_` + タイムスタンプ | `os.path.basename(image_path).split('.')[0]` |
| `NUMBA_THREADING_LAYER` | `forseq` | `omp` |
| `add_audio_to_video` 10秒制限 | あり | なし |
| `add_audio_to_video` fps引数 | あり | なし |
| generate_glbインポート | `from generateARKITGLBWithBlender` | `from tools.generateARKITGLBWithBlender` |
| ZIP作成 | `os.system('zip -r ...')` | 同じ（ただしOAC部分はpatoolib使用） |
| `lam.to(float32)` | なし（from_pretrainedが処理） | なし |
| `lam.eval()` | なし（from_pretrainedが処理?） | あり |

**app_lam.pyを正解と思ってはいけない。必ず公式app.pyを参照すること。**

---

## 4. Geminiの最終指示（決定済み・変更不可）

1. **GLB生成方式**: 公式`generate_glb`パイプラインを厳密に踏襲。自作スクリプトには絶対に戻さない
2. **enlarge_ratio**: `[1.0, 1.0]`を採用（公式のタイプミスに合わせない）
3. **vertex_order**: 公式の「OBJ + 90°回転 + ローカルZ値ソート」が絶対の正解
4. **マテリアル除去**: 不要（公式に合わせる）

---

## 5. 前セッションで行った修正の状態

### concierge_modal.py への修正（コミット済み）

| # | 修正内容 | 正しいか |
|---|---------|---------|
| F1 | `prepare_motion_seqs`に`max_squen_length=300`追加 | **正しい** |
| F2 | offset.plyの保存順序を公式に合わせた（mesh→ply→glb→anim） | **正しい** |
| F3 | `base_iid`を`chatting_avatar_YYYYMMDDHHMMSS`に変更 | **正しい** |
| D2 | モデルロードを`load_state_dict(strict=False)`に変更 | **間違い** — 公式は`from_pretrained`方式 |

### lam_avatar_batch.py への修正（コミット済み）

| # | 修正内容 | 正しいか |
|---|---------|---------|
| F3 | `base_iid`を`chatting_avatar_YYYYMMDDHHMMSS`に変更 | **正しい** |
| コメント修正 | OAC生成順序のコメント修正 | **正しい** |

### D2の問題

concierge_modal.pyの`_init_lam_pipeline`内で、モデルロードを手動copy_方式から`load_state_dict(strict=False)`に変更した。

しかし公式app.pyは:
```python
hf_model_cls = wrap_model_hub(model_dict["lam"])
model = hf_model_cls.from_pretrained(cfg.model_name)
```

`from_pretrained`方式。`load_state_dict`とは異なる。
Modal環境で`from_pretrained`が使えるかどうかの検証が必要。

---

## 6. 未対応の課題

### 優先度高
- **D2の修正が間違っている** — `from_pretrained`方式への変更、またはModal環境での互換方法の検討が必要
- **ZIP作成方法** — 現在`zipfile.ZipFile`だが公式は`os.system('zip -r ...')`。lam_avatar_batch.pyは既に`os.system`方式

### 優先度中
- **`add_audio_to_video`の10秒制限** — 公式にはあるがconcierge_modal.pyにはない

### 確認が必要
- **`from_pretrained`がModal環境で動作するか** — HuggingFace Hubからのダウンロードではなくローカルパスからのロードになるため
- **`lam.eval()`の呼び出し** — 公式にはないが、`from_pretrained`内部で呼ばれている可能性

---

## 7. ファイル構成

| ファイル | 役割 | 実行方法 |
|---------|------|---------|
| `concierge_modal.py` | Modal上のGradio UIサーバー | `modal serve concierge_modal.py` |
| `lam_avatar_batch.py` | **Modal上のバッチ処理（ユーザーが実際に使う）** | `modal run lam_avatar_batch.py --image-path ./input/input.jpg --param-json-path ./input/params.json` |
| `app_lam.py` | ローカルGradioアプリ（非公式コピー） | `python app_lam.py` |
| `app_concierge.py` | HF Spaces版（自作スクリプト使用・鳥の化け物バグあり） | 使わない |

**ユーザーが実行するのは`lam_avatar_batch.py`。**
`lam_avatar_batch.py`は`concierge_modal.py`の`_init_lam_pipeline`と`image`をインポートして使う。

### input/params.json
```json
{
  "shape_scale": 1.0,
  "motion_name": "talk"
}
```
※ `shape_scale`はコード内で未使用

---

## 8. gitコミット履歴（このセッション）

```
872bbc3 fix: correct OAC generation order comment in lam_avatar_batch.py
bd1ed9d fix: align Modal pipeline with official app.py (Choice A - Gemini directive)
11a1c7e docs: update modification plan v2 with app_concierge.py comparison
b46b443 docs: add modification plan for aligning Modal files with official ModelScope app.py
```

---

## 9. 参照すべきドキュメント

- `MODIFICATION_PLAN_20260228.md` — 修正計画書（v2、一部古い情報あり）
- この引継文（`SESSION_HANDOFF_20260228.md`）— 最新の正確な状態
