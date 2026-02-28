# 修正計画書: Modal版をModelScope公式app.pyに準拠させる
**作成日**: 2026-02-28
**作成者**: ClaudeCode（ユーザー・他AI確認用）
**対象ブランチ**: `claude/update-lam-modelscope-UQKxj`

---

## 1. 修正目的

ModelScope公式の `LAM_Large_Avatar_Model/app.py`（正常動作確認済み）のロジックを、
Modal実行環境の `concierge_modal.py` と `lam_avatar_batch.py` に正確に移植する。

---

## 2. Geminiコメントの独自検証結果

### Geminiの主張
> 「自作Blenderスクリプト (convert_and_order.py) が鳥のばけもの化の元凶」

### 検証結果: 現在のコードでは**既に修正済み**

現在の `concierge_modal.py:784` と `lam_avatar_batch.py:308` は、
いずれも公式の `tools.generateARKITGLBWithBlender.generate_glb` を使用しています。
Geminiのコメントは**過去の `app_concierge.py`**（現存しない古いファイル）に対する分析です。

ただし、Geminiの指摘の核心（「公式の generate_glb 関数を使え」）は正しく、
現在のコードは既にその方針に沿っています。

---

## 3. 公式app.pyとModal版の差異一覧（行単位比較）

以下は、公式 `LAM_Large_Avatar_Model/app.py` と現在の Modal版を
行単位で比較して発見した**実際の差異**です。

### 3.1 `prepare_motion_seqs` の `enlarge_ratio` パラメータ

| 項目 | 公式 app.py (382行目) | concierge_modal.py (753行目) | lam_avatar_batch.py (249行目) |
|------|----------------------|------------------------------|-------------------------------|
| enlarge_ratio | `[1.0, 1, 0]` | `[1.0, 1.0]` | `[1.0, 1.0]` |

**分析**:
- 公式の `[1.0, 1, 0]` は **Python構文上 `[1.0, 1, 0]` = 3要素のリスト** です。
- これはタイプミスの可能性が高い（`[1.0, 1.0]` の意図で `1.0` を `1, 0` と書き損じ）
- 仮に意図的な3要素だとしても、`prepare_motion_seqs`の内部実装次第で挙動が変わる

**対応方針**:
要確認事項。公式がこの値で正常動作しているなら、`[1.0, 1, 0]` に合わせるべきか判断が必要。
→ **ユーザー/Geminiに判断を委ねる**

### 3.2 `prepare_motion_seqs` の `max_squen_length` パラメータ

| 項目 | 公式 app.py (386行目) | concierge_modal.py | lam_avatar_batch.py (253行目) |
|------|----------------------|---------------------|-------------------------------|
| max_squen_length | `300` | **指定なし（デフォルト値に依存）** | `300` |

**分析**: concierge_modal.py の `prepare_motion_seqs` 呼び出し（753行目付近）に `max_squen_length=300` が**ない**。

**対応方針**: concierge_modal.py に `max_squen_length=300` を追加する。

### 3.3 `save_imgs_2_video` のフレーム数検証

| 項目 | 公式 app.py (207-222行目) | concierge_modal.py | lam_avatar_batch.py |
|------|--------------------------|---------------------|---------------------|
| フレーム数検証 | `cv2.VideoCapture` で保存後のフレーム数を検証 | なし | なし |

**分析**: 公式はビデオ保存後に `cv2.VideoCapture` でフレーム数が一致するか確認している。

**対応方針**: 品質検証として有用だが、ロジックの正確性には影響なし。低優先度。

### 3.4 `add_audio_to_video` の10秒制限

| 項目 | 公式 app.py (236-237行目) | concierge_modal.py (app_lam.py経由) | lam_avatar_batch.py (90-91行目) |
|------|--------------------------|--------------------------------------|-------------------------------|
| audio_clip 10秒制限 | `if audio_clip.duration > 10: audio_clip = audio_clip.subclip(0, 10)` | **なし** (app_lam.py:108-124 にはこの制限がない) | **あり** |

**分析**:
- 公式は音声を10秒にクリップしている
- concierge_modal.py は `app_lam.py:add_audio_to_video` を使っており、この制限がない
- lam_avatar_batch.py は独自実装で10秒制限あり

**対応方針**: concierge_modal.pyで公式に合わせて10秒制限を追加するか検討。

### 3.5 ZIP作成方法

| 項目 | 公式 app.py (427行目) | concierge_modal.py (809行目) | lam_avatar_batch.py (329行目) |
|------|----------------------|------------------------------|-------------------------------|
| ZIP作成 | `os.system('zip -r ...')` | `zipfile.ZipFile` (Python) | `os.system('zip -r ...')` |

**分析**:
- 公式は `os.system('zip -r {output_zip} {oac_dir}')` でシステムのzipコマンドを使用
- concierge_modal.py は Python の `zipfile` モジュールを使用
- lam_avatar_batch.py は公式と同じ `os.system('zip -r ...')`

**対応方針**:
機能的には同等だが、ZIPの内部構造（ディレクトリエントリの有無、圧縮レベル）が微妙に異なる可能性。
Chatting Avatar のパーサーがZIP構造に厳密なら問題になりうる。
→ **公式と同じ `os.system('zip -r ...')` に統一するか検討**

### 3.6 OAC ZIPの出力パス

| 項目 | 公式 app.py (409-410行目) | concierge_modal.py (776行目) | lam_avatar_batch.py (281行目) |
|------|--------------------------|------------------------------|-------------------------------|
| oac_dir | `os.path.join('./', base_iid)` (カレントディレクトリ直下) | `os.path.join(working_dir, "oac_export", base_iid)` | `os.path.join('./', base_iid)` |

**分析**:
- 公式はカレントディレクトリ直下に `chatting_avatar_YYYYMMDDHHMMSS/` を作成
- concierge_modal.py は `working_dir/oac_export/concierge/` に作成
- ZIPの中身のフォルダ名が変わるため、Chatting Avatar パーサーに影響する可能性

**対応方針**:
ZIPの中身のフォルダ名（arcname）が正しければパスは問題ない。要確認。

### 3.7 `base_iid` の命名

| 項目 | 公式 app.py (318行目) | concierge_modal.py (687行目) | lam_avatar_batch.py (280行目) |
|------|----------------------|------------------------------|-------------------------------|
| base_iid | `'chatting_avatar_' + datetime.now().strftime("%Y%m%d%H%M%S")` | `"concierge"` (固定) | `'avatar_' + datetime.now().strftime("%Y%m%d%H%M%S")` |

**分析**: ZIPの中身のフォルダ名に影響。Chatting Avatarがフォルダ名に依存しなければ無害。

**対応方針**: 公式に合わせて `chatting_avatar_YYYYMMDDHHMMSS` にする。

### 3.8 app_lam.py の `NUMBA_THREADING_LAYER`

| 項目 | 公式 app.py (658行目) | app_lam.py (533行目) |
|------|----------------------|----------------------|
| NUMBA_THREADING_LAYER | `forseq` | `omp` |

**分析**: 公式は `forseq`、ローカルの app_lam.py は `omp`。
concierge_modal.py は既に `forseq` を使用（正しい）。

**対応方針**: app_lam.py は今回のスコープ外（Modal版のみが対象）。

### 3.9 `_build_model` のモデルロード方法

| 項目 | 公式 app.py (641-648行目) | concierge_modal.py (401-481行目) |
|------|--------------------------|----------------------------------|
| ロード方法 | `wrap_model_hub(model_dict["lam"]).from_pretrained(cfg.model_name)` | `ModelLAM(**cfg.model)` + 手動safetensorsロード |

**分析**:
- 公式は `from_pretrained` を使う（HuggingFace Hub形式）
- concierge_modal.py は `ModelLAM(**cfg.model)` で構築後、手動でsafetensorsをロード
- 機能的には同等だが、`from_pretrained` が内部でdtype設定などを行っている可能性

**対応方針**:
これは重大な差異の可能性あり。`from_pretrained` と手動ロードで
重みの精度やデフォルトパラメータが異なる可能性を調査する必要がある。
→ **ユーザー/Geminiに判断を委ねる**

### 3.10 offset.ply の保存順序

| 項目 | 公式 app.py (411行目) | concierge_modal.py (794行目) |
|------|----------------------|------------------------------|
| 順序 | save_shaped_mesh → save_ply (offset) → generate_glb → animation copy | save_shaped_mesh → generate_glb → save_ply (offset) → animation copy |

**分析**:
- 公式: save_shaped_mesh → offset.ply → generate_glb → animation
- concierge_modal.py: save_shaped_mesh → generate_glb → offset.ply → animation
- 順序が異なる。generate_glb が内部で OBJ ファイルを読むため、offset.ply の生成順序自体は無関係のはず
- ただし、generate_glb の中で vertex_order.json が生成されるが、offset.ply の後か前かで影響があるかは要確認

**対応方針**: 公式と同じ順序に揃える（リスクなし）。

---

## 4. 修正計画（優先度順）

### 高優先度（ロジック差異・出力に影響する可能性）

| # | 対象ファイル | 修正内容 | 根拠 |
|---|-------------|---------|------|
| H1 | concierge_modal.py | `prepare_motion_seqs` に `max_squen_length=300` を追加 | 3.2: 公式との差異 |
| H2 | concierge_modal.py | offset.ply の保存順序を公式に合わせる | 3.10: 公式との差異 |
| H3 | concierge_modal.py | `base_iid` を `chatting_avatar_YYYYMMDDHHMMSS` に変更 | 3.7: 公式との差異 |

### 中優先度（動作に微妙な影響の可能性）

| # | 対象ファイル | 修正内容 | 根拠 |
|---|-------------|---------|------|
| M1 | concierge_modal.py | ZIP作成を `os.system('zip -r ...')` に変更するか検討 | 3.5: ZIP構造差異 |
| M2 | 全体 | `enlarge_ratio` パラメータの公式値 `[1.0, 1, 0]` について判断 | 3.1: 不明確 |
| M3 | concierge_modal.py | `_build_model` を公式の `from_pretrained` 方式に合わせるか検討 | 3.9: 重要だが影響範囲大 |

### 低優先度（品質改善、動作には影響なし）

| # | 対象ファイル | 修正内容 | 根拠 |
|---|-------------|---------|------|
| L1 | concierge_modal.py | `add_audio_to_video` に10秒制限追加 | 3.4: 公式との差異 |
| L2 | concierge_modal.py | ビデオ保存後のフレーム数検証追加 | 3.3: 品質チェック |

---

## 5. 判断を委ねる事項（ユーザー/Gemini確認必要）

1. **`enlarge_ratio` の値**: 公式の `[1.0, 1, 0]` はタイプミスか意図的か？
   - タイプミスなら現在の `[1.0, 1.0]` のままでOK
   - 意図的なら `[1.0, 1, 0]` に変更する

2. **`_build_model` 方式**: `from_pretrained` vs 手動ロード
   - `from_pretrained` に合わせると、内部のdtype設定等も公式に準拠する
   - ただし、現在の手動ロード + 明示的な `torch.float32` 設定で問題なければ変更不要

3. **ZIP作成方法**: `zipfile.ZipFile` vs `os.system('zip -r ...')`
   - Chatting AvatarのZIPパーサーが構造に厳密か不明
   - 安全策は公式に合わせること

---

## 6. 修正しない項目（理由付き）

| 項目 | 理由 |
|------|------|
| app_lam.py の NUMBA_THREADING_LAYER | 今回のスコープ外（Modal版のみ対象） |
| torch.compile 無効化処理 | Modal環境固有の対策であり公式には不要 |
| Gradio UIの差異 | UI層は移植対象外 |
| _shape_guard 関数 | 公式にはないが安全装置として有用（削除不要） |

---

## 7. 次のステップ

1. ユーザーと他のAI（Gemini/ChatGPT）がこの計画書を確認
2. 判断委託事項（セクション5）について方針決定
3. 方針決定後、ClaudeCodeが高優先度の修正から順に実装
