# ポストモーテム: 鳥の化け物（Bird Monster）問題

**日付**: 2026-03-05
**ステータス**: 修正適用済み、検証待ち
**影響範囲**: `app_modal.py` (Modal Image Build)
**累計デバッグ時間**: 150時間以上（ChatGPT, Gemini, Claude Code across 複数セッション）

---

## 概要

LAM (Large Avatar Model) をModalにデプロイした際、入力画像（日本人女性の実写）から
生成される3Dアバターが「鳥の化け物」のように著しく崩壊する問題が発生。
公式ModelScopeデモでは同一画像から正常な結果が得られるにも関わらず、
Modal版では一貫して異常な出力となった。

---

## 根本原因

**過去のClaudeセッションが、公式ModelScopeのwheels（プリビルドバイナリ）を
GitHubソースビルドに勝手に差し替えたこと。**

Claudeの「GitHubの公式ソースからビルドするのが正しいはず」という
誤った一般知識に基づく独断的な改ざんが原因。

ドキュメント（`HANDOFF_MODELSCOPE.md`, `SESSION_HANDOFF_20260228.md`）には
wheels使用が明記されていたが、読まずに変更された。

---

## 具体的な改ざん箇所と修正

### 1. xformers（最重要 — 鳥の化け物の最有力原因）

| | 公式ModelScope app.py | 改ざん後のModal版 | 修正後 |
|---|---|---|---|
| xformers | `pip uninstall -y xformers` | `pip install xformers==0.0.27.post2` | インストールしない + 明示的アンインストール |

**なぜ致命的か**:
- DINOv2エンコーダは xformers の有無で**異なるattention実装**を使う
  - xformersあり → `xformers.ops.memory_efficient_attention`
  - xformersなし → PyTorchネイティブattention
- 同じ重みファイルでも出力テンソルが微妙に異なる
- Gaussian Splatパラメータが狂い → メッシュ崩壊 → 鳥の化け物

### 2. nvdiffrast

| | 公式ModelScope | 改ざん後 | 修正後 |
|---|---|---|---|
| nvdiffrast | `./wheels/nvdiffrast-0.3.3.whl` | GitHubソースビルド | wheelsのwhlを使用 |

### 3. diff_gaussian_rasterization / simple_knn

| | 公式ModelScope | 改ざん後（一部） | 修正後 |
|---|---|---|---|
| diff_gaussian_rasterization | `./wheels/` のwhl | GitHubソースビルド（concierge_modal.py） | wheelsのwhlを使用 |
| simple_knn | `./wheels/` のwhl | GitHubソースビルド（concierge_modal.py） | wheelsのwhlを使用 |

### 4. numpy

| | 公式ModelScope | 改ざん後 | 修正後 |
|---|---|---|---|
| numpy | `1.23.0` | `1.26.4` | `1.23.0` |

---

## 修正内容（app_modal.py）

```
コミット: claude/fix-modelscope-wheels-mpGPD ブランチ
```

1. **xformersインストールを削除** — コメントで理由を明記
2. **nvdiffrast GitHubクローン+ビルドを削除** — wheelsから統一インストール
3. **全wheelsを公式app.pyと同じ順序・フラグで個別インストール**:
   - `diff_gaussian_rasterization-0.0.0 --force-reinstall`
   - `simple_knn-0.0.0 --force-reinstall`
   - `nvdiffrast-0.3.3 --force-reinstall`
   - `pytorch3d-0.7.8 --force-reinstall`
   - `fbx-2020.3.4 --force-reinstall`
4. **wheels後にxformersを明示的にアンインストール**（依存で入った場合の保険）
5. **numpyを1.23.0に変更**

---

## 必要なwheelsファイル（ローカル `C:/Users/hamad/LAM/wheels/` に配置）

すべて公式ModelScope環境からダウンロードしたもの:

- `pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl`
- `diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl`
- `simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl`
- `nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl`
- `fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl`

---

## 教訓

### AIアシスタントに対する警告

1. **ドキュメントを読め** — プロジェクトにはハンドオフ文書、指示書が存在する。
   一般知識で推測するな。
2. **「GitHubが正義」は嘘** — プリビルドwheelsには、ソースビルドでは再現できない
   環境固有のチューニングが含まれている場合がある。
3. **勝手に差し替えるな** — 「こっちの方が良い」という判断で動いているものを変えるな。
   特にバイナリレベルのパッケージは、ソースが同じでもビルド環境で動作が変わる。
4. **公式が動いているなら公式をトレースしろ** — 改良は動いてから。

### 技術的教訓

- xformersの有無でDINOv2の出力が変わることは、公式ドキュメントには明記されていない。
  しかし公式が `pip uninstall -y xformers` している事実が全てを語る。
- ModelScope環境のwheelsは、特定のCUDAバージョン・PyTorchバージョン・
  コンパイラフラグの組み合わせでビルドされている。
  GitHubソースからの再ビルドは同一バイナリを保証しない。

---

## 検証手順

```bash
# ローカルPCで実行（wheelsディレクトリにwhlファイルがあること）
modal run --force-build lam_avatar_batch.py --image-path ./input/input.jpg
```

成功基準: 入力画像の人物に似た、自然な3Dアバターが生成されること。
「鳥の化け物」にならないこと。

---

## 関連ファイル

- `app_modal.py` — 修正対象（Modal Image Build定義）
- `lam_avatar_batch.py` — バッチ実行エントリポイント（app_modal.pyのimageを使用）
- `concierge_modal.py` — Gradio UI版（別途修正が必要な可能性あり）
- `app_hf_space.py` — HuggingFace Space版（参考用）
- `HANDOFF_MODELSCOPE.md` — ModelScope仕様書
- `SESSION_HANDOFF_20260228.md` — 前回セッション引き継ぎ
