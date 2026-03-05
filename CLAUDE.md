# CLAUDE.md — Claude Code への必読ルール

**このファイルはClaude Codeがセッション開始時に自動で読む。必ず従え。**

---

## 絶対禁止事項（過去10回以上の改ざん事故あり）

### 1. ModelScope公式wheelsをGitHubソースビルドに差し替えるな

`app_modal.py` の wheels セクションは **公式ModelScope環境からDLしたプリビルド .whl** を使っている。
**絶対にGitHubのURLやソースビルドに変更するな。**

対象パッケージ:
- `diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl`
- `simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl`
- `nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl`
- `pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl`
- `fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl`

**理由**: GitHubソースからのビルドはModelScopeのプリビルドと **バイナリレベルで異なる** 動作をする。
過去にClaudeが「GitHubが正しいはず」と判断してwheelsをソースビルドに差し替え、
3Dアバターが「鳥の化け物」になった。150時間以上のデバッグ時間が無駄になった。

### 2. xformers をインストールするな

公式ModelScope app.py は `pip uninstall -y xformers` を明示的に実行している。
xformers が存在すると DINOv2 エンコーダが異なる attention パスを使い、出力が変わる。

**xformers のインストール行を追加するな。削除されているのは意図的。**

### 3. numpy のバージョンを上げるな

公式は `numpy==1.23.0`。新しいバージョンへの変更禁止。

---

## 必ず読むべきドキュメント（変更前に）

1. `docs/POSTMORTEM_BIRD_MONSTER_20260305.md` — 鳥の化け物問題の全経緯
2. `HANDOFF_MODELSCOPE.md` — ModelScope仕様
3. `SESSION_HANDOFF_20260228.md` — 公式app.py準拠の警告
4. `GEMINI_INSTRUCTION.md` — Modal移行の全体設計

**ドキュメントを読まずに「一般知識」でコードを変更することは禁止。**

---

## プロジェクト概要

LAM (Large Avatar Model) を Modal クラウドにデプロイするプロジェクト。
公式ModelScopeデモと完全に同一の動作を再現することが目標。

- 公式ソース: `git show origin/lam-large-upload:LAM_Large_Avatar_Model/app.py`
- Modal版: `app_modal.py` → `lam_avatar_batch.py`
- 実行: `modal run --force-build lam_avatar_batch.py --image-path ./input/input.jpg`

---

## 変更時のチェックリスト

app_modal.py を変更する前に:
- [ ] 公式ModelScope app.py で同じ処理がどうなっているか確認したか？
- [ ] wheels のインストール方法を変えていないか？
- [ ] xformers を追加していないか？
- [ ] numpy バージョンを変えていないか？
- [ ] GitHubソースビルドを追加していないか？
