#!/bin/bash
# support-base リポへの初期プッシュスクリプト (Windows Git Bash 対応)
#
# 使い方:
#   git clone -b claude/platform-design-docs-oEVkm https://github.com/mirai-gpro/LAM_gpro.git
#   cd LAM_gpro
#   bash scripts/push_support_base.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${TMPDIR:-/tmp}/support-base-work-$$"
mkdir -p "$WORK_DIR"

echo "=== support-base リポへプッシュ ==="

# 1. support-base リポをクローン
echo "[1/4] support-base をクローン..."
git clone https://github.com/mirai-gpro/support-base.git "$WORK_DIR/support-base"

# 2. ファイルをコピー
echo "[2/4] ファイルをコピー..."
TARGET="$WORK_DIR/support-base"

# .gitignore
cat > "$TARGET/.gitignore" << 'GITIGNORE'
__pycache__/
*.pyc
*.pyo
.env
.venv/
*.egg-info/
dist/
build/
GITIGNORE

# Python パッケージをコピー（cp -r で、rsync 不要）
cp -r "$REPO_ROOT/support_base" "$TARGET/support_base"

# __pycache__ を除去
find "$TARGET/support_base" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$TARGET/support_base" -name '*.pyc' -delete 2>/dev/null || true

# ルートに移すべきファイルを移動
mv "$TARGET/support_base/Dockerfile" "$TARGET/"
mv "$TARGET/support_base/requirements.txt" "$TARGET/"
mv "$TARGET/support_base/cloudbuild.yaml" "$TARGET/"

# Dockerfile の COPY パスを修正 (sed -i はGit Bashで動作)
sed -i 's|^COPY \. \./support_base/$|COPY support_base/ ./support_base/|' "$TARGET/Dockerfile"

# cloudbuild.yaml: dir 行を削除、トリガーコメントを更新
sed -i "/^    dir: 'support_base'$/d" "$TARGET/cloudbuild.yaml"
sed -i 's|--repo-name=LAM_gpro|--repo-name=support-base|' "$TARGET/cloudbuild.yaml"
sed -i "s|--build-config=support_base/cloudbuild.yaml|--build-config=cloudbuild.yaml|" "$TARGET/cloudbuild.yaml"
sed -i '/--included-files=/d' "$TARGET/cloudbuild.yaml"
sed -i 's|cd support_base|# ルートから実行|' "$TARGET/cloudbuild.yaml"
sed -i 's|--config=cloudbuild.yaml \.|--config=cloudbuild.yaml .|' "$TARGET/cloudbuild.yaml"

# 3. コミット
echo "[3/4] コミット..."
cd "$TARGET"
git add -A
git commit -m "feat: support-base プラットフォーム初期構成

LAM_gpro/support_base から独立リポジトリとして移植。
- FastAPI + Gemini Live API WebSocket 中継
- REST エンドポイント (Flask→FastAPI 変換済み)
- プラグインアーキテクチャ (gourmet モード)
- Cloud Run デプロイ設定"

# 4. プッシュ
echo "[4/4] プッシュ..."
git push -u origin main

echo ""
echo "=== 完了 ==="
echo "https://github.com/mirai-gpro/support-base"

# 後片付け
rm -rf "$WORK_DIR"
