#!/bin/bash
# support-base リポへの初期プッシュスクリプト
#
# 使い方:
#   1. LAM_gpro をローカルにクローン (platform-design-docs ブランチ)
#   2. このスクリプトを実行
#
#   git clone -b claude/platform-design-docs-oEVkm https://github.com/mirai-gpro/LAM_gpro.git
#   cd LAM_gpro
#   bash scripts/push_support_base.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR=$(mktemp -d)

echo "=== support-base リポへプッシュ ==="
echo "作業ディレクトリ: $WORK_DIR"

# 1. support-base リポをクローン
echo "[1/4] support-base をクローン..."
git clone https://github.com/mirai-gpro/support-base.git "$WORK_DIR/support-base"

# 2. ファイルをコピー
echo "[2/4] ファイルをコピー..."
TARGET="$WORK_DIR/support-base"

# ルートファイル
cp "$REPO_ROOT/support_base/Dockerfile" "$TARGET/"
cp "$REPO_ROOT/support_base/requirements.txt" "$TARGET/"
cp "$REPO_ROOT/support_base/cloudbuild.yaml" "$TARGET/"

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

# Python パッケージ
mkdir -p "$TARGET/support_base"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='Dockerfile' --exclude='requirements.txt' --exclude='cloudbuild.yaml' \
  "$REPO_ROOT/support_base/" "$TARGET/support_base/"

# Dockerfile の COPY パスを修正
sed -i 's|^COPY \. \./support_base/$|COPY support_base/ ./support_base/|' "$TARGET/Dockerfile"

# cloudbuild.yaml の dir を削除（リポルートからビルド）
sed -i "/^    dir: 'support_base'$/d" "$TARGET/cloudbuild.yaml"

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
