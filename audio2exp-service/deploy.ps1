# Audio2Expression Service デプロイスクリプト (PowerShell)

# 設定
$PROJECT_ID = "hp-support-477512"
$SERVICE_NAME = "audio2exp-service"
$REGION = "us-central1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Audio2Expression Service デプロイ" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# デバッグ: 変数確認
Write-Host "PROJECT_ID: $PROJECT_ID" -ForegroundColor Gray
Write-Host "IMAGE_NAME: $IMAGE_NAME" -ForegroundColor Gray
Write-Host ""

# 1. イメージビルド
Write-Host "[1/3] Dockerイメージをビルド中..." -ForegroundColor Yellow
gcloud builds submit --tag "$IMAGE_NAME" --project "$PROJECT_ID"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ビルドに失敗しました" -ForegroundColor Red
    exit 1
}
Write-Host "ビルド完了" -ForegroundColor Green
Write-Host ""

# 2. Cloud Runにデプロイ
Write-Host "[2/3] Cloud Runにデプロイ中..." -ForegroundColor Yellow
gcloud run deploy "$SERVICE_NAME" `
    --image "$IMAGE_NAME" `
    --platform managed `
    --region "$REGION" `
    --allow-unauthenticated `
    --memory 1Gi `
    --cpu 1 `
    --timeout 300 `
    --max-instances 10 `
    --project "$PROJECT_ID"

if ($LASTEXITCODE -ne 0) {
    Write-Host "デプロイに失敗しました" -ForegroundColor Red
    exit 1
}
Write-Host "デプロイ完了" -ForegroundColor Green
Write-Host ""

# 3. URLを取得
Write-Host "[3/3] サービスURLを取得中..." -ForegroundColor Yellow
$SERVICE_URL = gcloud run services describe "$SERVICE_NAME" `
    --region "$REGION" `
    --format 'value(status.url)' `
    --project "$PROJECT_ID"

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "デプロイが完了しました!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "サービスURL: $SERVICE_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "次のステップ:" -ForegroundColor Cyan
Write-Host "1. gourmet-support に環境変数を追加:"
Write-Host "   AUDIO2EXP_SERVICE_URL=$SERVICE_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. gourmet-support の deploy.ps1 を修正して再デプロイ"
Write-Host ""
