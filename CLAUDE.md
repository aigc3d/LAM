# Project Notes

## audio2exp-service デプロイ

### Cloud Run デプロイコマンド (PowerShell)
```powershell
gcloud run deploy audio2exp-service --source . --region us-central1 --memory 4Gi --cpu 2 --timeout 120 --min-instances 1 --max-instances 3 --set-env-vars "MODEL_DIR=/app/models,DEVICE=cpu"
```

### 重要な決定事項
- **メモリは4Gi必須**: 2Giでは3回メモリ不足で失敗。4Giで成功。2Giに戻さないこと。
- `--source .` でソースからビルド（Artifact Registry使用）
- `min-instances=1` でコールドスタート排除（Wav2Vec2ロードに時間がかかるため）

### ルール
- 重要な決定・変更は発生時点で即座にこのファイルに記録すること
- 推測で回答せず、必ずファイルや記録を確認してから回答すること
