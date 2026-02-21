#!/bin/bash
# Fix the corrupted model file in GCS
# The issue: GCS has a 356MB file but the correct file is 390MB

set -e

# Correct model file path
LOCAL_MODEL="/home/user/LAM_gpro/OpenAvatarChat/models/LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar"

# GCS destination
GCS_BUCKET="gs://hp-support-477512-models/audio2exp"

echo "=== Fixing GCS Model File ==="
echo ""

# Check local file
echo "[1/4] Checking local model file..."
if [ ! -f "$LOCAL_MODEL" ]; then
    echo "ERROR: Local model file not found: $LOCAL_MODEL"
    exit 1
fi

LOCAL_SIZE=$(stat -c%s "$LOCAL_MODEL" 2>/dev/null || stat -f%z "$LOCAL_MODEL")
echo "Local file size: $LOCAL_SIZE bytes ($(numfmt --to=iec $LOCAL_SIZE 2>/dev/null || echo "$LOCAL_SIZE"))"
echo "Local file hash: $(md5sum "$LOCAL_MODEL" | cut -d' ' -f1)"

# Verify local file works with PyTorch
echo ""
echo "[2/4] Verifying local file works with PyTorch..."
python3 -c "
import torch
checkpoint = torch.load('$LOCAL_MODEL', map_location='cpu', weights_only=False)
print(f'SUCCESS: Loaded checkpoint with {len(checkpoint[\"state_dict\"])} parameters')
" || { echo "ERROR: Local file is invalid"; exit 1; }

# Upload to GCS
echo ""
echo "[3/4] Uploading correct model to GCS..."
echo "Destination: $GCS_BUCKET/lam_audio2exp_streaming.tar"
gsutil cp "$LOCAL_MODEL" "$GCS_BUCKET/lam_audio2exp_streaming.tar"

# Verify upload
echo ""
echo "[4/4] Verifying GCS file..."
gsutil ls -la "$GCS_BUCKET/lam_audio2exp_streaming.tar"

echo ""
echo "=== DONE ==="
echo "The correct model file has been uploaded to GCS."
echo "Now redeploy the Cloud Run service to use the fixed model."
echo ""
echo "To redeploy, run:"
echo "  cd /home/user/LAM_gpro/audio2exp-service"
echo "  gcloud builds submit --config=cloudbuild.yaml"
