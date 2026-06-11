#!/bin/bash
# Local run script for audio2exp-service with real LAM Audio2Expression model
# This script sets up the correct paths for the OpenAvatarChat models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set environment variables for model paths
export LAM_A2E_PATH="$PROJECT_ROOT/OpenAvatarChat/src/handlers/avatar/lam/LAM_Audio2Expression"
export LAM_WEIGHT_PATH="$PROJECT_ROOT/OpenAvatarChat/models/LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar"
export WAV2VEC_PATH="$PROJECT_ROOT/OpenAvatarChat/models/wav2vec2-base-960h"

echo "========================================"
echo "Audio2Expression Service - Local Mode"
echo "========================================"
echo "LAM_A2E_PATH: $LAM_A2E_PATH"
echo "LAM_WEIGHT_PATH: $LAM_WEIGHT_PATH"
echo "WAV2VEC_PATH: $WAV2VEC_PATH"
echo "========================================"

# Check if paths exist
if [ ! -d "$LAM_A2E_PATH" ]; then
    echo "ERROR: LAM_Audio2Expression not found at $LAM_A2E_PATH"
    echo "Run: cd $PROJECT_ROOT/OpenAvatarChat && git submodule update --init src/handlers/avatar/lam/LAM_Audio2Expression"
    exit 1
fi

if [ ! -f "$LAM_WEIGHT_PATH" ]; then
    echo "ERROR: Model weights not found at $LAM_WEIGHT_PATH"
    echo "Run: wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar -P $PROJECT_ROOT/OpenAvatarChat/models/LAM_audio2exp/"
    echo "     tar -xzvf $PROJECT_ROOT/OpenAvatarChat/models/LAM_audio2exp/LAM_audio2exp_streaming.tar -C $PROJECT_ROOT/OpenAvatarChat/models/LAM_audio2exp"
    exit 1
fi

if [ ! -d "$WAV2VEC_PATH" ]; then
    echo "ERROR: wav2vec2 model not found at $WAV2VEC_PATH"
    echo "Run: git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/wav2vec2-base-960h.git $WAV2VEC_PATH"
    exit 1
fi

# Run the service
cd "$SCRIPT_DIR"
python app.py --host 0.0.0.0 --port 8283
