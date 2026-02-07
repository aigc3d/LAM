"""
Audio2Expression Service - Simplified and Robust Version
"""

import asyncio
import base64
import json
import os
import struct
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths - these are baked into the Docker image
LAM_A2E_PATH = os.environ.get("LAM_A2E_PATH", os.path.join(SCRIPT_DIR, "LAM_Audio2Expression"))
LAM_WEIGHT_PATH = os.environ.get("LAM_WEIGHT_PATH", os.path.join(SCRIPT_DIR, "models", "LAM_audio2exp_streaming.tar"))
WAV2VEC_PATH = os.environ.get("WAV2VEC_PATH", os.path.join(SCRIPT_DIR, "models", "wav2vec2-base-960h"))

# Add LAM path to sys.path
if os.path.exists(LAM_A2E_PATH):
    sys.path.insert(0, LAM_A2E_PATH)
    print(f"[Init] LAM_A2E_PATH: {LAM_A2E_PATH} (exists)")
else:
    print(f"[Init] LAM_A2E_PATH: {LAM_A2E_PATH} (NOT FOUND)")

# Global initialization status
init_status = {
    "step": "not_started",
    "error": None,
    "traceback": None
}

# ARKit 52 channel names
ARKIT_CHANNELS = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut"
]


class Audio2ExpressionEngine:
    """Simplified Audio2Expression wrapper"""

    def __init__(self):
        self.infer = None
        self.initialized = False
        self.model_sample_rate = 16000
        self.input_sample_rate = 24000
        self.init_error = None

    def initialize(self):
        """Initialize the model with step-by-step error tracking"""
        global init_status

        if self.initialized:
            return True

        try:
            # Step 1: Check LAM_A2E_PATH
            init_status["step"] = "check_lam_path"
            if not os.path.exists(LAM_A2E_PATH):
                raise FileNotFoundError(f"LAM_A2E_PATH not found: {LAM_A2E_PATH}")
            print(f"[Init] Step 1: LAM_A2E_PATH OK")

            # Step 2: Check model weights
            init_status["step"] = "check_weights"
            if not os.path.exists(LAM_WEIGHT_PATH):
                raise FileNotFoundError(f"Model weights not found: {LAM_WEIGHT_PATH}")
            print(f"[Init] Step 2: Model weights OK ({os.path.getsize(LAM_WEIGHT_PATH)} bytes)")

            # Step 3: Check wav2vec
            init_status["step"] = "check_wav2vec"
            if not os.path.exists(WAV2VEC_PATH):
                raise FileNotFoundError(f"wav2vec2 not found: {WAV2VEC_PATH}")
            print(f"[Init] Step 3: wav2vec2 OK")

            # Step 4: Import modules
            init_status["step"] = "import_modules"
            print("[Init] Step 4: Importing LAM modules...")
            from engines.defaults import default_config_parser, default_setup
            from engines.infer import INFER
            print("[Init] Step 4: Import OK")

            # Step 5: Setup paths
            init_status["step"] = "setup_paths"
            config_file = os.path.join(LAM_A2E_PATH, "configs", "lam_audio2exp_config_streaming.py")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")

            wav2vec_config = os.path.join(LAM_A2E_PATH, "configs", "wav2vec2_config.json")
            if not os.path.exists(wav2vec_config):
                raise FileNotFoundError(f"wav2vec config not found: {wav2vec_config}")

            save_path = os.path.join(SCRIPT_DIR, "exp", "audio2exp")
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "model"), exist_ok=True)
            print(f"[Init] Step 5: Paths OK")

            # Step 6: Build config
            init_status["step"] = "build_config"
            print("[Init] Step 6: Building config...")
            cfg_options = {
                "weight": LAM_WEIGHT_PATH,
                "save_path": save_path,
                "model": {
                    "backbone": {
                        "wav2vec2_config_path": wav2vec_config,
                        "pretrained_encoder_path": WAV2VEC_PATH,
                    }
                }
            }
            cfg = default_config_parser(config_file, cfg_options)
            print("[Init] Step 6: Config parsed")

            # Step 7: Setup config
            init_status["step"] = "setup_config"
            cfg = default_setup(cfg)
            print("[Init] Step 7: Config setup OK")

            # Step 8: Build model
            init_status["step"] = "build_model"
            print("[Init] Step 8: Building model (this may take a while)...")
            self.infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
            self.infer.model.eval()
            print("[Init] Step 8: Model built OK")

            # Step 9: Warmup
            init_status["step"] = "warmup"
            print("[Init] Step 9: Running warmup...")
            self.infer.infer_streaming_audio(
                audio=np.zeros([self.input_sample_rate], dtype=np.float32),
                ssr=self.input_sample_rate,
                context=None
            )
            print("[Init] Step 9: Warmup OK")

            self.initialized = True
            init_status["step"] = "completed"
            print("[Init] Model initialized successfully!")
            return True

        except Exception as e:
            self.init_error = str(e)
            init_status["error"] = str(e)
            init_status["traceback"] = traceback.format_exc()
            print(f"[Init] FAILED at step '{init_status['step']}': {e}")
            traceback.print_exc()
            return False

    def process_audio(self, audio: np.ndarray, context=None, ssr: int = None):
        if ssr is None:
            ssr = self.input_sample_rate

        if not self.initialized or self.infer is None:
            return self._mock_expression(audio), context

        result, new_context = self.infer.infer_streaming_audio(audio=audio, ssr=ssr, context=context)
        expression = result.get("expression")
        if expression is None:
            return None, new_context
        return expression.astype(np.float32), new_context

    def process_full_audio(self, audio: np.ndarray, sample_rate: int = None, frame_rate: int = 30):
        if sample_rate is None:
            sample_rate = self.input_sample_rate

        if not self.initialized or self.infer is None:
            return self._mock_expression(audio, frame_rate, sample_rate)

        chunk_samples = sample_rate
        all_expressions = []
        context = None

        for start in range(0, len(audio), chunk_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            if len(chunk) < sample_rate // 10:
                continue
            expression, context = self.process_audio(chunk, context, ssr=sample_rate)
            if expression is not None and len(expression) > 0:
                all_expressions.append(expression)

        if not all_expressions:
            audio_duration = len(audio) / sample_rate
            expected_frames = int(audio_duration * frame_rate)
            return np.zeros((max(1, expected_frames), 52), dtype=np.float32)

        return np.concatenate(all_expressions, axis=0).astype(np.float32)

    def _mock_expression(self, audio: np.ndarray, frame_rate: int = 30, sample_rate: int = None):
        if sample_rate is None:
            sample_rate = self.input_sample_rate
        if len(audio) == 0:
            return np.zeros((1, 52), dtype=np.float32)

        samples_per_frame = sample_rate // frame_rate
        num_frames = max(1, len(audio) // samples_per_frame)
        expression = np.zeros((num_frames, 52), dtype=np.float32)

        jaw_open_idx = ARKIT_CHANNELS.index("jawOpen")
        mouth_lower_left = ARKIT_CHANNELS.index("mouthLowerDownLeft")
        mouth_lower_right = ARKIT_CHANNELS.index("mouthLowerDownRight")

        for i in range(num_frames):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio))
            rms = np.sqrt(np.mean(audio[start:end] ** 2)) if end > start else 0
            expression[i, jaw_open_idx] = min(rms * 3, 0.5)
            expression[i, mouth_lower_left] = min(rms * 4, 0.6)
            expression[i, mouth_lower_right] = min(rms * 4, 0.6)

        return expression


# Global engine
engine = Audio2ExpressionEngine()
active_connections: Dict[str, WebSocket] = {}
session_batch_ids: Dict[str, int] = {}


class AudioRequest(BaseModel):
    audio_base64: str
    session_id: str
    is_start: bool = False
    is_final: bool = False
    audio_format: str = "pcm"
    sample_rate: int = 24000


class ExpressionResponse(BaseModel):
    session_id: str
    names: List[str]
    frames: List[dict]
    frame_rate: int = 30
    timestamp: float
    batch_id: int = 0


def create_jbin_bundle(expression, audio, expression_sample_rate=30, audio_sample_rate=24000, batch_id=0, start_of_batch=False, end_of_batch=False):
    if audio.dtype == np.float32:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)

    expression_f32 = np.ascontiguousarray(expression.astype(np.float32))
    expression_bytes = expression_f32.tobytes()
    audio_bytes = audio_int16.tobytes()

    descriptor = {
        "data_records": {
            "arkit_face": {
                "data_type": "float32", "data_offset": 0,
                "shape": list(expression_f32.shape),
                "channel_names": ARKIT_CHANNELS, "sample_rate": expression_sample_rate,
            },
            "avatar_audio": {
                "data_type": "int16", "data_offset": len(expression_bytes),
                "shape": [1, len(audio_int16)], "sample_rate": audio_sample_rate,
            }
        },
        "batch_id": batch_id, "start_of_batch": start_of_batch, "end_of_batch": end_of_batch
    }

    json_bytes = json.dumps(descriptor).encode('utf-8')
    header = b'JBIN' + struct.pack('<I', len(json_bytes)) + struct.pack('<I', len(expression_bytes) + len(audio_bytes))
    return header + json_bytes + expression_bytes + audio_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("[Startup] Audio2Expression Service Starting...")
    print(f"[Startup] LAM_A2E_PATH: {LAM_A2E_PATH}")
    print(f"[Startup] LAM_WEIGHT_PATH: {LAM_WEIGHT_PATH}")
    print(f"[Startup] WAV2VEC_PATH: {WAV2VEC_PATH}")
    print("=" * 60)

    # Initialize in background
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.initialize)

    print(f"[Startup] Initialization result: {'SUCCESS' if engine.initialized else 'FAILED'}")
    if not engine.initialized:
        print(f"[Startup] Error: {init_status.get('error')}")
    print("=" * 60)

    yield
    print("[Shutdown] Audio2Expression Service stopping...")


app = FastAPI(title="Audio2Expression Service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_initialized": engine.initialized,
        "mode": "inference" if engine.initialized else "mock",
        "init_step": init_status["step"],
        "init_error": init_status["error"]
    }


@app.get("/debug")
async def debug_info():
    return {
        "lam_a2e_path": LAM_A2E_PATH,
        "lam_a2e_exists": os.path.exists(LAM_A2E_PATH),
        "lam_weight_path": LAM_WEIGHT_PATH,
        "lam_weight_exists": os.path.exists(LAM_WEIGHT_PATH),
        "lam_weight_size": os.path.getsize(LAM_WEIGHT_PATH) if os.path.exists(LAM_WEIGHT_PATH) else 0,
        "wav2vec_path": WAV2VEC_PATH,
        "wav2vec_exists": os.path.exists(WAV2VEC_PATH),
        "engine_initialized": engine.initialized,
        "init_status": init_status,
        "script_dir_contents": os.listdir(SCRIPT_DIR) if os.path.exists(SCRIPT_DIR) else [],
        "models_dir_contents": os.listdir(os.path.join(SCRIPT_DIR, "models")) if os.path.exists(os.path.join(SCRIPT_DIR, "models")) else [],
    }


@app.post("/api/audio2expression", response_model=ExpressionResponse)
async def process_audio_endpoint(request: AudioRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        session_id = request.session_id
        if request.is_start or session_id not in session_batch_ids:
            session_batch_ids[session_id] = session_batch_ids.get(session_id, 0) + 1

        expression = engine.process_full_audio(audio_float, sample_rate=request.sample_rate)
        frames = [{"weights": row} for row in expression.tolist()]

        return ExpressionResponse(
            session_id=session_id, names=ARKIT_CHANNELS, frames=frames,
            frame_rate=30, timestamp=time.time(), batch_id=session_batch_ids.get(session_id, 0)
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.pop(session_id, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
