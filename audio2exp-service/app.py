"""
Audio2Expression Service for LAM Lip Sync - Cloud Run Optimized
Bypass DDP initialization and use /tmp for all file writes.
"""

import asyncio
import base64
import json
import os
import struct
import sys
import time
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Dict, List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch

# --- 1. Logger setup ---
# Cloud Run restricts filesystem writes. Use stdout only.
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("Audio2Expression")

# --- Path configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cloud Run: GCS FUSE mount path (primary), Docker-baked models (fallback)
MOUNT_PATH = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models")
MODEL_SUBDIR = os.environ.get("MODEL_SUBDIR", "audio2exp")

# LAM module path resolution
LAM_A2E_CANDIDATES = [
    os.environ.get("LAM_A2E_PATH"),
    os.path.join(SCRIPT_DIR, "LAM_Audio2Expression"),
]
LAM_A2E_PATH = None
for candidate in LAM_A2E_CANDIDATES:
    if candidate and os.path.exists(candidate):
        LAM_A2E_PATH = os.path.abspath(candidate)
        break

if LAM_A2E_PATH:
    sys.path.insert(0, LAM_A2E_PATH)
    logger.info(f"Added LAM_Audio2Expression to path: {LAM_A2E_PATH}")
else:
    logger.error("LAM_Audio2Expression not found!")

# --- ARKit 52 channels ---
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


# --- JBIN bundle generator ---
def create_jbin_bundle(expression: np.ndarray, audio: np.ndarray,
                       expression_sample_rate: int = 30,
                       audio_sample_rate: int = 24000,
                       batch_id: int = 0,
                       start_of_batch: bool = False,
                       end_of_batch: bool = False) -> bytes:
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
                "data_type": "float32",
                "data_offset": 0,
                "shape": list(expression_f32.shape),
                "channel_names": ARKIT_CHANNELS,
                "sample_rate": expression_sample_rate,
                "data_id": 0,
                "timeline_axis": 0,
                "channel_axis": 1
            },
            "avatar_audio": {
                "data_type": "int16",
                "data_offset": len(expression_bytes),
                "shape": [1, len(audio_int16)],
                "sample_rate": audio_sample_rate,
                "data_id": 1,
                "timeline_axis": 1
            }
        },
        "metadata": {},
        "events": [],
        "batch_id": batch_id,
        "start_of_batch": start_of_batch,
        "end_of_batch": end_of_batch
    }
    json_bytes = json.dumps(descriptor).encode('utf-8')
    header = (b'JBIN'
              + struct.pack('<I', len(json_bytes))
              + struct.pack('<I', len(expression_bytes) + len(audio_bytes)))
    return header + json_bytes + expression_bytes + audio_bytes


# --- 2. Inference Engine (Cloud Run optimized) ---
class Audio2ExpressionEngine:
    def __init__(self):
        self.infer = None
        self.initialized = False
        self.model_sample_rate = 16000
        self.input_sample_rate = 24000

    def _resolve_model_path(self, filename: str, subdir: str = None) -> str:
        """Resolve model file: FUSE mount first, then Docker-baked fallback."""
        candidates = []
        fuse_dir = os.path.join(MOUNT_PATH, MODEL_SUBDIR)
        if subdir:
            candidates.append(os.path.join(fuse_dir, subdir))
            candidates.append(os.path.join(SCRIPT_DIR, "models", subdir))
        else:
            candidates.append(os.path.join(fuse_dir, filename))
            candidates.append(os.path.join(SCRIPT_DIR, "models", filename))

        for path in candidates:
            if os.path.exists(path):
                logger.info(f"Found: {path}")
                return path
        logger.error(f"NOT FOUND: {filename} (searched {candidates})")
        return None

    def initialize(self):
        if self.initialized:
            return
        if not LAM_A2E_PATH:
            logger.error("Cannot initialize: LAM_A2E_PATH not found")
            return

        try:
            logger.info("Initializing Audio2Expression Engine...")

            from engines.defaults import default_config_parser
            from engines.infer import INFER

            # --- CRITICAL: Force DDP environment for single-process ---
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12345"

            # Resolve model paths
            lam_weight_path = self._resolve_model_path("LAM_audio2exp_streaming.tar")
            wav2vec_path = self._resolve_model_path(None, subdir="wav2vec2-base-960h")

            if not lam_weight_path:
                logger.error("LAM model weight not found. Aborting.")
                return
            if not wav2vec_path:
                logger.error("wav2vec2 model not found. Aborting.")
                return

            config_file = os.path.join(LAM_A2E_PATH, "configs",
                                       "lam_audio2exp_config_streaming.py")
            wav2vec_config = os.path.join(LAM_A2E_PATH, "configs",
                                          "wav2vec2_config.json")

            # --- CRITICAL: Config override to bypass DDP ---
            # save_path -> /tmp (only writable dir on Cloud Run)
            # This allows default_config_parser's os.makedirs() and cfg.dump() to succeed.
            save_path = "/tmp/audio2exp_logs"
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "model"), exist_ok=True)

            cfg_options = {
                "weight": lam_weight_path,
                "save_path": save_path,
                "model": {
                    "backbone": {
                        "wav2vec2_config_path": wav2vec_config,
                        "pretrained_encoder_path": wav2vec_path
                    }
                },
                "num_worker": 0,
                "batch_size": 1,
            }

            logger.info("Loading config with Cloud Run overrides...")
            cfg = default_config_parser(config_file, cfg_options)

            # --- CRITICAL: Skip default_setup() entirely ---
            # default_setup() calls comm.get_world_size(), batch_size asserts,
            # and num_worker calculations that are unnecessary for inference.
            # Instead, set the minimal required fields manually.
            cfg.device = torch.device('cpu')
            cfg.num_worker = 0
            cfg.num_worker_per_gpu = 0
            cfg.batch_size_per_gpu = 1
            cfg.batch_size_val_per_gpu = 1
            cfg.batch_size_test_per_gpu = 1

            logger.info("Building INFER model (skipping default_setup)...")
            self.infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))

            # Force CPU + eval mode
            self.infer.model.to(torch.device('cpu'))
            self.infer.model.eval()

            # Warmup inference
            logger.info("Running warmup inference...")
            dummy_audio = np.zeros(self.input_sample_rate, dtype=np.float32)
            self.infer.infer_streaming_audio(
                audio=dummy_audio, ssr=self.input_sample_rate, context=None
            )

            self.initialized = True
            logger.info("Model initialized successfully!")

        except Exception as e:
            logger.critical(f"Initialization FAILED: {e}")
            traceback.print_exc()
            self.initialized = False

    def process_full_audio(self, audio: np.ndarray,
                           sample_rate: int = 24000) -> np.ndarray:
        if not self.initialized:
            logger.warning("Model not initialized, returning mock expression.")
            return self._mock_expression(audio, sample_rate=sample_rate)

        chunk_samples = sample_rate
        all_expressions = []
        context = None

        try:
            for start in range(0, len(audio), chunk_samples):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                if len(chunk) < sample_rate // 10:
                    continue

                result, context = self.infer.infer_streaming_audio(
                    audio=chunk, ssr=sample_rate, context=context
                )
                expr = result.get("expression")
                if expr is not None:
                    all_expressions.append(expr.astype(np.float32))

            if not all_expressions:
                return np.zeros((1, 52), dtype=np.float32)

            return np.concatenate(all_expressions, axis=0)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._mock_expression(audio, sample_rate=sample_rate)

    def _mock_expression(self, audio: np.ndarray,
                         sample_rate: int = 24000) -> np.ndarray:
        frame_rate = 30
        samples_per_frame = sample_rate // frame_rate
        num_frames = max(1, len(audio) // samples_per_frame)
        return np.zeros((num_frames, 52), dtype=np.float32)


# --- FastAPI Setup ---
engine = Audio2ExpressionEngine()
active_connections: Dict[str, WebSocket] = {}
session_batch_ids: Dict[str, int] = {}
session_chunk_counts: Dict[str, int] = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Synchronous model load at startup
    engine.initialize()
    yield


app = FastAPI(title="Gourmet AI Concierge LipSync", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_initialized": engine.initialized,
        "mode": "inference" if engine.initialized else "mock",
        "mount_check": os.path.exists(os.path.join(MOUNT_PATH, MODEL_SUBDIR))
    }


@app.post("/api/audio2expression", response_model=ExpressionResponse)
async def process_audio_endpoint(request: AudioRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)

        if request.audio_format == "mp3":
            try:
                from pydub import AudioSegment
                import io
                seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes)).set_channels(1)
                audio_int16 = np.array(seg.get_array_of_samples(), dtype=np.int16)
                actual_sr = seg.frame_rate
            except ImportError:
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                actual_sr = request.sample_rate
        else:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            actual_sr = request.sample_rate

        audio_float = audio_int16.astype(np.float32) / 32768.0

        sid = request.session_id
        if request.is_start or sid not in session_batch_ids:
            session_batch_ids[sid] = session_batch_ids.get(sid, 0) + 1
            session_chunk_counts[sid] = 0

        batch_id = session_batch_ids[sid]
        session_chunk_counts[sid] = session_chunk_counts.get(sid, 0) + 1
        is_start_chunk = (session_chunk_counts[sid] == 1)

        expression = engine.process_full_audio(audio_float, sample_rate=actual_sr)

        if expression is None:
            raise HTTPException(status_code=500, detail="Inference failed")

        # Send JBIN via WebSocket if connected
        if sid in active_connections:
            await send_bundled_to_ws(
                active_connections[sid], expression, audio_int16, sid,
                batch_id=batch_id,
                start_of_batch=is_start_chunk,
                end_of_batch=request.is_final,
                audio_sample_rate=actual_sr
            )

        frames = [{"weights": row} for row in expression.tolist()]
        return ExpressionResponse(
            session_id=sid, names=ARKIT_CHANNELS, frames=frames,
            frame_rate=30, timestamp=time.time(), batch_id=batch_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def send_bundled_to_ws(ws: WebSocket, expression: np.ndarray,
                             audio: np.ndarray, session_id: str,
                             batch_id: int, start_of_batch: bool,
                             end_of_batch: bool, audio_sample_rate: int):
    try:
        jbin_data = create_jbin_bundle(
            expression, audio, 30, audio_sample_rate,
            batch_id, start_of_batch, end_of_batch
        )
        await ws.send_bytes(jbin_data)
    except Exception as e:
        logger.error(f"WS Send Error [{session_id}]: {e}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.pop(session_id, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
