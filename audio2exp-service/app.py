"""
Audio2Expression Service for LAM Lip Sync

Receives audio data from gourmet-sp TTS, processes with Audio2Expression,
and sends expression data to browser via WebSocket.

Architecture:
  gourmet-sp (TTS audio) → REST API → Audio2Expression → WebSocket → Browser (LAMAvatar)
"""

import asyncio
import base64
import json
import os
import sys
import time
from typing import Dict, Optional, List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add LAM_Audio2Expression to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAM_A2E_PATH = os.path.join(SCRIPT_DIR, "..", "OpenAvatarChat", "src", "handlers", "avatar", "lam", "LAM_Audio2Expression")
if os.path.exists(LAM_A2E_PATH):
    sys.path.insert(0, LAM_A2E_PATH)

app = FastAPI(title="Audio2Expression Service")

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Wrapper for LAM Audio2Expression inference"""

    def __init__(self):
        self.infer = None
        self.initialized = False
        self.sample_rate = 16000

    def initialize(self, model_path: str = None):
        """Initialize the Audio2Expression model"""
        if self.initialized:
            return

        try:
            from engines.defaults import default_config_parser, default_setup
            from engines.infer import INFER

            config_file = os.path.join(LAM_A2E_PATH, "configs", "lam_audio2exp_config_streaming.py")

            # Use default config or custom model path
            if model_path:
                cfg = default_config_parser(config_file, {"weight": model_path})
            else:
                cfg = default_config_parser(config_file, {})

            cfg = default_setup(cfg)
            self.infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
            self.infer.model.eval()

            # Warmup
            self.infer.infer_streaming_audio(
                audio=np.zeros([self.sample_rate], dtype=np.float32),
                ssr=self.sample_rate,
                context=None
            )

            self.initialized = True
            print("[Audio2Expression] Model initialized successfully")

        except Exception as e:
            print(f"[Audio2Expression] Initialization failed: {e}")
            print("[Audio2Expression] Running in mock mode")

    def process_audio(self, audio: np.ndarray, context: Optional[Dict] = None) -> tuple:
        """
        Process audio chunk and return expression data

        Args:
            audio: Audio samples (float32, 16kHz)
            context: Previous inference context for streaming

        Returns:
            (expression_data, new_context)
        """
        if not self.initialized or self.infer is None:
            # Mock mode: generate simple lip movement
            return self._mock_expression(audio), context

        result, new_context = self.infer.infer_streaming_audio(
            audio=audio,
            ssr=self.sample_rate,
            context=context
        )

        expression = result.get("expression")
        if expression is None:
            return None, new_context

        return expression.astype(np.float32), new_context

    def _mock_expression(self, audio: np.ndarray, frame_rate: int = 30) -> np.ndarray:
        """
        Generate mock expression data based on audio amplitude.
        Generates multiple frames for proper lip sync animation.

        Args:
            audio: Audio samples (float32, 16kHz)
            frame_rate: Output frame rate (default 30fps)

        Returns:
            Expression array with shape (num_frames, 52)
        """
        if len(audio) == 0:
            return np.zeros((1, 52), dtype=np.float32)

        # Calculate number of frames based on audio duration
        # Audio is 16kHz, so samples / 16000 = duration in seconds
        # frames = duration * frame_rate
        samples_per_frame = self.sample_rate // frame_rate  # 16000 / 30 = ~533 samples
        num_frames = max(1, len(audio) // samples_per_frame)

        # Get channel indices
        mouth_lower_down_left_idx = ARKIT_CHANNELS.index("mouthLowerDownLeft")
        mouth_lower_down_right_idx = ARKIT_CHANNELS.index("mouthLowerDownRight")
        mouth_dimple_left_idx = ARKIT_CHANNELS.index("mouthDimpleLeft")
        mouth_dimple_right_idx = ARKIT_CHANNELS.index("mouthDimpleRight")
        jaw_open_idx = ARKIT_CHANNELS.index("jawOpen")
        mouth_funnel_idx = ARKIT_CHANNELS.index("mouthFunnel")

        # Create expression array for all frames
        expression = np.zeros((num_frames, 52), dtype=np.float32)

        for frame_idx in range(num_frames):
            # Get audio window for this frame
            start_sample = frame_idx * samples_per_frame
            end_sample = min(start_sample + samples_per_frame, len(audio))
            audio_window = audio[start_sample:end_sample]

            # Calculate RMS for this window
            rms = np.sqrt(np.mean(audio_window ** 2)) if len(audio_window) > 0 else 0

            # Scale to mouth value (0.0 - 0.7)
            mouth_value = min(rms * 3.0, 0.7)

            # Apply expressions for this frame
            expression[frame_idx, mouth_lower_down_left_idx] = mouth_value
            expression[frame_idx, mouth_lower_down_right_idx] = mouth_value
            expression[frame_idx, mouth_dimple_left_idx] = mouth_value * 0.5
            expression[frame_idx, mouth_dimple_right_idx] = mouth_value * 0.5
            expression[frame_idx, jaw_open_idx] = mouth_value * 0.15
            expression[frame_idx, mouth_funnel_idx] = mouth_value * 0.05

        return expression


# Global engine instance
engine = Audio2ExpressionEngine()

# WebSocket connections
active_connections: Dict[str, WebSocket] = {}


class AudioRequest(BaseModel):
    """Request model for audio processing"""
    audio_base64: str  # Base64 encoded audio (PCM 16-bit, 16kHz)
    session_id: str
    is_final: bool = False


class ExpressionResponse(BaseModel):
    """Response model for expression data"""
    session_id: str
    channels: List[str]
    weights: List[List[float]]  # List of frames, each frame has 52 weights
    timestamp: float


@app.on_event("startup")
async def startup():
    """Initialize engine on startup"""
    # Skip initialization on Cloud Run (will use mock mode)
    # engine.initialize()
    print("[Audio2Expression] Starting in mock mode (Cloud Run)")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "initialized": engine.initialized,
        "mode": "inference" if engine.initialized else "mock"
    }


@app.post("/api/audio2expression", response_model=ExpressionResponse)
async def process_audio(request: AudioRequest):
    """
    Process audio and return expression data

    This endpoint is called by gourmet-sp backend after TTS synthesis.
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Convert to numpy array (assuming PCM 16-bit)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Process audio
        expression, _ = engine.process_audio(audio_float)

        if expression is None:
            raise HTTPException(status_code=500, detail="Failed to process audio")

        # Send to WebSocket if connected
        if request.session_id in active_connections:
            ws = active_connections[request.session_id]
            await send_expression_to_ws(ws, expression, request.session_id, request.is_final)

        return ExpressionResponse(
            session_id=request.session_id,
            channels=ARKIT_CHANNELS,
            weights=expression.tolist(),
            timestamp=time.time()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def send_expression_to_ws(ws: WebSocket, expression: np.ndarray, session_id: str, is_final: bool, frame_rate: int = 30):
    """Send expression data to WebSocket client with frame rate info"""
    try:
        data = {
            "type": "expression",
            "session_id": session_id,
            "channels": ARKIT_CHANNELS,
            "weights": expression.tolist(),
            "frame_rate": frame_rate,  # Frames per second for playback sync
            "frame_count": len(expression),  # Total number of frames
            "is_final": is_final,
            "timestamp": time.time()
        }
        await ws.send_json(data)
        print(f"[WebSocket] Sent {len(expression)} frames at {frame_rate}fps to {session_id}")
    except Exception as e:
        print(f"[WebSocket] Failed to send: {e}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time expression data streaming"""
    await websocket.accept()
    active_connections[session_id] = websocket
    print(f"[WebSocket] Client connected: {session_id}")

    try:
        while True:
            # Receive messages from client (e.g., for streaming audio)
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "audio":
                # Process streaming audio
                audio_base64 = message.get("audio")
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0

                    expression, _ = engine.process_audio(audio_float)
                    if expression is not None:
                        await send_expression_to_ws(
                            websocket,
                            expression,
                            session_id,
                            message.get("is_final", False)
                        )

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected: {session_id}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8283)
    parser.add_argument("--model-path", default=None, help="Path to model weights")
    args = parser.parse_args()

    if args.model_path:
        engine.initialize(args.model_path)

    uvicorn.run(app, host=args.host, port=args.port)
