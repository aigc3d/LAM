"""
Audio2Expression Service for LAM Lip Sync

Receives audio data from gourmet-sp TTS, processes with Audio2Expression,
and sends BUNDLED audio+expression data to browser via WebSocket.

Architecture (Official OpenAvatarChat approach):
  gourmet-sp (TTS audio) → REST API → Audio2Expression → Bundle(audio+expression) → WebSocket → Browser

Key: Audio and expression are bundled together in JBIN format for guaranteed sync.
The client uses audio playback position to determine which expression frame to show.
"""

import asyncio
import base64
import json
import os
import struct
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
LAM_A2E_PATH = os.path.join(SCRIPT_DIR, "..", "LAM_Audio2Expression")
if os.path.exists(LAM_A2E_PATH):
    sys.path.insert(0, LAM_A2E_PATH)
    print(f"[Audio2Expression] Added LAM_Audio2Expression to path: {LAM_A2E_PATH}")

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


def create_jbin_bundle(
    expression: np.ndarray,
    audio: np.ndarray,
    expression_sample_rate: int = 30,
    audio_sample_rate: int = 16000,
    batch_id: int = 0,
    start_of_batch: bool = False,
    end_of_batch: bool = False
) -> bytes:
    """
    Create JBIN format bundle with audio and expression data.

    JBIN Format:
    - 4 bytes: "JBIN" magic
    - 4 bytes: JSON size (little endian)
    - 4 bytes: Binary size (little endian)
    - N bytes: JSON descriptor
    - M bytes: Binary data (arkit_face + avatar_audio)

    Args:
        expression: Expression data (num_frames, 52)
        audio: Audio data (int16 or float32)
        expression_sample_rate: Expression frames per second
        audio_sample_rate: Audio samples per second
        batch_id: Batch ID for tracking speech segments
        start_of_batch: True if this is the first chunk of a speech
        end_of_batch: True if this is the last chunk of a speech

    Returns:
        JBIN formatted bytes
    """
    # Convert audio to int16 if float32
    if audio.dtype == np.float32:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)

    # Ensure expression is float32 and contiguous
    expression_f32 = np.ascontiguousarray(expression.astype(np.float32))

    # Calculate binary offsets and sizes
    expression_bytes = expression_f32.tobytes()
    audio_bytes = audio_int16.tobytes()

    expression_offset = 0
    audio_offset = len(expression_bytes)

    # Create descriptor JSON
    descriptor = {
        "data_records": {
            "arkit_face": {
                "data_type": "float32",
                "data_offset": expression_offset,
                "shape": list(expression_f32.shape),
                "channel_names": ARKIT_CHANNELS,
                "sample_rate": expression_sample_rate,
                "data_id": 0,
                "timeline_axis": 0,
                "channel_axis": 1
            },
            "avatar_audio": {
                "data_type": "int16",
                "data_offset": audio_offset,
                "shape": [1, len(audio_int16)],
                "sample_rate": audio_sample_rate,
                "data_id": 1,
                "timeline_axis": 1
            }
        },
        "metadata": {},
        "events": [],
        "batch_id": batch_id,
        "batch_name": f"speech_{batch_id}",
        "start_of_batch": start_of_batch,
        "end_of_batch": end_of_batch
    }

    # Encode JSON
    json_bytes = json.dumps(descriptor).encode('utf-8')

    # Calculate sizes
    json_size = len(json_bytes)
    bin_size = len(expression_bytes) + len(audio_bytes)

    # Build JBIN
    header = b'JBIN'
    header += struct.pack('<I', json_size)  # Little endian uint32
    header += struct.pack('<I', bin_size)   # Little endian uint32

    return header + json_bytes + expression_bytes + audio_bytes


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

            # Default weight path in cloned repo
            default_weight_path = os.path.join(LAM_A2E_PATH, "pretrained_models", "lam_audio2exp_flow")
            weight_path = model_path or default_weight_path

            # wav2vec config path
            wav2vec_config = os.path.join(LAM_A2E_PATH, "configs", "wav2vec2_config.json")

            print(f"[Audio2Expression] Config file: {config_file}")
            print(f"[Audio2Expression] Weight path: {weight_path}")

            # Use default config with weight path
            cfg = default_config_parser(config_file, {
                "weight": weight_path,
                "model": {
                    "backbone": {
                        "wav2vec2_config_path": wav2vec_config,
                    }
                }
            })

            cfg = default_setup(cfg)
            self.infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
            self.infer.model.eval()

            # Warmup
            print("[Audio2Expression] Running warmup inference...")
            self.infer.infer_streaming_audio(
                audio=np.zeros([self.sample_rate], dtype=np.float32),
                ssr=self.sample_rate,
                context=None
            )

            self.initialized = True
            print("[Audio2Expression] Model initialized successfully (CPU mode)")

        except Exception as e:
            import traceback
            print(f"[Audio2Expression] Initialization failed: {e}")
            traceback.print_exc()
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

# Batch tracking per session (for speech segment sync)
session_batch_ids: Dict[str, int] = {}
session_chunk_counts: Dict[str, int] = {}


class AudioRequest(BaseModel):
    """Request model for audio processing"""
    audio_base64: str  # Base64 encoded audio (PCM 16-bit, 16kHz or MP3)
    session_id: str
    is_start: bool = False  # True for first chunk of a speech
    is_final: bool = False  # True for last chunk of a speech
    audio_format: str = "pcm"  # "pcm" or "mp3"
    sample_rate: int = 16000  # Audio sample rate


class ExpressionResponse(BaseModel):
    """Response model for expression data"""
    session_id: str
    channels: List[str]
    weights: List[List[float]]  # List of frames, each frame has 52 weights
    timestamp: float
    batch_id: int = 0


@app.on_event("startup")
async def startup():
    """Initialize engine on startup"""
    # Try to initialize the real model (works on CPU)
    model_path = os.environ.get("AUDIO2EXP_MODEL_PATH")
    if model_path or os.path.exists(LAM_A2E_PATH):
        print("[Audio2Expression] Attempting to initialize model...")
        engine.initialize(model_path)
    else:
        print("[Audio2Expression] LAM_Audio2Expression not found, starting in mock mode")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "initialized": engine.initialized,
        "mode": "inference" if engine.initialized else "mock"
    }


@app.post("/api/audio2expression", response_model=ExpressionResponse)
async def process_audio_endpoint(request: AudioRequest):
    """
    Process audio and return expression data.
    Also sends BUNDLED audio+expression to WebSocket for guaranteed sync.

    This endpoint is called by gourmet-sp backend after TTS synthesis.
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Handle different audio formats
        if request.audio_format == "mp3":
            # MP3 needs to be decoded - try using pydub if available
            try:
                from pydub import AudioSegment
                import io
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                audio_int16 = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            except ImportError:
                # Fallback: assume it's already PCM
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            # PCM 16-bit
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Manage batch IDs for this session
        session_id = request.session_id
        if request.is_start or session_id not in session_batch_ids:
            session_batch_ids[session_id] = session_batch_ids.get(session_id, 0) + 1
            session_chunk_counts[session_id] = 0

        batch_id = session_batch_ids[session_id]
        session_chunk_counts[session_id] += 1
        is_start = session_chunk_counts[session_id] == 1

        # Process audio to get expression
        expression, _ = engine.process_audio(audio_float)

        if expression is None:
            raise HTTPException(status_code=500, detail="Failed to process audio")

        # Send BUNDLED data to WebSocket if connected
        if session_id in active_connections:
            ws = active_connections[session_id]
            await send_bundled_to_ws(
                ws,
                expression,
                audio_int16,
                session_id,
                batch_id=batch_id,
                start_of_batch=is_start,
                end_of_batch=request.is_final,
                audio_sample_rate=request.sample_rate
            )

        return ExpressionResponse(
            session_id=session_id,
            channels=ARKIT_CHANNELS,
            weights=expression.tolist(),
            timestamp=time.time(),
            batch_id=batch_id
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def send_bundled_to_ws(
    ws: WebSocket,
    expression: np.ndarray,
    audio: np.ndarray,
    session_id: str,
    batch_id: int,
    start_of_batch: bool,
    end_of_batch: bool,
    expression_sample_rate: int = 30,
    audio_sample_rate: int = 16000
):
    """
    Send BUNDLED audio+expression in JBIN format to WebSocket client.

    This is the official OpenAvatarChat synchronization approach:
    - Audio and expression are bundled together
    - Client plays audio and syncs expression based on playback position
    """
    try:
        # Create JBIN bundle
        jbin_data = create_jbin_bundle(
            expression=expression,
            audio=audio,
            expression_sample_rate=expression_sample_rate,
            audio_sample_rate=audio_sample_rate,
            batch_id=batch_id,
            start_of_batch=start_of_batch,
            end_of_batch=end_of_batch
        )

        # Send as binary
        await ws.send_bytes(jbin_data)

        audio_duration = len(audio) / audio_sample_rate
        print(f"[WebSocket] Sent JBIN bundle: {len(expression)} frames, {audio_duration:.2f}s audio, batch={batch_id}, start={start_of_batch}, end={end_of_batch}")

    except Exception as e:
        print(f"[WebSocket] Failed to send bundle: {e}")


# Legacy JSON-only send (for backward compatibility)
async def send_expression_to_ws(ws: WebSocket, expression: np.ndarray, session_id: str, is_final: bool, frame_rate: int = 30):
    """Send expression data to WebSocket client with frame rate info (legacy JSON format)"""
    try:
        data = {
            "type": "expression",
            "session_id": session_id,
            "channels": ARKIT_CHANNELS,
            "weights": expression.tolist(),
            "frame_rate": frame_rate,
            "frame_count": len(expression),
            "is_final": is_final,
            "timestamp": time.time()
        }
        await ws.send_json(data)
        print(f"[WebSocket] Sent {len(expression)} frames at {frame_rate}fps to {session_id}")
    except Exception as e:
        print(f"[WebSocket] Failed to send: {e}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time bundled audio+expression streaming"""
    await websocket.accept()
    active_connections[session_id] = websocket
    print(f"[WebSocket] Client connected: {session_id}")

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "audio":
                # Process streaming audio and send bundled response
                audio_base64 = message.get("audio")
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0

                    # Manage batch IDs
                    is_start = message.get("is_start", False)
                    is_final = message.get("is_final", False)

                    if is_start or session_id not in session_batch_ids:
                        session_batch_ids[session_id] = session_batch_ids.get(session_id, 0) + 1
                        session_chunk_counts[session_id] = 0

                    batch_id = session_batch_ids[session_id]
                    session_chunk_counts[session_id] += 1
                    chunk_is_start = session_chunk_counts[session_id] == 1

                    expression, _ = engine.process_audio(audio_float)
                    if expression is not None:
                        await send_bundled_to_ws(
                            websocket,
                            expression,
                            audio_int16,
                            session_id,
                            batch_id=batch_id,
                            start_of_batch=chunk_is_start,
                            end_of_batch=is_final
                        )

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif message.get("type") == "reset":
                # Reset batch tracking for this session
                if session_id in session_batch_ids:
                    del session_batch_ids[session_id]
                if session_id in session_chunk_counts:
                    del session_chunk_counts[session_id]
                print(f"[WebSocket] Session reset: {session_id}")

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected: {session_id}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]
        # Clean up session state
        if session_id in session_batch_ids:
            del session_batch_ids[session_id]
        if session_id in session_chunk_counts:
            del session_chunk_counts[session_id]


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
