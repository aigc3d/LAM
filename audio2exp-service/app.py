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

# Check multiple possible locations for LAM_Audio2Expression
LAM_A2E_CANDIDATES = [
    os.environ.get("LAM_A2E_PATH"),  # Environment variable takes priority
    os.path.join(SCRIPT_DIR, "LAM_Audio2Expression"),  # Docker container: /app/LAM_Audio2Expression
    os.path.join(SCRIPT_DIR, "..", "LAM_Audio2Expression"),  # Original location
    os.path.join(SCRIPT_DIR, "..", "OpenAvatarChat", "src", "handlers", "avatar", "lam", "LAM_Audio2Expression"),  # OpenAvatarChat submodule
]

LAM_A2E_PATH = None
for candidate in LAM_A2E_CANDIDATES:
    if candidate and os.path.exists(candidate):
        LAM_A2E_PATH = os.path.abspath(candidate)
        break

if LAM_A2E_PATH:
    sys.path.insert(0, LAM_A2E_PATH)
    print(f"[Audio2Expression] Added LAM_Audio2Expression to path: {LAM_A2E_PATH}")
else:
    print("[Audio2Expression] LAM_Audio2Expression not found in any of the expected locations")

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
    audio_sample_rate: int = 24000,  # Official default from AvatarLAMConfig
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
        # Internal model sample rate (from lam_audio2exp_config_streaming.py: audio_sr = 16000)
        self.model_sample_rate = 16000
        # API input sample rate (from official AvatarLAMConfig: audio_sample_rate = 24000)
        # The model's infer_streaming_audio handles resampling internally
        self.input_sample_rate = 24000

    def initialize(self, model_path: str = None, wav2vec_path: str = None):
        """Initialize the Audio2Expression model"""
        if self.initialized:
            return

        if LAM_A2E_PATH is None:
            print("[Audio2Expression] LAM_Audio2Expression not found, cannot initialize")
            return

        try:
            from engines.defaults import default_config_parser, default_setup
            from engines.infer import INFER

            config_file = os.path.join(LAM_A2E_PATH, "configs", "lam_audio2exp_config_streaming.py")

            # Model paths - check multiple locations
            weight_candidates = [
                model_path,  # Explicit path
                os.environ.get("LAM_WEIGHT_PATH"),  # Environment variable
                os.path.join(SCRIPT_DIR, "models", "lam_audio2exp_streaming.tar"),  # Docker: /app/models/
                os.path.join(SCRIPT_DIR, "..", "OpenAvatarChat", "models", "LAM_audio2exp", "pretrained_models", "lam_audio2exp_streaming.tar"),
                os.path.join(LAM_A2E_PATH, "pretrained_models", "lam_audio2exp_streaming.tar"),
            ]
            weight_path = None
            for candidate in weight_candidates:
                if candidate and os.path.exists(candidate):
                    weight_path = os.path.abspath(candidate)
                    break

            if not weight_path:
                print("[Audio2Expression] Model weights not found")
                return

            # wav2vec2 path - check multiple locations
            wav2vec_candidates = [
                wav2vec_path,  # Explicit path
                os.environ.get("WAV2VEC_PATH"),  # Environment variable
                os.path.join(SCRIPT_DIR, "models", "wav2vec2-base-960h"),  # Docker: /app/models/
                os.path.join(SCRIPT_DIR, "..", "OpenAvatarChat", "models", "wav2vec2-base-960h"),
            ]
            wav2vec_model_path = None
            for candidate in wav2vec_candidates:
                if candidate and os.path.exists(candidate):
                    wav2vec_model_path = os.path.abspath(candidate)
                    break

            # wav2vec config path
            wav2vec_config = os.path.join(LAM_A2E_PATH, "configs", "wav2vec2_config.json")

            print(f"[Audio2Expression] Config file: {config_file}")
            print(f"[Audio2Expression] Weight path: {weight_path}")
            print(f"[Audio2Expression] Wav2Vec2 path: {wav2vec_model_path}")
            print(f"[Audio2Expression] Wav2Vec2 config: {wav2vec_config}")

            # Ensure save_path directory exists for logging
            save_path = os.path.join(SCRIPT_DIR, "exp", "audio2exp")
            os.makedirs(os.path.join(save_path, "model"), exist_ok=True)

            # Build config with model paths
            cfg_options = {
                "weight": weight_path,
                "save_path": save_path,  # Override to use absolute path
                "model": {
                    "backbone": {
                        "wav2vec2_config_path": wav2vec_config,
                    }
                }
            }

            # If we have a local wav2vec2 model, use it
            if wav2vec_model_path:
                cfg_options["model"]["backbone"]["pretrained_encoder_path"] = wav2vec_model_path

            cfg = default_config_parser(config_file, cfg_options)

            cfg = default_setup(cfg)
            self.infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
            self.infer.model.eval()

            # Warmup with input_sample_rate (model handles resampling internally)
            print("[Audio2Expression] Running warmup inference...")
            self.infer.infer_streaming_audio(
                audio=np.zeros([self.input_sample_rate], dtype=np.float32),
                ssr=self.input_sample_rate,
                context=None
            )

            self.initialized = True
            print("[Audio2Expression] Model initialized successfully (CPU mode)")

        except Exception as e:
            import traceback
            print(f"[Audio2Expression] Initialization failed: {e}")
            traceback.print_exc()
            print("[Audio2Expression] Running in mock mode")

    def process_audio(self, audio: np.ndarray, context: Optional[Dict] = None, ssr: int = None) -> tuple:
        """
        Process audio chunk and return expression data

        Args:
            audio: Audio samples (float32)
            context: Previous inference context for streaming (DEFAULT_CONTEXT structure)
            ssr: Source sample rate of the audio (default: self.input_sample_rate)
                 The model will resample internally to model_sample_rate if needed

        Returns:
            (expression_data, new_context)
        """
        if ssr is None:
            ssr = self.input_sample_rate

        if not self.initialized or self.infer is None:
            # Mock mode: generate simple lip movement
            return self._mock_expression(audio), context

        result, new_context = self.infer.infer_streaming_audio(
            audio=audio,
            ssr=ssr,
            context=context
        )

        expression = result.get("expression")
        if expression is None:
            return None, new_context

        return expression.astype(np.float32), new_context

    def process_full_audio(self, audio: np.ndarray, sample_rate: int = None, frame_rate: int = 30) -> np.ndarray:
        """
        Process full audio by splitting into chunks for proper streaming inference.

        Following official OpenAvatarChat pattern:
        - Audio is sliced into 1-second chunks (slice_size = sample_rate * 1.0)
        - Each chunk is processed with streaming context maintained
        - Model handles resampling internally (ssr -> audio_sr)

        Args:
            audio: Full audio samples (float32)
            sample_rate: Audio sample rate (default: self.input_sample_rate = 24000)
            frame_rate: Output frame rate (default 30fps)

        Returns:
            Expression array with shape (num_frames, 52)
        """
        if sample_rate is None:
            sample_rate = self.input_sample_rate

        if not self.initialized or self.infer is None:
            # Mock mode: generate simple lip movement
            return self._mock_expression(audio, frame_rate, sample_rate)

        # Calculate expected output frames
        audio_duration = len(audio) / sample_rate
        expected_frames = int(audio_duration * frame_rate)

        if expected_frames <= 0:
            return np.zeros((1, 52), dtype=np.float32)

        # Process in chunks of 1 second (official: slice_size = audio_sample_rate * 1.0)
        # This maintains proper streaming context across chunks
        chunk_samples = sample_rate  # 1 second chunks at input sample rate

        all_expressions = []
        context = None

        for start in range(0, len(audio), chunk_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Skip very short chunks (less than 0.1 seconds)
            if len(chunk) < sample_rate // 10:
                continue

            # Pass sample rate to process_audio for proper resampling
            expression, context = self.process_audio(chunk, context, ssr=sample_rate)

            if expression is not None and len(expression) > 0:
                all_expressions.append(expression)

        if not all_expressions:
            return np.zeros((max(1, expected_frames), 52), dtype=np.float32)

        # Concatenate all expression chunks
        full_expression = np.concatenate(all_expressions, axis=0)

        print(f"[Audio2Expression] Processed {audio_duration:.2f}s audio -> {len(full_expression)} frames (expected ~{expected_frames})")

        return full_expression.astype(np.float32)

    def _mock_expression(self, audio: np.ndarray, frame_rate: int = 30, sample_rate: int = None) -> np.ndarray:
        """
        Generate mock expression data based on audio amplitude with dynamic variation.
        Generates multiple frames for proper lip sync animation.

        Args:
            audio: Audio samples (float32)
            frame_rate: Output frame rate (default 30fps)
            sample_rate: Audio sample rate (default: self.input_sample_rate)

        Returns:
            Expression array with shape (num_frames, 52)
        """
        if sample_rate is None:
            sample_rate = self.input_sample_rate

        if len(audio) == 0:
            return np.zeros((1, 52), dtype=np.float32)

        # Calculate number of frames based on audio duration
        samples_per_frame = sample_rate // frame_rate
        num_frames = max(1, len(audio) // samples_per_frame)

        # Get channel indices
        mouth_lower_down_left_idx = ARKIT_CHANNELS.index("mouthLowerDownLeft")
        mouth_lower_down_right_idx = ARKIT_CHANNELS.index("mouthLowerDownRight")
        mouth_dimple_left_idx = ARKIT_CHANNELS.index("mouthDimpleLeft")
        mouth_dimple_right_idx = ARKIT_CHANNELS.index("mouthDimpleRight")
        jaw_open_idx = ARKIT_CHANNELS.index("jawOpen")
        mouth_funnel_idx = ARKIT_CHANNELS.index("mouthFunnel")
        mouth_pucker_idx = ARKIT_CHANNELS.index("mouthPucker")
        mouth_smile_left_idx = ARKIT_CHANNELS.index("mouthSmileLeft")
        mouth_smile_right_idx = ARKIT_CHANNELS.index("mouthSmileRight")

        # Create expression array for all frames
        expression = np.zeros((num_frames, 52), dtype=np.float32)

        # Collect RMS values for normalization
        rms_values = []
        for frame_idx in range(num_frames):
            start_sample = frame_idx * samples_per_frame
            end_sample = min(start_sample + samples_per_frame, len(audio))
            audio_window = audio[start_sample:end_sample]
            rms = np.sqrt(np.mean(audio_window ** 2)) if len(audio_window) > 0 else 0
            rms_values.append(rms)

        # Normalize RMS to use full dynamic range
        rms_array = np.array(rms_values)
        rms_min = rms_array.min()
        rms_max = rms_array.max()
        rms_range = rms_max - rms_min if rms_max > rms_min else 1.0

        for frame_idx in range(num_frames):
            # Normalize RMS to 0-1 range for this audio clip
            normalized_rms = (rms_values[frame_idx] - rms_min) / rms_range

            # Add speech-like variation using sine waves at different frequencies
            time_factor = frame_idx / frame_rate
            variation1 = 0.3 * np.sin(2 * np.pi * 3.5 * time_factor)  # ~3.5 Hz syllable rate
            variation2 = 0.2 * np.sin(2 * np.pi * 7.0 * time_factor)  # Higher frequency detail
            variation3 = 0.1 * np.sin(2 * np.pi * 1.5 * time_factor)  # Slower breathing pattern

            # Combine normalized amplitude with speech variation
            base_value = normalized_rms * 0.5 + 0.1  # Base range 0.1-0.6
            mouth_value = base_value + variation1 * normalized_rms
            mouth_value = np.clip(mouth_value, 0.0, 0.7)

            # Jaw has different timing (slightly delayed, lower frequency)
            jaw_variation = 0.15 * np.sin(2 * np.pi * 2.5 * time_factor + 0.3)
            jaw_value = normalized_rms * 0.25 + jaw_variation * normalized_rms
            jaw_value = np.clip(jaw_value, 0.0, 0.4)

            # Apply expressions for this frame
            expression[frame_idx, mouth_lower_down_left_idx] = mouth_value
            expression[frame_idx, mouth_lower_down_right_idx] = mouth_value
            expression[frame_idx, mouth_dimple_left_idx] = mouth_value * 0.3
            expression[frame_idx, mouth_dimple_right_idx] = mouth_value * 0.3
            expression[frame_idx, jaw_open_idx] = jaw_value
            expression[frame_idx, mouth_funnel_idx] = mouth_value * 0.2 + variation2 * 0.1
            expression[frame_idx, mouth_pucker_idx] = max(0, variation3 * 0.15)
            expression[frame_idx, mouth_smile_left_idx] = max(0, variation1 * 0.1)
            expression[frame_idx, mouth_smile_right_idx] = max(0, variation1 * 0.1)

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
    audio_base64: str  # Base64 encoded audio (PCM 16-bit or MP3)
    session_id: str
    is_start: bool = False  # True for first chunk of a speech
    is_final: bool = False  # True for last chunk of a speech
    audio_format: str = "pcm"  # "pcm" or "mp3"
    sample_rate: int = 24000  # Audio sample rate (official default from AvatarLAMConfig)


class ExpressionResponse(BaseModel):
    """Response model for expression data - 公式形式に準拠"""
    session_id: str
    names: List[str]  # チャンネル名配列（公式と同じ）
    frames: List[dict]  # 各フレームは {"weights": [...]} 形式（公式と同じ）
    frame_rate: int = 30
    timestamp: float
    batch_id: int = 0


@app.on_event("startup")
async def startup():
    """Start background task to initialize engine when models are ready"""
    print("[Audio2Expression] Server started, waiting for models in background...")
    asyncio.create_task(initialize_model_when_ready())


async def initialize_model_when_ready():
    """Background task to initialize model when models are downloaded"""
    model_dir = os.path.join(SCRIPT_DIR, "models")
    ready_signal = os.path.join(model_dir, ".models_ready")
    model_path = os.environ.get("AUDIO2EXP_MODEL_PATH")

    # Wait for models to be ready (max 10 minutes)
    max_wait = 600  # seconds
    waited = 0
    check_interval = 2  # seconds

    while waited < max_wait:
        # Check if models are already available (e.g., pre-baked in image)
        if engine.initialized:
            print("[Audio2Expression] Model already initialized")
            return

        # Check for ready signal from start.sh
        if os.path.exists(ready_signal):
            print("[Audio2Expression] Models ready signal detected, initializing...")
            break

        # Check if model files exist directly (for pre-baked images)
        lam_model = os.path.join(model_dir, "lam_audio2exp_streaming.tar")
        wav2vec = os.path.join(model_dir, "wav2vec2-base-960h")
        if os.path.exists(lam_model) and os.path.exists(wav2vec):
            print("[Audio2Expression] Model files detected, initializing...")
            break

        await asyncio.sleep(check_interval)
        waited += check_interval
        if waited % 30 == 0:
            print(f"[Audio2Expression] Waiting for models... ({waited}s elapsed)")

    if waited >= max_wait:
        print("[Audio2Expression] Timeout waiting for models, running in mock mode")
        return

    # Initialize the model
    try:
        if LAM_A2E_PATH and os.path.exists(LAM_A2E_PATH):
            print("[Audio2Expression] Attempting to initialize model...")
            # Run initialization in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, engine.initialize, model_path)
            print("[Audio2Expression] Model initialization complete")
        else:
            print("[Audio2Expression] LAM_Audio2Expression not found, running in mock mode")
    except Exception as e:
        import traceback
        print(f"[Audio2Expression] Model initialization failed: {e}")
        traceback.print_exc()
        print("[Audio2Expression] Running in mock mode")


@app.get("/health")
async def health_check():
    """Health check endpoint - always returns ok for Cloud Run liveness probe"""
    model_dir = os.path.join(SCRIPT_DIR, "models")
    models_ready = os.path.exists(os.path.join(model_dir, ".models_ready"))

    if engine.initialized:
        model_status = "ready"
    elif models_ready:
        model_status = "initializing"
    else:
        model_status = "downloading"

    return {
        "status": "ok",  # Always ok for Cloud Run health check
        "model_initialized": engine.initialized,
        "model_status": model_status,
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
        actual_sample_rate = request.sample_rate
        if request.audio_format == "mp3":
            # MP3 needs to be decoded - try using pydub if available
            try:
                from pydub import AudioSegment
                import io
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                # Use the MP3's native sample rate (model handles resampling internally)
                actual_sample_rate = audio_segment.frame_rate
                audio_segment = audio_segment.set_channels(1)  # Convert to mono
                audio_int16 = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
                print(f"[Audio2Expression] Decoded MP3: {actual_sample_rate}Hz, {len(audio_int16)} samples")
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

        # Process full audio to get expression (handles long audio by chunking)
        # Pass the actual sample rate so the model can resample internally if needed
        expression = engine.process_full_audio(audio_float, sample_rate=actual_sample_rate)

        if expression is None or len(expression) == 0:
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
                audio_sample_rate=actual_sample_rate
            )

        # 公式形式に変換: frames = [{"weights": [...]}, ...]
        frames = [{"weights": row} for row in expression.tolist()]

        return ExpressionResponse(
            session_id=session_id,
            names=ARKIT_CHANNELS,
            frames=frames,
            frame_rate=30,
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
    audio_sample_rate: int = 24000  # Official default from AvatarLAMConfig
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

                    # Get sample rate from message (default to engine's input rate)
                    sample_rate = message.get("sample_rate", engine.input_sample_rate)

                    # Manage batch IDs
                    is_start = message.get("is_start", False)
                    is_final = message.get("is_final", False)

                    if is_start or session_id not in session_batch_ids:
                        session_batch_ids[session_id] = session_batch_ids.get(session_id, 0) + 1
                        session_chunk_counts[session_id] = 0

                    batch_id = session_batch_ids[session_id]
                    session_chunk_counts[session_id] += 1
                    chunk_is_start = session_chunk_counts[session_id] == 1

                    # Pass sample rate to process_audio
                    expression, _ = engine.process_audio(audio_float, ssr=sample_rate)
                    if expression is not None:
                        await send_bundled_to_ws(
                            websocket,
                            expression,
                            audio_int16,
                            session_id,
                            batch_id=batch_id,
                            start_of_batch=chunk_is_start,
                            end_of_batch=is_final,
                            audio_sample_rate=sample_rate
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
