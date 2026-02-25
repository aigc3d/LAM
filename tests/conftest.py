"""
共通テストフィクスチャ

A2Eサービスのテストで使用するフィクスチャを定義。
モデルファイル不要のCI実行を前提とする。
"""

import base64
import io
import struct
import wave

import numpy as np
import pytest


# --- ARKit 52 ブレンドシェイプ定義 ---

ARKIT_BLENDSHAPE_NAMES_INFER = [
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
    "tongueOut",
]

ARKIT_BLENDSHAPE_NAMES_FALLBACK = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]


def generate_wav_bytes(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """テスト用WAVバイト列を生成"""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def generate_silence_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """無音WAVバイト列を生成"""
    return generate_wav_bytes(duration_s=duration_s, sample_rate=sample_rate,
                              frequency=0.0, amplitude=0.0)


@pytest.fixture
def wav_440hz_1s():
    """1秒 440Hz 正弦波 WAV"""
    return generate_wav_bytes(duration_s=1.0, frequency=440.0)


@pytest.fixture
def wav_440hz_1s_base64():
    """1秒 440Hz 正弦波 WAV (base64)"""
    return base64.b64encode(generate_wav_bytes(duration_s=1.0, frequency=440.0)).decode()


@pytest.fixture
def wav_silence_1s():
    """1秒無音 WAV"""
    return generate_silence_wav_bytes(duration_s=1.0)


@pytest.fixture
def wav_silence_1s_base64():
    """1秒無音 WAV (base64)"""
    return base64.b64encode(generate_silence_wav_bytes(duration_s=1.0)).decode()


@pytest.fixture
def wav_speech_like_2s():
    """擬似音声 WAV (複数周波数)"""
    sr = 16000
    duration = 2.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # 基本周波数 + 倍音でスピーチらしい波形を生成
    signal = (
        0.4 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
        + 0.05 * np.sin(2 * np.pi * 1600 * t)
    )
    # エンベロープで発話区間を再現
    envelope = np.ones(n)
    envelope[:int(0.1 * sr)] = np.linspace(0, 1, int(0.1 * sr))
    envelope[int(1.5 * sr):int(1.7 * sr)] = 0.0  # 無音区間
    envelope[int(1.9 * sr):] = np.linspace(1, 0, n - int(1.9 * sr))
    signal *= envelope

    samples = (signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def wav_speech_like_2s_base64(wav_speech_like_2s):
    """擬似音声 WAV (base64)"""
    return base64.b64encode(wav_speech_like_2s).decode()


@pytest.fixture
def mock_a2e_response():
    """A2E APIの期待レスポンス形式"""
    n_frames = 30  # 1秒 @ 30fps
    frames = np.random.rand(n_frames, 52).astype(np.float32) * 0.5
    return {
        "names": ARKIT_BLENDSHAPE_NAMES_INFER,
        "frames": [frame.tolist() for frame in frames],
        "frame_rate": 30,
    }


@pytest.fixture
def sample_blendshape_frames():
    """テスト用ブレンドシェイプフレーム (母音パターン)"""
    # 「あ」パターン: jawOpen高、mouthFunnel低
    frame_a = np.zeros(52, dtype=np.float32)
    idx = {n: i for i, n in enumerate(ARKIT_BLENDSHAPE_NAMES_INFER)}
    frame_a[idx["jawOpen"]] = 0.7
    frame_a[idx["mouthLowerDownLeft"]] = 0.3
    frame_a[idx["mouthLowerDownRight"]] = 0.3

    # 「い」パターン: jawOpen低、mouthSmile高
    frame_i = np.zeros(52, dtype=np.float32)
    frame_i[idx["jawOpen"]] = 0.1
    frame_i[idx["mouthSmileLeft"]] = 0.5
    frame_i[idx["mouthSmileRight"]] = 0.5

    # 「う」パターン: jawOpen低、mouthPucker/Funnel高
    frame_u = np.zeros(52, dtype=np.float32)
    frame_u[idx["jawOpen"]] = 0.15
    frame_u[idx["mouthPucker"]] = 0.6
    frame_u[idx["mouthFunnel"]] = 0.4

    return {
        "a": frame_a,
        "i": frame_i,
        "u": frame_u,
        "names": ARKIT_BLENDSHAPE_NAMES_INFER,
        "idx": idx,
    }
