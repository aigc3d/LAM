"""
Test client for Audio2Expression Service

Usage:
  # Test with audio file
  python test_client.py --audio test.wav

  # Test with generated sine wave
  python test_client.py --generate

  # Test WebSocket connection
  python test_client.py --websocket --session test-session
"""

import argparse
import asyncio
import base64
import json
import numpy as np
import requests
import websockets


def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate test audio (sine wave with varying amplitude)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Varying amplitude to simulate speech
    amplitude = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))  # 2Hz modulation
    audio = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return audio.tobytes()


def load_audio_file(path: str) -> bytes:
    """Load audio file and convert to PCM 16-bit"""
    try:
        import librosa
        audio, sr = librosa.load(path, sr=16000)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except ImportError:
        print("librosa not installed, using wave module")
        import wave
        with wave.open(path, 'rb') as wf:
            return wf.readframes(wf.getnframes())


def test_rest_api(host: str, audio_bytes: bytes, session_id: str):
    """Test REST API endpoint"""
    print(f"\n=== Testing REST API: {host}/api/audio2expression ===")

    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    response = requests.post(
        f"{host}/api/audio2expression",
        json={
            "audio_base64": audio_base64,
            "session_id": session_id,
            "is_final": True
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"   Session ID: {data['session_id']}")
        print(f"   Channels: {len(data['channels'])} (52 expected)")
        print(f"   Frames: {len(data['weights'])}")
        if data['weights']:
            # Show some key channels
            channels = data['channels']
            weights = data['weights'][0]
            jaw_open_idx = channels.index('jawOpen') if 'jawOpen' in channels else -1
            if jaw_open_idx >= 0:
                print(f"   jawOpen: {weights[jaw_open_idx]:.4f}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")


async def test_websocket(host: str, session_id: str, audio_bytes: bytes):
    """Test WebSocket endpoint"""
    ws_url = host.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/{session_id}"

    print(f"\n=== Testing WebSocket: {ws_url} ===")

    try:
        async with websockets.connect(ws_url) as ws:
            print("✅ Connected!")

            # Send audio data
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            await ws.send(json.dumps({
                "type": "audio",
                "audio": audio_base64,
                "is_final": True
            }))
            print("   Sent audio data")

            # Receive expression data
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"   Received: {data['type']}")
                if data['type'] == 'expression':
                    print(f"   Channels: {len(data['channels'])}")
                    print(f"   Frames: {len(data['weights'])}")
            except asyncio.TimeoutError:
                print("   ⚠️ No response (timeout)")

    except Exception as e:
        print(f"❌ Connection failed: {e}")


def test_health(host: str):
    """Test health endpoint"""
    print(f"\n=== Testing Health: {host}/health ===")
    try:
        response = requests.get(f"{host}/health")
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Mode: {data['mode']}")
        print(f"   Initialized: {data['initialized']}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Audio2Expression Service")
    parser.add_argument("--host", default="http://localhost:8283", help="Service URL")
    parser.add_argument("--audio", help="Path to audio file (WAV)")
    parser.add_argument("--generate", action="store_true", help="Generate test audio")
    parser.add_argument("--websocket", action="store_true", help="Test WebSocket")
    parser.add_argument("--session", default="test-session", help="Session ID")
    parser.add_argument("--health", action="store_true", help="Test health endpoint only")
    args = parser.parse_args()

    # Health check
    test_health(args.host)

    if args.health:
        return

    # Prepare audio
    if args.audio:
        print(f"\nLoading audio from: {args.audio}")
        audio_bytes = load_audio_file(args.audio)
    else:
        print("\nGenerating test audio (1 second sine wave)")
        audio_bytes = generate_test_audio(1.0)

    print(f"Audio size: {len(audio_bytes)} bytes")

    # Test REST API
    test_rest_api(args.host, audio_bytes, args.session)

    # Test WebSocket
    if args.websocket:
        asyncio.run(test_websocket(args.host, args.session, audio_bytes))


if __name__ == "__main__":
    main()
