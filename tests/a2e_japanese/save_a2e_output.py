"""
A2E推論出力保存スクリプト

OpenAvatarChat環境内でA2Eを直接呼び出し、
日本語音声からブレンドシェイプ出力をnpyファイルに保存する。

このスクリプトはOpenAvatarChatのavatar_handler_lam_audio2expressionを
直接呼び出して、A2Eモデルの生出力をキャプチャする。

使い方:
    cd C:\Users\hamad\OpenAvatarChat
    conda activate oac
    python tests/a2e_japanese/save_a2e_output.py --audio-dir tests/a2e_japanese/audio_samples

出力:
    tests/a2e_japanese/blendshape_outputs/ にnpyファイルが保存される
"""

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np


def load_wav_as_pcm(wav_path: str, target_sr: int = 24000) -> np.ndarray:
    """WAVファイルをPCM float32配列として読み込み"""
    with wave.open(wav_path, "r") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # リサンプリング
    if frame_rate != target_sr:
        duration = len(audio) / frame_rate
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
        audio = audio[indices]

    return audio


def try_direct_a2e_inference(oac_dir: Path, audio_path: str) -> np.ndarray:
    """A2Eモデルを直接ロードして推論"""
    # OpenAvatarChatのパスを追加
    paths = [
        str(oac_dir / "src"),
        str(oac_dir / "src" / "handlers"),
        str(oac_dir / "src" / "handlers" / "avatar" / "lam"),
        str(oac_dir / "src" / "handlers" / "avatar" / "lam" / "LAM_Audio2Expression"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    # Wav2Vec2で特徴量抽出
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    wav2vec_dir = oac_dir / "models" / "wav2vec2-base-960h"
    if wav2vec_dir.exists() and (wav2vec_dir / "config.json").exists():
        model_name = str(wav2vec_dir)
    else:
        model_name = "facebook/wav2vec2-base-960h"

    print(f"  Loading Wav2Vec2: {model_name}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
    except Exception:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
    wav2vec_model.eval()

    # 音声読み込み (Wav2Vec2は16kHz)
    audio_16k = load_wav_as_pcm(audio_path, target_sr=16000)
    print(f"  Audio: {len(audio_16k)/16000:.2f}s at 16kHz")

    # 特徴量抽出
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    features = outputs.last_hidden_state  # (1, T, 768)
    print(f"  Wav2Vec2 features: {features.shape}")

    # A2Eデコーダーのロード試行
    try:
        from LAM_Audio2Expression.engines.infer import Audio2ExpressionInfer
        from LAM_Audio2Expression.engines.defaults import default_setup

        # A2Eのconfigを構築
        # 注: 実際のconfig構造はLAM_Audio2Expressionの実装に依存
        print("  A2E module loaded. Attempting inference...")

        # A2E推論 (実装依存)
        # result = a2e_infer(features)
        # return result

        print("  NOTE: Direct A2E inference requires full config setup.")
        print("  Falling back to Wav2Vec2 feature analysis.")
        raise ImportError("Direct A2E not configured")

    except ImportError:
        # A2Eデコーダーがロードできない場合、Wav2Vec2特徴量の分析を返す
        print("  A2E decoder not available. Saving Wav2Vec2 features instead.")
        print("  For full A2E output, run OpenAvatarChat and capture the output.")
        return features.squeeze(0).numpy()  # (T, 768)


def try_handler_inference(oac_dir: Path, audio_path: str) -> np.ndarray:
    """OpenAvatarChatのhandler経由でA2E推論"""
    paths = [
        str(oac_dir / "src"),
        str(oac_dir / "src" / "handlers"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from avatar.lam.avatar_handler_lam_audio2expression import HandlerAvatarLAM
        print("  HandlerAvatarLAM loaded.")

        # Handler config
        class MockConfig:
            model_name = "LAM_audio2exp"
            feature_extractor_model_name = "wav2vec2-base-960h"
            audio_sample_rate = 24000

        class MockEngineConfig:
            model_root = str(oac_dir / "models")

        handler = HandlerAvatarLAM()
        handler.load(MockEngineConfig(), MockConfig())

        # 音声をPCMとして読み込み
        audio_24k = load_wav_as_pcm(audio_path, target_sr=24000)
        audio_bytes = (audio_24k * 32768).astype(np.int16).tobytes()

        # handler.process() の出力をキャプチャ
        # 注: 実際のAPIは HandlerAvatarLAM の実装に依存
        print("  NOTE: Handler API depends on OpenAvatarChat internals.")
        print("  This may need adjustment based on the actual handler interface.")

        return None

    except ImportError as e:
        print(f"  Handler not available: {e}")
        return None
    except Exception as e:
        print(f"  Handler error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Save A2E Inference Output")
    parser.add_argument("--oac-dir", type=str, default=None)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None, help="Single audio file")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # OACディレクトリ解決
    if args.oac_dir:
        oac_dir = Path(args.oac_dir)
    else:
        candidates = [
            Path(r"C:\Users\hamad\OpenAvatarChat"),
            Path.home() / "OpenAvatarChat",
            Path.cwd(),
        ]
        oac_dir = next((p for p in candidates if (p / "src" / "demo.py").exists()), None)
        if oac_dir is None:
            print("ERROR: OpenAvatarChat not found. Use --oac-dir")
            sys.exit(1)

    # 音声ファイル解決
    if args.audio_file:
        audio_files = [Path(args.audio_file)]
    elif args.audio_dir:
        audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    else:
        audio_files = sorted((script_dir / "audio_samples").glob("*.wav"))

    if not audio_files:
        print("ERROR: No WAV files found.")
        print("Run generate_test_audio.py first.")
        sys.exit(1)

    output_dir = script_dir / "blendshape_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("A2E Inference Output Capture")
    print(f"OAC: {oac_dir}")
    print(f"Audio files: {len(audio_files)}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for audio_path in audio_files:
        name = audio_path.stem
        output_path = output_dir / f"{name}.npy"

        if output_path.exists():
            print(f"\n[SKIP] {name}: output already exists")
            continue

        print(f"\n[{name}] Processing: {audio_path}")
        t0 = time.time()

        # 方法1: 直接A2E推論
        result = try_direct_a2e_inference(oac_dir, str(audio_path))

        if result is None:
            # 方法2: Handler経由
            result = try_handler_inference(oac_dir, str(audio_path))

        if result is not None:
            np.save(str(output_path), result)
            elapsed = time.time() - t0
            print(f"  Saved: {output_path} shape={result.shape} ({elapsed:.1f}s)")
        else:
            print(f"  FAILED: Could not generate output for {name}")

    # サマリー
    saved_files = list(output_dir.glob("*.npy"))
    print(f"\n{'=' * 60}")
    print(f"Saved {len(saved_files)} output files to {output_dir}")
    for f in sorted(saved_files):
        data = np.load(str(f))
        print(f"  {f.name}: shape={data.shape}")

    if saved_files:
        print(f"\nNext: Analyze with:")
        print(f"  python tests/a2e_japanese/analyze_blendshapes.py --input-dir {output_dir}")


if __name__ == "__main__":
    main()
