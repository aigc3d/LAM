"""
A2E (Audio2Expression) 推論エンジン

Wav2Vec2 + A2Eデコーダーを使って、音声から52次元ARKitブレンドシェイプを生成。

モデル構成:
    - facebook/wav2vec2-base-960h: 音響特徴量抽出 (768次元)
    - 3DAIGC/LAM_audio2exp: 表情デコーダー (768→52次元)

入出力:
    Input:  base64エンコードされた音声 (MP3/WAV/PCM)
    Output: {names: [52 strings], frames: [[52 floats], ...], frame_rate: 30}
"""

import base64
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ARKit 52 ブレンドシェイプ名 (Apple公式仕様)
ARKIT_BLENDSHAPE_NAMES = [
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

# A2E出力のFPS (OpenAvatarChatのデフォルト)
A2E_OUTPUT_FPS = 30


class Audio2ExpressionEngine:
    """A2E推論エンジン - Wav2Vec2 + LAM A2Eデコーダー"""

    def __init__(self, model_dir: str = "./models", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self._ready = False

        # デバイス決定
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.device_name = self.device

        logger.info(f"[A2E Engine] Device: {self.device}")

        self._load_models()

    def _load_models(self):
        """Wav2Vec2 + A2Eデコーダーをロード"""
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        # ========================================
        # Wav2Vec2 ロード
        # ========================================
        wav2vec_dir = self.model_dir / "wav2vec2-base-960h"
        if wav2vec_dir.exists() and (wav2vec_dir / "config.json").exists():
            wav2vec_path = str(wav2vec_dir)
            logger.info(f"[A2E Engine] Loading Wav2Vec2 from local: {wav2vec_path}")
        else:
            wav2vec_path = "facebook/wav2vec2-base-960h"
            logger.info(f"[A2E Engine] Loading Wav2Vec2 from HuggingFace: {wav2vec_path}")

        try:
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        except Exception:
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )

        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path)
        self.wav2vec_model.to(self.device)
        self.wav2vec_model.eval()
        logger.info("[A2E Engine] Wav2Vec2 loaded")

        # ========================================
        # A2Eデコーダー ロード
        # ========================================
        self.a2e_decoder = None
        self._load_a2e_decoder(self.model_dir)

        self._ready = True
        logger.info("[A2E Engine] Ready")

    def _load_a2e_decoder(self, model_dir: Path):
        """
        LAM A2Eデコーダーのロード

        対応するディレクトリ構造:
          パターン1 (フラット): models/LAM_audio2exp_streaming.tar
          パターン2 (サブディレクトリ): models/LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar
          パターン3 (サブディレクトリ直下): models/LAM_audio2exp/LAM_audio2exp_streaming.tar
        """
        import torch

        # チェックポイントを探索
        checkpoint_path = None
        search_patterns = [
            # パターン1: models/ 直下にtar (フラット配置)
            model_dir / "LAM_audio2exp_streaming.tar",
            model_dir / "lam_audio2exp_streaming.tar",
            # パターン2: models/LAM_audio2exp/pretrained_models/
            model_dir / "LAM_audio2exp" / "pretrained_models" / "lam_audio2exp_streaming.tar",
            model_dir / "LAM_audio2exp" / "pretrained_models" / "LAM_audio2exp_streaming.tar",
            # パターン3: models/LAM_audio2exp/ 直下
            model_dir / "LAM_audio2exp" / "LAM_audio2exp_streaming.tar",
            model_dir / "LAM_audio2exp" / "lam_audio2exp_streaming.tar",
        ]

        for path in search_patterns:
            if path.exists():
                checkpoint_path = path
                break

        # パターンに一致しなければ、model_dir以下の全tarを検索
        if checkpoint_path is None:
            tar_files = list(model_dir.rglob("*audio2exp*.tar"))
            if tar_files:
                checkpoint_path = tar_files[0]

        if checkpoint_path is None:
            logger.warning(f"[A2E Engine] No A2E checkpoint found in {model_dir}")
            logger.warning("[A2E Engine] Searched patterns: models/*.tar, models/LAM_audio2exp/**/*.tar")
            logger.warning("[A2E Engine] Will use Wav2Vec2-based fallback")
            return

        logger.info(f"[A2E Engine] Found A2E checkpoint: {checkpoint_path}")

        # LAM_Audio2Expression のPythonモジュールパスを追加
        for lam_path in [
            model_dir / "LAM_Audio2Expression",
            model_dir / "LAM_audio2exp" / "LAM_Audio2Expression",
            model_dir.parent / "LAM_Audio2Expression",
        ]:
            if lam_path.exists() and str(lam_path) not in sys.path:
                sys.path.insert(0, str(lam_path))

        try:
            from engines.infer import Audio2ExpressionInfer

            self.a2e_decoder = Audio2ExpressionInfer()
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            self.a2e_decoder.load_state_dict(checkpoint)
            self.a2e_decoder.to(self.device)
            self.a2e_decoder.eval()
            logger.info("[A2E Engine] A2E decoder loaded successfully")

        except ImportError:
            logger.warning("[A2E Engine] LAM_Audio2Expression module not importable")
            logger.warning("[A2E Engine] Using Wav2Vec2-based fallback")
        except Exception as e:
            logger.warning(f"[A2E Engine] Failed to load A2E decoder: {e}")
            logger.warning("[A2E Engine] Using Wav2Vec2-based fallback")

    def is_ready(self) -> bool:
        return self._ready

    def process(self, audio_base64: str, audio_format: str = "mp3") -> dict:
        """
        音声を処理してブレンドシェイプ係数を生成

        Args:
            audio_base64: base64エンコードされた音声
            audio_format: 音声フォーマット (mp3, wav, pcm)

        Returns:
            {names: [52 strings], frames: [[52 floats], ...], frame_rate: int}
        """
        import torch

        # 1. 音声デコード → PCM 16kHz
        audio_pcm = self._decode_audio(audio_base64, audio_format)
        duration = len(audio_pcm) / 16000
        logger.info(f"[A2E Engine] Audio decoded: {duration:.2f}s at 16kHz")

        # 2. Wav2Vec2 特徴量抽出
        inputs = self.wav2vec_processor(
            audio_pcm, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.wav2vec_model(input_values)
            features = outputs.last_hidden_state  # (1, T, 768)

        logger.info(f"[A2E Engine] Wav2Vec2 features: {tuple(features.shape)}")

        # 3. A2E デコーダーで52次元ブレンドシェイプに変換
        if self.a2e_decoder is not None:
            blendshapes = self._run_a2e_decoder(features)
        else:
            blendshapes = self._wav2vec_to_blendshapes_fallback(features, duration)

        # 4. フレームレート調整 (A2E出力→30fps)
        frames = self._resample_to_fps(blendshapes, duration, A2E_OUTPUT_FPS)

        # 5. レスポンス構築
        return {
            "names": ARKIT_BLENDSHAPE_NAMES,
            "frames": frames,
            "frame_rate": A2E_OUTPUT_FPS
        }

    def _decode_audio(self, audio_base64: str, audio_format: str) -> np.ndarray:
        """base64音声をPCM float32 16kHzにデコード"""
        audio_bytes = base64.b64decode(audio_base64)

        if audio_format in ("mp3", "wav", "ogg", "flac"):
            # pydub で変換
            from pydub import AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0
        elif audio_format == "pcm":
            # 生PCM (int16, 16kHz, mono)
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

        return samples

    def _run_a2e_decoder(self, features) -> np.ndarray:
        """A2Eデコーダーで推論"""
        import torch

        with torch.no_grad():
            # A2Eデコーダーの入力形式に合わせる
            # 具体的なインターフェースはLAM_Audio2Expressionの実装に依存
            output = self.a2e_decoder(features)

            if isinstance(output, torch.Tensor):
                blendshapes = output.squeeze(0).cpu().numpy()
            elif isinstance(output, (list, tuple)):
                blendshapes = np.array(output, dtype=np.float32)
            else:
                blendshapes = output

        # (T, 52)であることを確認
        if blendshapes.ndim == 1:
            blendshapes = blendshapes.reshape(1, -1)
        if blendshapes.shape[-1] != 52:
            logger.warning(f"[A2E Engine] Unexpected output dim: {blendshapes.shape}")

        return blendshapes

    def _wav2vec_to_blendshapes_fallback(
        self, features, duration: float
    ) -> np.ndarray:
        """
        A2Eデコーダーがない場合のフォールバック:
        Wav2Vec2の特徴量からリップシンク関連のブレンドシェイプを近似生成。

        Wav2Vec2の768次元特徴量のエネルギーパターンを使って、
        jawOpen等のリップ関連ブレンドシェイプを駆動する。
        完全なA2Eデコーダーに比べて精度は劣るが、
        FFT音量ベースよりも正確なタイミングを提供する。
        """
        features_np = features.squeeze(0).cpu().numpy()  # (T, 768)
        n_frames = features_np.shape[0]

        # 全52次元を0で初期化
        blendshapes = np.zeros((n_frames, 52), dtype=np.float32)

        # Wav2Vec2特徴量からエネルギーを計算
        # 低周波帯(0-256): 母音に関連する音響特徴
        # 中周波帯(256-512): 子音に関連
        # 高周波帯(512-768): 摩擦音・破裂音
        low_energy = np.abs(features_np[:, :256]).mean(axis=1)
        mid_energy = np.abs(features_np[:, 256:512]).mean(axis=1)
        high_energy = np.abs(features_np[:, 512:]).mean(axis=1)

        # エネルギーを正規化 (0.0〜1.0)
        def normalize(x):
            x_min = x.min()
            x_max = x.max()
            if x_max - x_min < 1e-6:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        low_norm = normalize(low_energy)
        mid_norm = normalize(mid_energy)
        high_norm = normalize(high_energy)

        # 全体のスピーチ活性度
        speech_activity = normalize(low_energy + mid_energy + high_energy)

        # ブレンドシェイプ名→インデックスのマップ
        idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}

        # ========================================
        # リップシンク関連のブレンドシェイプを駆動
        # ========================================

        # jawOpen: 口の開き (低周波エネルギーに強く相関)
        blendshapes[:, idx["jawOpen"]] = np.clip(low_norm * 0.8, 0, 1)

        # mouthClose: jawOpenの逆
        blendshapes[:, idx["mouthClose"]] = np.clip(1.0 - low_norm * 0.8, 0, 1) * speech_activity

        # mouthFunnel: 「う」「お」の口の丸め (中周波で推定)
        funnel = np.clip(mid_norm * 0.5 - low_norm * 0.2, 0, 1)
        blendshapes[:, idx["mouthFunnel"]] = funnel

        # mouthPucker: 「う」のすぼめ
        blendshapes[:, idx["mouthPucker"]] = np.clip(funnel * 0.7, 0, 1)

        # mouthSmile: 「い」「え」の横開き (高周波が多い時)
        smile = np.clip(high_norm * 0.4 - mid_norm * 0.1, 0, 1)
        blendshapes[:, idx["mouthSmileLeft"]] = smile
        blendshapes[:, idx["mouthSmileRight"]] = smile

        # mouthLowerDown / mouthUpperUp: 母音の開き
        lower_down = np.clip(low_norm * 0.5, 0, 1)
        blendshapes[:, idx["mouthLowerDownLeft"]] = lower_down
        blendshapes[:, idx["mouthLowerDownRight"]] = lower_down
        upper_up = np.clip(low_norm * 0.3, 0, 1)
        blendshapes[:, idx["mouthUpperUpLeft"]] = upper_up
        blendshapes[:, idx["mouthUpperUpRight"]] = upper_up

        # mouthStretch: 口の横幅 (中〜高周波)
        stretch = np.clip((mid_norm + high_norm) * 0.25, 0, 1)
        blendshapes[:, idx["mouthStretchLeft"]] = stretch
        blendshapes[:, idx["mouthStretchRight"]] = stretch

        # ========================================
        # 非リップ関連（微細な表情）
        # ========================================

        # browInnerUp: 話す時の眉の動き
        blendshapes[:, idx["browInnerUp"]] = np.clip(speech_activity * 0.15, 0, 1)

        # cheekSquint: 笑顔時
        blendshapes[:, idx["cheekSquintLeft"]] = smile * 0.3
        blendshapes[:, idx["cheekSquintRight"]] = smile * 0.3

        # noseSneer: 発話の力み
        nose = np.clip(speech_activity * 0.1, 0, 1)
        blendshapes[:, idx["noseSneerLeft"]] = nose
        blendshapes[:, idx["noseSneerRight"]] = nose

        # 無音フレームではすべてをゼロに近づける
        silence_mask = speech_activity < 0.1
        blendshapes[silence_mask] *= 0.1

        # スムージング (3フレームの移動平均)
        if n_frames > 3:
            kernel = np.ones(3) / 3
            for i in range(52):
                blendshapes[:, i] = np.convolve(blendshapes[:, i], kernel, mode='same')

        logger.info(f"[A2E Engine] Fallback: {n_frames} frames generated, "
                    f"jawOpen range=[{blendshapes[:, idx['jawOpen']].min():.3f}, "
                    f"{blendshapes[:, idx['jawOpen']].max():.3f}]")

        return blendshapes

    def _resample_to_fps(
        self, blendshapes: np.ndarray, duration: float, target_fps: int
    ) -> list:
        """ブレンドシェイプを目標FPSにリサンプリング"""
        n_source = blendshapes.shape[0]
        n_target = max(1, int(duration * target_fps))

        if n_source == n_target:
            frames = blendshapes
        else:
            # 線形補間でリサンプリング
            source_indices = np.linspace(0, n_source - 1, n_target)
            frames = np.zeros((n_target, 52), dtype=np.float32)
            for i in range(52):
                frames[:, i] = np.interp(
                    source_indices, np.arange(n_source), blendshapes[:, i]
                )

        # Python list に変換 (JSON serializable)
        return [frame.tolist() for frame in frames]
