"""
A2E (Audio2Expression) 推論エンジン

LAM Audio2Expression INFER パイプラインを使って、
音声から52次元ARKitブレンドシェイプを生成。

モデル構成:
    - facebook/wav2vec2-base-960h: 音響特徴量抽出 (768次元)
    - 3DAIGC/LAM_audio2exp: 表情デコーダー (768→52次元)

優先順位:
    1. INFER パイプライン (LAM_Audio2Expression モジュール使用)
       → 完全な A2E 推論 + ポストプロセッシング
    2. Wav2Vec2 エネルギーベースフォールバック
       → モジュール未インストール時の近似生成

入出力:
    Input:  base64エンコードされた音声 (MP3/WAV/PCM)
    Output: {names: [52 strings], frames: [[52 floats], ...], frame_rate: 30}
"""

import base64
import io
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# INFER パイプラインが使用する ARKit 52 ブレンドシェイプ名
# (LAM_Audio2Expression/models/utils.py の ARKitBlendShape と同じ順序)
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

# フォールバック用の ARKit 名 (a2e_engine.py 独自の順序)
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

# A2E出力のFPS
A2E_OUTPUT_FPS = 30

# INFER パイプライン用の入力サンプルレート
INFER_INPUT_SAMPLE_RATE = 16000


class Audio2ExpressionEngine:
    """A2E推論エンジン - INFER パイプライン優先、Wav2Vec2 フォールバック"""

    def __init__(self, model_dir: str = "./models", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self._ready = False
        self._use_infer = False  # INFER パイプライン使用フラグ
        self._infer = None       # INFER パイプラインインスタンス
        self._infer_context = None  # ストリーミング推論のコンテキスト

        # デバイス決定
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.device_name = self.device

        logger.info(f"[A2E Engine] Device: {self.device}")

        self._initialize()

    def _initialize(self):
        """エンジン初期化 - INFER パイプラインを優先的にロード"""
        # 1. INFER パイプラインを試行
        if self._try_load_infer_pipeline():
            self._use_infer = True
            self._ready = True
            logger.info("[A2E Engine] Ready (INFER pipeline mode)")
            return

        # 2. フォールバック: Wav2Vec2 のみ
        logger.warning("[A2E Engine] INFER pipeline unavailable, loading Wav2Vec2 fallback")
        self._load_wav2vec_fallback()
        self._ready = True
        logger.info("[A2E Engine] Ready (Wav2Vec2 fallback mode)")

    def _find_lam_module(self) -> str:
        """LAM_Audio2Expression モジュールを探索して sys.path に追加"""
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            # 環境変数で指定
            os.environ.get("LAM_A2E_PATH"),
            # サービスディレクトリ直下 (Docker COPY)
            str(script_dir / "LAM_Audio2Expression"),
            # models ディレクトリ内
            str(self.model_dir / "LAM_Audio2Expression"),
            str(self.model_dir / "LAM_audio2exp" / "LAM_Audio2Expression"),
            # 親ディレクトリ
            str(self.model_dir.parent / "LAM_Audio2Expression"),
        ]

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                abs_path = os.path.abspath(candidate)
                if abs_path not in sys.path:
                    sys.path.insert(0, abs_path)
                logger.info(f"[A2E Engine] Found LAM_Audio2Expression: {abs_path}")
                return abs_path

        return None

    def _find_checkpoint(self) -> str:
        """
        A2E チェックポイントファイルを探索。

        HuggingFace からダウンロードした LAM_audio2exp_streaming.tar は
        gzip 圧縮の tar アーカイブで、中に pretrained_models/lam_audio2exp_streaming.tar
        (これが実際の PyTorch チェックポイント) が入っている。
        自動的に展開して内側のチェックポイントを返す。
        """
        import gzip
        import tarfile

        model_dir = self.model_dir

        # 実際の PyTorch チェックポイント (展開済み) を優先検索
        search_patterns = [
            model_dir / "pretrained_models" / "lam_audio2exp_streaming.tar",
            model_dir / "pretrained_models" / "LAM_audio2exp_streaming.tar",
            model_dir / "lam_audio2exp_streaming.pth",
            model_dir / "LAM_audio2exp_streaming.pth",
            model_dir / "LAM_audio2exp" / "pretrained_models" / "lam_audio2exp_streaming.tar",
            model_dir / "LAM_audio2exp" / "pretrained_models" / "LAM_audio2exp_streaming.tar",
        ]

        for path in search_patterns:
            if path.exists():
                return str(path)

        # 外側の gzip tar を見つけたら自動展開
        outer_candidates = [
            model_dir / "LAM_audio2exp_streaming.tar",
            model_dir / "lam_audio2exp_streaming.tar",
        ]
        for outer_path in outer_candidates:
            if outer_path.exists():
                try:
                    with tarfile.open(str(outer_path), "r:gz") as tf:
                        tf.extractall(path=str(model_dir))
                        logger.info(f"[A2E Engine] Extracted {outer_path}")
                    # 展開後に内側のチェックポイントを探索
                    inner = model_dir / "pretrained_models" / "lam_audio2exp_streaming.tar"
                    if inner.exists():
                        return str(inner)
                except Exception as e:
                    logger.warning(f"[A2E Engine] Failed to extract {outer_path}: {e}")

        # ワイルドカード検索
        tar_files = list(model_dir.rglob("*audio2exp*.tar"))
        # 外側の gzip tar は除外
        tar_files = [f for f in tar_files if f.stat().st_size < 400_000_000]
        if tar_files:
            return str(tar_files[0])
        pth_files = list(model_dir.rglob("*audio2exp*.pth"))
        if pth_files:
            return str(pth_files[0])

        return None

    def _find_wav2vec_dir(self) -> str:
        """wav2vec2-base-960h モデルディレクトリを探索"""
        candidates = [
            self.model_dir / "wav2vec2-base-960h",
        ]
        # GCS FUSE mount
        mount_path = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models")
        model_subdir = os.environ.get("MODEL_SUBDIR", "audio2exp")
        candidates.append(Path(mount_path) / model_subdir / "wav2vec2-base-960h")

        for path in candidates:
            if path.exists() and (path / "config.json").exists():
                return str(path)
        return None

    def _try_load_infer_pipeline(self) -> bool:
        """
        INFER パイプラインのロードを試行。

        old FastAPI app.py の実装をベースに:
        1. LAM_Audio2Expression モジュールを見つけて sys.path に追加
        2. default_config_parser で streaming config をパース
        3. INFER.build() でモデルをビルド
        4. warmup 推論を実行
        """
        import torch

        # 1. LAM_Audio2Expression モジュールを探索
        lam_path = self._find_lam_module()
        if not lam_path:
            logger.warning("[A2E Engine] LAM_Audio2Expression module not found")
            return False

        # 2. チェックポイントを探索
        checkpoint_path = self._find_checkpoint()
        if not checkpoint_path:
            logger.warning("[A2E Engine] No A2E checkpoint found")
            return False

        # 3. wav2vec2 ディレクトリを探索
        wav2vec_dir = self._find_wav2vec_dir()
        if not wav2vec_dir:
            logger.warning("[A2E Engine] wav2vec2-base-960h not found locally")
            # HuggingFace からダウンロードさせるためにデフォルト値を使用
            wav2vec_dir = "facebook/wav2vec2-base-960h"

        logger.info(f"[A2E Engine] Checkpoint: {checkpoint_path}")
        logger.info(f"[A2E Engine] Wav2Vec2: {wav2vec_dir}")

        try:
            from engines.defaults import default_config_parser
            from engines.infer import INFER

            # DDP 環境変数 (single-process 用)
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12345")

            # config ファイルのパス
            config_file = os.path.join(lam_path, "configs",
                                       "lam_audio2exp_config_streaming.py")
            if not os.path.exists(config_file):
                logger.warning(f"[A2E Engine] Config not found: {config_file}")
                return False

            # save_path (ログ出力先 - /tmp に設定)
            save_path = "/tmp/audio2exp_logs"
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "model"), exist_ok=True)

            # wav2vec2 config.json パスの解決
            if os.path.isdir(wav2vec_dir):
                wav2vec_config = os.path.join(wav2vec_dir, "config.json")
            else:
                # HuggingFace ID の場合、LAM モジュール内蔵の config を使用
                wav2vec_config = os.path.join(lam_path, "configs", "wav2vec2_config.json")

            # cfg_options: config のオーバーライド
            cfg_options = {
                "weight": checkpoint_path,
                "save_path": save_path,
                "model": {
                    "backbone": {
                        "wav2vec2_config_path": wav2vec_config,
                        "pretrained_encoder_path": wav2vec_dir,
                    }
                },
                "num_worker": 0,
                "batch_size": 1,
            }

            logger.info(f"[A2E Engine] Loading config: {config_file}")
            cfg = default_config_parser(config_file, cfg_options)

            # default_setup() をスキップ (DDP 関連の処理は不要)
            # 必要な設定を手動で設定
            cfg.device = torch.device(self.device)
            cfg.num_worker = 0
            cfg.num_worker_per_gpu = 0
            cfg.batch_size_per_gpu = 1
            cfg.batch_size_val_per_gpu = 1
            cfg.batch_size_test_per_gpu = 1

            logger.info("[A2E Engine] Building INFER model...")
            self._infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))

            # CPU + eval mode
            device = torch.device(self.device)
            self._infer.model.to(device)
            self._infer.model.eval()

            # Warmup 推論 (タイムアウト付き、失敗しても致命的ではない)
            logger.info("[A2E Engine] Running warmup inference (timeout=120s)...")
            import threading as _thr
            warmup_result = [None]  # [None]=running, [True]=ok, [Exception]=fail

            def _warmup():
                try:
                    dummy_audio = np.zeros(INFER_INPUT_SAMPLE_RATE, dtype=np.float32)
                    self._infer.infer_streaming_audio(
                        audio=dummy_audio, ssr=INFER_INPUT_SAMPLE_RATE, context=None
                    )
                    warmup_result[0] = True
                except Exception as exc:
                    warmup_result[0] = exc

            t = _thr.Thread(target=_warmup, daemon=True)
            t.start()
            t.join(timeout=120)
            if t.is_alive():
                logger.warning("[A2E Engine] Warmup timed out after 120s (non-fatal, inference may be slow on CPU)")
            elif isinstance(warmup_result[0], Exception):
                logger.warning(f"[A2E Engine] Warmup failed (non-fatal): {warmup_result[0]}")
            else:
                logger.info("[A2E Engine] Warmup succeeded")

            logger.info("[A2E Engine] INFER pipeline loaded successfully!")
            return True

        except ImportError as e:
            logger.warning(f"[A2E Engine] INFER import failed: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            logger.warning(f"[A2E Engine] INFER initialization failed: {e}")
            traceback.print_exc()
            return False

    def _load_wav2vec_fallback(self):
        """Wav2Vec2 フォールバックモードのロード"""
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        wav2vec_dir = self._find_wav2vec_dir()
        if wav2vec_dir:
            wav2vec_path = wav2vec_dir
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
        logger.info("[A2E Engine] Wav2Vec2 loaded (fallback mode)")

    def is_ready(self) -> bool:
        return self._ready

    def get_mode(self) -> str:
        """現在の推論モードを返す"""
        return "infer" if self._use_infer else "fallback"

    def process(self, audio_base64: str, audio_format: str = "mp3") -> dict:
        """
        音声を処理してブレンドシェイプ係数を生成

        Args:
            audio_base64: base64エンコードされた音声
            audio_format: 音声フォーマット (mp3, wav, pcm)

        Returns:
            {names: [52 strings], frames: [[52 floats], ...], frame_rate: int}
        """
        # 1. 音声デコード → PCM 16kHz
        audio_pcm = self._decode_audio(audio_base64, audio_format)
        duration = len(audio_pcm) / INFER_INPUT_SAMPLE_RATE
        logger.info(f"[A2E Engine] Audio decoded: {duration:.2f}s at 16kHz")

        # 2. 推論実行
        if self._use_infer:
            return self._process_with_infer(audio_pcm, duration)
        else:
            return self._process_with_fallback(audio_pcm, duration)

    def _process_with_infer(self, audio_pcm: np.ndarray, duration: float) -> dict:
        """
        INFER パイプラインで推論。

        infer_streaming_audio() を使用:
        - 音声をチャンクに分割
        - チャンクごとに推論 (コンテキスト引き継ぎ)
        - ポストプロセッシング込み (smooth_mouth, frame_blending,
          savitzky_golay, symmetrize, eye_blinks)
        """
        chunk_samples = INFER_INPUT_SAMPLE_RATE  # 1秒チャンク
        all_expressions = []
        context = None

        try:
            for start in range(0, len(audio_pcm), chunk_samples):
                end = min(start + chunk_samples, len(audio_pcm))
                chunk = audio_pcm[start:end]

                # 極端に短いチャンクはスキップ
                if len(chunk) < INFER_INPUT_SAMPLE_RATE // 10:
                    continue

                result, context = self._infer.infer_streaming_audio(
                    audio=chunk, ssr=INFER_INPUT_SAMPLE_RATE, context=context
                )
                expr = result.get("expression")
                if expr is not None:
                    all_expressions.append(expr.astype(np.float32))

            if not all_expressions:
                logger.warning("[A2E Engine] INFER produced no expression data")
                num_frames = max(1, int(duration * A2E_OUTPUT_FPS))
                expression = np.zeros((num_frames, 52), dtype=np.float32)
            else:
                expression = np.concatenate(all_expressions, axis=0)

            logger.info(f"[A2E Engine] INFER: {expression.shape[0]} frames, "
                        f"jawOpen range=[{expression[:, 24].min():.3f}, "
                        f"{expression[:, 24].max():.3f}]")  # jawOpen = index 24 in INFER order

            # フレームリストに変換
            frames = [frame.tolist() for frame in expression]

            return {
                "names": ARKIT_BLENDSHAPE_NAMES_INFER,
                "frames": frames,
                "frame_rate": A2E_OUTPUT_FPS,
            }

        except Exception as e:
            logger.error(f"[A2E Engine] INFER inference error: {e}")
            traceback.print_exc()
            # エラー時はフォールバック
            logger.warning("[A2E Engine] Falling back to Wav2Vec2 for this request")
            if hasattr(self, 'wav2vec_model'):
                return self._process_with_fallback(audio_pcm, duration)
            # Wav2Vec2 もない場合は空フレームを返す
            num_frames = max(1, int(duration * A2E_OUTPUT_FPS))
            return {
                "names": ARKIT_BLENDSHAPE_NAMES_INFER,
                "frames": [np.zeros(52).tolist()] * num_frames,
                "frame_rate": A2E_OUTPUT_FPS,
            }

    def _process_with_fallback(self, audio_pcm: np.ndarray, duration: float) -> dict:
        """Wav2Vec2 フォールバックで推論"""
        import torch

        inputs = self.wav2vec_processor(
            audio_pcm, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.wav2vec_model(input_values)
            features = outputs.last_hidden_state  # (1, T, 768)

        logger.info(f"[A2E Engine] Wav2Vec2 features: {tuple(features.shape)}")

        blendshapes = self._wav2vec_to_blendshapes_fallback(features, duration)
        frames = self._resample_to_fps(blendshapes, duration, A2E_OUTPUT_FPS)

        return {
            "names": ARKIT_BLENDSHAPE_NAMES_FALLBACK,
            "frames": frames,
            "frame_rate": A2E_OUTPUT_FPS,
        }

    def _decode_audio(self, audio_base64: str, audio_format: str) -> np.ndarray:
        """base64音声をPCM float32 16kHzにデコード"""
        audio_bytes = base64.b64decode(audio_base64)

        if audio_format in ("mp3", "wav", "ogg", "flac"):
            from pydub import AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0
        elif audio_format == "pcm":
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

        return samples

    def _wav2vec_to_blendshapes_fallback(
        self, features, duration: float
    ) -> np.ndarray:
        """
        A2Eデコーダーがない場合のフォールバック:
        Wav2Vec2の特徴量からリップシンク関連のブレンドシェイプを近似生成。
        """
        features_np = features.squeeze(0).cpu().numpy()  # (T, 768)
        n_frames = features_np.shape[0]

        blendshapes = np.zeros((n_frames, 52), dtype=np.float32)

        low_energy = np.abs(features_np[:, :256]).mean(axis=1)
        mid_energy = np.abs(features_np[:, 256:512]).mean(axis=1)
        high_energy = np.abs(features_np[:, 512:]).mean(axis=1)

        def normalize(x):
            x_min = x.min()
            x_max = x.max()
            if x_max - x_min < 1e-6:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        low_norm = normalize(low_energy)
        mid_norm = normalize(mid_energy)
        high_norm = normalize(high_energy)
        speech_activity = normalize(low_energy + mid_energy + high_energy)

        idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES_FALLBACK)}

        # リップシンク
        blendshapes[:, idx["jawOpen"]] = np.clip(low_norm * 0.8, 0, 1)
        blendshapes[:, idx["mouthClose"]] = np.clip(1.0 - low_norm * 0.8, 0, 1) * speech_activity
        funnel = np.clip(mid_norm * 0.5 - low_norm * 0.2, 0, 1)
        blendshapes[:, idx["mouthFunnel"]] = funnel
        blendshapes[:, idx["mouthPucker"]] = np.clip(funnel * 0.7, 0, 1)
        smile = np.clip(high_norm * 0.4 - mid_norm * 0.1, 0, 1)
        blendshapes[:, idx["mouthSmileLeft"]] = smile
        blendshapes[:, idx["mouthSmileRight"]] = smile
        lower_down = np.clip(low_norm * 0.5, 0, 1)
        blendshapes[:, idx["mouthLowerDownLeft"]] = lower_down
        blendshapes[:, idx["mouthLowerDownRight"]] = lower_down
        upper_up = np.clip(low_norm * 0.3, 0, 1)
        blendshapes[:, idx["mouthUpperUpLeft"]] = upper_up
        blendshapes[:, idx["mouthUpperUpRight"]] = upper_up
        stretch = np.clip((mid_norm + high_norm) * 0.25, 0, 1)
        blendshapes[:, idx["mouthStretchLeft"]] = stretch
        blendshapes[:, idx["mouthStretchRight"]] = stretch

        # 非リップ関連
        blendshapes[:, idx["browInnerUp"]] = np.clip(speech_activity * 0.15, 0, 1)
        blendshapes[:, idx["cheekSquintLeft"]] = smile * 0.3
        blendshapes[:, idx["cheekSquintRight"]] = smile * 0.3
        nose = np.clip(speech_activity * 0.1, 0, 1)
        blendshapes[:, idx["noseSneerLeft"]] = nose
        blendshapes[:, idx["noseSneerRight"]] = nose

        # 無音フレームは抑制
        silence_mask = speech_activity < 0.1
        blendshapes[silence_mask] *= 0.1

        # スムージング
        if n_frames > 3:
            kernel = np.ones(3) / 3
            for i in range(52):
                blendshapes[:, i] = np.convolve(blendshapes[:, i], kernel, mode='same')

        logger.info(f"[A2E Engine] Fallback: {n_frames} frames, "
                    f"jawOpen=[{blendshapes[:, idx['jawOpen']].min():.3f}, "
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
            source_indices = np.linspace(0, n_source - 1, n_target)
            frames = np.zeros((n_target, 52), dtype=np.float32)
            for i in range(52):
                frames[:, i] = np.interp(
                    source_indices, np.arange(n_source), blendshapes[:, i]
                )

        return [frame.tolist() for frame in frames]
