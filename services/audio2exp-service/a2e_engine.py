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
        """エンジン初期化 - INFER パイプラインを優先的にロード（タイムアウト付き）"""
        import threading

        # INFER パイプラインのロードタイムアウト (秒)
        # tar事前解凍済みでも INFER.build() にCPUで10-15分かかる
        # 以前の成功実績: ENGINE_LOAD_TIMEOUT=1500 で完走
        infer_timeout = int(os.environ.get("INFER_LOAD_TIMEOUT", "1500"))
        logger.info(f"[A2E Engine] INFER pipeline load timeout: {infer_timeout}s")

        # 1. INFER パイプラインをタイムアウト付きスレッドで試行
        infer_result = [None]  # None=running, True=ok, Exception=fail

        def _try_infer():
            try:
                infer_result[0] = self._try_load_infer_pipeline()
            except Exception as e:
                infer_result[0] = e

        t = threading.Thread(target=_try_infer, daemon=True)
        t.start()
        t.join(timeout=infer_timeout)

        if t.is_alive():
            logger.warning(
                f"[A2E Engine] INFER pipeline timed out after {infer_timeout}s "
                f"(model build too slow on CPU). Falling back to Wav2Vec2."
            )
        elif isinstance(infer_result[0], Exception):
            logger.warning(f"[A2E Engine] INFER pipeline failed: {infer_result[0]}")
        elif infer_result[0] is True:
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

        Non-Streaming フルモデル (lam_audio2exp.tar) を優先検索。
        フォールバックとして Streaming モデルも検索。

        HuggingFace からダウンロードした tar は gzip 圧縮の tar アーカイブで、
        中に pretrained_models/lam_audio2exp.tar (PyTorch チェックポイント) が
        入っている。自動的に展開して内側のチェックポイントを返す。
        """
        import tarfile

        model_dir = self.model_dir

        # 1. Non-Streaming フルモデル (優先)
        full_model_patterns = [
            model_dir / "pretrained_models" / "lam_audio2exp.tar",
            model_dir / "pretrained_models" / "LAM_audio2exp.tar",
            model_dir / "lam_audio2exp.pth",
            model_dir / "LAM_audio2exp" / "pretrained_models" / "lam_audio2exp.tar",
            model_dir / "LAM_audio2exp" / "pretrained_models" / "LAM_audio2exp.tar",
        ]
        for path in full_model_patterns:
            if path.exists():
                logger.info(f"[A2E Engine] Found Non-Streaming full model: {path}")
                return str(path)

        # 2. 外側の gzip tar を展開 (Non-Streaming 優先)
        outer_candidates = [
            (model_dir / "LAM_audio2exp.tar", "lam_audio2exp.tar"),
            (model_dir / "lam_audio2exp.tar", "lam_audio2exp.tar"),
            (model_dir / "LAM_audio2exp_streaming.tar", "lam_audio2exp_streaming.tar"),
            (model_dir / "lam_audio2exp_streaming.tar", "lam_audio2exp_streaming.tar"),
        ]
        for outer_path, inner_name in outer_candidates:
            if outer_path.exists():
                try:
                    with tarfile.open(str(outer_path), "r:gz") as tf:
                        tf.extractall(path=str(model_dir))
                        logger.info(f"[A2E Engine] Extracted {outer_path}")
                    inner = model_dir / "pretrained_models" / inner_name
                    if inner.exists():
                        return str(inner)
                except Exception as e:
                    logger.warning(f"[A2E Engine] Failed to extract {outer_path}: {e}")

        # 3. Streaming モデル (フォールバック)
        streaming_patterns = [
            model_dir / "pretrained_models" / "lam_audio2exp_streaming.tar",
            model_dir / "pretrained_models" / "LAM_audio2exp_streaming.tar",
            model_dir / "lam_audio2exp_streaming.pth",
            model_dir / "LAM_audio2exp" / "pretrained_models" / "lam_audio2exp_streaming.tar",
        ]
        for path in streaming_patterns:
            if path.exists():
                logger.warning(f"[A2E Engine] Non-Streaming model not found, falling back to Streaming: {path}")
                return str(path)

        # 4. ワイルドカード検索
        tar_files = list(model_dir.rglob("*audio2exp*.tar"))
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
            import time as _time

            t_import = _time.time()
            logger.info("[A2E Engine] Importing INFER modules...")
            from engines.defaults import default_config_parser
            from engines.infer import INFER
            logger.info(f"[A2E Engine] Import done in {_time.time() - t_import:.1f}s")

            # DDP 環境変数 (single-process 用)
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12345")

            # config ファイルのパス — チェックポイント名からモデル種別を判定
            is_streaming = "streaming" in checkpoint_path.lower()
            if is_streaming:
                config_name = "lam_audio2exp_config_streaming.py"
                logger.info("[A2E Engine] Using STREAMING config (lightweight model)")
            else:
                config_name = "lam_audio2exp_config.py"
                logger.info("[A2E Engine] Using NON-STREAMING config (full model with 6-layer Transformer)")

            config_file = os.path.join(lam_path, "configs", config_name)
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
                # ポストプロセッシング強化 (Streaming config のデフォルトを上書き)
                "movement_smooth": True,
                "brow_movement": True,
            }

            t_cfg = _time.time()
            logger.info(f"[A2E Engine] Loading config: {config_file}")
            cfg = default_config_parser(config_file, cfg_options)
            logger.info(f"[A2E Engine] Config loaded in {_time.time() - t_cfg:.1f}s")

            # default_setup() をスキップ (DDP 関連の処理は不要)
            # 必要な設定を手動で設定
            cfg.device = torch.device(self.device)
            cfg.num_worker = 0
            cfg.num_worker_per_gpu = 0
            cfg.batch_size_per_gpu = 1
            cfg.batch_size_val_per_gpu = 1
            cfg.batch_size_test_per_gpu = 1

            t_build = _time.time()
            logger.info("[A2E Engine] Building INFER model (this may take several minutes on CPU)...")
            self._infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
            logger.info(f"[A2E Engine] INFER.build() done in {_time.time() - t_build:.1f}s")

            # CPU + eval mode
            device = torch.device(self.device)
            logger.info("[A2E Engine] Moving model to device and setting eval mode...")
            self._infer.model.to(device)
            self._infer.model.eval()
            logger.info(f"[A2E Engine] Model ready on {device}")

            # Warmup 推論 (WARMUP_TIMEOUT 環境変数で制御)
            # WARMUP_TIMEOUT=0 でスキップ（成功事例のデプロイパラメータ）
            warmup_timeout = int(os.environ.get("WARMUP_TIMEOUT", "120"))
            if warmup_timeout == 0:
                logger.info("[A2E Engine] Warmup SKIPPED (WARMUP_TIMEOUT=0)")
            else:
                logger.info(f"[A2E Engine] Running warmup inference (batch mode, timeout={warmup_timeout}s)...")
                import threading as _thr
                warmup_result = [None]  # [None]=running, [True]=ok, [Exception]=fail

                def _warmup():
                    try:
                        dummy_audio = np.zeros(INFER_INPUT_SAMPLE_RATE, dtype=np.float32)
                        self._infer.infer_batch_audio(
                            audio=dummy_audio, ssr=INFER_INPUT_SAMPLE_RATE
                        )
                        warmup_result[0] = True
                    except Exception as exc:
                        warmup_result[0] = exc

                t = _thr.Thread(target=_warmup, daemon=True)
                t.start()
                t.join(timeout=warmup_timeout)
                if t.is_alive():
                    logger.warning(f"[A2E Engine] Warmup timed out after {warmup_timeout}s (non-fatal, inference may be slow on CPU)")
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
        INFER パイプラインで推論 (バッチモード)。

        infer_batch_audio() を使用:
        - 音声全体を一括でモデルに入力 (チャンク分割なし)
        - 完全版ポストプロセッシング (smooth_mouth_movements,
          apply_random_brow_movement, savitzky_golay, symmetrize, eye_blinks)
        """
        try:
            result = self._infer.infer_batch_audio(
                audio=audio_pcm, ssr=INFER_INPUT_SAMPLE_RATE
            )
            expression = result.get("expression")

            if expression is None or len(expression) == 0:
                logger.warning("[A2E Engine] INFER produced no expression data")
                num_frames = max(1, int(duration * A2E_OUTPUT_FPS))
                expression = np.zeros((num_frames, 52), dtype=np.float32)

            logger.info(f"[A2E Engine] INFER batch: {expression.shape[0]} frames, "
                        f"jawOpen range=[{expression[:, 24].min():.3f}, "
                        f"{expression[:, 24].max():.3f}]")

            frames = [frame.tolist() for frame in expression]

            return {
                "names": ARKIT_BLENDSHAPE_NAMES_INFER,
                "frames": frames,
                "frame_rate": A2E_OUTPUT_FPS,
            }

        except Exception as e:
            logger.error(f"[A2E Engine] INFER batch inference error: {e}")
            traceback.print_exc()
            logger.warning("[A2E Engine] Falling back to Wav2Vec2 for this request")
            if hasattr(self, 'wav2vec_model'):
                return self._process_with_fallback(audio_pcm, duration)
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
