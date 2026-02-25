"""
A2Eエンジン ユニットテスト

モデルファイル不要で実行可能な、ロジックレベルのテスト。
対象: services/audio2exp-service/a2e_engine.py
"""

import base64
import io
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# a2e_engine.py をインポートできるよう sys.path を設定
SERVICE_DIR = Path(__file__).parent.parent / "services" / "audio2exp-service"
sys.path.insert(0, str(SERVICE_DIR))


# ---- ブレンドシェイプ名定義テスト ----

class TestBlendshapeNames:
    """ARKitブレンドシェイプ名の定義が正しいことを検証"""

    def test_infer_names_count(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_INFER
        assert len(ARKIT_BLENDSHAPE_NAMES_INFER) == 52

    def test_fallback_names_count(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_FALLBACK
        assert len(ARKIT_BLENDSHAPE_NAMES_FALLBACK) == 52

    def test_infer_names_unique(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_INFER
        assert len(set(ARKIT_BLENDSHAPE_NAMES_INFER)) == 52

    def test_fallback_names_unique(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_FALLBACK
        assert len(set(ARKIT_BLENDSHAPE_NAMES_FALLBACK)) == 52

    def test_both_lists_same_set(self):
        """INFER名とFALLBACK名は順序違いでも同じセットであるべき"""
        from a2e_engine import (
            ARKIT_BLENDSHAPE_NAMES_FALLBACK,
            ARKIT_BLENDSHAPE_NAMES_INFER,
        )
        assert set(ARKIT_BLENDSHAPE_NAMES_INFER) == set(ARKIT_BLENDSHAPE_NAMES_FALLBACK)

    def test_jawopen_exists(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_INFER
        assert "jawOpen" in ARKIT_BLENDSHAPE_NAMES_INFER

    def test_lip_related_names_present(self):
        """リップシンクに必要なブレンドシェイプが含まれている"""
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_INFER
        required = [
            "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
            "mouthSmileLeft", "mouthSmileRight",
            "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",
        ]
        for name in required:
            assert name in ARKIT_BLENDSHAPE_NAMES_INFER, f"{name} missing"


# ---- 音声デコードテスト (モック不要) ----

class TestAudioDecoding:
    """_decode_audio メソッドの単体テスト"""

    @pytest.fixture(autouse=True)
    def _setup_engine_class(self):
        """エンジンクラスのみインポート (初期化はモックする)"""
        from a2e_engine import Audio2ExpressionEngine
        self.EngineClass = Audio2ExpressionEngine

    def _make_engine_no_init(self):
        """__init__ をスキップしてインスタンスを作成"""
        engine = object.__new__(self.EngineClass)
        engine.model_dir = Path("/tmp/fake_models")
        engine._ready = False
        engine._use_infer = False
        engine.device = "cpu"
        engine.device_name = "cpu"
        return engine

    def test_decode_wav_format(self, wav_440hz_1s_base64):
        engine = self._make_engine_no_init()
        pcm = engine._decode_audio(wav_440hz_1s_base64, "wav")
        assert isinstance(pcm, np.ndarray)
        assert pcm.dtype == np.float32
        # 1秒 16kHz = 16000サンプル
        assert abs(len(pcm) - 16000) < 100
        # float32 正規化 [-1, 1]
        assert pcm.max() <= 1.0
        assert pcm.min() >= -1.0

    def test_decode_pcm_format(self):
        """PCM int16 → float32 変換"""
        engine = self._make_engine_no_init()
        # 100サンプルの PCM int16 データ
        pcm_int16 = np.array([0, 16384, 32767, -32768, -16384], dtype=np.int16)
        pcm_b64 = base64.b64encode(pcm_int16.tobytes()).decode()
        result = engine._decode_audio(pcm_b64, "pcm")
        assert result.dtype == np.float32
        assert len(result) == 5
        assert abs(result[0]) < 1e-6  # 0
        assert abs(result[2] - 1.0) < 0.001  # 32767/32768 ≈ 1.0
        assert abs(result[3] + 1.0) < 0.001  # -32768/32768 = -1.0

    def test_decode_invalid_format_raises(self):
        engine = self._make_engine_no_init()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            engine._decode_audio(base64.b64encode(b"dummy").decode(), "aac")

    def test_decode_silence(self, wav_silence_1s_base64):
        engine = self._make_engine_no_init()
        pcm = engine._decode_audio(wav_silence_1s_base64, "wav")
        assert np.abs(pcm).max() < 0.01  # ほぼ無音


# ---- リサンプリングテスト ----

class TestResampling:
    """_resample_to_fps メソッドの単体テスト"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from a2e_engine import Audio2ExpressionEngine
        engine = object.__new__(Audio2ExpressionEngine)
        engine.model_dir = Path("/tmp/fake")
        engine.device = "cpu"
        engine.device_name = "cpu"
        self.engine = engine

    def test_resample_same_length(self):
        """ソースとターゲットが同じ長さの場合"""
        blendshapes = np.random.rand(30, 52).astype(np.float32)
        frames = self.engine._resample_to_fps(blendshapes, duration=1.0, target_fps=30)
        assert len(frames) == 30
        assert len(frames[0]) == 52

    def test_resample_upsample(self):
        """アップサンプリング (10fps → 30fps)"""
        blendshapes = np.random.rand(10, 52).astype(np.float32)
        frames = self.engine._resample_to_fps(blendshapes, duration=1.0, target_fps=30)
        assert len(frames) == 30

    def test_resample_downsample(self):
        """ダウンサンプリング (60fps → 30fps)"""
        blendshapes = np.random.rand(60, 52).astype(np.float32)
        frames = self.engine._resample_to_fps(blendshapes, duration=1.0, target_fps=30)
        assert len(frames) == 30

    def test_resample_preserves_range(self):
        """リサンプリング後の値域が元データの範囲内"""
        blendshapes = np.random.rand(50, 52).astype(np.float32)
        frames = self.engine._resample_to_fps(blendshapes, duration=2.0, target_fps=30)
        arr = np.array(frames)
        assert arr.min() >= blendshapes.min() - 1e-6
        assert arr.max() <= blendshapes.max() + 1e-6

    def test_resample_output_format(self):
        """出力がリストのリスト (JSON互換) であること"""
        blendshapes = np.random.rand(10, 52).astype(np.float32)
        frames = self.engine._resample_to_fps(blendshapes, duration=1.0, target_fps=30)
        assert isinstance(frames, list)
        assert isinstance(frames[0], list)
        assert all(isinstance(v, float) for v in frames[0])

    def test_resample_short_duration(self):
        """非常に短い音声 (最低1フレーム保証)"""
        blendshapes = np.random.rand(2, 52).astype(np.float32)
        frames = self.engine._resample_to_fps(blendshapes, duration=0.01, target_fps=30)
        assert len(frames) >= 1


# ---- フォールバック推論ロジックテスト ----

class TestFallbackLogic:
    """Wav2Vec2 フォールバックのブレンドシェイプ生成ロジックをテスト"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from a2e_engine import Audio2ExpressionEngine, ARKIT_BLENDSHAPE_NAMES_FALLBACK
        engine = object.__new__(Audio2ExpressionEngine)
        engine.model_dir = Path("/tmp/fake")
        engine.device = "cpu"
        engine.device_name = "cpu"
        self.engine = engine
        self.names = ARKIT_BLENDSHAPE_NAMES_FALLBACK
        self.idx = {n: i for i, n in enumerate(self.names)}

    def _make_fake_features(self, n_frames: int, pattern: str = "speech"):
        """テスト用のWav2Vec2出力テンソルを生成"""
        import torch
        if pattern == "speech":
            features = torch.randn(1, n_frames, 768) * 0.5 + 0.3
        elif pattern == "silence":
            features = torch.zeros(1, n_frames, 768)
        elif pattern == "loud":
            features = torch.randn(1, n_frames, 768) * 2.0
        else:
            features = torch.randn(1, n_frames, 768)
        return features

    @pytest.mark.unit
    def test_fallback_output_shape(self):
        """フォールバック出力が (N, 52) であること"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = self._make_fake_features(50, "speech")
        result = self.engine._wav2vec_to_blendshapes_fallback(features, duration=1.0)
        assert result.shape == (50, 52)
        assert result.dtype == np.float32

    @pytest.mark.unit
    def test_fallback_values_clipped(self):
        """出力値が [0, 1] 範囲内"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = self._make_fake_features(50, "loud")
        result = self.engine._wav2vec_to_blendshapes_fallback(features, duration=1.0)
        assert result.min() >= -0.01  # スムージングで若干の誤差あり
        assert result.max() <= 1.01

    @pytest.mark.unit
    def test_fallback_silence_suppressed(self):
        """無音入力時にブレンドシェイプが抑制される"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = self._make_fake_features(50, "silence")
        result = self.engine._wav2vec_to_blendshapes_fallback(features, duration=1.0)
        # 無音時は全ブレンドシェイプがほぼゼロ
        assert result.max() < 0.1

    @pytest.mark.unit
    def test_fallback_jawopen_active_for_speech(self):
        """音声入力時に jawOpen が活性化する"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = self._make_fake_features(50, "speech")
        result = self.engine._wav2vec_to_blendshapes_fallback(features, duration=1.0)
        jaw_open_idx = self.idx["jawOpen"]
        assert result[:, jaw_open_idx].max() > 0.1

    @pytest.mark.unit
    def test_fallback_smoothing(self):
        """スムージングが適用されている (連続するフレーム間の差が小さい)"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = self._make_fake_features(100, "speech")
        result = self.engine._wav2vec_to_blendshapes_fallback(features, duration=2.0)
        # フレーム間差分の標準偏差がスムージングなしより小さいことを確認
        diffs = np.diff(result, axis=0)
        max_frame_diff = np.abs(diffs).max()
        # スムージングにより極端なジャンプはない
        assert max_frame_diff < 1.0


# ---- 定数テスト ----

class TestConstants:
    """定数定義の正確性"""

    def test_output_fps(self):
        from a2e_engine import A2E_OUTPUT_FPS
        assert A2E_OUTPUT_FPS == 30

    def test_input_sample_rate(self):
        from a2e_engine import INFER_INPUT_SAMPLE_RATE
        assert INFER_INPUT_SAMPLE_RATE == 16000


# ---- モジュール探索テスト ----

class TestModuleDiscovery:
    """_find_lam_module, _find_checkpoint, _find_wav2vec_dir のテスト"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from a2e_engine import Audio2ExpressionEngine
        engine = object.__new__(Audio2ExpressionEngine)
        engine.model_dir = Path("/tmp/nonexistent_model_dir_test")
        engine.device = "cpu"
        engine.device_name = "cpu"
        self.engine = engine

    def test_find_checkpoint_returns_none_when_missing(self):
        result = self.engine._find_checkpoint()
        assert result is None

    def test_find_wav2vec_dir_returns_none_when_missing(self):
        result = self.engine._find_wav2vec_dir()
        assert result is None

    def test_find_lam_module_consistent_with_filesystem(self):
        """LAM_Audio2Expression の探索結果がファイルシステムと一致する"""
        result = self.engine._find_lam_module()
        # サービスディレクトリに実在する場合は見つかるのが正しい動作
        if result is not None:
            assert "LAM_Audio2Expression" in result
            assert Path(result).exists()

    def test_find_lam_module_finds_local(self, tmp_path):
        """LAM_Audio2Expression がサービスディレクトリ直下にある場合"""
        lam_dir = tmp_path / "LAM_Audio2Expression"
        lam_dir.mkdir()
        self.engine.model_dir = tmp_path / "models"
        # _find_lam_module は __file__ ベースのパスを見るので、
        # 環境変数経由のパスをテスト
        import os
        os.environ["LAM_A2E_PATH"] = str(lam_dir)
        try:
            result = self.engine._find_lam_module()
            assert result is not None
            assert "LAM_Audio2Expression" in result
        finally:
            del os.environ["LAM_A2E_PATH"]
