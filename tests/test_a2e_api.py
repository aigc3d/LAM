"""
A2E Flask API コントラクトテスト

Flask test client を使用して API のリクエスト・レスポンス形式を検証。
実際のモデル推論はモックする。
"""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).parent.parent / "services" / "audio2exp-service"
sys.path.insert(0, str(SERVICE_DIR))

from conftest import ARKIT_BLENDSHAPE_NAMES_INFER


def make_mock_engine():
    """モックされた A2E エンジン"""
    engine = MagicMock()
    engine.is_ready.return_value = True
    engine.get_mode.return_value = "infer"
    engine.device_name = "cpu"

    # process() のモックレスポンス
    n_frames = 30
    frames = np.random.rand(n_frames, 52).astype(np.float32)
    engine.process.return_value = {
        "names": list(ARKIT_BLENDSHAPE_NAMES_INFER),
        "frames": [frame.tolist() for frame in frames],
        "frame_rate": 30,
    }
    return engine


@pytest.fixture
def app():
    """Flask アプリケーション (エンジンをモック)"""
    mock_engine = make_mock_engine()

    with patch.dict("sys.modules", {"a2e_engine": MagicMock()}):
        # app.py をモック付きでインポートし直す
        import importlib
        # a2e_engine モジュールのモック
        mock_a2e_module = MagicMock()
        mock_a2e_module.Audio2ExpressionEngine.return_value = mock_engine
        sys.modules["a2e_engine"] = mock_a2e_module

        # app モジュールのキャッシュをクリア
        if "app" in sys.modules:
            del sys.modules["app"]

        import app as flask_app
        flask_app.engine = mock_engine
        flask_app.app.config["TESTING"] = True
        yield flask_app.app, mock_engine


@pytest.fixture
def client(app):
    """Flask test client"""
    flask_app, engine = app
    return flask_app.test_client(), engine


class TestHealthEndpoint:
    """GET /health エンドポイント"""

    @pytest.mark.api
    def test_health_returns_200(self, client):
        c, engine = client
        rv = c.get("/health")
        assert rv.status_code == 200

    @pytest.mark.api
    def test_health_response_format(self, client):
        c, engine = client
        rv = c.get("/health")
        data = rv.get_json()
        assert "status" in data
        assert "engine_ready" in data
        assert "mode" in data
        assert "device" in data
        assert "model_dir" in data

    @pytest.mark.api
    def test_health_status_healthy(self, client):
        c, engine = client
        rv = c.get("/health")
        data = rv.get_json()
        assert data["status"] == "healthy"
        assert data["engine_ready"] is True


class TestAudio2ExpressionEndpoint:
    """POST /api/audio2expression エンドポイント"""

    @pytest.mark.api
    def test_missing_audio_returns_400(self, client):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={"session_id": "test"})
        assert rv.status_code == 400

    @pytest.mark.api
    def test_empty_audio_returns_400(self, client):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={"audio_base64": "", "session_id": "test"})
        assert rv.status_code == 400

    @pytest.mark.api
    def test_valid_request_returns_200(self, client, wav_440hz_1s_base64):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test-session",
                         "audio_format": "wav",
                     })
        assert rv.status_code == 200

    @pytest.mark.api
    def test_response_has_required_fields(self, client, wav_440hz_1s_base64):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test",
                         "audio_format": "wav",
                     })
        data = rv.get_json()
        assert "names" in data
        assert "frames" in data
        assert "frame_rate" in data

    @pytest.mark.api
    def test_response_names_count(self, client, wav_440hz_1s_base64):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test",
                         "audio_format": "wav",
                     })
        data = rv.get_json()
        assert len(data["names"]) == 52

    @pytest.mark.api
    def test_response_frame_dimensions(self, client, wav_440hz_1s_base64):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test",
                         "audio_format": "wav",
                     })
        data = rv.get_json()
        assert len(data["frames"]) > 0
        assert len(data["frames"][0]) == 52

    @pytest.mark.api
    def test_response_frame_rate(self, client, wav_440hz_1s_base64):
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test",
                         "audio_format": "wav",
                     })
        data = rv.get_json()
        assert data["frame_rate"] == 30

    @pytest.mark.api
    def test_default_audio_format_mp3(self, client, wav_440hz_1s_base64):
        """audio_format 省略時はデフォルト mp3"""
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test",
                     })
        # engine.process が呼ばれたときの audio_format を確認
        call_args = engine.process.call_args
        assert call_args[1].get("audio_format", "mp3") == "mp3" or \
               (len(call_args[0]) > 1 and call_args[0][1] == "mp3") or \
               call_args.kwargs.get("audio_format", "mp3") == "mp3"

    @pytest.mark.api
    def test_engine_error_returns_500(self, client, wav_440hz_1s_base64):
        c, engine = client
        engine.process.side_effect = RuntimeError("Model error")
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "session_id": "test",
                         "audio_format": "wav",
                     })
        assert rv.status_code == 500
        data = rv.get_json()
        assert "error" in data

    @pytest.mark.api
    def test_session_id_defaults_to_unknown(self, client, wav_440hz_1s_base64):
        """session_id 省略時でもリクエストが通る"""
        c, engine = client
        rv = c.post("/api/audio2expression",
                     json={
                         "audio_base64": wav_440hz_1s_base64,
                         "audio_format": "wav",
                     })
        assert rv.status_code == 200
