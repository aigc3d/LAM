"""
ブレンドシェイプ データ形式バリデーションテスト

A2E出力の52次元ARKitブレンドシェイプデータが
フロントエンド (gourmet-sp) の期待形式と整合するかを検証。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).parent.parent / "services" / "audio2exp-service"
sys.path.insert(0, str(SERVICE_DIR))

from conftest import ARKIT_BLENDSHAPE_NAMES_FALLBACK, ARKIT_BLENDSHAPE_NAMES_INFER


# ---- Apple ARKit 公式仕様との整合性 ----

# Apple ARKit 公式 52 ブレンドシェイプ (アルファベット順ではなく機能別グループ)
ARKIT_OFFICIAL_NAMES = {
    # 目
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # 顎
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    # 口
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    # 眉
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    # 頬
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    # 鼻
    "noseSneerLeft", "noseSneerRight",
    # 舌
    "tongueOut",
}


class TestARKitCompliance:
    """Apple ARKit 52ブレンドシェイプ仕様との整合"""

    def test_official_count(self):
        assert len(ARKIT_OFFICIAL_NAMES) == 52

    def test_infer_matches_arkit(self):
        assert set(ARKIT_BLENDSHAPE_NAMES_INFER) == ARKIT_OFFICIAL_NAMES

    def test_fallback_matches_arkit(self):
        assert set(ARKIT_BLENDSHAPE_NAMES_FALLBACK) == ARKIT_OFFICIAL_NAMES

    def test_a2e_engine_infer_names_match_arkit(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_INFER as engine_names
        assert set(engine_names) == ARKIT_OFFICIAL_NAMES

    def test_a2e_engine_fallback_names_match_arkit(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_FALLBACK as engine_names
        assert set(engine_names) == ARKIT_OFFICIAL_NAMES


# ---- INFER パイプラインのインデックスマッピング ----

class TestINFERIndexMapping:
    """INFER パイプラインのブレンドシェイプインデックスが正しいことを検証。
    a2e_engine.py:428 の jawOpen=index 24 が正しいか確認。"""

    def test_jawopen_index_in_infer_order(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_INFER
        assert ARKIT_BLENDSHAPE_NAMES_INFER[24] == "jawOpen"

    def test_jawopen_index_in_fallback_order(self):
        from a2e_engine import ARKIT_BLENDSHAPE_NAMES_FALLBACK
        idx = ARKIT_BLENDSHAPE_NAMES_FALLBACK.index("jawOpen")
        assert idx == 17  # fallback order


# ---- レスポンス形式テスト ----

class TestResponseFormat:
    """API レスポンスのデータ形式が期待通りか検証"""

    def test_mock_response_structure(self, mock_a2e_response):
        data = mock_a2e_response
        assert "names" in data
        assert "frames" in data
        assert "frame_rate" in data

    def test_mock_response_names_type(self, mock_a2e_response):
        data = mock_a2e_response
        assert isinstance(data["names"], list)
        assert all(isinstance(n, str) for n in data["names"])

    def test_mock_response_frames_type(self, mock_a2e_response):
        data = mock_a2e_response
        assert isinstance(data["frames"], list)
        assert all(isinstance(f, list) for f in data["frames"])
        assert all(isinstance(v, float) for v in data["frames"][0])

    def test_mock_response_json_serializable(self, mock_a2e_response):
        """レスポンスがJSON直列化可能"""
        json_str = json.dumps(mock_a2e_response)
        parsed = json.loads(json_str)
        assert len(parsed["names"]) == 52
        assert len(parsed["frames"]) > 0

    def test_frames_values_in_range(self, mock_a2e_response):
        """フレーム値が 0～1 の範囲内"""
        data = mock_a2e_response
        for frame in data["frames"]:
            for val in frame:
                assert 0.0 <= val <= 1.0, f"Value {val} out of [0, 1] range"


# ---- フロントエンド統合テスト ----

class TestFrontendIntegration:
    """フロントエンド (vrm-expression-manager.ts) が期待するデータ形式との整合"""

    def test_expression_manager_mapping(self, sample_blendshape_frames):
        """ExpressionManager のマッピングロジック再現:
        jawOpen × 0.6 + (mouthLowerDownL + mouthLowerDownR) / 2 × 0.2
        + (mouthUpperUpL + mouthUpperUpR) / 2 × 0.1
        + mouthFunnel × 0.05 + mouthPucker × 0.05
        → mouthOpenness (0.0 ~ 1.0)
        """
        idx = sample_blendshape_frames["idx"]
        frame_a = sample_blendshape_frames["a"]

        jaw_open = frame_a[idx["jawOpen"]]
        lower_down = (frame_a[idx["mouthLowerDownLeft"]] + frame_a[idx["mouthLowerDownRight"]]) / 2
        upper_up = (frame_a[idx["mouthUpperUpLeft"]] + frame_a[idx["mouthUpperUpRight"]]) / 2
        funnel = frame_a[idx["mouthFunnel"]]
        pucker = frame_a[idx["mouthPucker"]]

        mouth_openness = (
            jaw_open * 0.6
            + lower_down * 0.2
            + upper_up * 0.1
            + funnel * 0.05
            + pucker * 0.05
        )
        assert 0.0 <= mouth_openness <= 1.0
        # 「あ」は口が大きく開くので openness が高い
        assert mouth_openness > 0.3

    def test_vowel_a_pattern(self, sample_blendshape_frames):
        """「あ」: jawOpen が高い"""
        idx = sample_blendshape_frames["idx"]
        frame = sample_blendshape_frames["a"]
        assert frame[idx["jawOpen"]] > 0.5

    def test_vowel_i_pattern(self, sample_blendshape_frames):
        """「い」: mouthSmile が高い、jawOpen が低い"""
        idx = sample_blendshape_frames["idx"]
        frame = sample_blendshape_frames["i"]
        assert frame[idx["jawOpen"]] < 0.3
        assert frame[idx["mouthSmileLeft"]] > 0.3
        assert frame[idx["mouthSmileRight"]] > 0.3

    def test_vowel_u_pattern(self, sample_blendshape_frames):
        """「う」: mouthPucker/Funnel が高い"""
        idx = sample_blendshape_frames["idx"]
        frame = sample_blendshape_frames["u"]
        assert frame[idx["mouthPucker"]] > 0.3
        assert frame[idx["mouthFunnel"]] > 0.2

    def test_lam_avatar_controller_format(self, mock_a2e_response):
        """lamAvatarController.queueExpressionFrames() が期待する形式:
        frames: [{name: weight}, ...] の配列
        """
        data = mock_a2e_response
        # フロントエンドの変換ロジック再現
        converted_frames = []
        for frame_weights in data["frames"]:
            frame_dict = {}
            for name, weight in zip(data["names"], frame_weights):
                frame_dict[name] = weight
            converted_frames.append(frame_dict)

        assert len(converted_frames) == len(data["frames"])
        assert "jawOpen" in converted_frames[0]
        assert isinstance(converted_frames[0]["jawOpen"], float)


# ---- INFER/Fallback 名前順序一貫性 ----

class TestNameOrderConsistency:
    """INFER と Fallback で名前順序が異なることの影響テスト"""

    def test_name_order_differs(self):
        """INFER と Fallback の名前順序は異なる (意図的な設計)"""
        assert ARKIT_BLENDSHAPE_NAMES_INFER != ARKIT_BLENDSHAPE_NAMES_FALLBACK

    def test_name_lookup_by_dict(self):
        """名前→インデックスの辞書ルックアップで順序差を吸収できる"""
        infer_idx = {n: i for i, n in enumerate(ARKIT_BLENDSHAPE_NAMES_INFER)}
        fallback_idx = {n: i for i, n in enumerate(ARKIT_BLENDSHAPE_NAMES_FALLBACK)}

        # jawOpen は両方に存在するが、異なるインデックス
        assert infer_idx["jawOpen"] != fallback_idx["jawOpen"]
        # 名前からアクセスすれば正しい値が取れる
        assert "jawOpen" in infer_idx
        assert "jawOpen" in fallback_idx

    def test_frontend_uses_names_not_indices(self, mock_a2e_response):
        """フロントエンドは names 配列を使ってマッピングするため、
        順序の違いは問題にならない"""
        data = mock_a2e_response
        # names と frames を zip して dict にする (フロントエンドのロジック)
        frame_dict = dict(zip(data["names"], data["frames"][0]))
        assert "jawOpen" in frame_dict
