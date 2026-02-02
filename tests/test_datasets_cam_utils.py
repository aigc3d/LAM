# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for lam.datasets.cam_utils module.
"""

import pytest
import torch
import math

from lam.datasets.cam_utils import (
    compose_extrinsic_R_T,
    compose_extrinsic_RT,
    decompose_extrinsic_R_T,
    decompose_extrinsic_RT,
    get_normalized_camera_intrinsics,
    build_camera_principle,
    center_looking_at_camera_pose,
    surrounding_views_linspace,
    create_intrinsics,
)


class TestComposeExtrinsic:
    """Tests for extrinsic matrix composition functions."""

    def test_compose_extrinsic_R_T_shape(self, cpu_device):
        """Test compose_extrinsic_R_T output shape."""
        batch_size = 4
        R = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        T = torch.zeros(batch_size, 3)

        E = compose_extrinsic_R_T(R, T)

        assert E.shape == (batch_size, 4, 4)

    def test_compose_extrinsic_R_T_identity(self, cpu_device):
        """Test compose_extrinsic_R_T with identity rotation and zero translation."""
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros(1, 3)

        E = compose_extrinsic_R_T(R, T)

        expected = torch.eye(4).unsqueeze(0)
        assert torch.allclose(E, expected)

    def test_compose_extrinsic_RT_shape(self, cpu_device):
        """Test compose_extrinsic_RT output shape."""
        batch_size = 4
        RT = torch.zeros(batch_size, 3, 4)
        RT[:, :3, :3] = torch.eye(3)

        E = compose_extrinsic_RT(RT)

        assert E.shape == (batch_size, 4, 4)

    def test_compose_extrinsic_RT_last_row(self, cpu_device):
        """Test that compose_extrinsic_RT adds correct last row [0,0,0,1]."""
        RT = torch.rand(2, 3, 4)

        E = compose_extrinsic_RT(RT)

        expected_last_row = torch.tensor([[0, 0, 0, 1]], dtype=RT.dtype)
        assert torch.allclose(E[:, 3, :], expected_last_row.repeat(2, 1))


class TestDecomposeExtrinsic:
    """Tests for extrinsic matrix decomposition functions."""

    def test_decompose_extrinsic_RT_shape(self, cpu_device):
        """Test decompose_extrinsic_RT output shape."""
        batch_size = 4
        E = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)

        RT = decompose_extrinsic_RT(E)

        assert RT.shape == (batch_size, 3, 4)

    def test_decompose_extrinsic_R_T_shapes(self, cpu_device):
        """Test decompose_extrinsic_R_T output shapes."""
        batch_size = 4
        E = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)

        R, T = decompose_extrinsic_R_T(E)

        assert R.shape == (batch_size, 3, 3)
        assert T.shape == (batch_size, 3)

    def test_compose_decompose_round_trip(self, cpu_device):
        """Test that compose and decompose are inverse operations."""
        R_orig = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        T_orig = torch.rand(2, 3)

        E = compose_extrinsic_R_T(R_orig, T_orig)
        R_recovered, T_recovered = decompose_extrinsic_R_T(E)

        assert torch.allclose(R_orig, R_recovered)
        assert torch.allclose(T_orig, T_recovered)


class TestCameraIntrinsics:
    """Tests for camera intrinsics functions."""

    def test_get_normalized_camera_intrinsics(self, cpu_device):
        """Test get_normalized_camera_intrinsics returns correct values."""
        # intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
        intrinsics = torch.tensor([
            [[500, 500], [256, 256], [512, 512]]
        ], dtype=torch.float32)

        fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)

        # Normalized values should be fx/width, etc.
        assert torch.allclose(fx, torch.tensor([500/512]))
        assert torch.allclose(fy, torch.tensor([500/512]))
        assert torch.allclose(cx, torch.tensor([256/512]))
        assert torch.allclose(cy, torch.tensor([256/512]))

    def test_get_normalized_camera_intrinsics_batch(self, cpu_device):
        """Test get_normalized_camera_intrinsics with batch."""
        batch_size = 4
        intrinsics = torch.tensor([
            [[500, 500], [256, 256], [512, 512]]
        ], dtype=torch.float32).repeat(batch_size, 1, 1)

        fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)

        assert fx.shape == (batch_size,)
        assert fy.shape == (batch_size,)
        assert cx.shape == (batch_size,)
        assert cy.shape == (batch_size,)

    def test_create_intrinsics_with_c(self, cpu_device):
        """Test create_intrinsics with center parameter."""
        intrinsics = create_intrinsics(f=1.0, c=0.5)

        assert intrinsics.shape == (3, 2)
        # fx, fy normalized
        assert intrinsics[0, 0] == 1.0  # fx
        assert intrinsics[0, 1] == 1.0  # fy
        # cx, cy normalized
        assert intrinsics[1, 0] == 0.5  # cx
        assert intrinsics[1, 1] == 0.5  # cy

    def test_create_intrinsics_with_cx_cy(self, cpu_device):
        """Test create_intrinsics with separate cx, cy parameters."""
        intrinsics = create_intrinsics(f=1.0, cx=0.3, cy=0.7)

        assert intrinsics[1, 0] == 0.3  # cx
        assert intrinsics[1, 1] == 0.7  # cy

    def test_create_intrinsics_c_and_cx_cy_conflict(self, cpu_device):
        """Test create_intrinsics raises error when c and cx/cy both provided."""
        with pytest.raises(AssertionError):
            create_intrinsics(f=1.0, c=0.5, cx=0.3, cy=0.7)


class TestBuildCameraPrinciple:
    """Tests for build_camera_principle function."""

    def test_build_camera_principle_shape(self, cpu_device):
        """Test build_camera_principle output shape."""
        batch_size = 4
        RT = torch.zeros(batch_size, 3, 4)
        RT[:, :3, :3] = torch.eye(3)
        intrinsics = torch.tensor([
            [[500, 500], [256, 256], [512, 512]]
        ], dtype=torch.float32).repeat(batch_size, 1, 1)

        camera = build_camera_principle(RT, intrinsics)

        # Output should be [RT flattened (12) + fx, fy, cx, cy (4)] = 16
        assert camera.shape == (batch_size, 16)


class TestCenterLookingAtCameraPose:
    """Tests for center_looking_at_camera_pose function."""

    def test_center_looking_at_camera_pose_shape(self, cpu_device):
        """Test center_looking_at_camera_pose output shape."""
        camera_position = torch.tensor([[0, 0, 2]], dtype=torch.float32)

        extrinsics = center_looking_at_camera_pose(camera_position)

        assert extrinsics.shape == (1, 3, 4)

    def test_center_looking_at_camera_pose_batch(self, cpu_device):
        """Test center_looking_at_camera_pose with batch."""
        camera_positions = torch.rand(8, 3)

        extrinsics = center_looking_at_camera_pose(camera_positions)

        assert extrinsics.shape == (8, 3, 4)

    def test_center_looking_at_camera_pose_orthonormal(self, cpu_device):
        """Test that rotation part is orthonormal."""
        camera_position = torch.tensor([[2, 0, 0]], dtype=torch.float32)

        extrinsics = center_looking_at_camera_pose(camera_position)

        # Extract rotation matrix
        R = extrinsics[0, :, :3]

        # Check orthonormality: R^T * R = I
        RtR = torch.mm(R.T, R)
        assert torch.allclose(RtR, torch.eye(3), atol=1e-5)


class TestSurroundingViewsLinspace:
    """Tests for surrounding_views_linspace function."""

    def test_surrounding_views_linspace_shape(self, cpu_device):
        """Test surrounding_views_linspace output shape."""
        n_views = 8

        extrinsics = surrounding_views_linspace(n_views)

        assert extrinsics.shape == (n_views, 3, 4)

    def test_surrounding_views_linspace_different_views(self, cpu_device):
        """Test surrounding_views_linspace with different view counts."""
        for n_views in [4, 8, 16, 32]:
            extrinsics = surrounding_views_linspace(n_views)
            assert extrinsics.shape == (n_views, 3, 4)

    def test_surrounding_views_linspace_radius(self, cpu_device):
        """Test surrounding_views_linspace camera distance."""
        radius = 3.0
        height = 0.0

        extrinsics = surrounding_views_linspace(n_views=4, radius=radius, height=height)

        # Camera positions are in the last column
        positions = extrinsics[:, :, 3]
        distances = positions.norm(dim=-1)

        # All cameras should be at the same distance (approximately radius)
        assert torch.allclose(distances, torch.full_like(distances, radius), atol=1e-5)

    def test_surrounding_views_linspace_zero_views_raises(self, cpu_device):
        """Test that zero views raises assertion error."""
        with pytest.raises(AssertionError):
            surrounding_views_linspace(n_views=0)

    def test_surrounding_views_linspace_negative_radius_raises(self, cpu_device):
        """Test that negative radius raises assertion error."""
        with pytest.raises(AssertionError):
            surrounding_views_linspace(n_views=4, radius=-1.0)

    def test_surrounding_views_linspace_device(self, cpu_device):
        """Test surrounding_views_linspace respects device parameter."""
        extrinsics = surrounding_views_linspace(n_views=4, device=cpu_device)
        assert extrinsics.device == cpu_device
