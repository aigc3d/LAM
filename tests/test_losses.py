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
Tests for lam.losses module.
"""

import pytest
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from lam.losses.pixelwise import PixelLoss
from lam.losses.tvloss import TVLoss


class TestPixelLoss:
    """Tests for the PixelLoss class."""

    def test_pixel_loss_mse_initialization(self):
        """Test PixelLoss initializes with MSE loss by default."""
        loss = PixelLoss(option='mse')
        assert loss.loss_fn is not None

    def test_pixel_loss_l1_initialization(self):
        """Test PixelLoss initializes with L1 loss."""
        loss = PixelLoss(option='l1')
        assert loss.loss_fn is not None

    def test_pixel_loss_invalid_option(self):
        """Test PixelLoss raises error for invalid option."""
        with pytest.raises(NotImplementedError, match="Unknown pixel loss option"):
            PixelLoss(option='invalid')

    def test_pixel_loss_mse_forward(self, random_image_tensor, cpu_device):
        """Test PixelLoss forward pass with MSE."""
        loss_fn = PixelLoss(option='mse')
        x = random_image_tensor(batch_size=2, num_views=4, channels=3, height=32, width=32)
        y = random_image_tensor(batch_size=2, num_views=4, channels=3, height=32, width=32)

        loss = loss_fn(x, y)

        assert loss.ndim == 0  # Scalar output
        assert loss >= 0  # Loss should be non-negative

    def test_pixel_loss_l1_forward(self, random_image_tensor, cpu_device):
        """Test PixelLoss forward pass with L1."""
        loss_fn = PixelLoss(option='l1')
        x = random_image_tensor(batch_size=2, num_views=4, channels=3, height=32, width=32)
        y = random_image_tensor(batch_size=2, num_views=4, channels=3, height=32, width=32)

        loss = loss_fn(x, y)

        assert loss.ndim == 0  # Scalar output
        assert loss >= 0  # Loss should be non-negative

    def test_pixel_loss_identical_images(self, cpu_device):
        """Test PixelLoss returns zero for identical images."""
        loss_fn = PixelLoss(option='mse')
        x = torch.rand(2, 4, 3, 32, 32)
        y = x.clone()

        loss = loss_fn(x, y)

        assert loss.item() < 1e-6  # Should be very close to zero

    def test_pixel_loss_batch_reduction(self, random_image_tensor, cpu_device):
        """Test PixelLoss reduces across batch correctly."""
        loss_fn = PixelLoss(option='mse')

        # Single batch
        x1 = random_image_tensor(batch_size=1, num_views=4, channels=3, height=32, width=32)
        y1 = random_image_tensor(batch_size=1, num_views=4, channels=3, height=32, width=32)
        loss1 = loss_fn(x1, y1)

        # Larger batch
        x2 = random_image_tensor(batch_size=4, num_views=4, channels=3, height=32, width=32)
        y2 = random_image_tensor(batch_size=4, num_views=4, channels=3, height=32, width=32)
        loss2 = loss_fn(x2, y2)

        # Both should return scalar
        assert loss1.ndim == 0
        assert loss2.ndim == 0

    def test_pixel_loss_different_sizes(self, cpu_device):
        """Test PixelLoss with different image sizes."""
        loss_fn = PixelLoss(option='mse')

        # Small images
        x_small = torch.rand(2, 2, 3, 16, 16)
        y_small = torch.rand(2, 2, 3, 16, 16)
        loss_small = loss_fn(x_small, y_small)

        # Large images
        x_large = torch.rand(2, 2, 3, 64, 64)
        y_large = torch.rand(2, 2, 3, 64, 64)
        loss_large = loss_fn(x_large, y_large)

        assert loss_small.ndim == 0
        assert loss_large.ndim == 0


class TestTVLoss:
    """Tests for the TVLoss class."""

    def test_tv_loss_initialization(self):
        """Test TVLoss initializes correctly."""
        loss = TVLoss()
        assert loss is not None

    def test_tv_loss_forward(self, random_image_tensor, cpu_device):
        """Test TVLoss forward pass."""
        loss_fn = TVLoss()
        x = random_image_tensor(batch_size=2, num_views=4, channels=3, height=32, width=32)

        loss = loss_fn(x)

        assert loss.ndim == 0  # Scalar output
        assert loss >= 0  # Loss should be non-negative

    def test_tv_loss_constant_image(self, cpu_device):
        """Test TVLoss returns zero for constant image."""
        loss_fn = TVLoss()
        # Constant image (no variation)
        x = torch.ones(2, 4, 3, 32, 32)

        loss = loss_fn(x)

        assert loss.item() < 1e-6  # Should be very close to zero

    def test_tv_loss_high_variation(self, cpu_device):
        """Test TVLoss returns higher value for high variation image."""
        loss_fn = TVLoss()

        # Low variation (constant)
        x_low = torch.ones(2, 4, 3, 32, 32)
        loss_low = loss_fn(x_low)

        # High variation (checkerboard pattern)
        x_high = torch.zeros(2, 4, 3, 32, 32)
        x_high[..., ::2, ::2] = 1.0
        x_high[..., 1::2, 1::2] = 1.0
        loss_high = loss_fn(x_high)

        assert loss_high > loss_low

    def test_tv_loss_different_batch_sizes(self, cpu_device):
        """Test TVLoss with different batch sizes."""
        loss_fn = TVLoss()

        x1 = torch.rand(1, 2, 3, 32, 32)
        x2 = torch.rand(4, 2, 3, 32, 32)

        loss1 = loss_fn(x1)
        loss2 = loss_fn(x2)

        assert loss1.ndim == 0
        assert loss2.ndim == 0

    def test_numel_excluding_first_dim(self, cpu_device):
        """Test numel_excluding_first_dim helper method."""
        loss_fn = TVLoss()
        x = torch.rand(4, 3, 32, 32)

        numel = loss_fn.numel_excluding_first_dim(x)
        expected = 3 * 32 * 32

        assert numel == expected

    def test_tv_loss_gradients(self, cpu_device):
        """Test TVLoss produces gradients."""
        loss_fn = TVLoss()
        x = torch.rand(2, 4, 3, 32, 32, requires_grad=True)

        loss = loss_fn(x)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPixelLossGradients:
    """Tests for gradient computation of PixelLoss."""

    def test_pixel_loss_mse_gradients(self, cpu_device):
        """Test MSE PixelLoss produces gradients."""
        loss_fn = PixelLoss(option='mse')
        x = torch.rand(2, 4, 3, 32, 32, requires_grad=True)
        y = torch.rand(2, 4, 3, 32, 32)

        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_pixel_loss_l1_gradients(self, cpu_device):
        """Test L1 PixelLoss produces gradients."""
        loss_fn = PixelLoss(option='l1')
        x = torch.rand(2, 4, 3, 32, 32, requires_grad=True)
        y = torch.rand(2, 4, 3, 32, 32)

        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
