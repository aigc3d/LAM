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
Pytest configuration and fixtures for LAM tests.
"""

import pytest
import sys

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "requires_numpy: mark test as requiring NumPy"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require unavailable dependencies."""
    if not TORCH_AVAILABLE:
        skip_torch = pytest.mark.skip(reason="PyTorch not installed")
        # Only skip tests in files that explicitly need torch
        torch_test_files = [
            "test_losses",
            "test_models",
            "test_utils_scheduler",
            "test_datasets_cam_utils",
        ]
        for item in items:
            # Check if the test file needs torch
            for torch_file in torch_test_files:
                if torch_file in str(item.fspath):
                    item.add_marker(skip_torch)
                    break


@pytest.fixture
def device():
    """Return available device (GPU if available, else CPU)."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    return torch.device("cpu")


@pytest.fixture
def random_image_tensor():
    """Generate a random image tensor with shape [N, M, C, H, W]."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    def _generate(batch_size=2, num_views=4, channels=3, height=64, width=64):
        return torch.rand(batch_size, num_views, channels, height, width)
    return _generate


@pytest.fixture
def random_rgba_image():
    """Generate a random RGBA image as numpy array."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy not installed")
    def _generate(height=256, width=256):
        return np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
    return _generate


@pytest.fixture
def sample_optimizer():
    """Create a sample optimizer for scheduler tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    def _create(lr=0.001):
        model = torch.nn.Linear(10, 10)
        return torch.optim.Adam(model.parameters(), lr=lr)
    return _create
