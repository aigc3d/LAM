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
Tests for lam.models.transformer module.
"""

import pytest
import torch

from lam.models.transformer import TransformerDecoder


class TestTransformerDecoder:
    """Tests for the TransformerDecoder class."""

    def test_basic_block_type_initialization(self):
        """Test TransformerDecoder initializes with basic block type."""
        decoder = TransformerDecoder(
            block_type='basic',
            num_layers=2,
            num_heads=4,
            inner_dim=64
        )
        assert decoder.block_type == 'basic'
        assert len(decoder.layers) == 2

    def test_cond_block_type_initialization(self):
        """Test TransformerDecoder initializes with cond block type."""
        decoder = TransformerDecoder(
            block_type='cond',
            num_layers=2,
            num_heads=4,
            inner_dim=64,
            cond_dim=128
        )
        assert decoder.block_type == 'cond'
        assert len(decoder.layers) == 2

    def test_cond_mod_block_type_initialization(self):
        """Test TransformerDecoder initializes with cond_mod block type."""
        decoder = TransformerDecoder(
            block_type='cond_mod',
            num_layers=2,
            num_heads=4,
            inner_dim=64,
            cond_dim=128,
            mod_dim=32
        )
        assert decoder.block_type == 'cond_mod'
        assert len(decoder.layers) == 2

    def test_invalid_block_type(self):
        """Test TransformerDecoder raises error for invalid block type."""
        with pytest.raises(AssertionError, match="Unsupported block type"):
            TransformerDecoder(
                block_type='invalid',
                num_layers=2,
                num_heads=4,
                inner_dim=64
            )

    def test_basic_forward(self, cpu_device):
        """Test TransformerDecoder forward pass with basic block."""
        batch_size = 2
        seq_len = 16
        inner_dim = 64

        decoder = TransformerDecoder(
            block_type='basic',
            num_layers=2,
            num_heads=4,
            inner_dim=inner_dim
        )

        x = torch.rand(batch_size, seq_len, inner_dim)
        output = decoder(x)

        assert output.shape == x.shape

    def test_cond_forward(self, cpu_device):
        """Test TransformerDecoder forward pass with condition block."""
        batch_size = 2
        seq_len = 16
        cond_len = 8
        inner_dim = 64
        cond_dim = 128

        decoder = TransformerDecoder(
            block_type='cond',
            num_layers=2,
            num_heads=4,
            inner_dim=inner_dim,
            cond_dim=cond_dim
        )

        x = torch.rand(batch_size, seq_len, inner_dim)
        cond = torch.rand(batch_size, cond_len, cond_dim)
        output = decoder(x, cond=cond)

        assert output.shape == x.shape

    def test_cond_mod_forward(self, cpu_device):
        """Test TransformerDecoder forward pass with condition and modulation."""
        batch_size = 2
        seq_len = 16
        cond_len = 8
        inner_dim = 64
        cond_dim = 128
        mod_dim = 32

        decoder = TransformerDecoder(
            block_type='cond_mod',
            num_layers=2,
            num_heads=4,
            inner_dim=inner_dim,
            cond_dim=cond_dim,
            mod_dim=mod_dim
        )

        x = torch.rand(batch_size, seq_len, inner_dim)
        cond = torch.rand(batch_size, cond_len, cond_dim)
        mod = torch.rand(batch_size, mod_dim)
        output = decoder(x, cond=cond, mod=mod)

        assert output.shape == x.shape

    def test_runtime_integrity_basic(self, cpu_device):
        """Test runtime integrity check for basic block."""
        decoder = TransformerDecoder(
            block_type='basic',
            num_layers=2,
            num_heads=4,
            inner_dim=64
        )

        x = torch.rand(2, 16, 64)

        # Basic block should not accept cond or mod
        with pytest.raises(AssertionError):
            decoder(x, cond=torch.rand(2, 8, 64))

    def test_runtime_integrity_cond(self, cpu_device):
        """Test runtime integrity check for condition block."""
        decoder = TransformerDecoder(
            block_type='cond',
            num_layers=2,
            num_heads=4,
            inner_dim=64,
            cond_dim=128
        )

        x = torch.rand(2, 16, 64)

        # Cond block requires cond, should fail without it
        with pytest.raises(AssertionError):
            decoder(x)

    def test_gradients(self, cpu_device):
        """Test TransformerDecoder produces gradients."""
        decoder = TransformerDecoder(
            block_type='cond',
            num_layers=2,
            num_heads=4,
            inner_dim=64,
            cond_dim=128
        )

        x = torch.rand(2, 16, 64, requires_grad=True)
        cond = torch.rand(2, 8, 128, requires_grad=True)

        output = decoder(x, cond=cond)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert cond.grad is not None

    def test_different_layer_counts(self, cpu_device):
        """Test TransformerDecoder with different layer counts."""
        for num_layers in [1, 4, 8]:
            decoder = TransformerDecoder(
                block_type='basic',
                num_layers=num_layers,
                num_heads=4,
                inner_dim=64
            )
            assert len(decoder.layers) == num_layers

            x = torch.rand(2, 16, 64)
            output = decoder(x)
            assert output.shape == x.shape

    def test_norm_layer(self, cpu_device):
        """Test TransformerDecoder has proper normalization layer."""
        decoder = TransformerDecoder(
            block_type='basic',
            num_layers=2,
            num_heads=4,
            inner_dim=64
        )

        assert isinstance(decoder.norm, torch.nn.LayerNorm)
        assert decoder.norm.normalized_shape == (64,)

    def test_mod_block_not_implemented(self):
        """Test that mod-only block type raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="modulation without condition"):
            TransformerDecoder(
                block_type='mod',
                num_layers=2,
                num_heads=4,
                inner_dim=64,
                mod_dim=32
            )
