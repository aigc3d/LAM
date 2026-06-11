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
Tests for lam.models.block module.
"""

import pytest
import torch

from lam.models.block import BasicBlock, ConditionBlock, ConditionModulationBlock


class TestBasicBlock:
    """Tests for the BasicBlock class."""

    def test_basic_block_initialization(self):
        """Test BasicBlock initializes correctly."""
        block = BasicBlock(
            inner_dim=64,
            num_heads=4,
            eps=1e-6
        )
        assert block is not None
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)
        assert isinstance(block.self_attn, torch.nn.MultiheadAttention)
        assert isinstance(block.mlp, torch.nn.Sequential)

    def test_basic_block_forward(self, cpu_device):
        """Test BasicBlock forward pass."""
        batch_size = 2
        seq_len = 16
        inner_dim = 64

        block = BasicBlock(
            inner_dim=inner_dim,
            num_heads=4,
            eps=1e-6
        )

        x = torch.rand(batch_size, seq_len, inner_dim)
        output = block(x)

        assert output.shape == x.shape

    def test_basic_block_output_shape(self, cpu_device):
        """Test BasicBlock maintains input shape."""
        block = BasicBlock(inner_dim=128, num_heads=8, eps=1e-6)

        # Test various input shapes
        for batch_size in [1, 4]:
            for seq_len in [8, 32]:
                x = torch.rand(batch_size, seq_len, 128)
                output = block(x)
                assert output.shape == x.shape

    def test_basic_block_gradients(self, cpu_device):
        """Test BasicBlock produces gradients."""
        block = BasicBlock(inner_dim=64, num_heads=4, eps=1e-6)
        x = torch.rand(2, 16, 64, requires_grad=True)

        output = block(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_basic_block_with_dropout(self, cpu_device):
        """Test BasicBlock with dropout."""
        block = BasicBlock(
            inner_dim=64,
            num_heads=4,
            eps=1e-6,
            attn_drop=0.1,
            mlp_drop=0.1
        )

        x = torch.rand(2, 16, 64)

        # Should work in eval mode
        block.eval()
        output = block(x)
        assert output.shape == x.shape

    def test_basic_block_mlp_ratio(self, cpu_device):
        """Test BasicBlock with custom MLP ratio."""
        inner_dim = 64
        mlp_ratio = 2.0
        block = BasicBlock(
            inner_dim=inner_dim,
            num_heads=4,
            eps=1e-6,
            mlp_ratio=mlp_ratio
        )

        # Check MLP hidden dimension
        mlp_linear1 = block.mlp[0]
        assert mlp_linear1.out_features == int(inner_dim * mlp_ratio)


class TestConditionBlock:
    """Tests for the ConditionBlock class."""

    def test_condition_block_initialization(self):
        """Test ConditionBlock initializes correctly."""
        block = ConditionBlock(
            inner_dim=64,
            cond_dim=128,
            num_heads=4,
            eps=1e-6
        )
        assert block is not None
        assert isinstance(block.cross_attn, torch.nn.MultiheadAttention)
        assert isinstance(block.self_attn, torch.nn.MultiheadAttention)

    def test_condition_block_forward(self, cpu_device):
        """Test ConditionBlock forward pass."""
        batch_size = 2
        seq_len = 16
        cond_len = 8
        inner_dim = 64
        cond_dim = 128

        block = ConditionBlock(
            inner_dim=inner_dim,
            cond_dim=cond_dim,
            num_heads=4,
            eps=1e-6
        )

        x = torch.rand(batch_size, seq_len, inner_dim)
        cond = torch.rand(batch_size, cond_len, cond_dim)
        output = block(x, cond)

        assert output.shape == x.shape

    def test_condition_block_different_cond_lengths(self, cpu_device):
        """Test ConditionBlock with different condition sequence lengths."""
        inner_dim = 64
        cond_dim = 128
        block = ConditionBlock(
            inner_dim=inner_dim,
            cond_dim=cond_dim,
            num_heads=4,
            eps=1e-6
        )

        x = torch.rand(2, 16, inner_dim)

        # Test with different condition lengths
        for cond_len in [4, 16, 32]:
            cond = torch.rand(2, cond_len, cond_dim)
            output = block(x, cond)
            assert output.shape == x.shape

    def test_condition_block_gradients(self, cpu_device):
        """Test ConditionBlock produces gradients."""
        block = ConditionBlock(
            inner_dim=64,
            cond_dim=128,
            num_heads=4,
            eps=1e-6
        )

        x = torch.rand(2, 16, 64, requires_grad=True)
        cond = torch.rand(2, 8, 128, requires_grad=True)

        output = block(x, cond)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert cond.grad is not None


class TestConditionModulationBlock:
    """Tests for the ConditionModulationBlock class."""

    def test_condition_modulation_block_initialization(self):
        """Test ConditionModulationBlock initializes correctly."""
        block = ConditionModulationBlock(
            inner_dim=64,
            cond_dim=128,
            mod_dim=32,
            num_heads=4,
            eps=1e-6
        )
        assert block is not None
        assert isinstance(block.cross_attn, torch.nn.MultiheadAttention)
        assert isinstance(block.self_attn, torch.nn.MultiheadAttention)

    def test_condition_modulation_block_forward(self, cpu_device):
        """Test ConditionModulationBlock forward pass."""
        batch_size = 2
        seq_len = 16
        cond_len = 8
        inner_dim = 64
        cond_dim = 128
        mod_dim = 32

        block = ConditionModulationBlock(
            inner_dim=inner_dim,
            cond_dim=cond_dim,
            mod_dim=mod_dim,
            num_heads=4,
            eps=1e-6
        )

        x = torch.rand(batch_size, seq_len, inner_dim)
        cond = torch.rand(batch_size, cond_len, cond_dim)
        mod = torch.rand(batch_size, mod_dim)
        output = block(x, cond, mod)

        assert output.shape == x.shape

    def test_condition_modulation_block_gradients(self, cpu_device):
        """Test ConditionModulationBlock produces gradients."""
        block = ConditionModulationBlock(
            inner_dim=64,
            cond_dim=128,
            mod_dim=32,
            num_heads=4,
            eps=1e-6
        )

        x = torch.rand(2, 16, 64, requires_grad=True)
        cond = torch.rand(2, 8, 128, requires_grad=True)
        mod = torch.rand(2, 32, requires_grad=True)

        output = block(x, cond, mod)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert cond.grad is not None
        assert mod.grad is not None
