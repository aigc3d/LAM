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
Tests for lam.utils.scheduler module.
"""

import pytest
import math
import torch
from lam.utils.scheduler import CosineWarmupScheduler


class TestCosineWarmupScheduler:
    """Tests for the CosineWarmupScheduler class."""

    def test_scheduler_initialization(self, sample_optimizer):
        """Test scheduler initializes correctly."""
        optimizer = sample_optimizer(lr=0.001)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=100,
            max_iters=1000
        )

        assert scheduler.warmup_iters == 100
        assert scheduler.max_iters == 1000
        assert scheduler.initial_lr == 1e-10

    def test_warmup_phase_starts_from_initial_lr(self, sample_optimizer):
        """Test that warmup starts from initial_lr."""
        base_lr = 0.001
        initial_lr = 1e-10
        optimizer = sample_optimizer(lr=base_lr)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=100,
            max_iters=1000,
            initial_lr=initial_lr
        )

        # At step 1, LR should be close to initial_lr
        lrs = scheduler.get_lr()
        expected_lr = initial_lr + (base_lr - initial_lr) * 1 / 100
        assert abs(lrs[0] - expected_lr) < 1e-10

    def test_warmup_phase_linear_increase(self, sample_optimizer):
        """Test linear learning rate increase during warmup."""
        base_lr = 0.001
        initial_lr = 0.0
        warmup_iters = 10
        optimizer = sample_optimizer(lr=base_lr)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            max_iters=100,
            initial_lr=initial_lr
        )

        # Step through warmup phase
        prev_lr = 0
        for i in range(warmup_iters):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # LR should increase
            assert current_lr >= prev_lr
            prev_lr = current_lr

    def test_warmup_reaches_base_lr(self, sample_optimizer):
        """Test that warmup reaches base_lr at the end of warmup phase."""
        base_lr = 0.001
        warmup_iters = 10
        optimizer = sample_optimizer(lr=base_lr)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            max_iters=100,
            initial_lr=0.0
        )

        # Step to end of warmup
        for _ in range(warmup_iters):
            scheduler.step()

        # At warmup end, LR should equal base_lr
        assert abs(scheduler.get_last_lr()[0] - base_lr) < 1e-8

    def test_cosine_decay_phase(self, sample_optimizer):
        """Test cosine decay after warmup phase."""
        base_lr = 0.001
        warmup_iters = 10
        max_iters = 100
        optimizer = sample_optimizer(lr=base_lr)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            initial_lr=0.0
        )

        # Complete warmup
        for _ in range(warmup_iters):
            scheduler.step()

        # Step through cosine phase and verify decay
        prev_lr = scheduler.get_last_lr()[0]
        for _ in range(max_iters - warmup_iters - 1):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # LR should decrease during cosine decay
            assert current_lr <= prev_lr + 1e-10  # Small tolerance
            prev_lr = current_lr

    def test_cosine_decay_reaches_zero(self, sample_optimizer):
        """Test that cosine decay reaches near zero at max_iters."""
        base_lr = 0.001
        warmup_iters = 10
        max_iters = 100
        optimizer = sample_optimizer(lr=base_lr)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            initial_lr=0.0
        )

        # Step to max_iters
        for _ in range(max_iters):
            scheduler.step()

        # At max_iters, LR should be very close to 0
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < 1e-6

    def test_multiple_param_groups(self):
        """Test scheduler with multiple parameter groups."""
        model1 = torch.nn.Linear(10, 10)
        model2 = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam([
            {'params': model1.parameters(), 'lr': 0.001},
            {'params': model2.parameters(), 'lr': 0.01}
        ])

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=10,
            max_iters=100
        )

        # Verify both param groups have correct learning rates
        lrs = scheduler.get_lr()
        assert len(lrs) == 2

    def test_custom_initial_lr(self, sample_optimizer):
        """Test scheduler with custom initial learning rate."""
        custom_initial_lr = 1e-5
        optimizer = sample_optimizer(lr=0.001)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=100,
            max_iters=1000,
            initial_lr=custom_initial_lr
        )

        assert scheduler.initial_lr == custom_initial_lr
