# -*- coding: utf-8 -*-

import math
import torch
from torch import nn


# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=0.01, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_2 = (y - y_hat).pow(2).sum(dim=-1, keepdim=False)
        # delta = delta_2.sqrt()
        delta = delta_2.clamp(min=1e-6).sqrt()
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - C
        )
        return loss.mean()
