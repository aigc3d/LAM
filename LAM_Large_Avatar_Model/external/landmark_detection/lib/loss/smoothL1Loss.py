import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self, scale=0.01):
        super(SmoothL1Loss, self).__init__()
        self.scale = scale
        self.EPSILON = 1e-10

    def __repr__(self):
        return "SmoothL1Loss()"

    def forward(self, output: torch.Tensor, groundtruth: torch.Tensor, reduction='mean'):
        """
            input:  b x n x 2
            output: b x n x 1 => 1
        """
        if output.dim() == 4:
            shape = output.shape
            groundtruth = groundtruth.reshape(shape[0], shape[1], 1, shape[3])

        delta_2 = (output - groundtruth).pow(2).sum(dim=-1, keepdim=False)
        delta = delta_2.clamp(min=1e-6).sqrt()
        # delta = torch.sqrt(delta_2 + self.EPSILON)
        loss = torch.where( \
            delta_2 < self.scale * self.scale, \
            0.5 / self.scale * delta_2, \
            delta - 0.5 * self.scale)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss
