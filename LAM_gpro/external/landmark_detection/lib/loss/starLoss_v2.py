import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .smoothL1Loss import SmoothL1Loss
from .wingLoss import WingLoss


def get_channel_sum(input):
    temp = torch.sum(input, dim=3)
    output = torch.sum(temp, dim=2)
    return output


def expand_two_dimensions_at_end(input, dim1, dim2):
    input = input.unsqueeze(-1).unsqueeze(-1)
    input = input.expand(-1, -1, dim1, dim2)
    return input


class STARLoss_v2(nn.Module):
    def __init__(self, w=1, dist='smoothl1', num_dim_image=2, EPSILON=1e-5):
        super(STARLoss_v2, self).__init__()
        self.w = w
        self.num_dim_image = num_dim_image
        self.EPSILON = EPSILON
        self.dist = dist
        if self.dist == 'smoothl1':
            self.dist_func = SmoothL1Loss()
        elif self.dist == 'l1':
            self.dist_func = F.l1_loss
        elif self.dist == 'l2':
            self.dist_func = F.mse_loss
        elif self.dist == 'wing':
            self.dist_func = WingLoss()
        else:
            raise NotImplementedError

    def __repr__(self):
        return "STARLoss()"

    def _make_grid(self, h, w):
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h - 1) * 2 - 1,
            torch.arange(w).float() / (w - 1) * 2 - 1)
        return yy, xx

    def weighted_mean(self, heatmap):
        batch, npoints, h, w = heatmap.shape

        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        yy_coord = (yy * heatmap).sum([2, 3])  # batch x npoints
        xx_coord = (xx * heatmap).sum([2, 3])  # batch x npoints
        coords = torch.stack([xx_coord, yy_coord], dim=-1)
        return coords

    def unbiased_weighted_covariance(self, htp, means, num_dim_image=2, EPSILON=1e-5):
        batch_size, num_points, height, width = htp.shape

        yv, xv = self._make_grid(height, width)
        xv = Variable(xv)
        yv = Variable(yv)

        if htp.is_cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        xmean = means[:, :, 0]
        xv_minus_mean = xv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(xmean, height,
                                                                                                 width)  # [batch_size, 68, 64, 64]
        ymean = means[:, :, 1]
        yv_minus_mean = yv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(ymean, height,
                                                                                                 width)  # [batch_size, 68, 64, 64]
        wt_xv_minus_mean = xv_minus_mean
        wt_yv_minus_mean = yv_minus_mean

        wt_xv_minus_mean = wt_xv_minus_mean.view(batch_size * num_points, height * width)  # [batch_size*68, 4096]
        wt_xv_minus_mean = wt_xv_minus_mean.view(batch_size * num_points, 1, height * width)  # [batch_size*68, 1, 4096]
        wt_yv_minus_mean = wt_yv_minus_mean.view(batch_size * num_points, height * width)  # [batch_size*68, 4096]
        wt_yv_minus_mean = wt_yv_minus_mean.view(batch_size * num_points, 1, height * width)  # [batch_size*68, 1, 4096]
        vec_concat = torch.cat((wt_xv_minus_mean, wt_yv_minus_mean), 1)  # [batch_size*68, 2, 4096]

        htp_vec = htp.view(batch_size * num_points, 1, height * width)
        htp_vec = htp_vec.expand(-1, 2, -1)

        covariance = torch.bmm(htp_vec * vec_concat, vec_concat.transpose(1, 2))  # [batch_size*68, 2, 2]
        covariance = covariance.view(batch_size, num_points, num_dim_image, num_dim_image)  # [batch_size, 68, 2, 2]

        V_1 = htp.sum([2, 3]) + EPSILON  # [batch_size, 68]
        V_2 = torch.pow(htp, 2).sum([2, 3]) + EPSILON  # [batch_size, 68]

        denominator = V_1 - (V_2 / V_1)
        covariance = covariance / expand_two_dimensions_at_end(denominator, num_dim_image, num_dim_image)

        return covariance

    def ambiguity_guided_decompose(self, error, evalues, evectors):
        bs, npoints = error.shape[:2]
        normal_vector = evectors[:, :, 0]
        tangent_vector = evectors[:, :, 1]
        normal_error = torch.matmul(normal_vector.unsqueeze(-2), error.unsqueeze(-1))
        tangent_error = torch.matmul(tangent_vector.unsqueeze(-2), error.unsqueeze(-1))
        normal_error = normal_error.squeeze(dim=-1)
        tangent_error = tangent_error.squeeze(dim=-1)
        normal_dist = self.dist_func(normal_error, torch.zeros_like(normal_error).to(normal_error), reduction='none')
        tangent_dist = self.dist_func(tangent_error, torch.zeros_like(tangent_error).to(tangent_error), reduction='none')
        normal_dist = normal_dist.reshape(bs, npoints, 1)
        tangent_dist = tangent_dist.reshape(bs, npoints, 1)
        dist = torch.cat((normal_dist, tangent_dist), dim=-1)
        scale_dist = dist / torch.sqrt(evalues + self.EPSILON)
        scale_dist = scale_dist.sum(-1)
        return scale_dist

    def eigenvalue_restriction(self, evalues, batch, npoints):
        eigen_loss = torch.abs(evalues.view(batch, npoints, 2)).sum(-1)
        return eigen_loss

    def forward(self, heatmap, groundtruth):
        """
            heatmap:     b x n x 64 x 64
            groundtruth: b x n x 2
            output:      b x n x 1 => 1
        """
        # normalize
        bs, npoints, h, w = heatmap.shape
        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)
        heatmap = heatmap / heatmap_sum.view(bs, npoints, 1, 1)

        means = self.weighted_mean(heatmap)  # [bs, 68, 2]
        covars = self.unbiased_weighted_covariance(heatmap, means)  # covars [bs, 68, 2, 2]

        # TODO: GPU-based eigen-decomposition
        # https://github.com/pytorch/pytorch/issues/60537
        _covars = covars.view(bs * npoints, 2, 2).cpu()
        evalues, evectors = _covars.symeig(eigenvectors=True)  # evalues [bs * 68, 2], evectors [bs * 68, 2, 2]
        evalues = evalues.view(bs, npoints, 2).to(heatmap)
        evectors = evectors.view(bs, npoints, 2, 2).to(heatmap)

        # STAR Loss
        # Ambiguity-guided Decomposition
        loss_trans = self.ambiguity_guided_decompose(groundtruth - means, evalues, evectors)
        # Eigenvalue Restriction
        loss_eigen = self.eigenvalue_restriction(evalues, bs, npoints)
        star_loss = loss_trans + self.w * loss_eigen

        return star_loss.mean()
