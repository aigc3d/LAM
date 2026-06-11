import torch
import torch.nn as nn
from torch.autograd import Variable


def get_channel_sum(input):
    temp = torch.sum(input, dim=3)
    output = torch.sum(temp, dim=2)
    return output


def expand_two_dimensions_at_end(input, dim1, dim2):
    input = input.unsqueeze(-1).unsqueeze(-1)
    input = input.expand(-1, -1, dim1, dim2)
    return input


class TestTimePCA(nn.Module):
    def __init__(self):
        super(TestTimePCA, self).__init__()

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

    def forward(self, heatmap, groudtruth):

        batch, npoints, h, w = heatmap.shape

        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)
        heatmap = heatmap / heatmap_sum.view(batch, npoints, 1, 1)

        # means [batch_size, 68, 2]
        means = self.weighted_mean(heatmap)

        # covars [batch_size, 68, 2, 2]
        covars = self.unbiased_weighted_covariance(heatmap, means)

        # eigenvalues [batch_size * 68, 2] , eigenvectors [batch_size * 68, 2, 2]
        covars = covars.view(batch * npoints, 2, 2).cpu()
        evalues, evectors = covars.symeig(eigenvectors=True)
        evalues = evalues.view(batch, npoints, 2)
        evectors = evectors.view(batch, npoints, 2, 2)
        means = means.cpu()

        results = [dict() for _ in range(batch)]
        for i in range(batch):
            results[i]['pred'] = means[i].numpy().tolist()
            results[i]['gt'] = groudtruth[i].cpu().numpy().tolist()
            results[i]['evalues'] = evalues[i].numpy().tolist()
            results[i]['evectors'] = evectors[i].numpy().tolist()

        return results
