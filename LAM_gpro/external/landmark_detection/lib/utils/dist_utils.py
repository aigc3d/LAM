import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns


def get_channel_sum(input):
    """
        Generates the sum of each channel of the input
        input  = batch_size x 68 x 64 x 64
        output = batch_size x 68
    """
    temp = torch.sum(input, dim=3)
    output = torch.sum(temp, dim=2)

    return output


def expand_two_dimensions_at_end(input, dim1, dim2):
    """
        Adds two more dimensions to the end of the input
        input = batch_size x 68
        output= batch_size x 68 x dim1 x dim2
    """
    input = input.unsqueeze(-1).unsqueeze(-1)
    input = input.expand(-1, -1, dim1, dim2)

    return input


class Distribution(object):
    def __init__(self, heatmaps, num_dim_dist=2, EPSILON=1e-5, is_normalize=True):
        self.heatmaps = heatmaps
        self.num_dim_dist = num_dim_dist
        self.EPSILON = EPSILON
        self.is_normalize = is_normalize
        batch, npoints, h, w = heatmaps.shape
        # normalize
        heatmap_sum = torch.clamp(heatmaps.sum([2, 3]), min=1e-6)
        self.heatmaps = heatmaps / heatmap_sum.view(batch, npoints, 1, 1)

        # means [batch_size x 68 x 2]
        self.mean = self.get_spatial_mean(self.heatmaps)
        # covars [batch_size x 68 x 2 x 2]
        self.covars = self.get_covariance_matrix(self.heatmaps, self.mean)

        _covars = self.covars.view(batch * npoints, 2, 2).cpu()
        evalues, evectors = _covars.symeig(eigenvectors=True)
        # eigenvalues [batch_size x 68 x 2]
        self.evalues = evalues.view(batch, npoints, 2).to(heatmaps)
        # eignvectors [batch_size x 68 x 2 x 2]
        self.evectors = evectors.view(batch, npoints, 2, 2).to(heatmaps)

    def __repr__(self):
        return "Distribution()"

    def plot(self, heatmap, mean, evalues, evectors):
        # heatmap is not normalized
        plt.figure(0)
        if heatmap.is_cuda:
            heatmap, mean = heatmap.cpu(), mean.cpu()
            evalues, evectors = evalues.cpu(), evectors.cpu()
        sns.heatmap(heatmap, cmap="RdBu_r")
        for evalue, evector in zip(evalues, evectors):
            plt.arrow(mean[0], mean[1], evalue * evector[0], evalue * evector[1],
                      width=0.2, shape="full")
        plt.show()

    def easy_plot(self, index):
        # index = (num of batch_size, num of num_points)
        num_bs, num_p = index
        heatmap = self.heatmaps[num_bs, num_p]
        mean = self.mean[num_bs, num_p]
        evalues = self.evalues[num_bs, num_p]
        evectors = self.evectors[num_bs, num_p]
        self.plot(heatmap, mean, evalues, evectors)

    def project_and_scale(self, pts, eigenvalues, eigenvectors):
        batch_size, npoints, _ = pts.shape
        proj_pts = torch.matmul(pts.view(batch_size, npoints, 1, 2), eigenvectors)
        scale_proj_pts = proj_pts.view(batch_size, npoints, 2) / torch.sqrt(eigenvalues)
        return scale_proj_pts

    def _make_grid(self, h, w):
        if self.is_normalize:
            yy, xx = torch.meshgrid(
                torch.arange(h).float() / (h - 1) * 2 - 1,
                torch.arange(w).float() / (w - 1) * 2 - 1)
        else:
            yy, xx = torch.meshgrid(
                torch.arange(h).float(),
                torch.arange(w).float()
            )

        return yy, xx

    def get_spatial_mean(self, heatmap):
        batch, npoints, h, w = heatmap.shape

        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        yy_coord = (yy * heatmap).sum([2, 3])  # batch x npoints
        xx_coord = (xx * heatmap).sum([2, 3])  # batch x npoints
        coords = torch.stack([xx_coord, yy_coord], dim=-1)
        return coords

    def get_covariance_matrix(self, htp, means):
        """
            Covariance calculation from the normalized heatmaps
            Reference https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
            The unbiased estimate is given by
            Unbiased covariance =
                  ___
                  \
                  /__ w_i (x_i - \mu_i)^T (x_i - \mu_i)

              ___________________________________________

                            V_1 - (V_2/V_1)

                        ___                 ___
                        \                   \
            where V_1 = /__ w_i   and V_2 = /__ w_i^2


            Input:
                htp =        batch_size x 68 x 64 x 64
                means =      batch_size x 68 x 2

            Output:
                covariance = batch_size x 68 x 2 x 2
        """
        batch_size = htp.shape[0]
        num_points = htp.shape[1]
        height = htp.shape[2]
        width = htp.shape[3]

        yv, xv = self._make_grid(height, width)
        xv = Variable(xv)
        yv = Variable(yv)

        if htp.is_cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        xmean = means[:, :, 0]
        xv_minus_mean = xv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(xmean, height,
                                                                                                 width)  # batch_size x 68 x 64 x 64
        ymean = means[:, :, 1]
        yv_minus_mean = yv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(ymean, height,
                                                                                                 width)  # batch_size x 68 x 64 x 64

        # These are the unweighted versions
        wt_xv_minus_mean = xv_minus_mean
        wt_yv_minus_mean = yv_minus_mean

        wt_xv_minus_mean = wt_xv_minus_mean.view(batch_size * num_points, height * width)  # batch_size*68 x 4096
        wt_xv_minus_mean = wt_xv_minus_mean.view(batch_size * num_points, 1,
                                                 height * width)  # batch_size*68 x 1    x 4096
        wt_yv_minus_mean = wt_yv_minus_mean.view(batch_size * num_points, height * width)  # batch_size*68 x 4096
        wt_yv_minus_mean = wt_yv_minus_mean.view(batch_size * num_points, 1,
                                                 height * width)  # batch_size*68 x 1    x 4096
        vec_concat = torch.cat((wt_xv_minus_mean, wt_yv_minus_mean), 1)  # batch_size*68 x 2    x 4096

        htp_vec = htp.view(batch_size * num_points, 1, height * width)
        htp_vec = htp_vec.expand(-1, 2, -1)

        # Torch batch matrix multiplication
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Also use the heatmap as the weights at one place now
        covariance = torch.bmm(htp_vec * vec_concat, vec_concat.transpose(1, 2))  # batch_size*68 x 2    x 2
        covariance = covariance.view(batch_size, num_points, self.num_dim_dist,
                                     self.num_dim_dist)  # batch_size    x 68   x 2   x 2

        V_1 = get_channel_sum(htp) + self.EPSILON  # batch_size x 68
        V_2 = get_channel_sum(torch.pow(htp, 2))  # batch_size x 68
        denominator = V_1 - (V_2 / V_1)

        covariance = covariance / expand_two_dimensions_at_end(denominator, self.num_dim_dist, self.num_dim_dist)

        return (covariance)
