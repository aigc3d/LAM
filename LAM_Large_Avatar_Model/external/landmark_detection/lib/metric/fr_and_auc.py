import numpy as np
from scipy.integrate import simps


class FR_AUC:
    def __init__(self, data_definition):
        self.data_definition = data_definition
        if data_definition == '300W':
            self.thresh = 0.05
        else:
            self.thresh = 0.1

    def __repr__(self):
        return "FR_AUC()"

    def test(self, nmes, thres=None, step=0.0001):
        if thres is None:
            thres = self.thresh

        num_data = len(nmes)
        xs = np.arange(0, thres + step, step)
        ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
        fr = 1.0 - ys[-1]
        auc = simps(ys, x=xs) / thres
        return [round(fr, 4), round(auc, 6)]
