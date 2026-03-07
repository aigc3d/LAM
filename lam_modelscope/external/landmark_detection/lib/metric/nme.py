import torch
import numpy as np

class NME:
    def __init__(self, nme_left_index, nme_right_index):
        self.nme_left_index = nme_left_index
        self.nme_right_index = nme_right_index

    def __repr__(self):
        return "NME()"

    def get_norm_distance(self, landmarks):
        assert isinstance(self.nme_right_index, list), 'the nme_right_index is not list.'
        assert isinstance(self.nme_left_index, list), 'the nme_left, index is not list.'
        right_pupil = landmarks[self.nme_right_index, :].mean(0)
        left_pupil = landmarks[self.nme_left_index, :].mean(0)
        norm_distance = np.linalg.norm(right_pupil - left_pupil)
        return norm_distance

    def test(self, label_pd, label_gt):
        nme_list = []
        label_pd = label_pd.data.cpu().numpy()
        label_gt = label_gt.data.cpu().numpy()

        for i in range(label_gt.shape[0]):
            landmarks_gt = label_gt[i]
            landmarks_pv = label_pd[i]
            if isinstance(self.nme_right_index, list):
                norm_distance = self.get_norm_distance(landmarks_gt)
            elif isinstance(self.nme_right_index, int):
                norm_distance = np.linalg.norm(landmarks_gt[self.nme_left_index] - landmarks_gt[self.nme_right_index])
            else:
                raise NotImplementedError
            landmarks_delta = landmarks_pv - landmarks_gt
            nme = (np.linalg.norm(landmarks_delta, axis=1) / norm_distance).mean()
            nme_list.append(nme)
            # sum_nme += nme
            # total_cnt += 1
        return nme_list
