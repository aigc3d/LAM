import torch
import torch.nn.functional as F

class Accuracy:
    def __init__(self):
        pass

    def __repr__(self):
        return "Accuracy()"

    def test(self, label_pd, label_gt, ignore_label=-1):
        correct_cnt = 0
        total_cnt = 0
        with torch.no_grad():
            label_pd = F.softmax(label_pd, dim=1)
            label_pd = torch.max(label_pd, 1)[1]
            label_gt = label_gt.long()
            c = (label_pd == label_gt)
            correct_cnt = torch.sum(c).item()
            total_cnt = c.size(0) - torch.sum(label_gt==ignore_label).item()
        return correct_cnt, total_cnt
