import torch.nn as nn

from models.losses import build_criteria
from .builder import MODELS, build_model

@MODELS.register_module()
class DefaultEstimator(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        pred_exp = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(pred_exp, input_dict["gt_exp"])
            return dict(loss=loss)
        # eval
        elif "gt_exp" in input_dict.keys():
            loss = self.criteria(pred_exp, input_dict["gt_exp"])
            return dict(loss=loss, pred_exp=pred_exp)
        # infer
        else:
            return dict(pred_exp=pred_exp)
