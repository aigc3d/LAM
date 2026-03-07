import torch.nn as nn

def count_parameters_in_MB(model):
    if isinstance(model, nn.Module):
        return sum(v.numel() for v in model.parameters()) / 1e6
    else:
        return sum(v.numel() for v in model) / 1e6