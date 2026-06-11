# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from accelerate.logging import get_logger

logger = get_logger(__name__)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        inner_channels, 
        use_clstoken=False,
        out_channel=1024,
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in inner_channels
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.output_conv = nn.Conv2d(sum(inner_channels) , out_channel, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            
            out.append(x)
        
        fusion_feats = torch.cat(out, dim=1)        

        fusion_feats = self.output_conv(fusion_feats)
        
        return fusion_feats


class Dinov2FusionWrapper(nn.Module):
    """
    Dinov2FusionWrapper using original implementation, hacked with modulation.
    """
    def __init__(self, model_name: str, modulation_dim: int = None, freeze: bool = True, encoder_feat_dim: int = 384):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = self._build_dinov2(model_name, modulation_dim=modulation_dim)
        
        self.intermediate_layer_idx_info = {
            'dinov2_vits14_reg': [2, 5, 8, 11],
            'dinov2_vitb14_reg': [2, 5, 8, 11], 
            'dinov2_vitl14_reg': [4, 11, 17, 23], 
            'dinov2_vitg14_reg': [9, 19, 29, 39]
        }
        
        self.intermediate_layer_idx = self.intermediate_layer_idx_info[model_name]
        self.fusion_head = DPTHead(in_channels=self.model.embed_dim, 
                                   inner_channels=[self.model.embed_dim] * 4, 
                                   out_channel=encoder_feat_dim)

        if freeze:
            if modulation_dim is not None:
                raise ValueError("Modulated Dinov2 requires training, freezing is not allowed.")
            self._freeze()


    def _freeze(self):
        # logger.warning(f"======== Freezing Dinov2FusionWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov2(model_name: str, modulation_dim: int = None, pretrained: bool = True):
        from importlib import import_module
        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        # logger.debug(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model

    # @torch.compile  # DISABLED: causes bird-monster on Modal L4 GPU
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized
        
        patch_h, patch_w = image.shape[-2] // self.model.patch_size, image.shape[-1] // self.model.patch_size
        
        features = self.model.get_intermediate_layers(image, self.intermediate_layer_idx, return_class_token=True)
        
        out_local = self.fusion_head(features,  patch_h, patch_w)

        out_global = None
        if out_global is not None:
            ret = torch.cat([out_local.permute(0, 2, 3, 1).flatten(1, 2), out_global.unsqueeze(1)], dim=1)
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)
        return ret
