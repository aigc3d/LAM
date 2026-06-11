import os
import torch
import inspect
import warnings
import torchvision
from .stylematte import StyleMatte

class StyleMatteEngine(torch.nn.Module):
    def __init__(self, device='cpu',human_matting_path='./pretrain_model/matting/stylematte_synth.pt'):
        super().__init__()
        self._device = device
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._init_models(human_matting_path)

    def _init_models(self,_ckpt_path):
        # load dict
        state_dict = torch.load(_ckpt_path, map_location='cpu')
        # build model
        model = StyleMatte()
        model.load_state_dict(state_dict)
        self.model = model.to(self._device).eval()
    
    @torch.no_grad()
    def forward(self, input_image, return_type='matting', background_rgb=1.0):
        if not hasattr(self, 'model'):
            self._init_models()
        if input_image.max() > 2.0:
            warnings.warn('Image should be normalized to [0, 1].')
        _, ori_h, ori_w = input_image.shape
        input_image = input_image.to(self._device).float()
        image = input_image.clone()
        # resize
        if max(ori_h, ori_w) > 1024:
            scale = 1024.0 / max(ori_h, ori_w)
            resized_h, resized_w = int(ori_h * scale), int(ori_w * scale)
            image = torchvision.transforms.functional.resize(image, (resized_h, resized_w), antialias=True)
        else:
            resized_h, resized_w = ori_h, ori_w
        # padding
        if resized_h % 8 != 0 or resized_w % 8 != 0:
            image = torchvision.transforms.functional.pad(image, ((8-resized_w % 8)%8, (8-resized_h % 8)%8, 0, 0, ), padding_mode='reflect')
        # normalize and forwarding
        image = self.normalize(image)[None]
        predict = self.model(image)[0]
        # undo padding
        predict = predict[:, -resized_h:, -resized_w:]
        # undo resize
        if resized_h != ori_h or resized_w != ori_w:
            predict = torchvision.transforms.functional.resize(predict, (ori_h, ori_w), antialias=True)
        
        if return_type == 'alpha':
            return predict[0]
        elif return_type == 'matting':
            predict = predict.expand(3, -1, -1)
            matting_image = input_image.clone()
            background_rgb = matting_image.new_ones(matting_image.shape) * background_rgb
            matting_image = matting_image * predict + (1-predict) * background_rgb
            return matting_image, predict[0]
        elif return_type == 'all':
            predict = predict.expand(3, -1, -1)
            background_rgb = input_image.new_ones(input_image.shape) * background_rgb
            foreground_image = input_image * predict + (1-predict) * background_rgb
            background_image = input_image * (1-predict) + predict * background_rgb
            return foreground_image, background_image
        else:
            raise NotImplementedError
