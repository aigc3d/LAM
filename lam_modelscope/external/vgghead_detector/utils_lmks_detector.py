#################################################
# written by wangduomin@xiaobing.ai             #
# modified by xg.chu@outlook.com                #
#################################################
import os
import torch
import numpy as np
import torchvision
os.environ["GLOG_minloglevel"] ="2"

class LmksDetector(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.size = 256
        self._device = device
        # model
        model = LandmarkDetector(model_path)
        self.model = model.to(self._device).eval()
        
    def _transform(self, image, bbox):
        assert bbox[3]-bbox[1] == bbox[2]-bbox[0], 'Bounding box should be square.'
        c_image = torchvision.transforms.functional.crop(image, bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
        c_image = torchvision.transforms.functional.resize(c_image, (self.size, self.size), antialias=True)
        c_image = torchvision.transforms.functional.normalize(c_image/255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return c_image[None], self.size / (bbox[3]-bbox[1])

    @torch.no_grad()
    def forward(self, image, bbox):
        assert image.dim() == 3, 'Input must be a 3D tensor.'
        if image.max() < 2.0:
            print('Image Should be in 0-255 range, but found in 0-1 range.')
        bbox = expand_bbox(bbox, ratio=1.38)
        # image_bbox = torchvision.utils.draw_bounding_boxes(image.cpu().to(torch.uint8), bbox[None], width=3, colors='green')
        # torchvision.utils.save_image(image_bbox/255.0, 'image_bbox.jpg')
        c_image, scale = self._transform(image.to(self._device), bbox)
        landmarks = self.model(c_image).squeeze(0) / scale
        landmarks = landmarks + bbox[:2][None]
        landmarks = mapping_lmk98_to_lmk70(landmarks)
        return landmarks


def mapping_lmk98_to_lmk70(lmk98):
    lmk70 = lmk98[[
        0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 
        33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 
        51, 52, 53, 54, 55, 56, 57, 58, 59, 
        60, 61, 63, 64, 65, 67, 
        68, 69, 71, 72, 73, 75, 
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 
        86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97
    ]]
    return lmk70


def expand_bbox(bbox, ratio=1.0):
    xmin, ymin, xmax, ymax = bbox
    cenx, ceny = ((xmin + xmax) / 2).long(), ((ymin + ymax) / 2).long()
    extend_size = torch.sqrt((ymax - ymin + 1) * (xmax - xmin + 1)) * ratio
    xmine, xmaxe = cenx - extend_size // 2, cenx + extend_size // 2
    ymine, ymaxe = ceny - extend_size // 2, ceny + extend_size // 2
    return torch.stack([xmine, ymine, xmaxe, ymaxe]).long()


# ------------------------------------------------------------------------------
# Reference: https://github.com/HRNet/HRNet-Image-Classification
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = [ 'hrnet18s', 'hrnet18', 'hrnet32' ]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):

    def __init__(self, num_modules, num_branches, block, 
            num_blocks, num_channels, fuse_method, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.num_modules = num_modules
        self.num_branches = num_branches
        self.block = block
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.fuse_method = fuse_method

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # layer1        
        num_channels, num_blocks = self.num_channels[0][0], self.num_blocks[0][0]
        self.layer1 = self._make_layer(self.block[0], 64, num_channels, num_blocks)
        stage1_out_channel = self.block[0].expansion*num_channels
        # layer2
        num_channels, num_blocks = self.num_channels[1], self.num_blocks[1]
        num_channels = [
            num_channels[i] * self.block[1].expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(1, num_channels)
        # layer3
        num_channels, num_blocks = self.num_channels[2], self.num_blocks[2]
        num_channels = [
            num_channels[i] * self.block[2].expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(2, num_channels)
        # layer4
        num_channels, num_blocks = self.num_channels[3], self.num_blocks[3]
        num_channels = [
            num_channels[i] * self.block[3].expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(3, num_channels, multi_scale_output=True)
        self._out_channels = sum(pre_stage_channels)
        
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], ),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, ),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, ),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, stage_index, in_channels,
                    multi_scale_output=True):
        num_modules = self.num_modules[stage_index]
        num_branches = self.num_branches[stage_index]
        num_blocks = self.num_blocks[stage_index]
        num_channels = self.num_channels[stage_index]
        block = self.block[stage_index]
        fuse_method = self.fuse_method[stage_index]
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      in_channels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            in_channels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.num_branches[1]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.num_branches[2]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.num_branches[3]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        kwargs = {
            'size': tuple(y_list[0].shape[-2:]), 
            'mode': 'bilinear', 'align_corners': False,
        }
        return torch.cat([F.interpolate(y,**kwargs) for y in y_list], 1)

def hrnet18s(pretrained=True, **kwargs):
    model = HighResolutionNet(
        num_modules = [1, 1, 3, 2],
        num_branches = [1, 2, 3, 4],
        block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_blocks = [(2,), (2,2), (2,2,2), (2,2,2,2)],
        num_channels = [(64,), (18,36), (18,36,72), (18,36,72,144)],
        fuse_method = ['SUM', 'SUM', 'SUM', 'SUM'],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet_w18s']), strict=False)
    return model

def hrnet18(pretrained=False, **kwargs):
    model = HighResolutionNet(
        num_modules = [1, 1, 4, 3],
        num_branches = [1, 2, 3, 4],
        block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_blocks = [(4,), (4,4), (4,4,4), (4,4,4,4)],
        num_channels = [(64,), (18,36), (18,36,72), (18,36,72,144)],
        fuse_method = ['SUM', 'SUM', 'SUM', 'SUM'],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet18']), strict=False)
    return model

def hrnet32(pretrained=False, **kwargs):
    model = HighResolutionNet(
        num_modules = [1, 1, 4, 3],
        num_branches = [1, 2, 3, 4],
        block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_blocks = [(4,), (4,4), (4,4,4), (4,4,4,4)],
        num_channels = [(64,), (32,64), (32,64,128), (32,64,128,256)],
        fuse_method = ['SUM', 'SUM', 'SUM', 'SUM'],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet32']), strict=False)
    return model


class BinaryHeadBlock(nn.Module):
    """BinaryHeadBlock
    """
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, out_channels*2, 1, bias=False),
        )
        
    def forward(self, input):
        N, C, H, W = input.shape
        return self.layers(input).view(N, 2, -1, H, W)

def heatmap2coord(heatmap, topk=9):
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N,C,1,-1).topk(topk, dim=-1)
    coord = torch.cat([index%W, index//W], dim=2)
    return (coord*F.softmax(score, dim=-1)).sum(-1)

class BinaryHeatmap2Coordinate(nn.Module):
    """BinaryHeatmap2Coordinate
    """
    def __init__(self, stride=4.0, topk=5, **kwargs):
        super(BinaryHeatmap2Coordinate, self).__init__()
        self.topk = topk
        self.stride = stride
        
    def forward(self, input):
        return self.stride * heatmap2coord(input[:,1,...], self.topk)
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'topk={}, '.format(self.topk)
        format_string += 'stride={}'.format(self.stride)
        format_string += ')'
        return format_string

class HeatmapHead(nn.Module):
    """HeatmapHead
    """
    def __init__(self):
        super(HeatmapHead, self).__init__()
        self.decoder = BinaryHeatmap2Coordinate(
            topk=9,
            stride=4.0,
        )
        self.head = BinaryHeadBlock(
            in_channels=270,
            proj_channels=270,
            out_channels=98,
        )
        
    def forward(self, input):
        heatmap = self.head(input)
        ldmk = self.decoder(heatmap)
        return heatmap[:,1,...], ldmk


class LandmarkDetector(nn.Module):
    def __init__(self, model_path):
        super(LandmarkDetector, self).__init__()

        self.backbone = HighResolutionNet(
            num_modules = [1, 1, 4, 3],
            num_branches = [1, 2, 3, 4],
            block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
            num_blocks = [(4,), (4,4), (4,4,4), (4,4,4,4)],
            num_channels = [(64,), (18,36), (18,36,72), (18,36,72,144)],
            fuse_method = ['SUM', 'SUM', 'SUM', 'SUM']
        )

        self.heatmap_head = HeatmapHead()

        self.load_state_dict(torch.load(model_path, map_location='cpu'))

    def forward(self, img):
        heatmap, landmark = self.heatmap_head(self.backbone(img))

        return landmark
