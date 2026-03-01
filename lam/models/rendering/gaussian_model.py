import os
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from lam.models.rendering.utils.typing import *
from lam.models.rendering.utils.utils import trunc_exp, MLP
from einops import rearrange, repeat


inverse_sigmoid = lambda x: np.log(x / (1 - x))


class GaussianModel:
    def __init__(self, xyz=None, opacity=None, rotation=None, scaling=None, shs=None, offset=None, ply_path=None, sh2rgb=False) -> None:
        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs
        self.offset: Tensor = offset
        if ply_path is not None:
            self.load_ply(ply_path, sh2rgb=sh2rgb)

    def update_shs(self, shs):
        self.shs = shs
        
    def to_cuda(self):
        self.xyz = self.xyz.cuda()
        self.opacity = self.opacity.cuda()
        self.rotation = self.rotation.cuda()
        self.scaling = self.scaling.cuda()
        self.shs = self.shs.cuda()
        self.offset = self.offset.cuda()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        if len(self.shs.shape) == 2:
            features_dc = self.shs[:, :3].unsqueeze(1)
            features_rest = self.shs[:, 3:].unsqueeze(1)
        else:
            features_dc = self.shs[:, :1]
            features_rest = self.shs[:, 1:]
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, rgb2sh=False, offset2xyz=False):
        if offset2xyz:
            xyz = self.offset.detach().cpu().float().numpy()
        else:
            xyz = self.xyz.detach().cpu().float().numpy()
        normals = np.zeros_like(xyz)
        if len(self.shs.shape) == 2:
            features_dc = self.shs[:, :3].unsqueeze(1).float()
            features_rest = self.shs[:, 3:].unsqueeze(1).float()
        else:
            features_dc = self.shs[:, :1].float()
            features_rest = self.shs[:, 1:].float()
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        if rgb2sh:
            from lam.models.rendering.utils.sh_utils import RGB2SH
            f_dc = RGB2SH(f_dc)
        opacities = inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().float().numpy())
        scale = np.log(self.scaling.detach().cpu().float().numpy())
        rotation = self.rotation.detach().cpu().float().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_nodeact(self, path, rgb2sh=False):
        xyz = self.xyz.detach().cpu().float().numpy()
        normals = np.zeros_like(xyz)
        if len(self.shs.shape) == 2:
            features_dc = self.shs[:, :3].unsqueeze(1).float()
            features_rest = self.shs[:, 3:].unsqueeze(1).float()
        else:
            features_dc = self.shs[:, :1].float()
            features_rest = self.shs[:, 1:].float()
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        if rgb2sh:
            from lam.models.rendering.utils.sh_utils import RGB2SH
            f_dc = RGB2SH(f_dc)
        opacities = self.opacity.detach().cpu().float().numpy()
        scale = self.scaling.detach().cpu().float().numpy()
        rotation = self.rotation.detach().cpu().float().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, sh2rgb=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        self.sh_degree = 0
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cpu").requires_grad_(False))
        self.features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous().requires_grad_(False))
        if sh2rgb:
            from lam.models.rendering.utils.sh_utils import SH2RGB
            self.features_dc = SH2RGB(self.features_dc)
        self.features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cpu").transpose(1, 2).contiguous().requires_grad_(False))
        self.shs = torch.cat([self.features_dc, self.features_rest], dim=1)
        self.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cpu").requires_grad_(False))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cpu").requires_grad_(False))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cpu").requires_grad_(False))
        self.offset = nn.Parameter(torch.zeros_like(self.xyz).requires_grad_(False))
        if sh2rgb:
            self.opacity = nn.functional.sigmoid(self.opacity)
            self.scaling = trunc_exp(self.scaling)

        self.active_sh_degree = self.sh_degree
