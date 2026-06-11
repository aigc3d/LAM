# Copyright (c) 2023-2024, Zexin He
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


from collections import defaultdict
import os
import glob
from typing import Union
import random
import numpy as np
import torch
# from megfile import smart_path_join, smart_open
import json
from PIL import Image
import cv2

from lam.datasets.base import BaseDataset
from lam.datasets.cam_utils import build_camera_standard, build_camera_principle, camera_normalization_objaverse
from lam.utils.proxy import no_proxy
from typing import Optional, Union

__all__ = ['VideoHeadDataset']


class VideoHeadDataset(BaseDataset):

    def __init__(self, root_dirs: str, meta_path: Optional[Union[str, list]],
                 sample_side_views: int,
                 render_image_res_low: int, render_image_res_high: int, render_region_size: int,
                 source_image_res: int,
                 repeat_num=1,
                 crop_range_ratio_hw=[1.0, 1.0],
                 aspect_standard=1.0,  # h/w
                 enlarge_ratio=[0.8, 1.2],
                 debug=False,
                 is_val=False,
                 **kwargs):
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        if not (isinstance(render_region_size, list) or isinstance(render_region_size, tuple)): 
            render_region_size = render_region_size, render_region_size  # [H, W]
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        
        self.uids = self.uids * repeat_num
        self.crop_range_ratio_hw = crop_range_ratio_hw
        self.debug = debug
        self.aspect_standard = aspect_standard
        
        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = enlarge_ratio
        print(f"VideoHeadDataset, data_len:{len(self.uids)}, repeat_num:{repeat_num}, debug:{debug}, is_val:{is_val}")
        self.multiply = kwargs.get("multiply", 14)
        # set data deterministic
        self.is_val = is_val

    @staticmethod
    def _load_pose(frame_info, transpose_R=False):
        c2w = torch.eye(4)
        c2w = np.array(frame_info["transform_matrix"])
        c2w[:3, 1:3] *= -1
        c2w = torch.FloatTensor(c2w)
        """
        if transpose_R:
            w2c = torch.inverse(c2w)
            w2c[:3, :3] = w2c[:3, :3].transpose(1, 0).contiguous()
            c2w = torch.inverse(w2c)
        """
        
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = frame_info["fl_x"]
        intrinsic[1, 1] = frame_info["fl_y"]
        intrinsic[0, 2] = frame_info["cx"]
        intrinsic[1, 2] = frame_info["cy"]
        intrinsic = intrinsic.float()
        
        return c2w, intrinsic

    def img_center_padding(self, img_np, pad_ratio):
        
        ori_w, ori_h = img_np.shape[:2]
        
        w = round((1 + pad_ratio) * ori_w)
        h = round((1 + pad_ratio) * ori_h)
        
        if len(img_np.shape) > 2:
            img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
        else:
            img_pad_np = np.zeros((w, h), dtype=np.uint8)
        offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
        img_pad_np[offset_h: offset_h + img_np.shape[0]:, offset_w: offset_w + img_np.shape[1]] = img_np
        
        return img_pad_np
    
    def resize_image_keepaspect_np(self, img, max_tgt_size):
        """
        similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio  
        """
        h, w = img.shape[:2]
        ratio = max_tgt_size / max(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)

    def center_crop_according_to_mask(self, img, mask, aspect_standard, enlarge_ratio):
        """ 
            img: [H, W, 3]
            mask: [H, W]
        """ 
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("empty mask")

        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)
        
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        
        half_w = max(abs(center_x - x_min), abs(center_x -  x_max))
        half_h = max(abs(center_y - y_min), abs(center_y -  y_max))
        aspect = half_h / half_w

        if aspect >= aspect_standard:                
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) >  0.01:
            enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
            enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
            enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
            enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
            enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
            half_h, half_w = round(enlarge_ratio_cur * half_h), round(enlarge_ratio_cur * half_w)
            
        assert half_h <= center_y
        assert half_w <= center_x
        assert abs(half_h / half_w - aspect_standard) < 0.03
        
        offset_x = center_x - half_w
        offset_y = center_y - half_h
        
        new_img = img[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        new_mask = mask[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        
        return  new_img, new_mask, offset_x, offset_y        
        
    def load_rgb_image_with_aug_bg(self, rgb_path, mask_path, bg_color, pad_ratio, max_tgt_size, aspect_standard, enlarge_ratio,
                                   render_tgt_size, multiply, intr):
        rgb = np.array(Image.open(rgb_path))
        interpolation = cv2.INTER_AREA
        if rgb.shape[0] != 1024 and rgb.shape[0] == rgb.shape[1]:
            rgb = cv2.resize(rgb, (1024, 1024), interpolation=interpolation)
        if pad_ratio > 0:
            rgb = self.img_center_padding(rgb, pad_ratio)
        
        rgb = rgb / 255.0
        if mask_path is not None:
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path)) > 180
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                assert pad_ratio == 0
                # if pad_ratio > 0:
                #     mask = self.img_center_padding(mask, pad_ratio)
                # mask = mask / 255.0
            else:
                # print("no mask file")
                mask = (rgb >= 0.99).sum(axis=2) == 3
                mask = np.logical_not(mask)
                # erode
                mask = (mask * 255).astype(np.uint8)
                kernel_size, iterations = 3, 7
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=iterations) / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
    
        # crop image to enlarge face area.
        try:
            rgb, mask, offset_x, offset_y = self.center_crop_according_to_mask(rgb, mask, aspect_standard, enlarge_ratio)
        except Exception as ex:
            print(rgb_path, mask_path, ex)

        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

        # resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = self.calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], 
                                                                         aspect_standard=aspect_standard,
                                                                         tgt_size=render_tgt_size, multiply=multiply)
        rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
        
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)

        return rgb, mask, intr
            
    def scale_intrs(self, intrs, ratio_x, ratio_y):
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y  
        return intrs
    
    def uniform_sample_in_chunk(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk)//2])
        return select_list
    
    def calc_new_tgt_size(self, cur_hw, tgt_size, multiply):
        ratio = tgt_size / min(cur_hw)
        tgt_size = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
        tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def calc_new_tgt_size_by_aspect(self, cur_hw, aspect_standard, tgt_size, multiply):
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
        tgt_size = tgt_size * aspect_standard, tgt_size
        tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x
    
    def load_flame_params(self, flame_file_path, teeth_bs=None):
        
        flame_param = dict(np.load(flame_file_path), allow_pickle=True)

        flame_param_tensor = {}
        flame_param_tensor['expr'] = torch.FloatTensor(flame_param['expr'])[0]
        flame_param_tensor['rotation'] = torch.FloatTensor(flame_param['rotation'])[0]
        flame_param_tensor['neck_pose'] = torch.FloatTensor(flame_param['neck_pose'])[0]
        flame_param_tensor['jaw_pose'] = torch.FloatTensor(flame_param['jaw_pose'])[0]
        flame_param_tensor['eyes_pose'] = torch.FloatTensor(flame_param['eyes_pose'])[0]
        flame_param_tensor['translation'] = torch.FloatTensor(flame_param['translation'])[0]
        if teeth_bs is not None:
            flame_param_tensor['teeth_bs'] = torch.FloatTensor(teeth_bs)
            # flame_param_tensor['expr'] = torch.cat([flame_param_tensor['expr'], flame_param_tensor['teeth_bs']], dim=0)
    
        return flame_param_tensor
    
    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        crop_ratio_h, crop_ratio_w = self.crop_range_ratio_hw
        
        uid = self.uids[idx]
        if len(uid.split('/')) == 1:
            uid = os.path.join(self.root_dirs, uid)
        mode_str = "train" if not self.is_val else "test"
        transforms_json = os.path.join(uid, f"transforms_{mode_str}.json")
        
        with open(transforms_json) as fp:
            data = json.load(fp)    
        cor_flame_path = transforms_json.replace('transforms_{}.json'.format(mode_str),'canonical_flame_param.npz')
        flame_param = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_param['shape'])
        # data['static_offset'] = flame_param['static_offset']
                        
        all_frames = data["frames"]

        sample_total_views = self.sample_side_views + 1
        if len(all_frames) >= self.sample_side_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(len(all_frames)))
                else:
                    replace = len(all_frames) < sample_total_views
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(len(all_frames)))
                else:
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = len(all_frames) < sample_total_views
                frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > 1:
                    frame_id_list = np.linspace(0, len(all_frames) - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for i in range(sample_total_views)]
        
        cam_id_list = frame_id_list
        
        assert self.sample_side_views + 1 == len(frame_id_list)

        # source images
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        flame_params = []
        teeth_bs_pth = os.path.join(uid, "tracked_teeth_bs.npz")
        use_teeth = False
        if os.path.exists(teeth_bs_pth) and use_teeth:
            teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
        else:
            teeth_bs_lst = None
        for cam_id, frame_id in zip(cam_id_list, frame_id_list):
            frame_info = all_frames[frame_id]
            frame_path = os.path.join(uid, frame_info["file_path"])
            if 'nersemble' in frame_path or "tiktok_v34" in frame_path:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])
            else:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])

            teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
            flame_path = os.path.join(uid, frame_info["flame_param_path"])
            flame_param = self.load_flame_params(flame_path, teeth_bs)

            # if cam_id == 0:
            #     shape_param = flame_param["betas"]

            c2w, ori_intrinsic = self._load_pose(frame_info, transpose_R="nersemble" in frame_path)

            bg_color = random.choice([0.0, 0.5, 1.0])  # 1.0
            # if self.is_val:
            #     bg_color = 1.0       
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(frame_path, mask_path=mask_path,
                                                                    bg_color=bg_color, 
                                                                    pad_ratio=0,
                                                                    max_tgt_size=None,
                                                                    aspect_standard=self.aspect_standard,
                                                                    enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                                                                    render_tgt_size=self.render_image_res,
                                                                    multiply=16,
                                                                    intr=ori_intrinsic.clone())
            c2ws.append(c2w)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            intrs.append(intrinsic)
            flame_params.append(flame_param)
            masks.append(mask)

        c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]

        flame_params_tmp = defaultdict(list)
        for flame in flame_params:
            for k, v in flame.items():
                flame_params_tmp[k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        flame_params = flame_params_tmp
        # TODO check different betas for same person
        flame_params["betas"] = shape_param
        
        # reference images
        prob_refidx = np.ones(self.sample_side_views + 1)
        if not self.is_val:
            prob_refidx[0] = 0.5  # front_prob
        else:
            prob_refidx[0] = 1.0
        # print(frame_id_list, kinect_color_list, prob_refidx[0])
        prob_refidx[1:] = (1 - prob_refidx[0]) / len(prob_refidx[1:])
        ref_idx = np.random.choice(self.sample_side_views + 1, p=prob_refidx)
        cam_id_source_list = cam_id_list[ref_idx: ref_idx + 1]
        frame_id_source_list = frame_id_list[ref_idx: ref_idx + 1]

        source_c2ws, source_intrs, source_rgbs, source_flame_params = [], [], [], []
        for cam_id, frame_id in zip(cam_id_source_list, frame_id_source_list):
            frame_info = all_frames[frame_id]
            frame_path = os.path.join(uid, frame_info["file_path"])
            if 'nersemble' in frame_path:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])
            else:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
            flame_path = os.path.join(uid, frame_info["flame_param_path"])
            
            teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
            flame_param = self.load_flame_params(flame_path, teeth_bs)

            c2w, ori_intrinsic = self._load_pose(frame_info)
            
            # bg_color = 1.0
            # bg_color = 0.0
            bg_color = random.choice([0.0, 0.5, 1.0])   # 1. 
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(frame_path, mask_path=mask_path, 
                                                                    bg_color=bg_color,
                                                                    pad_ratio=0,
                                                                    max_tgt_size=None, 
                                                                    aspect_standard=self.aspect_standard,
                                                                    enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                                                                    render_tgt_size=self.source_image_res,
                                                                    multiply=self.multiply,
                                                                    intr=ori_intrinsic.clone())

            source_c2ws.append(c2w)
            source_intrs.append(intrinsic)
            source_rgbs.append(rgb)
            source_flame_params.append(flame_param)

        source_c2ws = torch.stack(source_c2ws, dim=0)
        source_intrs = torch.stack(source_intrs, dim=0)
        source_rgbs = torch.cat(source_rgbs, dim=0)

        flame_params_tmp = defaultdict(list)
        for flame in source_flame_params:
            for k, v in flame.items():
                flame_params_tmp['source_'+k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        source_flame_params = flame_params_tmp
        # TODO check different betas for same person
        source_flame_params["source_betas"] = shape_param
    
        render_image = rgbs
        render_mask = masks
        tgt_size = render_image.shape[2:4]   # [H, W]
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 1.1, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 1.1, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,  # [N1, 4, 4]
            'source_intrs': source_intrs,  # [N1, 4, 4]
            'source_rgbs': source_rgbs.clamp(0, 1),   # [N1, 3, H, W]
            'render_image': render_image.clamp(0, 1), # [N, 3, H, W]
            'render_mask': render_mask.clamp(0, 1), #[ N, 1, H, W]
            'c2ws': c2ws,  # [N, 4, 4]
            'intrs': intrs,  # [N, 4, 4]
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),  # [N, 2]
            'render_bg_colors': bg_colors, # [N, 3]
            'pytorch3d_transpose_R': torch.Tensor(["nersemble" in frame_path]), # [1]
        }
        
        #['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas']
        # 'flame_params': flame_params, # dict: body_pose:[N, 21, 3], 
        ret.update(flame_params)
        ret.update(source_flame_params)
            
        return ret

def gen_valid_id_json():
    root_dir = "./train_data/vfhq_vhap/export"
    save_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid_id_list = []
    for file in os.listdir(root_dir):
        if not file.startswith("."):
            valid_id_list.append(file)
    print(len(valid_id_list), valid_id_list[:2])
    with open(save_path, "w") as fp:
        json.dump(valid_id_list, fp)


def gen_valid_id_json():
    root_dir = "./train_data/vfhq_vhap/export"
    mask_root_dir = "./train_data/vfhq_vhap/mask"
    save_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid_id_list = []
    for file in os.listdir(root_dir):
        if not file.startswith(".") and ".txt" not in file:
            valid_id_list.append(file)
    print("raw:", len(valid_id_list), valid_id_list[:2])

    mask_valid_id_list = []
    for file in os.listdir(mask_root_dir):
        if not file.startswith(".") and ".txt" not in file:
            mask_valid_id_list.append(file)
    print("mask:", len(mask_valid_id_list), mask_valid_id_list[:2])

    valid_id_list = list(set(valid_id_list).intersection(set(mask_valid_id_list)))
    print("intesection:", len(mask_valid_id_list), mask_valid_id_list[:2])

    with open(save_path, "w") as fp:
        json.dump(valid_id_list, fp)
        
    save_train_path = "./train_data/vfhq_vhap/label/valid_id_train_list.json"
    save_val_path = "./train_data/vfhq_vhap/label/valid_id_val_list.json"
    valid_id_list = sorted(valid_id_list)
    idxs = np.linspace(0, len(valid_id_list)-1, num=20, endpoint=True).astype(np.int64)
    valid_id_train_list = []
    valid_id_val_list = []
    for i in range(len(valid_id_list)):
        if i in idxs:
            valid_id_val_list.append(valid_id_list[i])
        else:
            valid_id_train_list.append(valid_id_list[i])

    print(len(valid_id_train_list), len(valid_id_val_list), valid_id_val_list)
    with open(save_train_path, "w") as fp:
        json.dump(valid_id_train_list, fp)
        
    with open(save_val_path, "w") as fp:
        json.dump(valid_id_val_list, fp)


if __name__ == "__main__":
    import trimesh
    import cv2
    root_dir = "./train_data/vfhq_vhap/export"
    meta_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    dataset = VideoHeadDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=15,
                    render_image_res_low=512, render_image_res_high=512,
                    render_region_size=(512, 512), source_image_res=512,
                    enlarge_ratio=[0.8, 1.2],
                    debug=False, is_val=False)

    from lam.models.rendering.flame_model.flame import FlameHeadSubdivided

    # subdivided flame 
    subdivide = 2
    flame_sub_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=True,
        add_shoulder=False,
        flame_model_path='pretrained_models/human_model_files/flame_assets/flame/flame2023.pkl',
        flame_lmk_embedding_path="pretrained_models/human_model_files/flame_assets/flame/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path="pretrained_models/human_model_files/flame_assets/flame/head_template_mesh.obj",
        flame_parts_path="pretrained_models/human_model_files/flame_assets/flame/FLAME_masks.pkl",
        subdivide_num=subdivide,
        teeth_bs_flag=False,
    ).cuda()
        
    source_key = "source_rgbs"
    render_key = "render_image"
        
    for idx, data in enumerate(dataset):
        import boxx
        boxx.tree(data)
        if idx > 0:
            exit(0)
        os.makedirs("debug_vis/dataloader", exist_ok=True)
        for i in range(data[source_key].shape[0]):
            cv2.imwrite(f"debug_vis/dataloader/{source_key}_{i}_b{idx}.jpg", ((data[source_key][i].permute(1, 2, 0).numpy()[:, :, (2, 1, 0)] * 255).astype(np.uint8)))
            
        for i in range(data[render_key].shape[0]):
            cv2.imwrite(f"debug_vis/dataloader/rgbs{i}_b{idx}.jpg", ((data[render_key][i].permute(1, 2, 0).numpy()[:, :, (2, 1, 0)] * 255).astype(np.uint8)))
            

        save_root = "./debug_vis/dataloader"
        os.makedirs(save_root, exist_ok=True)

        shape = data['betas'].to('cuda')
        flame_param = {}
        flame_param['expr'] = data['expr'].to('cuda')
        flame_param['rotation'] = data['rotation'].to('cuda')
        flame_param['neck'] = data['neck_pose'].to('cuda')
        flame_param['jaw'] = data['jaw_pose'].to('cuda')
        flame_param['eyes'] = data['eyes_pose'].to('cuda')
        flame_param['translation'] = data['translation'].to('cuda')


        v_cano = flame_sub_model.get_cano_verts(
            shape.unsqueeze(0)
        )
        ret = flame_sub_model.animation_forward(
            v_cano.repeat(flame_param['expr'].shape[0], 1, 1),
            shape.unsqueeze(0).repeat(flame_param['expr'].shape[0], 1),
            flame_param['expr'],
            flame_param['rotation'],
            flame_param['neck'],
            flame_param['jaw'],
            flame_param['eyes'],
            flame_param['translation'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            # static_offset=batch_data['static_offset'].to('cuda'),
            static_offset=None,
        )

        import boxx
        boxx.tree(data)
        boxx.tree(ret)
        
        for i in range(ret["animated"].shape[0]):
            mesh = trimesh.Trimesh()
            mesh.vertices = np.array(ret["animated"][i].cpu().squeeze())
            mesh.faces = np.array(flame_sub_model.faces.cpu().squeeze())
            mesh.export(f'{save_root}/animated_sub{subdivide}_{i}.obj')

            intr = data["intrs"][i]
            from lam.models.rendering.utils.vis_utils import render_mesh
            cam_param = {"focal": torch.tensor([intr[0, 0], intr[1, 1]]), 
                        "princpt": torch.tensor([intr[0, 2], intr[1, 2]])}
            render_shape = data[render_key].shape[2:] # int(cam_param['princpt'][1]* 2), int(cam_param['princpt'][0] * 2)
            
            face = flame_sub_model.faces.cpu().squeeze().numpy()
            vertices = ret["animated"][i].cpu().squeeze()
            
            c2ws = data["c2ws"][i]
            w2cs = torch.inverse(c2ws)
            if data['pytorch3d_transpose_R'][0] > 0:
                R = w2cs[:3, :3].transpose(1, 0)
            else:
                R = w2cs[:3, :3]
            T = w2cs[:3, 3]
            vertices = vertices @ R + T
            mesh_render, is_bkg = render_mesh(vertices, face, cam_param=cam_param, 
                                            bkg=np.ones((render_shape[0],render_shape[1], 3), dtype=np.float32) * 255, 
                                            return_bg_mask=True)
            
            rgb_mesh = mesh_render.astype(np.uint8)
            t_image = (data[render_key][i].permute(1, 2, 0)*255).numpy().astype(np.uint8)
            
            blend_ratio = 0.7
            vis_img = np.concatenate([rgb_mesh, t_image, (blend_ratio * rgb_mesh + (1 -  blend_ratio) * t_image).astype(np.uint8)], axis=1)
            cam_idx = int(data.get('cam_idxs', [i for j in range(16)])[i])

            cv2.imwrite(os.path.join(save_root, f"render_{cam_idx}.jpg"), vis_img[:, :, (2, 1, 0)])
