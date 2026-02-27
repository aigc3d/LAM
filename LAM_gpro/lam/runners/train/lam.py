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


import os
import math
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.utils import make_grid
from einops import rearrange, repeat
from accelerate.logging import get_logger
from taming.modules.losses.vqperceptual import hinge_d_loss

from .base_trainer import Trainer
from lam.utils.profiler import DummyProfiler
from lam.runners import REGISTRY_RUNNERS
from lam.utils.hf_hub import wrap_model_hub
from safetensors.torch import load_file
from pytorch3d.ops.knn import knn_points
import torch.nn.functional as F

logger = get_logger(__name__)

# torch.autograd.set_detect_anomaly(True)


from omegaconf import OmegaConf
@REGISTRY_RUNNERS.register('train.lam')
class LAMTrainer(Trainer):

    EXP_TYPE: str = 'lam'

    def __init__(self):
        super().__init__()
        
        self.model = self._build_model(self.cfg)
        if self.has_disc:
            self.model_disc = self._build_model_disc(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        if self.has_disc:
            self.optimizer_disc = self._build_optimizer(self.model_disc, self.cfg)
            
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        if self.has_disc:
            self.scheduler_disc = self._build_scheduler(self.optimizer_disc, self.cfg)
        self.pixel_loss_fn, self.perceptual_loss_fn, self.tv_loss_fn = self._build_loss_fn(self.cfg)
        self.only_sym_conf = 2
        print("==="*16*3, "\n"+"only_sym_conf:", self.only_sym_conf, "\n"+"==="*16*3)
        
        
    def _build_model(self, cfg):
        assert cfg.experiment.type == 'lrm', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from lam.models import ModelLAM
        model = ModelLAM(**cfg.model)

        # resume
        if len(self.cfg.train.resume) > 0:
            resume = self.cfg.train.resume
            print("==="*16*3)
            self.accelerator.print("loading pretrained weight from:", resume)
            if resume.endswith('safetensors'):
                ckpt = load_file(resume, device='cpu')
            else:
                ckpt = torch.load(resume, map_location='cpu')
            state_dict = model.state_dict()
            for k, v in ckpt.items():
                if k in state_dict:
                    if state_dict[k].shape == v.shape:
                        state_dict[k].copy_(v)
                    else:
                        self.accelerator.print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
                else:
                    self.accelerator.print(f"WARN] unexpected param {k}: {v.shape}")
            self.accelerator.print("Finish loading ckpt:", resume, "\n"+"==="*16*3)
        return model

    def _build_model_disc(self, cfg):
        if cfg.model.disc.type == "pix2pix":
            from lam.models.discriminator import NLayerDiscriminator, weights_init
            model = NLayerDiscriminator(input_nc=cfg.model.disc.in_channels,
                                        n_layers=cfg.model.disc.num_layers,
                                        use_actnorm=cfg.model.disc.use_actnorm
                                        ).apply(weights_init)

        elif cfg.model.disc.type == "vqgan":
            from lam.models.discriminator import Discriminator
            model = Discriminator(in_channels=cfg.model.disc.in_channels,
                                  cond_channels=0, hidden_channels=512,
                                  depth=cfg.model.disc.depth)
        elif cfg.model.disc.type == "stylegan":
            from lam.models.gan.stylegan_discriminator import SingleDiscriminatorV2, SingleDiscriminator
            from lam.models.gan.stylegan_discriminator_torch import Discriminator
        
            model = Discriminator(512, channel_multiplier=2)
            
            model.input_size = cfg.model.disc.img_res
        else:
            raise NotImplementedError
        return model

    def _build_optimizer(self, model: nn.Module, cfg):
        decay_params, no_decay_params = [], []
        
        # add all bias and LayerNorm params to no_decay_params
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )
        
        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == 'cosine':
            from lam.utils.scheduler import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _build_dataloader(self, cfg):
        # dataset class
        from lam.datasets import MixerDataset
        gaga_track_type = cfg.dataset.get("gaga_track_type", "vfhq_gagtrack")
        sample_aug_views = cfg.dataset.get("sample_aug_views", 0)

        # build dataset
        load_normal = cfg.train.loss.get("normal_weight", False) > 0. if hasattr(cfg.train.loss, "normal_weight") else False
        load_normal = load_normal or (cfg.train.loss.get("surfel_normal_weight", False) > 0. if hasattr(cfg.train.loss, "surfel_normal_weight") else False)
        print("==="*16*3, "\nload_normal:", load_normal)
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            repeat_num=cfg.dataset.repeat_num if hasattr(cfg.dataset, "repeat_num") else 1,
            multiply=cfg.dataset.multiply if hasattr(cfg.dataset, "multiply") else 14,
            debug=cfg.dataset.debug if hasattr(cfg.dataset, "debug") else False,
            is_val=False,
            gaga_track_type=gaga_track_type,
            sample_aug_views=sample_aug_views,
            load_albedo=cfg.model.get("render_albedo", False) if hasattr(cfg.model, "render_albedo") else False,
            load_normal=load_normal,
        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            repeat_num=cfg.dataset.repeat_num if hasattr(cfg.dataset, "repeat_num") else 1,
            multiply=cfg.dataset.multiply if hasattr(cfg.dataset, "multiply") else 14,
            debug=cfg.dataset.debug if hasattr(cfg.dataset, "debug") else False,
            is_val=True,
            gaga_track_type=gaga_track_type,
            sample_aug_views=sample_aug_views,
            load_albedo=cfg.model.get("render_albedo", False) if hasattr(cfg.model, "render_albedo") else False,
            load_normal=load_normal,
        )

        # build data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.dataset.num_train_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.dataset.num_val_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=False,
        )

        return train_loader, val_loader

    def _build_loss_fn(self, cfg):
        from lam.losses import PixelLoss, LPIPSLoss, TVLoss
        pixel_loss_fn = PixelLoss(option=cfg.train.loss.get("pixel_loss_fn", "mse"))
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)
            
        if cfg.model.get("use_conf_map", False):
            assert cfg.train.loss.get("head_pl", False), "Set head_pl in train.loss to true to use faceperceptualloss when using conf_map."
        tv_loss_fn = TVLoss()
        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn

    def register_hooks(self):
        pass

    def get_flame_params(self, data, is_source=False):
        flame_params = {}        
        flame_keys = ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas',\
                      'rotation', 'neck_pose', 'eyes_pose', 'translation', "teeth_bs"]
        if is_source:
            flame_keys = ['source_'+item for item in flame_keys]
        for k, v in data.items():
            if k in flame_keys:
                # print(k, v.shape)
                flame_params[k] = data[k]
        return flame_params
    
    def cross_copy(self, data):
        B = data.shape[0]
        assert data.shape[1] == 1
        new_data = []
        for i in range(B):
            B_i = [data[i]]
            for j in range(B):
                if j != i:
                    B_i.append(data[j])
            new_data.append(torch.concat(B_i, dim=0))
        new_data = torch.stack(new_data, dim=0)
        
        return new_data
    
    def prepare_cross_render_data(self, data):
        B, N_v, C, H, W = data['render_image'].shape
        assert N_v == 1
        
        # cross copy      
        data["c2ws"] = self.cross_copy(data["c2ws"])
        data["intrs"] = self.cross_copy(data["intrs"])
        data["render_full_resolutions"] = self.cross_copy(data["render_full_resolutions"])          
        data["render_image"] = self.cross_copy(data["render_image"])
        data["render_mask"] = self.cross_copy(data["render_mask"])
        data["render_bg_colors"] = self.cross_copy(data["render_bg_colors"])
        flame_params = self.get_flame_params(data)
        for key in flame_params.keys():
            if "betas" not in key:
                data[key] = self.cross_copy(data[key])
        source_flame_params = self.get_flame_params(data, is_source=True)
        for key in source_flame_params.keys():
            if "betas" not in key:
                data[key] = self.cross_copy(data[key])
                
        return data    
    
    def get_loss_weight(self, loss_weight):
        if isinstance(loss_weight, str) and ":" in loss_weight:
            start_step, start_value, end_value, end_step = map(float, loss_weight.split(":"))
            current_step = self.global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
            return value
        elif isinstance(loss_weight, (float, int)):
            return loss_weight
        else:
            raise NotImplementedError                
        
    def forward_loss_local_step(self, data):
        render_image = data['render_image']
        render_albedo = data.get('render_albedo', None)
        render_mask = data['render_mask']
        render_normal = data.get('render_normal', None)
        B, N_v, C, H, W = render_image.shape
        flame_params = self.get_flame_params(data)
        source_flame_params = self.get_flame_params(data, is_source=True)
        
        # forward
        outputs = self.model(
                    image=data['source_rgbs'], 
                    source_c2ws=data['source_c2ws'], 
                    source_intrs=data['source_intrs'], 
                    render_c2ws=data['c2ws'], 
                    render_intrs=data['intrs'], 
                    render_bg_colors=data['render_bg_colors'], 
                    flame_params=flame_params,
                    source_flame_params=source_flame_params,
                    render_images=render_image,
                    data = data
        )
        
        # loss calculation
        loss = 0.
        loss_pixel = None
        loss_perceptual = None
        loss_mask = None 
        extra_loss_dict = {}

        num_aug_view = self.cfg.dataset.get("sample_aug_views", 0)
        real_num_view = data["real_num_view"] - num_aug_view

        conf_sigma_l1 = outputs.get("conf_sigma_l1", None)
        conf_sigma_percl = outputs.get("conf_sigma_percl", None)
        if self.cfg.model.use_sym_proj:
            real_num_view *= 2
            if self.cfg.model.use_conf_map:
                conf_sigma_l1 = rearrange(conf_sigma_l1, "b v (c r) h w -> b (v r) c h w", r=2)[:, :real_num_view]
                conf_sigma_percl = rearrange(conf_sigma_percl, "b v (c r) h w -> b (v r) c h w", r=2)[:, :real_num_view]
            render_image = repeat(data['render_image'], "b v c h w -> b (v r) c h w", r=2)
            render_albedo = repeat(render_albedo, "b v c h w -> b (v r) c h w", r=2) if render_albedo is not None else None
            render_mask = repeat(data['render_mask'], "b v c h w -> b (v r) c h w", r=2)
            if "render_normal" in data.keys():
                render_normal = repeat(data['render_normal'], "b v c h w -> b (v r) c h w", r=2)
            for k, v in data.items():
                if "bbox" in k:
                    data[k] = repeat(v, "b v c -> b (v r) c", r=2)

        only_sym_conf = self.only_sym_conf

        if self.get_loss_weight(self.cfg.train.loss.get("masked_pixel_weight", 0)) > 0.:
            gt_rgb = render_image[:, :real_num_view] * render_mask[:, :real_num_view] + 1.0 * (1 - render_mask[:, :real_num_view])
            pred_rgb = outputs['comp_rgb'][:, :real_num_view] * render_mask[:, :real_num_view] + 1.0 * (1 - render_mask[:, :real_num_view])
            
            loss_pixel = self.pixel_loss_fn(pred_rgb, gt_rgb, conf_sigma_l1, only_sym_conf=only_sym_conf) * self.get_loss_weight(self.cfg.train.loss.masked_pixel_weight)
            loss += loss_pixel

            # using same weight
            loss_perceptual = self.perceptual_loss_fn(pred_rgb, gt_rgb, conf_sigma=conf_sigma_percl, only_sym_conf=only_sym_conf) * self.get_loss_weight(self.cfg.train.loss.masked_pixel_weight)
            loss += loss_perceptual

        if  self.get_loss_weight(self.cfg.train.loss.pixel_weight) > 0.:
            total_loss_pixel = loss_pixel
            if (hasattr(self.cfg.train.loss, 'rgb_weight') and self.get_loss_weight(self.cfg.train.loss.rgb_weight) > 0.) or not hasattr(self.cfg.train.loss, "rgb_weight"):
                loss_pixel = self.pixel_loss_fn(
                    outputs['comp_rgb'][:, :real_num_view], render_image[:, :real_num_view], conf_sigma=conf_sigma_l1, only_sym_conf=only_sym_conf
                ) * self.get_loss_weight(self.cfg.train.loss.pixel_weight)
                loss += loss_pixel
            if total_loss_pixel is not None:
                loss_pixel += total_loss_pixel

        if  self.get_loss_weight(self.cfg.train.loss.perceptual_weight) > 0.:
            total_loss_perceptual = loss_perceptual
            if (hasattr(self.cfg.train.loss, 'rgb_weight') and self.get_loss_weight(self.cfg.train.loss.rgb_weight) > 0.) or not hasattr(self.cfg.train.loss, "rgb_weight"):
                loss_perceptual = self.perceptual_loss_fn(
                    outputs['comp_rgb'][:, :real_num_view], render_image[:, :real_num_view], conf_sigma=conf_sigma_percl, only_sym_conf=only_sym_conf
                ) * self.get_loss_weight(self.cfg.train.loss.perceptual_weight)
                loss += loss_perceptual
            if total_loss_perceptual is not None:
                loss_perceptual += total_loss_perceptual

        if  self.get_loss_weight(self.cfg.train.loss.mask_weight) > 0. and 'comp_mask' in outputs.keys():
            loss_mask = self.pixel_loss_fn(outputs['comp_mask'][:, :real_num_view],  render_mask[:, :real_num_view], conf_sigma=conf_sigma_l1, only_sym_conf=only_sym_conf
                                           ) * self.get_loss_weight(self.cfg.train.loss.mask_weight)
            loss += loss_mask
        
        if hasattr(self.cfg.train.loss, 'offset_reg_weight') and self.get_loss_weight(self.cfg.train.loss.offset_reg_weight) > 0.:
            loss_offset_reg = 0
            for b_idx in range(len(outputs['3dgs'])):
                loss_offset_reg += torch.nn.functional.mse_loss(outputs['3dgs'][b_idx][0].offset.float(), torch.zeros_like(outputs['3dgs'][b_idx][0].offset.float()))
            loss_offset_reg = loss_offset_reg / len(outputs['3dgs'])
            loss += loss_offset_reg * self.get_loss_weight(self.cfg.train.loss.offset_reg_weight)   
        else:
            loss_offset_reg = None

        return outputs, loss, loss_pixel, loss_perceptual, loss_offset_reg, loss_mask, extra_loss_dict

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer, discriminator_weight=1):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight
    
    def disc_preprocess(self, img):
        # reshape [B, N_v, C, H, W] to [B*N_v, C, H, W]
        img = torch.flatten(img, 0, 1)
        # img = rearrange(img, 'b n c h w -> (b n) c h w')
        # convert 0-1 to -1-1
        img = 2 * img - 1
        
        if hasattr(self.accelerator.unwrap_model(self.model_disc), "input_size"):
            tgt_size = self.accelerator.unwrap_model(self.model_disc).input_size
            img = nn.functional.interpolate(img, (tgt_size, tgt_size))
        img = img.float()
            
        return img
    
    def forward_to_get_loss_with_gen_loss(self, data):
        # forward to loss
        outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, extra_loss_dict = self.forward_loss_local_step(data)

        with torch.autocast(device_type=outs["comp_rgb"].device.type, dtype=torch.float32):
            logits_fake = self.model_disc(self.disc_preprocess(outs["comp_rgb"]))
        
        loss_gen = -torch.mean(logits_fake)
        
        try:            
            if loss < 1e-5:
                d_weight = self.cfg.model.disc.disc_weight
            else:
                nll_loss = loss_pixel
                if nll_loss is None:
                    nll_loss = loss
                d_weight = self.calculate_adaptive_weight(nll_loss, loss_gen,
                                                          last_layer=self.accelerator.unwrap_model(self.model).get_last_layer(), 
                                                          discriminator_weight=self.cfg.model.disc.disc_weight)
        except RuntimeError:
            print("*************Error when calculate_adaptive_weight************")
            d_weight = torch.tensor(0.0)
            
        disc_factor = self.adopt_weight(1.0, self.global_step, threshold=self.cfg.model.disc.disc_iter_start)
        # print(disc_factor, d_weight)
        
        loss += disc_factor * d_weight * loss_gen
        
        # backward
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, loss_gen, extra_loss_dict
        

    def forward_to_get_loss(self, data):
        # forward to loss
        outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, extra_loss_dict = self.forward_loss_local_step(data)
                
        # backward
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, extra_loss_dict


    def forward_disc_loss_local_step(self, pred_img, gt_img):
        # detach gradient of pred_img
        with torch.autocast(device_type=pred_img.device.type, dtype=torch.float32):
            logits_real = self.model_disc(self.disc_preprocess(gt_img).detach())
            logits_fake = self.model_disc(self.disc_preprocess(pred_img).detach()) 
            
        loss_disc = hinge_d_loss(logits_real, logits_fake)
        return loss_disc


    def forward_to_get_disc_loss(self, pred_img, gt_img):
        # forward to loss
        loss_disc = self.forward_disc_loss_local_step(pred_img, gt_img)

        disc_factor = self.adopt_weight(1.0, self.global_step, threshold=self.cfg.model.disc.disc_iter_start)
        loss = disc_factor * loss_disc
            
        # backward
        self.accelerator.backward(loss)
        
        if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
            self.accelerator.clip_grad_norm_(self.model_disc.parameters(), self.cfg.train.optim.clip_grad_norm)
            
        self.optimizer_disc.step()
        self.optimizer_disc.zero_grad()

        return loss_disc

    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile, iepoch: int):

        self.model.train()
        if self.has_disc:
            self.model_disc.train()

        local_step_losses = []
        global_step_losses = []
        local_step_extra_losses = []
        global_step_extra_losses = []
        extra_loss_keys = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        loss_disc = None
        for idx, data in enumerate(loader):
            data["source_rgbs"] = data["source_rgbs"].to(self.weight_dtype)
            if self.has_disc and hasattr(self.cfg.model.disc, "cross_render") and self.cfg.model.disc.cross_render:
                data = self.prepare_cross_render_data(data)
                data["real_num_view"] = 1
            else:
                data["real_num_view"] = data["render_image"].shape[1]
            
            logger.debug(f"======== Starting global step {self.global_step} ========")
            
            if not self.has_disc:
                disc_step = False
                with self.accelerator.accumulate(self.model):
                    outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, extra_loss_dict = self.forward_to_get_loss(data)
                                        
                    # track local losses
                    loss_disc, loss_gen = None, None
                    local_step_losses.append(torch.stack([
                        _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                        for _loss in [loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, loss_disc, loss_gen]
                    ]))
                    extra_loss_keys = sorted(list(extra_loss_dict.keys()))
                    if len(extra_loss_keys) > 0:
                        local_step_extra_losses.append(torch.stack([
                            extra_loss_dict[k].detach() if extra_loss_dict[k] is not None else torch.tensor(float('nan'), device=self.device)
                            for k in extra_loss_keys
                        ]))
            else:
                disc_step = (idx % 5) == 0 or (iepoch * len(loader) + idx < 100 and idx % 2 == 0)
                local_step_losses_bak = torch.zeros(6, device=data["source_rgbs"].device)
                if not disc_step:
                    with self.accelerator.accumulate(self.model):
                        # generator step
                        outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, loss_gen, extra_loss_dict = self.forward_to_get_loss_with_gen_loss(data)
                        # track local losses
                        local_step_losses.append(torch.stack([
                            _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                            for _loss in [loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, loss_gen, loss_disc]
                        ]))
                        local_step_losses_bak = local_step_losses[-1].detach()
                        torch.cuda.empty_cache()
                        extra_loss_keys = sorted(list(extra_loss_dict.keys()))
                        if len(extra_loss_keys) > 0:
                            local_step_extra_losses.append(torch.stack([
                                extra_loss_dict[k].detach() if extra_loss_dict[k] is not None else torch.tensor(float('nan'), device=self.device)
                                for k in extra_loss_keys
                            ]))
                else:
                    with self.accelerator.accumulate(self.model_disc):
                        # discriminator step
                        outs, _, _, _, _, _, _ = self.forward_loss_local_step(data)
                        loss_disc = self.forward_to_get_disc_loss(pred_img=outs["comp_rgb"],
                                                                  gt_img=data["render_image"])
                        local_step_losses.append(torch.concat([local_step_losses_bak[:6], loss_disc.unsqueeze(0)], dim=0))
                        torch.cuda.empty_cache()

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                if not disc_step:
                    self.scheduler.step()
                if self.has_disc and disc_step:
                    self.scheduler_disc.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                if len(extra_loss_keys) > 0:
                    global_step_extra_loss = self.accelerator.gather(torch.stack(local_step_extra_losses)).mean(dim=0).cpu()
                    global_step_extra_loss_items = global_step_extra_loss.unbind()
                else:
                    global_step_extra_loss = None
                    global_step_extra_loss_items = []
                loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, loss_gen, loss_disc_  = global_step_loss.unbind()
                loss_kwargs = {
                    'loss': loss.item(),
                    'loss_pixel': loss_pixel.item(),
                    'loss_perceptual': loss_perceptual.item(),
                    'loss_tv': loss_tv.item(),
                    'loss_mask': loss_mask.item(),
                    'loss_disc': loss_disc_.item(),
                    'loss_gen': loss_gen.item(),
                }
                for k, loss in zip(extra_loss_keys, global_step_extra_loss_items):
                    loss_kwargs[k] = loss.item()
                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **loss_kwargs
                )
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)
                local_step_extra_losses = []
                global_step_extra_losses.append(global_step_extra_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                description = '[TRAIN STEP]' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
                pbar.set_description(description)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                    if self.has_disc:
                        self.model_disc.train()
                if (self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0) or (self.global_step < 1000 and self.global_step % 20 == 0):
                    conf_sigma_l1 = outs.get('conf_sigma_l1', None)
                    conf_sigma_l1 = conf_sigma_l1.cpu() if conf_sigma_l1 is not None else None
                    conf_sigma_percl = outs.get('conf_sigma_percl', None)
                    conf_sigma_percl = conf_sigma_percl.cpu() if conf_sigma_percl is not None else None
                    self.log_image_monitor(
                        step=self.global_step, split='train',
                        renders=outs['comp_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        conf_sigma_l1=conf_sigma_l1, conf_sigma_percl=conf_sigma_percl,
                        gts=data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )
                    if 'comp_mask' in outs.keys():
                        self.log_image_monitor(
                            step=self.global_step, split='train',
                            renders=outs['comp_mask'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                            gts=data['render_mask'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                            prefix="_mask",
                        )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_perceptual, epoch_loss_tv, epoch_loss_mask, epoch_loss_disc, epoch_loss_gen = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_perceptual': epoch_loss_perceptual.item(),
            'loss_tv': epoch_loss_tv.item(),
            'loss_mask': epoch_loss_mask.item(),
            'loss_disc': epoch_loss_disc.item(),
            'loss_gen': epoch_loss_gen.item(),
        }
        if len(extra_loss_keys) > 0:
            epoch_extra_losses = torch.stack(global_step_extra_losses).mean(dim=0)
            for k, v in zip(extra_loss_keys, epoch_extra_losses.unbind()):
                epoch_loss_dict[k] = v.item()
        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
        ) as pbar:

            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=100,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                )),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) if self.cfg.logger.enable_profiler else DummyProfiler()
            
            with profiler:
                self.optimizer.zero_grad()
                if self.has_disc:
                    self.optimizer_disc.zero_grad()
                for iepoch in range(self.current_epoch, self.cfg.train.epochs):

                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler, iepoch=iepoch)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            # final checkpoint and evaluation
            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: int = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        running_extra_losses = []
        extra_loss_keys = []
        sample_data, sample_outs = None, None

        for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):
            data["source_rgbs"] = data["source_rgbs"].to(self.weight_dtype)
            if self.has_disc and hasattr(self.cfg.model.disc, "cross_render") and self.cfg.model.disc.cross_render:
                data = self.prepare_cross_render_data(data)
                data["real_num_view"] = 1
            else:
                data["real_num_view"] = data["render_image"].shape[1]
                
            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            outs, loss, loss_pixel, loss_perceptual, loss_tv, loss_mask, extra_loss_dict = self.forward_loss_local_step(data)
            extra_loss_dict = sorted(list(extra_loss_dict.keys()))
            sample_data, sample_outs = data, outs

            running_losses.append(torch.stack([
                _loss if _loss is not None else torch.tensor(float('nan'), device=self.device)
                for _loss in [loss, loss_pixel, loss_perceptual, loss_tv, loss_mask]
            ]))
            if len(extra_loss_keys) > 0:
                running_extra_losses.append(torch.stack([
                    extra_loss_dict[k] if extra_loss_dict[k] is not None else torch.tensor(float('nan'), device=self.device)
                    for k in extra_loss_keys
                ]))

            # log each step
            conf_sigma_l1 = sample_outs.get('conf_sigma_l1', None)
            conf_sigma_l1 = conf_sigma_l1.cpu() if conf_sigma_l1 is not None else None
            conf_sigma_percl = sample_outs.get('conf_sigma_percl', None)
            conf_sigma_percl = conf_sigma_percl.cpu() if conf_sigma_percl is not None else None
            self.log_image_monitor_each_process(
                step=self.global_step, split='val',
                renders=sample_outs['comp_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                conf_sigma_l1=conf_sigma_l1, conf_sigma_percl=conf_sigma_percl,
                prefix=f"_{len(running_losses)}_rank{self.accelerator.process_index}"
            )
            if "comp_mask" in sample_outs.keys():
                self.log_image_monitor_each_process(
                    step=self.global_step, split='val',
                    renders=sample_outs['comp_mask'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    gts=sample_data['render_mask'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    prefix=f"_mask_{len(running_losses)}_rank{self.accelerator.process_index}"
                )
            
        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        total_loss, total_loss_pixel, total_loss_perceptual, total_loss_offset, total_loss_mask = total_losses.unbind()
        total_loss_dict = {
            'loss': total_loss.item(),
            'loss_pixel': total_loss_pixel.item(),
            'loss_perceptual': total_loss_perceptual.item(),
            'loss_offset': total_loss_offset.item(),
            'loss_mask': total_loss_mask.item(),
        }
        if len(extra_loss_keys) > 0:
            total_extra_losses = self.accelerator.gather(torch.stack(running_extra_losses)).mean(dim=0).cpu()
            for k, v in zip(extra_loss_keys, total_extra_losses.unbind()):
                total_loss_dict[k] = v.item()

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
        else:
            self.log_scalar_kwargs(
                step=self.global_step, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )

    def log_image_monitor_each_process(
        self, epoch: int = None, step: int = None, split: str = None,
        renders: torch.Tensor = None, gts: torch.Tensor = None, prefix=None,
        conf_sigma_l1: torch.Tensor = None, conf_sigma_percl: torch.Tensor = None
        ):
        M = renders.shape[1]
        if gts.shape[1] != M:
            gts = repeat(gts, "b v c h w -> b (v r) c h w", r=2)
        merged = torch.stack([renders, gts], dim=1)[0].view(-1, *renders.shape[2:])
        renders, gts = renders.view(-1, *renders.shape[2:]), gts.view(-1, *gts.shape[2:])
        renders, gts, merged = make_grid(renders, nrow=M), make_grid(gts, nrow=M), make_grid(merged, nrow=M)
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        split = split + prefix if prefix is not None else split
        log_img_dict = {
            f'Images_split{split}/rendered': renders.unsqueeze(0),
            f'Images_split{split}/gt': gts.unsqueeze(0),
            f'Images_split{split}/merged': merged.unsqueeze(0),
        }
        if conf_sigma_l1 is not None:
            EPS = 1e-7
            vis_conf_l1 = 1/(1+conf_sigma_l1.detach()+EPS).cpu()
            vis_conf_percl = 1/(1+conf_sigma_percl.detach()+EPS).cpu()
            vis_conf_l1, vis_conf_percl = rearrange(vis_conf_l1, "b v (r c) h w -> (b v r) c h w", r=2), rearrange(vis_conf_percl, "b v (r c) h w -> (b v r) c h w", r=2)
            vis_conf_l1, vis_conf_percl = repeat(vis_conf_l1, "b c1 h w-> b (c1 c2) h w", c2=3), repeat(vis_conf_percl, "b c1 h w -> b (c1 c2) h w", c2=3)
            vis_conf_l1, vis_conf_percl = make_grid(vis_conf_l1, nrow=M), make_grid(vis_conf_percl, nrow=M)
            log_img_dict[f'Images_split{split}/conf_l1'] = vis_conf_l1.unsqueeze(0)
            log_img_dict[f'Images_split{split}/conf_percl'] = vis_conf_percl.unsqueeze(0)

        self.log_images_each_process(log_img_dict, log_progress, {"imwrite_image": False})

     
    @Trainer.control('on_main_process')
    def log_image_monitor(
        self, epoch: int = None, step: int = None, split: str = None,
        renders: torch.Tensor = None, gts: torch.Tensor = None, prefix=None,
        conf_sigma_l1: torch.Tensor = None, conf_sigma_percl: torch.Tensor = None
        ):
        self.log_image_monitor_each_process(epoch, step, split, renders, gts, prefix, conf_sigma_l1, conf_sigma_percl)
