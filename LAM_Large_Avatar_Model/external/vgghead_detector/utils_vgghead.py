#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from Orest Kupyn (University of Oxford).

import torch
import torchvision

def reproject_vertices(flame_model, vgg_results):
    # flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0)
    vertices, _ = flame_model(
        shape_params=vgg_results['shapecode'], 
        expression_params=vgg_results['expcode'], 
        pose_params=vgg_results['posecode'],
        verts_sclae=1.0
    )
    vertices[:, :, 2] += 0.05 # MESH_OFFSET_Z
    vgg_landmarks3d = flame_model._vertices2landmarks(vertices)
    vgg_transform_results = vgg_results['transform']
    rotation_mat = rot_mat_from_6dof(vgg_transform_results['rotation_6d']).type(vertices.dtype)
    translation = vgg_transform_results['translation'][:, None, :]
    scale = torch.clamp(vgg_transform_results['scale'][:, None], 1e-8)
    rot_vertices = vertices.clone()
    rot_vertices = torch.matmul(rotation_mat.unsqueeze(1), rot_vertices.unsqueeze(-1))[..., 0]
    vgg_landmarks3d = torch.matmul(rotation_mat.unsqueeze(1), vgg_landmarks3d.unsqueeze(-1))[..., 0]
    proj_vertices = (rot_vertices * scale) + translation
    vgg_landmarks3d = (vgg_landmarks3d * scale) + translation
    
    trans_padding, trans_scale = vgg_results['normalize']['padding'], vgg_results['normalize']['scale']
    proj_vertices[:, :, 0] -= trans_padding[:, 0, None]
    proj_vertices[:, :, 1] -= trans_padding[:, 1, None]
    proj_vertices = proj_vertices / trans_scale[:, None, None]
    vgg_landmarks3d[:, :, 0] -= trans_padding[:, 0, None]
    vgg_landmarks3d[:, :, 1] -= trans_padding[:, 1, None]
    vgg_landmarks3d = vgg_landmarks3d / trans_scale[:, None, None]
    return proj_vertices.float()[..., :2], vgg_landmarks3d.float()[..., :2]


def rot_mat_from_6dof(v: torch.Tensor) -> torch.Tensor:
    assert v.shape[-1] == 6
    v = v.view(-1, 6)
    vx, vy = v[..., :3].clone(), v[..., 3:].clone()

    b1 = torch.nn.functional.normalize(vx, dim=-1)
    b3 = torch.nn.functional.normalize(torch.cross(b1, vy, dim=-1), dim=-1)
    b2 = -torch.cross(b1, b3, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def nms(boxes_xyxy, scores, flame_params,
        confidence_threshold: float = 0.5, iou_threshold: float = 0.5, 
        top_k: int = 1000, keep_top_k: int = 100
    ):
    for pred_bboxes_xyxy, pred_bboxes_conf, pred_flame_params in zip(
            boxes_xyxy.detach().float(),
            scores.detach().float(),
            flame_params.detach().float(),
    ):
        pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
        conf_mask = pred_bboxes_conf >= confidence_threshold

        pred_bboxes_conf = pred_bboxes_conf[conf_mask]
        pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask]
        pred_flame_params = pred_flame_params[conf_mask]

        # Filter all predictions by self.nms_top_k
        if pred_bboxes_conf.size(0) > top_k:
            topk_candidates = torch.topk(pred_bboxes_conf, k=top_k, largest=True, sorted=True)
            pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
            pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
            pred_flame_params = pred_flame_params[topk_candidates.indices]

        # NMS
        idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf, iou_threshold=iou_threshold)

        final_bboxes = pred_bboxes_xyxy[idx_to_keep][: keep_top_k]  # [Instances, 4]
        final_scores = pred_bboxes_conf[idx_to_keep][: keep_top_k]  # [Instances, 1]
        final_params = pred_flame_params[idx_to_keep][: keep_top_k]  # [Instances, Flame Params]
        return final_bboxes, final_scores, final_params
