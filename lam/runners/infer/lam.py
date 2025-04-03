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
import traceback
import time
import torch
import os
import argparse
import mcubes
import trimesh
import numpy as np
from PIL import Image
from glob import glob
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from accelerate.logging import get_logger

from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image


from .base_inferrer import Inferrer
from lam.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from lam.utils.logging import configure_logger
from lam.runners import REGISTRY_RUNNERS
from lam.utils.video import images_to_video
from lam.utils.hf_hub import wrap_model_hub
from lam.models.modeling_lam import ModelLAM
from safetensors.torch import load_file
import moviepy.editor as mpy
from tools.flame_tracking_single_image import FlameTrackingSingleImage


logger = get_logger(__name__)


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split('_')[-1])

        cfg.save_tmp_dump = os.path.join("exps", 'save_tmp', _relative_path)
        cfg.image_dump = os.path.join("exps", 'images', _relative_path)
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)
        
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, 'images'))
        cfg.setdefault('video_dump', os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join("dumps", cli_cfg.model_name, 'meshes'))
    
    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register('infer.lam')
class LAMInferrer(Inferrer):

    EXP_TYPE: str = 'lam'

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        """
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )
        """

        self.model: LAMInferrer = self._build_model(self.cfg).to(self.device)

        self.flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./pretrain_model/68_keypoints_model.pkl',
                                             vgghead_model_path='./pretrain_model/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./pretrain_model/matting/stylematte_synth.pt',
                                             facebox_model_path='./pretrain_model/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True,
                                             args = self.cfg)


    def _build_model(self, cfg):
        """
        from lam.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        """
        from lam.models import ModelLAM
        model = ModelLAM(**cfg.model)

        resume = os.path.join(cfg.model_name, "model.safetensors")
        print("==="*16*3)
        print("loading pretrained weight from:", resume)
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
                    print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
            else:
                print(f"WARN] unexpected param {k}: {v.shape}")
        print("finish loading pretrained weight from:", resume)
        print("==="*16*3)
        return model

    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):
        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, batch_size=N, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes

    def infer_video(self, planes: torch.Tensor, frame_size: int, render_size: int, render_views: int, render_fps: int, dump_video_path: str):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(n_views=render_views, batch_size=N, device=self.device)
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        render_bg_colors = torch.ones(N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32) * 0. # 1.

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i:i+frame_size],
                    anchors=render_anchors[:, i:i+frame_size],
                    resolutions=render_resolutions[:, i:i+frame_size],
                    bg_colors=render_bg_colors[:, i:i+frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        # dump
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        for k, v in frames.items():
            if k == 'images_rgb':
                images_to_video(
                    images=v[0],
                    output_path=dump_video_path,
                    fps=render_fps,
                    gradio_codec=self.cfg.app_enabled,
                )

    def infer_mesh(self, planes: torch.Tensor, mesh_size: int, mesh_thres: float, dump_mesh_path: str):
        grid_out = self.model.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_size,
        )
        
        vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
        vtx = vtx / (mesh_size - 1) * 2 - 1

        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
        vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # dump
        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)

    def add_audio_to_video(self, video_path, out_path, audio_path):
        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
        print(f"Audio added successfully at {out_path}")

    def save_imgs_2_video(self, img_lst, v_pth, fps):
        from moviepy.editor import ImageSequenceClip
        images = [image.astype(np.uint8) for image in img_lst]
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(v_pth, codec='libx264')
        print(f"Video saved successfully at {v_pth}")
    
    def infer_single(self, image_path: str,
                     motion_seqs_dir, 
                     motion_img_dir,
                     motion_video_read_fps,
                     export_video: bool, 
                     export_mesh: bool, 
                     dump_tmp_dir:str,  # require by extracting motion seq from video, to save some results
                     dump_image_dir:str,
                     dump_video_path: str, 
                     dump_mesh_path: str,
                     gaga_track_type: str):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        aspect_standard = 1.0/1.0
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False
        save_ply = self.cfg.get("save_ply", False)  # False
        save_img = self.cfg.get("save_img", False)  # False
        rendered_bg = 1.
        ref_bg = 1.
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        if ref_bg < 1.:
            if "VFHQ_TEST" in image_path:
                mask_path = image_path.replace("/VFHQ_TEST/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
            else:
                mask_path = image_path.replace("/vfhq_test_nooffset_export/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
        if not os.path.exists(mask_path):
            print("Warning: Mask path not exists:", mask_path)
            mask_path = None
        else:
            print("load mask from:", mask_path)

        image, _, _, shape_param = preprocess_image(image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=ref_bg, 
                                             max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                             render_tgt_size=source_size, multiply=14, need_mask=True, get_shape_param=True)
        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "refer_" + os.path.basename(image_path))
        vis_ref_img = (image[0].permute(1, 2 ,0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)
        # prepare motion seq
        test_sample=self.cfg.get("test_sample", False)
        # test_sample=True
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, motion_img_dir, save_root=dump_tmp_dir, fps=motion_video_read_fps,
                                            bg_color=rendered_bg, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
                                            render_image_res=render_size,  multiply=16, 
                                            need_mask=motion_img_need_mask, vis_motion=vis_motion, 
                                            shape_param=shape_param, test_sample=test_sample, cross_id=self.cfg.get("cross_id", False), src_driven=src_driven)

        # return

        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        start_time = time.time()
        device="cuda"
        dtype=torch.float32
        # dtype=torch.bfloat16
        self.model.to(dtype)
        print("start to inference...................")
        with torch.no_grad():
            # TODO check device and dtype
            res = self.model.infer_single_view(image.unsqueeze(0).to(device, dtype), None, None, 
                                               render_c2ws=motion_seq["render_c2ws"].to(device),
                                               render_intrs=motion_seq["render_intrs"].to(device),
                                               render_bg_colors=motion_seq["render_bg_colors"].to(device),
                                               flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()})

        print(f"time elapsed: {time.time() - start_time}")
        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        only_pred = rgb
        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)
            import cv2
            vis_ref_img = np.tile(cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            blend_ratio = 0.7
            blend_res = ((1 -  blend_ratio) * rgb + blend_ratio * motion_seq["vis_motion_render"]).astype(np.uint8)
            # rgb = np.concatenate([rgb, motion_seq["vis_motion_render"], blend_res, vis_ref_img], axis=2)
            rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)
            
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        # images_to_video(rgb, output_path=dump_video_path, fps=render_fps, gradio_codec=False, verbose=True)
        self.save_imgs_2_video(rgb, dump_video_path, render_fps)
        base_vid = motion_seqs_dir.strip('/').split('/')[-1]
        audio_path = os.path.join(motion_seqs_dir, base_vid+".wav")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        self.add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        if save_img and dump_image_dir is not None:
            for i in range(rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(only_pred[i]).save(save_file)
                if save_ply and dump_mesh_path is not None:
                    res["3dgs"][i][0][0].save_ply(os.path.join(dump_image_dir, f"{i:04d}.ply"))

            dump_cano_dir = "./exps/cano_gs/"
            if not os.path.exists(dump_cano_dir):
                os.system(f"mkdir -p {dump_cano_dir}")
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + ".ply")
            # res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=False)
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + "_gs_offset.ply")
            res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=False, offset2xyz=True)
            # res['cano_gs_lst'][0].save_ply("tmp.ply", rgb2sh=False, offset2xyz=True)

            def save_color_points(points, colors, sv_pth, sv_fd="debug_vis/dataloader/"):
                points = points.squeeze().detach().cpu().numpy()
                colors = colors.squeeze().detach().cpu().numpy()
                sv_pth = os.path.join(sv_fd, sv_pth)
                if not os.path.exists(sv_fd):
                    os.system(f"mkdir -p {sv_fd}")
                with open(sv_pth, 'w') as of:
                    for point, color in zip(points, colors):
                        print('v', point[0], point[1], point[2], color[0], color[1], color[2], file=of)
 
            # save canonical color point clouds
            save_color_points(res['cano_gs_lst'][0].xyz, res["cano_gs_lst"][0].shs[:, 0, :], "framework_img.obj", sv_fd=dump_cano_dir) 

            # Export the template mesh to an OBJ file
            import trimesh
            vtxs = res['cano_gs_lst'][0].xyz - res['cano_gs_lst'][0].offset
            vtxs = vtxs.detach().cpu().numpy() 
            faces = self.model.renderer.flame_model.faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vtxs, faces=faces)
            mesh.export(os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_shaped_mesh.obj'))

            # Export textured deformed mesh
            import lam.models.rendering.utils.mesh_utils as mesh_utils
            vtxs = res['cano_gs_lst'][0].xyz.detach().cpu()
            faces = self.model.renderer.flame_model.faces.detach().cpu()
            colors = res['cano_gs_lst'][0].shs.squeeze(1).detach().cpu()
            pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_textured_mesh.obj')
            print("Save textured mesh to:", pth)
            mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")

    def infer(self):
        image_paths = []
        # hard code
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths = [self.cfg.image_input]
        else:
            # ids = sorted(os.listdir(self.cfg.image_input))
            # image_paths = [os.path.join(self.cfg.image_input, e, "images/00000_00.png") for e in ids]
            image_paths = glob(os.path.join(self.cfg.image_input, "*.jpg"))
            omit_prefix = self.cfg.image_input

        gaga_track_type = ""

        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):
            try:

                # preprocess input image: segmentation, flame params estimation
                return_code = self.flametracking.preprocess(image_path)
                assert (return_code == 0), "flametracking preprocess failed!"
                return_code = self.flametracking.optimize()
                assert (return_code == 0), "flametracking optimize failed!"
                return_code, output_dir = self.flametracking.export()
                assert (return_code == 0), "flametracking export failed!"

                image_path = os.path.join(output_dir, "images/00000_00.png")
                # mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")

                motion_seqs_dir = self.cfg.motion_seqs_dir
                print("motion_seqs_dir:", motion_seqs_dir)
                # prepare dump paths
                image_name = os.path.basename(image_path)
                uid = image_name.split('.')[0]
                subdir_path = os.path.dirname(image_path).replace(omit_prefix, '')
                subdir_path = subdir_path[1:] if subdir_path.startswith('/') else subdir_path
                # hard code
                subdir_path = gaga_track_type
                uid = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                print("subdir_path and uid:", subdir_path, uid)
                dump_video_path = os.path.join(
                    self.cfg.video_dump,
                    subdir_path,
                    f'{uid}.mp4',
                )
                dump_image_dir = os.path.join(
                    self.cfg.image_dump,
                    subdir_path,
                    f'{uid}'
                )
                dump_tmp_dir = os.path.join(
                    self.cfg.image_dump,
                    subdir_path,
                    "tmp_res"
                )
                dump_mesh_path = os.path.join(
                    self.cfg.mesh_dump,
                    subdir_path,
                    # f'{uid}.ply',
                )
                os.makedirs(dump_image_dir, exist_ok=True)
                os.makedirs(dump_tmp_dir, exist_ok=True)
                os.makedirs(dump_mesh_path, exist_ok=True)

                # if os.path.exists(dump_video_path):
                #     print(f"skip:{image_path}")
                #     continue

                self.infer_single(
                    image_path,
                    motion_seqs_dir=motion_seqs_dir,
                    motion_img_dir=self.cfg.motion_img_dir,
                    motion_video_read_fps=self.cfg.motion_video_read_fps,
                    export_video=self.cfg.export_video,
                    export_mesh=self.cfg.export_mesh, 
                    dump_tmp_dir=dump_tmp_dir,
                    dump_image_dir=dump_image_dir,
                    dump_video_path=dump_video_path, 
                    dump_mesh_path=dump_mesh_path,
                    gaga_track_type=gaga_track_type
                    )
            except:
                traceback.print_exc()
