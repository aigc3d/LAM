# Copyright (c) 2024-2025, Yisheng He, Yuan Dong
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
import cv2
import base64
import subprocess

import gradio as gr
import numpy as np
from PIL import Image
import argparse
from omegaconf import OmegaConf

import torch
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
import moviepy.editor as mpy
from lam.utils.ffmpeg_utils import images_to_video
import sys
from flame_tracking_single_image import FlameTrackingSingleImage

try:
    import spaces
except:
    pass


def launch_pretrained():
    from huggingface_hub import snapshot_download, hf_hub_download
    hf_hub_download(repo_id='DyrusQZ/LHM_Runtime',
                    repo_type='model',
                    filename='assets.tar',
                    local_dir='./')
    os.system('tar -xvf assets.tar && rm assets.tar')
    hf_hub_download(repo_id='DyrusQZ/LHM_Runtime',
                    repo_type='model',
                    filename='LHM-0.5B.tar',
                    local_dir='./')
    os.system('tar -xvf LHM-0.5B.tar && rm LHM-0.5B.tar')
    hf_hub_download(repo_id='DyrusQZ/LHM_Runtime',
                    repo_type='model',
                    filename='LHM_prior_model.tar',
                    local_dir='./')
    os.system('tar -xvf LHM_prior_model.tar && rm LHM_prior_model.tar')


def launch_env_not_compile_with_cuda():
    os.system('pip install chumpy')
    os.system('pip uninstall -y basicsr')
    os.system('pip install git+https://github.com/hitsz-zuoqi/BasicSR/')
    os.system('pip install numpy==1.23.0')
    os.system(
        'pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html'
    )


def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error('No image selected or uploaded!')


def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir


def init_preprocessor():
    from lam.utils.preprocess import Preprocessor
    global preprocessor
    preprocessor = Preprocessor()


def preprocess_fn(image_in: np.ndarray, remove_bg: bool, recenter: bool,
                  working_dir):
    image_raw = os.path.join(working_dir.name, 'raw.png')
    with Image.fromarray(image_in) as img:
        img.save(image_raw)
    image_out = os.path.join(working_dir.name, 'rembg.png')
    success = preprocessor.preprocess(image_path=image_raw,
                                      save_path=image_out,
                                      rmbg=remove_bg,
                                      recenter=recenter)
    assert success, f'Failed under preprocess_fn!'
    return image_out


def get_image_base64(path):
    with open(path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'


def save_imgs_2_video(imgs, v_pth, fps):
    img_lst = [imgs[i] for i in range(imgs.shape[0])]
    # Convert the list of NumPy arrays to a list of ImageClip objects
    clips = [mpy.ImageClip(img).set_duration(0.1) for img in img_lst]  # 0.1 seconds per frame

    # Concatenate the ImageClips into a single VideoClip
    video = mpy.concatenate_videoclips(clips, method="compose")

    # Write the VideoClip to a file
    video.write_videofile(v_pth, fps=fps)  # setting fps to 10 as example


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def demo_lam(flametracking, lam, cfg):

    # @spaces.GPU(duration=80)
    def core_fn(image_path: str, video_params, working_dir):
        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)
        
        base_vid = os.path.basename(video_params).split(".")[0]
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        base_iid = os.path.basename(image_path).split('.')[0]
        image_path = os.path.join("./assets/sample_input", base_iid, "images/00000_00.png")

        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_image_path = os.path.join(working_dir.name, "output.png")

        # prepare dump paths
        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.split(".")[0]
        subdir_path = os.path.dirname(image_raw).replace(omit_prefix, "")
        subdir_path = (
            subdir_path[1:] if subdir_path.startswith("/") else subdir_path
        )
        print("subdir_path and uid:", subdir_path, uid)

        motion_seqs_dir = flame_params_dir

        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)

        print(image_raw, motion_seqs_dir, dump_image_dir, dump_video_path)

        dump_tmp_dir = dump_image_dir

        if os.path.exists(dump_video_path):
            return dump_image_path, dump_video_path

        motion_img_need_mask = cfg.get("motion_img_need_mask", False)  # False
        vis_motion = cfg.get("vis_motion", False)  # False

        # preprocess input image: segmentation, flame params estimation
        return_code = flametracking.preprocess(image_raw)
        assert (return_code == 0), "flametracking preprocess failed!"
        return_code = flametracking.optimize()
        assert (return_code == 0), "flametracking optimize failed!"
        return_code, output_dir = flametracking.export()
        assert (return_code == 0), "flametracking export failed!"

        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        print(image_path, mask_path)

        aspect_standard = 1.0/1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30
        # prepare reference image
        image, _, _, shape_param = preprocess_image(image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1., 
                                             max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                             render_tgt_size=source_size, multiply=14, need_mask=True, get_shape_param=True)

        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "output.png")
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # prepare motion seq
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
                                            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
                                            render_image_res=render_size,  multiply=16, 
                                            need_mask=motion_img_need_mask, vis_motion=vis_motion, 
                                            shape_param=shape_param, test_sample=False, cross_id=False, src_driven=src_driven)

        # start inference
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32
        print("start to inference...................")
        with torch.no_grad():
            # TODO check device and dtype
            res = lam.infer_single_view(image.unsqueeze(0).to(device, dtype), None, None, 
                                        render_c2ws=motion_seq["render_c2ws"].to(device),
                                        render_intrs=motion_seq["render_intrs"].to(device),
                                        render_bg_colors=motion_seq["render_bg_colors"].to(device),
                                        flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()})

        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask = res["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        if vis_motion:
            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :, :], 
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        save_imgs_2_video(rgb, dump_video_path, render_fps)
        # images_to_video(rgb, output_path=dump_video_path, fps=30, gradio_codec=False, verbose=True)

        return dump_image_path, dump_video_path

    with gr.Blocks(analytics_enabled=False) as demo:

        logo_url = './assets/images/logo.png'
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(f"""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1> <img src="{logo_base64}" style='height:35px; display:inline-block;'/> LAM: Large Avatar Model for One-shot Animatable Gaussian Head</h1>
            </div>
            </div>
            """)
        gr.HTML(
            """<p><h4 style="color: red;"> Notes: Inputing front-face images or face orientation close to the driven signal gets better results.</h4></p>"""
        )

        # DISPLAY
        with gr.Row():

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='lam_input_image'):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label='Input Image',
                                                   image_mode='RGB',
                                                   height=480,
                                                   width=270,
                                                   sources='upload',
                                                   type='filepath', # 'numpy',
                                                   elem_id='content_image')
                # EXAMPLES
                with gr.Row():
                    examples = [
                        ['assets/sample_input/2w01/images/2w01.png'],
                        ['assets/sample_input/2w02/images/2w02.png'],
                        ['assets/sample_input/2w03/images/2w03.png'],
                        ['assets/sample_input/2w04/images/2w04.png'],
                    ]
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image],
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Tabs(elem_id='lam_input_video'):
                    with gr.TabItem('Input Video'):
                        with gr.Row():
                            video_input = gr.Video(label='Input Video',
                                                   height=480,
                                                   width=270,
                                                   interactive=False)

                examples = [
                    './assets/sample_motion/export/clip1/clip1.mp4',
                    './assets/sample_motion/export/clip2/clip2.mp4',
                    './assets/sample_motion/export/clip3/clip3.mp4',
                ]

                gr.Examples(
                    examples=examples,
                    inputs=[video_input],
                    examples_per_page=20,
                )
            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='lam_processed_image'):
                    with gr.TabItem('Processed Image'):
                        with gr.Row():
                            processed_image = gr.Image(
                                label='Processed Image',
                                image_mode='RGBA',
                                type='filepath',
                                elem_id='processed_image',
                                height=480,
                                width=270,
                                interactive=False)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='lam_render_video'):
                    with gr.TabItem('Rendered Video'):
                        with gr.Row():
                            output_video = gr.Video(label='Rendered Video',
                                                    format='mp4',
                                                    height=480,
                                                    width=270,
                                                    autoplay=True)

        # SETTING
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                submit = gr.Button('Generate',
                                   elem_id='lam_generate',
                                   variant='primary')

        working_dir = gr.State()
        submit.click(
            fn=assert_input_image,
            inputs=[input_image],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=core_fn,
            inputs=[input_image, video_input,
                    working_dir],  # video_params refer to smpl dir
            outputs=[processed_image, output_video],
        )

        demo.queue()
        demo.launch()


def _build_model(cfg):
    from lam.models import model_dict
    from lam.utils.hf_hub import wrap_model_hub

    hf_model_cls = wrap_model_hub(model_dict["lam"])
    model = hf_model_cls.from_pretrained(cfg.model_name)

    return model

def launch_gradio_app():

    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME':
        './exps/releases/lam/lam-20k/step_045500/',
        'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    cfg, _ = parse_configs()
    lam = _build_model(cfg)
    lam.to('cuda')

    flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./pretrain_model/68_keypoints_model.pkl',
                                             vgghead_model_path='./pretrain_model/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./pretrain_model/matting/stylematte_synth.pt',
                                             facebox_model_path='./pretrain_model/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True)

    demo_lam(flametracking, lam, cfg)


if __name__ == '__main__':
    # launch_pretrained()
    # launch_env_not_compile_with_cuda()
    launch_gradio_app()
