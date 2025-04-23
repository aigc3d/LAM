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

import os
import cv2
import sys
import base64
import subprocess

import gradio as gr
import numpy as np
from PIL import Image
import argparse
from omegaconf import OmegaConf

import torch
import zipfile
from glob import glob
import moviepy.editor as mpy
from lam.utils.ffmpeg_utils import images_to_video
from tools.flame_tracking_single_image import FlameTrackingSingleImage
# from gradio_gaussian_render import gaussian_render
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

try:
    import spaces
except:
    pass

from pathlib import Path


torch._dynamo.config.suppress_errors = True  # 禁用动态编译错误
torch._dynamo.config.disable = True
h5_rendering = False  # True


def launch_env_not_compile_with_cuda():
    os.system('pip install chumpy')
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


def doRender():
     print('do render')


def save_images2video(img_lst, v_pth, fps):
    from moviepy.editor import ImageSequenceClip
    # Ensure all images are in uint8 format
    images = [image.astype(np.uint8) for image in img_lst]
    
    # Create an ImageSequenceClip from the list of images
    clip = ImageSequenceClip(images, fps=fps)
    
    # Write the clip to a video file
    clip.write_videofile(v_pth, codec='libx264')
    
    print(f"Video saved successfully at {v_pth}")


def add_audio_to_video(video_path, out_path, audio_path):
    # Import necessary modules from moviepy
    from moviepy.editor import VideoFileClip, AudioFileClip

    # Load video file into VideoFileClip object
    video_clip = VideoFileClip(video_path)

    # Load audio file into AudioFileClip object
    audio_clip = AudioFileClip(audio_path)

    # Attach audio clip to video clip (replaces existing audio)
    video_clip_with_audio = video_clip.set_audio(audio_clip)

    # Export final video with audio using standard codecs
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')

    print(f"Audio added successfully at {out_path}")

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

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def create_zip_archive(output_zip='runtime_/h5_render_data.zip', base_vid="nice"):
    flame_params_pth = os.path.join("./assets/sample_motion/export", base_vid, "flame_params.json")
    file_lst = [
        'runtime_data/lbs_weight_20k.json', 'runtime_data/offset.ply', 'runtime_data/skin.glb',
        'runtime_data/vertex_order.json', 'runtime_data/bone_tree.json', 
        flame_params_pth
    ]
    try:
        # Create a new ZIP file in write mode
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            # List all files in the specified directory
            for file_path in file_lst:
                zipf.write(file_path, arcname=os.path.join("h5_render_data", os.path.basename(file_path)))
        print(f"Archive created successfully: {output_zip}")
    except Exception as e:
        print(f"An error occurred: {e}")


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
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        print("image_path:", image_path, "\n"+"mask_path:", mask_path)

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
        src = Path(image_path).parent.parent.name
        driven = Path(motion_seqs_dir).parent.name
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
        
        # save h5 rendering info
        if h5_rendering:
            h5_fd = "./runtime_data"
            lam.renderer.flame_model.save_h5_info(shape_param.unsqueeze(0).cuda(), fd=h5_fd)
            res['cano_gs_lst'][0].save_ply(os.path.join(h5_fd, "offset.ply"), rgb2sh=False, offset2xyz=True)
            cmd = "thirdparties/blender/blender --background --python 'tools/generateGLBWithBlender_v2.py'"
            os.system(cmd)
            create_zip_archive(output_zip='runtime_data/h5_render_data.zip', base_vid=base_vid)

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

        save_images2video(rgb, dump_video_path, render_fps)
        audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid+".wav")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)

        return dump_image_path, dump_video_path_wa

    with gr.Blocks(analytics_enabled=False) as demo:

        logo_url = './assets/images/logo.jpeg'
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
                        ['assets/sample_input/barbara.jpg'],
                        ['assets/sample_input/cluo.jpg'],
                        ['assets/sample_input/dufu.jpg'],
                        ['assets/sample_input/james.png'],
                        ['assets/sample_input/libai.jpg'],
                        ['assets/sample_input/messi.png'],
                        ['assets/sample_input/speed.jpg'],
                        ['assets/sample_input/status.png'],
                        ['assets/sample_input/zhouxingchi.jpg'],
                    ]
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image],
                        examples_per_page=20,
                    )

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='lam_input_video'):
                    with gr.TabItem('Input Video'):
                        with gr.Row():
                            video_input = gr.Video(label='Input Video',
                                                   height=480,
                                                   width=270,
                                                   interactive=False)

                examples = glob("./assets/sample_motion/export/*/*.mp4")
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

        if h5_rendering:
            gr.set_static_paths("runtime_data/")
            assetPrefix = 'gradio_api/file=runtime_data/'
            with gr.Row():
                gs = gaussian_render(width = 300, height = 400, assets = assetPrefix + 'h5_render_data.zip')
            with gr.Row():
                renderButton = gr.Button('H5 Rendering')
                renderButton.click(doRender, js='''() => window.start()''')

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
    from lam.models import ModelLAM
    from safetensors.torch import load_file

    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")
    print("="*100)
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
    print("="*100)
    return model


def launch_gradio_app():

    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME':
        './model_zoo/lam_models/releases/lam/lam-20k/step_045500/',
        'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    cfg, _ = parse_configs()
    lam = _build_model(cfg)
    lam.to('cuda')

    flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
                                             vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
                                             facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True)

    demo_lam(flametracking, lam, cfg)


if __name__ == '__main__':
    # launch_env_not_compile_with_cuda()
    launch_gradio_app()
