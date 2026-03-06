# Copyright (c) 2023-2024, Qi Zuo
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
os.system('rm -rf /data-nvme/zerogpu-offload/')
os.system('pip install numpy==1.23.0')
os.system('pip install ./wheels/pytorch3d-0.7.3-cp310-cp310-linux_x86_64.whl')

import argparse
import base64
import time

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import gradio as gr
import spaces
from flame_tracking_single_image import FlameTrackingSingleImage
from ffmpeg_utils import images_to_video

# torch._dynamo.config.disable = True


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
            os.path.basename(cli_cfg.model_name).split('_')[-1],
        )

        cfg.save_tmp_dump = os.path.join('exps', 'save_tmp', _relative_path)
        cfg.image_dump = os.path.join('exps', 'images', _relative_path)
        cfg.video_dump = os.path.join('exps', 'videos',
                                      _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault('save_tmp_dump',
                       os.path.join('exps', cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault('image_dump',
                       os.path.join('exps', cli_cfg.model_name, 'images'))
        cfg.setdefault('video_dump',
                       os.path.join('dumps', cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump',
                       os.path.join('dumps', cli_cfg.model_name, 'meshes'))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault('logger', 'INFO')

    assert cfg.model_name is not None, 'model_name is required'

    return cfg, cfg_train



def launch_pretrained():
    from huggingface_hub import snapshot_download, hf_hub_download
    hf_hub_download(repo_id='yuandong513/flametracking_model',
                    repo_type='model',
                    filename='pretrain_model.tar',
                    local_dir='./')
    os.system('tar -xf pretrain_model.tar && rm pretrain_model.tar')

def animation_infer(renderer, gs_model_list, query_points, smplx_params,
                    render_c2ws, render_intrs, render_bg_colors):
    '''Inference code avoid repeat forward.
    '''
    render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
        render_intrs[0, 0, 0, 2] * 2)
    # render target views
    render_res_list = []
    num_views = render_c2ws.shape[1]
    start_time = time.time()

    # render target views
    render_res_list = []

    for view_idx in range(num_views):
        render_res = renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            renderer.get_single_view_smpl_data(smplx_params, view_idx),
            render_c2ws[:, view_idx:view_idx + 1],
            render_intrs[:, view_idx:view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx:view_idx + 1],
        )
        render_res_list.append(render_res)
    print(
        f'time elpased(animate gs model per frame):{(time.time() -  start_time)/num_views}'
    )

    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            if isinstance(v[0], torch.Tensor):
                out[k].append(v.detach().cpu())
            else:
                out[k].append(v)
    for k, v in out.items():
        # print(f"out key:{k}")
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in ['comp_rgb', 'comp_mask', 'comp_depth']:
                out[k] = out[k][0].permute(
                    0, 2, 3,
                    1)  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
        else:
            out[k] = v
    return out


def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error('No image selected or uploaded!')


def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def get_image_base64(path):
    with open(path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'


def demo_lhm(flametracking):
    @spaces.GPU(duration=80)
    def core_fn(image: str, video_params, working_dir):
        image_raw = os.path.join(working_dir.name, 'raw.png')
        with Image.fromarray(image) as img:
            img.save(image_raw)

        base_vid = os.path.basename(video_params).split('_')[0]

        dump_video_path = os.path.join(working_dir.name, 'output.mp4')
        dump_image_path = os.path.join(working_dir.name, 'output.png')

        # prepare dump paths
        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.split('.')[0]
        subdir_path = os.path.dirname(image_raw).replace(omit_prefix, '')
        subdir_path = (subdir_path[1:]
                       if subdir_path.startswith('/') else subdir_path)
        print('==> subdir_path and uid:', subdir_path, uid)

        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)

        print('==> path:', image_raw, dump_image_dir, dump_video_path)

        dump_tmp_dir = dump_image_dir

        return_code = flametracking.preprocess(image_raw)
        return_code = flametracking.optimize()
        return_code, output_dir = flametracking.export()

        print("==> output_dir:", output_dir)


        save_ref_img_path = os.path.join(dump_tmp_dir, 'output.png')
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() *
                       255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # rendering !!!!
        start_time = time.time()
        batch_dict = dict()

        rgb = cv2.imread(os.path.join(output_dir,'images/00000_00.png'))

        for i in range(30):
            images_to_video(
                rgb,
                output_path=dump_video_path,
                fps=30,
                gradio_codec=False,
                verbose=True,
            )

        return dump_image_path, dump_video_path

    _TITLE = '''LHM: Large Animatable Human Model'''

    _DESCRIPTION = '''
        <strong>Reconstruct a human avatar in 0.2 seconds with A100!</strong>
    '''

    with gr.Blocks(analytics_enabled=False, delete_cache=[3600, 3600]) as demo:

        # </div>
        logo_url = './asset/logo.jpeg'
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(f"""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1> <img src="{logo_base64}" style='height:35px; display:inline-block;'/> Large Animatable Human Model </h1>
            </div>
            </div>
            """)

        gr.HTML("""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; margin: 20px; gap: 10px;">
                <a class="flex-item" href="https://arxiv.org/abs/2503.10625" target="_blank">
                    <img src="https://img.shields.io/badge/Paper-arXiv-darkred.svg" alt="arXiv Paper">
                </a>
                <a class="flex-item" href="https://lingtengqiu.github.io/LHM/" target="_blank">
                    <img src="https://img.shields.io/badge/Project-LHM-blue" alt="Project Page">
                </a>
                <a class="flex-item" href="https://github.com/aigc3d/LHM" target="_blank">
                    <img src="https://img.shields.io/github/stars/aigc3d/LHM?label=Github%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                </a>
                <a class="flex-item" href="https://www.youtube.com/watch?v=tivEpz_yiEo" target="_blank">
                    <img src="https://img.shields.io/badge/Youtube-Video-red.svg" alt="Video">
                </a>
            </div>
            """)

        gr.HTML(
            """<p><h4 style="color: red;"> Notes: Please input full-body image in case of detection errors. We simplify the pipeline in spaces: 1) using Rembg instead of SAM2; 2) limit the output video length to 10s; For best visual quality, try the inference code on Github instead.</h4></p>"""
        )

        # DISPLAY
        with gr.Row():

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='openlrm_input_image'):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label='Input Image',
                                                   image_mode='RGB',
                                                   height=480,
                                                   width=270,
                                                   sources='upload',
                                                   type='numpy',
                                                   elem_id='content_image')
                # EXAMPLES
                with gr.Row():
                    examples = [
                        ['asset/sample_input/00000.png'],
                    ]
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image],
                        examples_per_page=10,
                    )

            with gr.Column():
                with gr.Tabs(elem_id='openlrm_input_video'):
                    with gr.TabItem('Input Video'):
                        with gr.Row():
                            video_input = gr.Video(label='Input Video',
                                                   height=480,
                                                   width=270,
                                                   interactive=False)

                examples = [
                    './asset/sample_input/demo.mp4',
                ]

                gr.Examples(
                    examples=examples,
                    inputs=[video_input],
                    examples_per_page=20,
                )
            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='openlrm_processed_image'):
                    with gr.TabItem('Processed Image'):
                        with gr.Row():
                            processed_image = gr.Image(
                                label='Processed Image',
                                image_mode='RGB',
                                type='filepath',
                                elem_id='processed_image',
                                height=480,
                                width=270,
                                interactive=False)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='openlrm_render_video'):
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
                                   elem_id='openlrm_generate',
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

        demo.queue(max_size=1)
        demo.launch()


def launch_gradio_app():

    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME':
        './exps/releases/video_human_benchmark/human-lrm-500M/step_060000/',
        'APP_INFER': './configs/inference/human-lrm-500M.yaml',
        'APP_TYPE': 'infer.human_lrm',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./pretrain_model/68_keypoints_model.pkl',
                                             vgghead_model_path='./pretrain_model/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./pretrain_model/matting/stylematte_synth.pt',
                                             facebox_model_path='./pretrain_model/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True)


    demo_lhm(flametracking)


if __name__ == '__main__':
    launch_pretrained()
    launch_gradio_app()

