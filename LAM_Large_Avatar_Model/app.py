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

os.system("rm -rf /data-nvme/zerogpu-offload/")
print("Blender file exist {}".format(os.path.exists('./blender-4.0.2-linux-x64.tar.xz')))
os.system('tar -xf ./blender-4.0.2-linux-x64.tar.xz')
os.system('chmod +x ./blender-4.0.2-linux-x64/blender')
os.system("pip install patool")
os.system("pip uninstall -y xformers")
os.system("pip install chumpy")
# os.system("pip uninstall -y basicsr")
os.system("pip install Cython")
os.system("pip install ./wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl --force-reinstall")
os.system("pip install ./wheels/simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl --force-reinstall")
# os.system("pip install ./wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl --force-reinstall")
# os.system("pip install nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling --force-reinstall")
os.system("pip install ./external/nvdiffrast/")
os.system("pip install iopath")
# os.system("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html --force-reinstall")
os.system("pip install ./wheels/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl --force-reinstall")
# os.system("pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121")
os.system("pip install numpy==1.23.0")
os.system("pip install oss2")

print("Run install FBX SDK ..............................3")
os.system('pip install ./wheels/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl')

print("Install FBX SDK Finished..............................3")

# import sys
# sys.path.insert(0, os.path.abspath('tools'))
# sys.path.insert(0, os.path.abspath('./'))

import oss2
import cv2
import base64
import subprocess
from datetime import datetime
import argparse
from glob import glob
import gradio as gr
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import moviepy.editor as mpy
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
from lam.utils.ffmpeg_utils import images_to_video

# import spaces


# def conver_oac_file():
#     print("Conver oac file ......")
#     from fbx_tools.generateARKITGLBWithBlender import convert_ascii_to_binary
#     from pathlib import Path
#     temp_files = {"ascii":Path('./assets/sampe_oac/template_file.fbx'),
#                   "binary":Path('./assets/sampe_oac/template_file_binary.fbx')}
#     convert_ascii_to_binary(temp_files["ascii"], temp_files["binary"])
#     return temp_files["binary"]

def compile_module(subfolder, script):
    try:
        # Save the current working directory
        current_dir = os.getcwd()
        # Change directory to the subfolder
        os.chdir(os.path.join(current_dir, subfolder))
        # Run the compilation command
        result = subprocess.run(
            ["sh", script],
            capture_output=True,
            text=True,
            check=True
        )
        # Print the compilation output
        print("Compilation output:", result.stdout)

    except Exception as e:
        # Print any error that occurred
        print(f"An error occurred: {e}")
    finally:
        # Ensure returning to the original directory
        os.chdir(current_dir)
        print("Returned to the original directory.")


# compile flame_tracking dependence submodule
compile_module("external/landmark_detection/FaceBoxesV2/utils/", "make.sh")
from flame_tracking_single_image import FlameTrackingSingleImage

def upload2oss(enable_oac_file, filepath):

    if(enable_oac_file):

        print("Uploading {} ... to {} ...".format(filepath,os.path.join('virutalbuy-public','share/aigc3d/LAM_Chatting_Avatar')))
        access_key_id = os.getenv('key_id')
        access_key_secret = os.getenv('key_secret')

        endpoint = 'http://oss-cn-hangzhou.aliyuncs.com'
        bucket_name = 'virutalbuy-public'

        object_name = os.path.join('share/aigc3d/LAM_Chatting_Avatar',filepath.split('/')[-1])
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        try:
            result = bucket.put_object_from_file(object_name, filepath)
            print("Upload Successful. HTTP Status Code:", result.status)
        except oss2.exceptions as e:
            print("Upload failed:", str(e))
    else:
        pass


def launch_pretrained():
    from huggingface_hub import snapshot_download, hf_hub_download
    # launch pretrained for flame tracking.
    hf_hub_download(repo_id='yuandong513/flametracking_model',
                    repo_type='model',
                    filename='pretrain_model.tar',
                    local_dir='./')
    os.system('tar -xf pretrain_model.tar && rm pretrain_model.tar')
    # launch human model files
    hf_hub_download(repo_id='3DAIGC/LAM-assets',
                    repo_type='model',
                    filename='LAM_human_model.tar',
                    local_dir='./')
    os.system('tar -xf LAM_human_model.tar && rm LAM_human_model.tar')
    # launch pretrained for LAM
    model_dir = hf_hub_download(repo_id="3DAIGC/LAM-20K", repo_type="model", local_dir="./exps/releases/lam/lam-20k/step_045500/", filename="config.json")
    print(model_dir)
    model_dir = hf_hub_download(repo_id="3DAIGC/LAM-20K", repo_type="model", local_dir="./exps/releases/lam/lam-20k/step_045500/", filename="model.safetensors")
    print(model_dir)
    model_dir = hf_hub_download(repo_id="3DAIGC/LAM-20K", repo_type="model", local_dir="./exps/releases/lam/lam-20k/step_045500/", filename="README.md")
    print(model_dir)
    # launch example for LAM
    hf_hub_download(repo_id='3DAIGC/LAM-assets',
                    repo_type='model',
                    filename='LAM_assets.tar',
                    local_dir='./')
    os.system('tar -xf LAM_assets.tar && rm LAM_assets.tar')
    hf_hub_download(repo_id='3DAIGC/LAM-assets',
                    repo_type='model',
                    filename='config.json',
                    local_dir='./tmp/')


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


def save_imgs_2_video(imgs, v_pth, fps=30):                                                                                                                                               
    # moviepy example                                                                                                                                                                        
    from moviepy.editor import ImageSequenceClip, VideoFileClip                                                                                                                              
    images = [image.astype(np.uint8) for image in imgs]                                                                                                                                      
    clip = ImageSequenceClip(images, fps=fps)                                                                                                                                                
    # final_duration = len(images) / fps                                                                                                                                                     
    # clip = clip.subclip(0, final_duration)                                                                                                                                                 
    clip = clip.subclip(0, len(images) / fps)                                                                                                                                                
    clip.write_videofile(v_pth, codec='libx264')                                            
                                                                                                                                                                                             
    import cv2                                                                                                                                                                               
    cap = cv2.VideoCapture(v_pth)                                                           
    nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)                                                  
    if nf != len(images):                                                                                                                                                                    
        print("="*100+f"\n{v_pth} moviepy saved video frame error."+"\n"+"="*100)                                                                                                            
    print(f"Video saved successfully at {v_pth}")   
    

def add_audio_to_video(video_path, out_path, audio_path, fps=30):
    # Import necessary modules from moviepy
    from moviepy.editor import VideoFileClip, AudioFileClip

    # Load video file into VideoFileClip object
    video_clip = VideoFileClip(video_path)

    # Load audio file into AudioFileClip object
    audio_clip = AudioFileClip(audio_path)

    # Hard code clip audio
    if audio_clip.duration > 10:
        audio_clip = audio_clip.subclip(0, 10)

    # Attach audio clip to video clip (replaces existing audio)
    video_clip_with_audio = video_clip.set_audio(audio_clip)

    # Export final video with audio using standard codecs
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac', fps=fps)

    print(f"Audio added successfully at {out_path}")


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    parser.add_argument("--blender_path", type=str,
                        default='./blender-4.0.2-linux-x64/blender' ,
                        help="Path to Blender executable")

    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg.blender_path = args.blender_path
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


def demo_lam(flametracking, lam, cfg):
    # @spaces.GPU(duration=80)
    def core_fn(image_path: str, video_params, working_dir, enable_oac_file):
        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)

        base_vid = os.path.basename(video_params).split(".")[0]
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        base_iid = 'chatting_avatar_'+datetime.now().strftime("%Y%m%d%H%M%S")

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
        # """
        return_code = flametracking.preprocess(image_raw)
        assert (return_code == 0), "flametracking preprocess failed!"
        return_code = flametracking.optimize()
        assert (return_code == 0), "flametracking optimize failed!"
        return_code, output_dir = flametracking.export()
        assert (return_code == 0), "flametracking export failed!"
        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        print("image_path:", image_path, "\n" + "mask_path:", mask_path)

        aspect_standard = 1.0 / 1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30
        # prepare reference image
        image, _, _, shape_param = preprocess_image(image_path, mask_path=mask_path, intr=None, pad_ratio=0,
                                                    bg_color=1.,
                                                    max_tgt_size=None, aspect_standard=aspect_standard,
                                                    enlarge_ratio=[1.0, 1.0],
                                                    render_tgt_size=source_size, multiply=14, need_mask=True,
                                                    get_shape_param=True)

        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "output.png")
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # prepare motion seq
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
                                         bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1, 0],
                                         render_image_res=render_size, multiply=16,
                                         need_mask=motion_img_need_mask, vis_motion=vis_motion,
                                         shape_param=shape_param, test_sample=False, cross_id=False,
                                         src_driven=src_driven, max_squen_length=300)

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
                                        flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()})
        output_zip_path = ''
        download_command = ''
        # save h5 rendering info
        if enable_oac_file:
            try:
                from generateARKITGLBWithBlender import generate_glb
                from pathlib import Path
                import shutil
                import patoolib

                oac_dir = os.path.join('./', base_iid)
                saved_head_path = lam.renderer.flame_model.save_shaped_mesh(shape_param.unsqueeze(0).cuda(), fd=oac_dir)
                res['cano_gs_lst'][0].save_ply(os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True)
                generate_glb(
                    input_mesh=Path(saved_head_path),
                    template_fbx=Path("./assets/sample_oac/template_file.fbx"),
                    output_glb=Path(os.path.join(oac_dir, "skin.glb")),
                    blender_exec=Path(cfg.blender_path)
                )
                shutil.copy(
                    src='./assets/sample_oac/animation.glb',
                    dst=os.path.join(oac_dir, 'animation.glb')
                )
                os.remove(saved_head_path)

                output_zip_path = os.path.join('./', base_iid + '.zip')
                if os.path.exists(output_zip_path):
                    os.remove(output_zip_path)
                os.system('zip -r {} {}'.format(output_zip_path,oac_dir))
                # original_cwd = os.getcwd()
                # oac_parent_dir = os.path.dirname(oac_dir)
                # base_iid_dir = os.path.basename(oac_dir)
                # os.chdir(oac_parent_dir)
                # try:
                #     patoolib.create_archive(
                #         archive=os.path.abspath(output_zip_path),
                #         filenames=[base_iid_dir],
                #         verbosity=-1,
                #         program='zip'
                #     )
                # finally:
                #     os.chdir(original_cwd)
                shutil.rmtree(oac_dir)
                download_command = 'wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/LAM_Chatting_Avatar/' + \
                                   output_zip_path.split('/')[-1]

            except Exception as e:
                output_zip_path = f"Archive creation failed: {str(e)}"

        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask = res["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        if vis_motion:
            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :,
                :],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        print("==="*36, "\nrgb length:", rgb.shape, render_fps, "==="*36)
        save_imgs_2_video(rgb, dump_video_path, render_fps)
        # images_to_video(rgb, output_path=dump_video_path, fps=30, gradio_codec=False, verbose=True)
        audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid + ".wav")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)


        return dump_image_path, dump_video_path_wa, output_zip_path, download_command

    def core_fn_space(image_path: str, video_params, working_dir):
        return core_fn(image_path, video_params, working_dir)

    with gr.Blocks(analytics_enabled=False) as demo:

        logo_url = './assets/images/logo.jpeg'
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(f"""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1> <img src="{logo_base64}" style='height:35px; display:inline-block;'/>  Large Avatar Model for One-shot Animatable Gaussian Head</h1>
            </div>
            </div>
            """)
        
        gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; margin: 20px; gap: 10px;">
                <a class="flex-item" href="https://arxiv.org/abs/2502.17796" target="_blank">
                    <img src="https://img.shields.io/badge/Paper-arXiv-darkred.svg" alt="arXiv Paper">
                </a>                      
                <a class="flex-item" href="https://aigc3d.github.io/projects/LAM/" target="_blank">
                    <img src="https://img.shields.io/badge/Project-LAM-blue" alt="Project Page">
                </a>
                <a class="flex-item" href="https://github.com/aigc3d/LAM" target="_blank">
                    <img src="https://img.shields.io/github/stars/aigc3d/LAM?label=Github%20★&logo=github&color=C8C" alt="badge-github-stars">
                </a>
                <a class="flex-item" href="https://youtu.be/FrfE3RYSKhk" target="_blank">
                    <img src="https://img.shields.io/badge/Youtube-Video-red.svg" alt="Video">
                </a>
            </div>
            """
        )

        
        gr.HTML("""<div style="margin-top: -10px">
            <p style="margin: 4px 0; line-height: 1.2"><h4 style="color: black; margin: 2px 0">Notes1: Inputing front-face images or face orientation close to the driven signal gets better results.</h4></p>
            <p style="margin: 4px 0; line-height: 1.2"><h4 style="color: black; margin: 2px 0">Notes2: Using LAM-20K model (lower quality than premium LAM-80K) to mitigate processing latency.</h4></p>
        </div>""")




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
                                                   type='filepath',  # 'numpy',
                                                   elem_id='content_image')
                # EXAMPLES
                with gr.Row():
                    examples = [
                        ['assets/sample_input/messi.png'],
                        ['assets/sample_input/status.png'],
                        ['assets/sample_input/james.png'],
                        ['assets/sample_input/cluo.jpg'],
                        ['assets/sample_input/dufu.jpg'],
                        ['assets/sample_input/libai.jpg'],
                        ['assets/sample_input/barbara.jpg'],
                        ['assets/sample_input/pop.png'],
                        ['assets/sample_input/musk.jpg'],
                        ['assets/sample_input/speed.jpg'],
                        ['assets/sample_input/zhouxingchi.jpg'],
                    ]
                gr.Examples(
                    examples=examples,
                    inputs=[input_image],
                    examples_per_page=20
                )


            with gr.Column():
                with gr.Tabs(elem_id='lam_input_video'):
                    with gr.TabItem('Input Video'):
                        with gr.Row():
                            video_input = gr.Video(label='Input Video',
                                                   height=480,
                                                   width=270,
                                                   interactive=False)

                examples = ['./assets/sample_motion/export/Speeding_Scandal/Speeding_Scandal.mp4', 
                            './assets/sample_motion/export/Look_In_My_Eyes/Look_In_My_Eyes.mp4', 
                            './assets/sample_motion/export/D_ANgelo_Dinero/D_ANgelo_Dinero.mp4', 
                            './assets/sample_motion/export/Michael_Wayne_Rosen/Michael_Wayne_Rosen.mp4', 
                            './assets/sample_motion/export/I_Am_Iron_Man/I_Am_Iron_Man.mp4', 
                            './assets/sample_motion/export/Anti_Drugs/Anti_Drugs.mp4', 
                            './assets/sample_motion/export/Pen_Pineapple_Apple_Pen/Pen_Pineapple_Apple_Pen.mp4', 
                            './assets/sample_motion/export/Taylor_Swift/Taylor_Swift.mp4', 
                            './assets/sample_motion/export/GEM/GEM.mp4', 
                             './assets/sample_motion/export/The_Shawshank_Redemption/The_Shawshank_Redemption.mp4'
                            ]
                print("Video example list {}".format(examples))

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
                enable_oac_file = gr.Checkbox(label="Export ZIP file for Chatting Avatar",
                                              value=False,
                                              visible=os.path.exists(cfg.blender_path))
                submit = gr.Button('Generate',
                                   elem_id='lam_generate',
                                   variant='primary')
                download_command = gr.Textbox(
                    label="Download ZIP file for Chatting Avatar",
                    interactive=False,
                    placeholder="Download ZIP file for Chatting Avatar ...",
                    visible=os.path.exists(cfg.blender_path)
                )

        output_zip_textbox = gr.Textbox(visible=False)
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
                    working_dir, enable_oac_file],  # video_params refer to smpl dir
            outputs=[processed_image, output_video, output_zip_textbox, download_command],
        ).success(
            fn=upload2oss,
            inputs=[enable_oac_file,output_zip_textbox]
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
        'NUMBA_THREADING_LAYER': 'forseq',
    })

    cfg, _ = parse_configs()
    lam = _build_model(cfg)
    lam.to('cuda')

    flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./pretrained_models/68_keypoints_model.pkl',
                                             vgghead_model_path='./pretrained_models/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./pretrained_models/matting/stylematte_synth.pt',
                                             facebox_model_path='./pretrained_models/FaceBoxesV2.pth',
                                             detect_iris_landmarks=False)

    demo_lam(flametracking, lam, cfg)


if __name__ == '__main__':
    # launch_pretrained()
    launch_gradio_app()
