import os
import sys
import shutil
import modal
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import argparse
from omegaconf import OmegaConf

# --- Modal インフラ設定 ---

app = modal.App("lam-concierge")
storage_vol = modal.Volume.from_name("lam-storage", create_if_missing=True)
output_vol = modal.Volume.from_name("concierge-output", create_if_missing=True)

# 指示書記載の Image Build 定義をそのまま流用
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "tree",
        "libusb-1.0-0", "build-essential",
        "gcc", "g++", "ninja-build",
        "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libxkbcommon0", "libsm6",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy==1.26.4'",
    )
    .run_commands(
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        "pip install xformers==0.0.27.post2 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CC": "gcc",
        "CXX": "g++",
        "CXXFLAGS": "-std=c++17",
        "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
        "TORCHDYNAMO_DISABLE": "1",
    })
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "CHUMPY_INIT=$(python -c \"import importlib.util; print(importlib.util.find_spec('chumpy').origin)\") && "
        "sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/"
        "from numpy import nan, inf; import numpy; bool = numpy.bool_; int = numpy.int_; "
        "float = numpy.float64; complex = numpy.complex128; object = numpy.object_; "
        "unicode = numpy.str_; str = numpy.str_/' "
        "\"$CHUMPY_INIT\" && "
        "find $(dirname \"$CHUMPY_INIT\") -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true",
        # pytorch3d: ローカル wheels/ から後段でインストールする（URL不要）
    )
    .pip_install(
        "gradio==4.44.0", "gradio_client==1.3.0", "fastapi",
        "omegaconf==2.3.0", "pandas", "scipy<1.14.0",
        "opencv-python-headless==4.9.0.80", "imageio[ffmpeg]",
        "moviepy==1.0.3", "rembg[gpu]", "scikit-image", "pillow",
        "huggingface_hub>=0.24.0", "filelock", "typeguard",
        "transformers==4.44.2", "diffusers==0.30.3", "accelerate==0.34.2",
        "tyro==0.8.0", "mediapipe==0.10.21", "tensorboard", "rich",
        "loguru", "Cython", "PyMCubes", "trimesh", "einops", "plyfile",
        "jaxtyping", "ninja", "patool", "safetensors", "decord",
        "numpy==1.26.4",
    )
    .run_commands(
        "pip install onnxruntime-gpu==1.18.1 "
        "--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
    )
    # NOTE: pytorch3d, diff_gaussian_rasterization, simple_knn, fbx は
    # 全てローカル wheels/ から後段でインストールする。
    # URL ダウンロードや GitHub ソースビルドは使用しない。
    .run_commands(
        "wget -q https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender",
        "tar xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1",
        "ln -sf /opt/blender/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
    .run_commands(
        "git clone https://github.com/aigc3d/LAM.git /root/LAM",
        "sed -i 's/dtype=np\\.int)/dtype=np.intp)/' "
        "/root/LAM/external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.pyx",
        "cd /root/LAM/external/landmark_detection/FaceBoxesV2/utils/nms && "
        "python -c \""
        "from setuptools import setup, Extension; "
        "from Cython.Build import cythonize; "
        "import numpy; "
        "setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), "
        "include_dirs=[numpy.get_include()])\" "
        "build_ext --inplace",
    )
    .run_commands(
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/models/modeling_lam.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/models/encoders/dinov2_fusion_wrapper.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/losses/tvloss.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/losses/pixelwise.py",
    )
    .run_commands(
        "python -c \""
        "import torch; "
        "url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'; "
        "torch.hub.load_state_dict_from_url(url, map_location='cpu'); "
        "print('DINOv2 cached OK')\"",
    )
)

def _precompile_nvdiffrast():
    import torch
    import nvdiffrast.torch as dr
    print("nvdiffrast pre-compiled OK")

image = image.run_function(_precompile_nvdiffrast)

# ============================================================
# ModelScope Official Wheels (REQUIRED)
# ============================================================
# The ./wheels/ directory MUST contain the official ModelScope pre-built wheels:
#   - pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
#   - diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
#   - simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl
#   - fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl
# These are the ONLY source for these packages. No URL fallback.
_APP_DIR = Path(__file__).resolve().parent
_wheels_dir = _APP_DIR / "wheels"
_whl_files = list(_wheels_dir.glob("*.whl")) if _wheels_dir.is_dir() else []
if not _whl_files:
    raise RuntimeError(
        f"[ABORT] No .whl files found in {_wheels_dir}/. "
        "You must place the official ModelScope wheels "
        "(pytorch3d, diff_gaussian_rasterization, simple_knn, fbx) "
        "in the wheels/ directory before building. "
        "See README or handoff doc for download instructions."
    )
print(f"[WHEELS] Found {len(_whl_files)} wheels in {_wheels_dir}:")
for w in _whl_files:
    print(f"  - {w.name}")

image = (
    image
    .add_local_dir(str(_wheels_dir), remote_path="/tmp/modelscope_wheels")
    .run_commands(
        "echo '[WHEELS] Installing ModelScope official wheels...'",
        "for whl in /tmp/modelscope_wheels/*.whl; do "
        "  [ -f \"$whl\" ] && pip install \"$whl\" --force-reinstall --no-deps && "
        "  echo \"  Installed: $(basename $whl)\"; "
        "done",
        "echo '[WHEELS] Done.'",
    )
)

# --- 写経セクション: app.py ヘルパー関数 ---

def save_imgs_2_video(imgs, v_pth, fps=30):
    from moviepy.editor import ImageSequenceClip
    images = [image.astype(np.uint8) for image in imgs]
    clip = ImageSequenceClip(images, fps=fps)
    clip = clip.subclip(0, len(images) / fps)
    clip.write_videofile(v_pth, codec='libx264')
    print(f"Video saved successfully at {v_pth}")

def add_audio_to_video(video_path, out_path, audio_path, fps=30):
    from moviepy.editor import VideoFileClip, AudioFileClip
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    if audio_clip.duration > 10:
        audio_clip = audio_clip.subclip(0, 10)
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac', fps=fps)
    print(f"Audio added successfully at {out_path}")

# --- Modal用パスセットアップ ---

def _setup_model_paths():
    """Volume上のディレクトリを/root/LAM配下にリンク"""
    import shutil
    lam_root = "/root/LAM"
    vol_lam = "/vol/lam-storage/LAM"
    for subdir in ["model_zoo", "assets", "pretrained_models"]:
        src = os.path.join(vol_lam, subdir)
        dst = os.path.join(lam_root, subdir)
        if not os.path.isdir(src): continue
        if os.path.islink(dst): os.unlink(dst)
        elif os.path.isdir(dst): shutil.rmtree(dst)
        os.symlink(src, dst)
    pretrained_hm = os.path.join(lam_root, "pretrained_models", "human_model_files")
    model_zoo_hpm = os.path.join(lam_root, "model_zoo", "human_parametric_models")
    if not os.path.exists(pretrained_hm) and os.path.isdir(model_zoo_hpm):
        os.makedirs(os.path.dirname(pretrained_hm), exist_ok=True)
        os.symlink(model_zoo_hpm, pretrained_hm)

# --- 写経セクション: コアロジック ---

def parse_configs():
    # app.py 248-306行の写経
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    parser.add_argument("--blender_path", type=str,
                        default='/usr/local/bin/blender', # [MODAL変更] Blender 4.2インストールパス
                        help="Path to Blender executable")
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg.blender_path = args.blender_path
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")
    args.config = args.infer if args.config is None else args.config
    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try: cfg.src_head_size = cfg_train.dataset.src_head_size
        except: cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split("_")[-1])
        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp"))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault("video_dump", os.path.join("dumps", cli_cfg.model_name, "videos"))
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))
    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")
    assert cfg.model_name is not None, "model_name is required"
    return cfg, cfg_train

def _build_model(cfg):
    # app.py 641-648行をそのままコピー
    from lam.models import model_dict
    from lam.utils.hf_hub import wrap_model_hub
    hf_model_cls = wrap_model_hub(model_dict["lam"])
    model = hf_model_cls.from_pretrained(cfg.model_name)
    return model

def _init_lam_pipeline():
    """app.pyのlaunch_gradio_app()を写経"""
    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")
    _setup_model_paths()
    # app.py 652-672行の写経
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME': './model_zoo/lam_models/releases/lam/lam-20k/step_045500/', # [MODAL変更] パス
        'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'forseq',
    })
    cfg, _ = parse_configs()
    lam = _build_model(cfg)
    lam.to('cuda')
    sys.path.insert(0, "/root/LAM/tools") # [MODAL変更] importパス解決
    from flame_tracking_single_image import FlameTrackingSingleImage
    flametracking = FlameTrackingSingleImage(
        output_dir='tracking_output',
        alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl', # [MODAL変更] パス
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd', # [MODAL変更] パス
        human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt', # [MODAL変更] パス
        facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth', # [MODAL変更] パス
        detect_iris_landmarks=False,
    )
    return cfg, lam, flametracking

# --- Modal Generator クラス ---

@app.cls(gpu="L4", image=image,
         volumes={"/vol/output": output_vol, "/vol/lam-storage": storage_vol},
         timeout=600)
class Generator:
    @modal.enter()
    def setup(self):
        self.cfg, self.lam, self.flametracking = _init_lam_pipeline()

    @modal.method()
    def generate(self, image_bytes, motion_name, enable_oac_file):
        import tempfile
        from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
        
        working_dir_path = "/tmp/work"
        os.makedirs(working_dir_path, exist_ok=True)
        
        image_raw = os.path.join(working_dir_path, "raw.png")
        with open(image_raw, "wb") as f:
            f.write(image_bytes)
        
        # core_fn() 316-471行の写経
        base_vid = motion_name
        flame_params_dir = os.path.join("./model_zoo/sample_motion/export", base_vid, "flame_param") # [MODAL変更] パス
        base_iid = 'chatting_avatar_'+datetime.now().strftime("%Y%m%d%H%M%S")
        
        dump_video_path = os.path.join(working_dir_path, "output.mp4")
        dump_image_path = os.path.join(working_dir_path, "output.png")
        
        motion_seqs_dir = flame_params_dir
        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)
        dump_tmp_dir = dump_image_dir
        
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)
        vis_motion = self.cfg.get("vis_motion", False)
        
        # Preprocess
        return_code = self.flametracking.preprocess(image_raw)
        assert (return_code == 0), "flametracking preprocess failed!"
        return_code = self.flametracking.optimize()
        assert (return_code == 0), "flametracking optimize failed!"
        return_code, output_dir = self.flametracking.export()
        assert (return_code == 0), "flametracking export failed!"
        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        
        aspect_standard = 1.0 / 1.0
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = 30
        
        # preprocess_image 写経
        image_tensor, _, _, shape_param = preprocess_image(
            image_path, mask_path=mask_path, intr=None, pad_ratio=0,
            bg_color=1., max_tgt_size=None, aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0], render_tgt_size=source_size,
            multiply=14, need_mask=True, get_shape_param=True
        )
        
        vis_ref_img = (image_tensor[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(os.path.join(dump_tmp_dir, "output.png"))
        
        # prepare_motion_seqs 写経
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(
            motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size, multiply=16,
            need_mask=motion_img_need_mask, vis_motion=vis_motion,
            shape_param=shape_param, test_sample=False, cross_id=False,
            src_driven=src_driven, max_squen_length=300
        )
        
        # infer_single_view 写経
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32
        with torch.no_grad():
            res = self.lam.infer_single_view(
                image_tensor.unsqueeze(0).to(device, dtype), None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()}
            )
            
        output_zip_path = ''
        if enable_oac_file:
            try:
                from tools.generateARKITGLBWithBlender import generate_glb # [MODAL変更] importパス
                oac_dir = os.path.join('/tmp', base_iid)
                saved_head_path = self.lam.renderer.flame_model.save_shaped_mesh(shape_param.unsqueeze(0).cuda(), fd=oac_dir)
                res['cano_gs_lst'][0].save_ply(os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True)
                generate_glb(
                    input_mesh=Path(saved_head_path),
                    template_fbx=Path("./model_zoo/sample_oac/template_file.fbx"), # [MODAL変更] パス
                    output_glb=Path(os.path.join(oac_dir, "skin.glb")),
                    blender_exec=Path("/usr/local/bin/blender") # [MODAL変更] パス
                )
                shutil.copy(
                    src='./model_zoo/sample_oac/animation.glb', # [MODAL変更] パス
                    dst=os.path.join(oac_dir, 'animation.glb')
                )
                os.remove(saved_head_path)
                output_zip_name = base_iid + '.zip'
                output_zip_path = os.path.join('/vol/output', output_zip_name)
                os.system(f'cd /tmp && zip -r {output_zip_path} {base_iid}')
                shutil.rmtree(oac_dir)
            except Exception as e:
                print(f"OAC error: {e}")

        # RGB後処理 写経
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        
        save_imgs_2_video(rgb, dump_video_path, render_fps)
        audio_path = os.path.join("./model_zoo/sample_motion/export", base_vid, base_vid + ".wav") # [MODAL変更] パス
        dump_video_path_wa = os.path.join("/vol/output", base_iid + "_audio.mp4")
        add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        output_vol.commit()
        
        return os.path.basename(dump_video_path_wa), (base_iid + '.zip' if enable_oac_file else None)

# --- Gradio UI (CPU) ---

@app.function(image=image, volumes={"/vol/output": output_vol})
@modal.web_endpoint(method="GET", label="lam-ui")
def web():
    import gradio as gr
    
    def predict(image_file, motion_video, enable_oac):
        if image_file is None: return None, None
        with open(image_file, "rb") as f:
            img_bytes = f.read()
        
        # モーションファイル名を取得
        motion_name = os.path.basename(motion_video).replace(".mp4", "")
        
        video_name, zip_name = Generator().generate.remote(img_bytes, motion_name, enable_oac)
        
        video_path = f"/vol/output/{video_name}"
        zip_path = f"/vol/output/{zip_name}" if zip_name else None
        return video_path, zip_path

    # app.pyのUI構成を簡略化して再現
    with gr.Blocks() as demo:
        gr.Markdown("# LAM: Large Avatar Model (Modal Edition)")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="filepath", label="Input Image")
                # app.pyのモーションリストを再現
                motion_choice = gr.Dropdown(
                    choices=[
                        "Speeding_Scandal", "Look_In_My_Eyes", "D_ANgelo_Dinero", 
                        "Michael_Wayne_Rosen", "I_Am_Iron_Man", "Anti_Drugs", 
                        "Pen_Pineapple_Apple_Pen", "Taylor_Swift", "GEM", "The_Shawshank_Redemption"
                    ],
                    value="Speeding_Scandal", label="Motion Template"
                )
                enable_oac = gr.Checkbox(label="Export ZIP for Chatting Avatar")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_video = gr.Video(label="Rendered Video")
                out_zip = gr.File(label="Output ZIP")
        
        btn.click(predict, [input_img, motion_choice, enable_oac], [out_video, out_zip])
    
    return demo