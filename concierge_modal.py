# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Modal environment and pipeline initialization for LAM avatar generation.

import os
import modal

MINUTES = 60

# ---------------------------------------------------------------------------
# Modal Volume: holds model weights, assets, and sample motions
# ---------------------------------------------------------------------------
vol = modal.Volume.from_name("lam-model-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal Image: full build environment for LAM inference
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    # System deps: Blender, FBX SDK, GL libs, ffmpeg, zip
    .apt_install(
        "git", "wget", "unzip", "xz-utils", "libgl1", "libglib2.0-0",
        "libsm6", "libxrender1", "libxext6", "ffmpeg", "zip",
    )
    # Blender 4.0.2 (for GLB conversion via generateARKITGLBWithBlender)
    .run_commands(
        "wget -q https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz"
        " -O /tmp/blender.tar.xz",
        "tar xf /tmp/blender.tar.xz -C /opt/",
        "ln -s /opt/blender-4.0.2-linux-x64/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
    # FBX SDK (required by generateARKITGLBWithBlender)
    .run_commands(
        "pip install --no-cache-dir fbx-sdk-py",
    )
    # Core Python deps
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "safetensors", "omegaconf", "trimesh", "Pillow",
        "opencv-python-headless", "scipy", "einops",
        "transformers", "accelerate", "huggingface_hub",
        "moviepy==1.0.3", "imageio[ffmpeg]",
        "chumpy", "numpy==1.23.0",
        "kiui", "plyfile",
    )
    # pytorch3d (pre-built wheel for py310/cu121/pt251)
    .pip_install(
        "pytorch3d",
        find_links="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html",
    )
    # DINOv2 dependencies
    .pip_install("xformers==0.0.29.post1")
    # Copy the LAM source tree into the container
    .copy_local_dir(".", "/root/LAM", ignore=[
        ".git", "__pycache__", "*.pyc", "output", "exps",
        "train_data", ".venv", "node_modules",
    ])
    .env({
        "NUMBA_THREADING_LAYER": "forksafe",
        "PYTHONPATH": "/root/LAM",
        "BLENDER_PATH": "/usr/local/bin/blender",
    })
)

# ---------------------------------------------------------------------------
# Pipeline initialisation (called once per container warm-up)
# ---------------------------------------------------------------------------

def _init_lam_pipeline():
    """
    Initialise and return (cfg, lam, flametracking).

    Mirrors the official app_lam.py launch_gradio_app() init sequence.
    """
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    os.chdir("/root/LAM")

    # ---- env ----
    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
        "APP_TYPE": "infer.lam",
        "NUMBA_THREADING_LAYER": "forksafe",
    })

    # ---- parse config (simplified, no argparse) ----
    cfg = OmegaConf.create()
    cfg_train = OmegaConf.load("./configs/inference/lam-20k-8gpu.yaml")
    cfg.source_size = cfg_train.dataset.source_image_res          # 512
    cfg.render_size = cfg_train.dataset.render_image.high          # 512
    cfg.src_head_size = getattr(cfg_train.dataset, "src_head_size", 112)
    cfg.motion_video_read_fps = 30
    cfg.blender_path = os.environ.get("BLENDER_PATH", "/usr/local/bin/blender")
    cfg.model_name = os.environ["APP_MODEL_NAME"]

    model_name = cfg.model_name
    _relative_path = os.path.join(
        cfg_train.experiment.parent,
        cfg_train.experiment.child,
        os.path.basename(model_name).split("_")[-1],
    )
    cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
    cfg.image_dump = os.path.join("exps", "images", _relative_path)
    cfg.video_dump = os.path.join("exps", "videos", _relative_path)
    cfg.model = cfg_train.model

    # ---- build model ----
    from lam.models import ModelLAM

    model = ModelLAM(**cfg.model)
    resume = os.path.join(model_name, "model.safetensors")
    print("=" * 80)
    print(f"Loading pretrained weights from: {resume}")
    if resume.endswith("safetensors"):
        ckpt = load_file(resume, device="cpu")
    else:
        ckpt = torch.load(resume, map_location="cpu")
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"[WARN] shape mismatch {k}: ckpt {v.shape} vs model {state_dict[k].shape}")
        else:
            print(f"[WARN] unexpected key {k}: {v.shape}")
    print(f"Finished loading weights from: {resume}")
    print("=" * 80)

    lam = model
    lam.to("cuda")
    lam.eval()

    # ---- FLAME tracking ----
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    flametracking = FlameTrackingSingleImage(
        output_dir="output/tracking",
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        detect_iris_landmarks=False,
    )

    return cfg, lam, flametracking
