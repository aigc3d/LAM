"""
concierge_modal.py - Concierge ZIP Generator on Modal (FINAL FIXED VERSION v2)
=====================================================

Updates:
1. Fixed "FileNotFoundError" in Gradio UI by handling file uploads correctly.
2. Kept all quality fixes (Official Blender script, Xformers, Weight loading).
3. Cost optimization settings applied (timeout=600).

Usage:
  modal run concierge_modal.py                              # Gradio UI
  modal deploy concierge_modal.py                           # Production
"""

import os
import sys
import modal

app = modal.App("concierge-zip-generator")

# Persistent storage for generated ZIPs
output_vol = modal.Volume.from_name("concierge-output", create_if_missing=True)
OUTPUT_VOL_PATH = "/vol/output"

# Model weights storage (uploaded from official ModelScope)
storage_vol = modal.Volume.from_name("lam-storage")
STORAGE_VOL_PATH = "/vol/lam-storage"

# ============================================================
# Modal Image Build
# ============================================================
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "tree",
        "libusb-1.0-0", "build-essential",
        "gcc", "g++", "ninja-build",
        # Blender runtime deps
        "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libxkbcommon0", "libsm6",
    )
    # Base Python
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy==1.26.4'",
    )
    # PyTorch 2.4.0 + CUDA 12.1
    .run_commands(
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    # xformers: Required for DINOv2 attention accuracy
    .run_commands(
        "pip install xformers==0.0.27.post2 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    # CUDA build environment
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
    # CUDA extensions
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        # Patch chumpy for NumPy 1.24+ (removed numpy.bool/int/float/complex/object/unicode/str)
        # Patch chumpy source AND delete __pycache__ so Python doesn't use stale bytecode
        "CHUMPY_INIT=$(python -c \"import importlib.util; print(importlib.util.find_spec('chumpy').origin)\") && "
        "sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/"
        "from numpy import nan, inf; import numpy; bool = numpy.bool_; int = numpy.int_; "
        "float = numpy.float64; complex = numpy.complex128; object = numpy.object_; "
        "unicode = numpy.str_; str = numpy.str_/' "
        "\"$CHUMPY_INIT\" && "
        "find $(dirname \"$CHUMPY_INIT\") -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true",
        "pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation",
    )
    # Python dependencies
    .pip_install(
        "gradio==4.44.0",
        "gradio_client==1.3.0",
        "fastapi",
        "omegaconf==2.3.0",
        "pandas",
        "scipy<1.14.0",
        "opencv-python-headless==4.9.0.80",
        "imageio[ffmpeg]",
        "moviepy==1.0.3",
        "rembg[gpu]",
        "scikit-image",
        "pillow",
        "huggingface_hub>=0.24.0",
        "filelock",
        "typeguard",
        "transformers==4.44.2",
        "diffusers==0.30.3",
        "accelerate==0.34.2",
        "tyro==0.8.0",
        "mediapipe==0.10.21",
        "tensorboard",
        "rich",
        "loguru",
        "Cython",
        "PyMCubes",
        "trimesh",
        "einops",
        "plyfile",
        "jaxtyping",
        "ninja",
        "patool",
        "safetensors",
        "decord",
        "numpy==1.26.4",
    )
    # onnxruntime-gpu for CUDA 12 (cuDNN 8.x compatible; PyPI default is CUDA 11)
    .run_commands(
        "pip install onnxruntime-gpu==1.18.1 "
        "--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
    )
    # diff-gaussian-rasterization — patch CUDA 12.1 header issues then build
    .run_commands(
        "git clone https://github.com/ashawkey/diff-gaussian-rasterization.git /tmp/dgr && "
        "find /tmp/dgr -name '*.cu' -exec sed -i '1i #include <cfloat>' {} + && "
        "find /tmp/dgr -name '*.h' -path '*/cuda_rasterizer/*' -exec sed -i '1i #include <cstdint>' {} + && "
        "pip install /tmp/dgr --no-build-isolation && "
        "rm -rf /tmp/dgr",
    )
    # simple-knn — patch cfloat for CUDA 12.1 then build
    .run_commands(
        "git clone https://github.com/camenduru/simple-knn.git /tmp/simple-knn && "
        "sed -i '1i #include <cfloat>' /tmp/simple-knn/simple_knn.cu && "
        "pip install /tmp/simple-knn --no-build-isolation && "
        "rm -rf /tmp/simple-knn",
    )
    # nvdiffrast — JIT compilation at runtime (requires -devel image)
    .run_commands(
        "pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation",
    )
    # FBX SDK
    .run_commands(
        "pip install https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl",
    )
    # Blender 4.2 LTS
    .run_commands(
        "wget -q https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender",
        "tar xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1",
        "ln -sf /opt/blender/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
    # Clone LAM and build cpu_nms
    .run_commands(
        "git clone https://github.com/aigc3d/LAM.git /root/LAM",
        # Patch cpu_nms.pyx: replace deprecated np.int with np.intp for NumPy 1.24+
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
    # BIRD-MONSTER FIX: Remove @torch.compile from cloned LAM source files.
    # These decorators cause silent numerical corruption on Modal L4 GPUs.
    # Confirmed on CUDA 11.8 + PyTorch 2.3.0; retained for CUDA 12.1 + PyTorch 2.4.0
    # as a safety measure. Setting TORCHDYNAMO_DISABLE=1 at runtime is NOT sufficient
    # because the decorator wraps the function at import time.
    .run_commands(
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED: causes bird-monster on Modal/' "
        "/root/LAM/lam/models/modeling_lam.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED: causes bird-monster on Modal/' "
        "/root/LAM/lam/models/encoders/dinov2_fusion_wrapper.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED: causes bird-monster on Modal/' "
        "/root/LAM/lam/losses/tvloss.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED: causes bird-monster on Modal/' "
        "/root/LAM/lam/losses/pixelwise.py",
        "echo '[BIRD-FIX] Removed all @torch.compile decorators from LAM source'",
    )
    # Pre-download DINOv2 pretrained weights during image build to avoid
    # runtime download dependency from dl.fbaipublicfiles.com
    .run_commands(
        "python -c \""
        "import torch; "
        "url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'; "
        "torch.hub.load_state_dict_from_url(url, map_location='cpu'); "
        "print('DINOv2 ViT-L/14 weights cached OK')\"",
    )
)


def _precompile_nvdiffrast():
    """Pre-compile nvdiffrast CUDA kernels during image build to avoid cold-start delay."""
    import torch
    import nvdiffrast.torch as dr
    print("nvdiffrast pre-compiled OK")


image = image.run_function(_precompile_nvdiffrast)


def _download_missing_models():
    import subprocess
    from huggingface_hub import snapshot_download, hf_hub_download

    os.chdir("/root/LAM")

    # LAM-20K model weights
    target = "/root/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500"
    if not os.path.isfile(os.path.join(target, "model.safetensors")):
        print("[1/4] Downloading LAM-20K model weights...")
        snapshot_download(
            repo_id="3DAIGC/LAM-20K",
            local_dir=target,
            local_dir_use_symlinks=False,
        )

    # FLAME tracking models
    if not os.path.isfile("/root/LAM/model_zoo/flame_tracking_models/FaceBoxesV2.pth"):
        print("[2/4] Downloading FLAME tracking models (thirdparty_models.tar)...")
        hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type="model",
            filename="thirdparty_models.tar",
            local_dir="/root/LAM/",
        )
        subprocess.run(
            "tar -xf thirdparty_models.tar && rm thirdparty_models.tar",
            shell=True, cwd="/root/LAM", check=True,
        )

    # FLAME parametric model
    if not os.path.isfile("/root/LAM/model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl"):
        print("[3/4] Downloading FLAME parametric model (LAM_human_model.tar)...")
        hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type="model",
            filename="LAM_human_model.tar",
            local_dir="/root/LAM/",
        )
        subprocess.run(
            "tar -xf LAM_human_model.tar && rm LAM_human_model.tar",
            shell=True, cwd="/root/LAM", check=True,
        )
        src = "/root/LAM/assets/human_parametric_models"
        dst = "/root/LAM/model_zoo/human_parametric_models"
        if os.path.isdir(src) and not os.path.exists(dst):
            subprocess.run(["cp", "-r", src, dst], check=True)

    # LAM assets
    if not os.path.isfile("/root/LAM/model_zoo/sample_motion/export/talk/flame_param/00000.npz"):
        print("[4/4] Downloading LAM assets (sample motions)...")
        hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type="model",
            filename="LAM_assets.tar",
            local_dir="/root/LAM/",
        )
        subprocess.run(
            "tar -xf LAM_assets.tar && rm LAM_assets.tar",
            shell=True, cwd="/root/LAM", check=True,
        )
        for subdir in ["sample_oac", "sample_motion"]:
            src = f"/root/LAM/assets/{subdir}"
            dst = f"/root/LAM/model_zoo/{subdir}"
            if os.path.isdir(src) and not os.path.exists(dst):
                subprocess.run(["cp", "-r", src, dst], check=True)

    # sample_oac
    if not os.path.isfile("/root/LAM/model_zoo/sample_oac/template_file.fbx"):
        print("[+] Downloading sample_oac (FBX/GLB templates)...")
        subprocess.run(
            "wget -q https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar"
            " -O /root/LAM/sample_oac.tar",
            shell=True, check=True,
        )
        subprocess.run(
            "mkdir -p /root/LAM/model_zoo/sample_oac && "
            "tar -xf /root/LAM/sample_oac.tar -C /root/LAM/model_zoo/ && "
            "rm /root/LAM/sample_oac.tar",
            shell=True, check=True,
        )

    print("Model downloads complete.")


# Model weights are served from lam-storage volume at runtime (no image bake-in needed).
# image = image.run_function(_download_missing_models)

if os.path.isdir("./tools"):
    image = image.add_local_dir("./tools", remote_path="/root/LAM/tools")
# Overlay local lam/ directory to ensure code consistency with this repo.
# The git clone may have a different version than our local fork.
if os.path.isdir("./lam"):
    image = image.add_local_dir("./lam", remote_path="/root/LAM/lam")

dl_image = modal.Image.debian_slim(python_version="3.10").pip_install("fastapi")
ui_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "gradio>=4.0", "fastapi", "uvicorn",
)


# ============================================================
# Pipeline Functions
# ============================================================

def _setup_model_paths():
    """Symlink model_zoo/assets/tmp_assets from lam-storage volume into /root/LAM."""
    import shutil as _shutil
    lam_root = "/root/LAM"
    vol_lam = os.path.join(STORAGE_VOL_PATH, "LAM")

    # Link top-level dirs from volume
    for subdir in ["model_zoo", "assets", "tmp_assets"]:
        src = os.path.join(vol_lam, subdir)
        dst = os.path.join(lam_root, subdir)
        if not os.path.isdir(src):
            continue
        # Remove stale dir/link from git clone or previous run
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            _shutil.rmtree(dst)
        os.symlink(src, dst)
        print(f"  [volume] {dst} -> {src}")

    # Bridge official model path (exps/) if weights live there on the volume
    vol_exps_model = os.path.join(vol_lam, "exps", "releases", "lam", "lam-20k", "step_045500")
    our_model_dir = os.path.join(lam_root, "model_zoo", "lam_models", "releases", "lam", "lam-20k", "step_045500")
    if os.path.isdir(vol_exps_model) and not os.path.exists(our_model_dir):
        os.makedirs(os.path.dirname(our_model_dir), exist_ok=True)
        os.symlink(vol_exps_model, our_model_dir)
        print(f"  [volume] {our_model_dir} -> {vol_exps_model}")

    # FLAME subdirectory structure bridging
    model_zoo = os.path.join(lam_root, "model_zoo")
    hpm = os.path.join(model_zoo, "human_parametric_models")
    if os.path.isdir(hpm):
        flame_subdir = os.path.join(hpm, "flame_assets", "flame")
        flame_assets_dir = os.path.join(hpm, "flame_assets")
        if os.path.isdir(flame_assets_dir) and not os.path.exists(flame_subdir):
            if os.path.isfile(os.path.join(flame_assets_dir, "flame2023.pkl")):
                os.symlink(flame_assets_dir, flame_subdir)

        flame_vhap = os.path.join(hpm, "flame_vhap")
        if not os.path.exists(flame_vhap):
            for candidate in [flame_subdir, flame_assets_dir]:
                if os.path.isdir(candidate):
                    os.symlink(candidate, flame_vhap)
                    break


def _init_lam_pipeline():
    """Initialize FLAME tracking and LAM model. Called once per container."""
    import torch
    import torch._dynamo

    # ============================================================
    # CRITICAL: Disable torch.compile/dynamo BEFORE any lam imports.
    # @torch.compile decorators on ModelLAM.forward_latent_points and
    # Dinov2FusionWrapper.forward are evaluated at IMPORT time (class
    # definition). If we monkey-patch torch.compile AFTER the import,
    # the decorators have already created wrapper objects that can
    # silently corrupt computation on Modal L4 GPUs.
    # ============================================================
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()

    _original_torch_compile = torch.compile
    def _noop_compile(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f
    torch.compile = _noop_compile
    print("[BIRD-FIX] torch.compile patched to no-op BEFORE imports")

    # Set up working directory and paths BEFORE importing lam modules
    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")
    _setup_model_paths()

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
        "APP_TYPE": "infer.lam",
        "NUMBA_THREADING_LAYER": "forseq",  # Match official ModelScope app.py
    })

    # NOW import lam modules (with torch.compile safely disabled)
    from safetensors.torch import load_file as _load_safetensors
    from lam.models import ModelLAM
    from app_lam import parse_configs

    # Parse config
    cfg, _ = parse_configs()

    print("Loading LAM model...")
    model_cfg = cfg.model
    lam = ModelLAM(**model_cfg)

    # Restore original torch.compile
    torch.compile = _original_torch_compile

    # Verify torch.compile is truly disabled on critical methods
    fwd_lp = getattr(lam, 'forward_latent_points', None)
    if fwd_lp is not None:
        is_compiled = hasattr(fwd_lp, '_torchdynamo_orig_callable') or \
                      hasattr(fwd_lp, '__wrapped__') or \
                      'OptimizedModule' in type(fwd_lp).__name__ or \
                      '_TorchCompile' in type(fwd_lp).__name__
        print(f"[BIRD-FIX] forward_latent_points type: {type(fwd_lp).__name__}, compiled={is_compiled}")
    enc_fwd = getattr(lam.encoder, 'forward', None) if hasattr(lam, 'encoder') else None
    if enc_fwd is not None:
        is_compiled = hasattr(enc_fwd, '_torchdynamo_orig_callable') or \
                      'OptimizedModule' in type(enc_fwd).__name__
        print(f"[BIRD-FIX] encoder.forward type: {type(enc_fwd).__name__}, compiled={is_compiled}")

    ckpt_path = os.path.join(cfg.model_name, "model.safetensors")
    print(f"Loading checkpoint: {ckpt_path}")

    # --- WEIGHT LOADING: use load_state_dict (matches app_concierge.py working version) ---
    ckpt = _load_safetensors(ckpt_path, device="cpu")
    missing, unexpected = lam.load_state_dict(ckpt, strict=False)
    print(f"Weight loading summary:")
    print(f"  Checkpoint keys: {len(ckpt)}")
    print(f"  Model keys:      {len(lam.state_dict())}")
    print(f"  Missing in model (not in ckpt): {len(missing)}")
    print(f"  Unexpected (in ckpt, not in model): {len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")
    print("=" * 100)

    lam.to("cuda")
    lam.to(torch.float32)  # CRITICAL: match official inference (lam/runners/infer/lam.py:337-339)
    lam.eval()
    print(f"[DTYPE-FIX] Model dtype after .to(float32): {next(lam.parameters()).dtype}")

    # Initialize FLAME tracking
    from tools.flame_tracking_single_image import FlameTrackingSingleImage
    print("Initializing FLAME tracking...")
    flametracking = FlameTrackingSingleImage(
        output_dir="output/tracking",
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        detect_iris_landmarks=False,
    )

    return cfg, lam, flametracking


def _track_video_to_motion(video_path, flametracking, working_dir, status_callback=None):
    """Process a custom motion video through VHAP FLAME tracking."""
    import cv2
    import numpy as np
    import torch
    import torchvision
    from pathlib import Path

    def report(msg):
        if status_callback:
            status_callback(msg)
        print(msg)

    report("  Extracting video frames...")
    frames_root = os.path.join(working_dir, "video_tracking", "preprocess")
    sequence_name = "custom_motion"
    sequence_dir = os.path.join(frames_root, sequence_name)

    images_dir = os.path.join(sequence_dir, "images")
    alpha_dir = os.path.join(sequence_dir, "alpha_maps")
    landmark_dir = os.path.join(sequence_dir, "landmark2d")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(landmark_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    target_fps = min(30, video_fps) if video_fps > 0 else 30
    frame_interval = max(1, int(round(video_fps / target_fps)))
    max_frames = 300

    report(f"  Video: sampling every {frame_interval} frame(s)")

    all_landmarks = []
    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or processed_count >= max_frames:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)

        try:
            from tools.flame_tracking_single_image import expand_bbox
            _, bbox, _ = flametracking.vgghead_encoder(frame_tensor, processed_count)
            if bbox is None:
                frame_idx += 1
                continue
        except Exception:
            frame_idx += 1
            continue

        bbox = expand_bbox(bbox, scale=1.65).long()
        cropped = torchvision.transforms.functional.crop(
            frame_tensor, top=bbox[1], left=bbox[0],
            height=bbox[3] - bbox[1], width=bbox[2] - bbox[0],
        )
        cropped = torchvision.transforms.functional.resize(cropped, (1024, 1024), antialias=True)

        cropped_matted, mask = flametracking.matting_engine(
            cropped / 255.0, return_type="matting", background_rgb=1.0,
        )
        cropped_matted = cropped_matted.cpu() * 255.0
        saved_image = np.round(cropped_matted.permute(1, 2, 0).numpy()).astype(np.uint8)[:, :, ::-1]

        fname = f"{processed_count:05d}.png"
        cv2.imwrite(os.path.join(images_dir, fname), saved_image)
        cv2.imwrite(
            os.path.join(alpha_dir, fname.replace(".png", ".jpg")),
            (np.ones_like(saved_image) * 255).astype(np.uint8),
        )

        saved_image_rgb = saved_image[:, :, ::-1]
        detections, _ = flametracking.detector.detect(saved_image_rgb, 0.8, 1)
        frame_landmarks = None
        for det in detections:
            x1, y1 = det[2], det[3]
            x2, y2 = x1 + det[4], y1 + det[5]
            scale = max(x2 - x1, y2 - y1) / 180
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            face_lmk = flametracking.alignment.analyze(
                saved_image_rgb, float(scale), float(cx), float(cy),
            )
            normalized = np.zeros((face_lmk.shape[0], 3))
            normalized[:, :2] = face_lmk / 1024
            frame_landmarks = normalized
            break

        if frame_landmarks is None:
            frame_idx += 1
            continue

        all_landmarks.append(frame_landmarks)
        processed_count += 1
        frame_idx += 1

    cap.release()
    torch.cuda.empty_cache()

    if processed_count == 0:
        raise RuntimeError("No valid face frames found in video")

    stacked_landmarks = np.stack(all_landmarks, axis=0)
    np.savez(
        os.path.join(landmark_dir, "landmarks.npz"),
        bounding_box=[],
        face_landmark_2d=stacked_landmarks,
    )

    report("  Running VHAP FLAME tracking...")
    from vhap.config.base import (
        BaseTrackingConfig, DataConfig, ModelConfig, RenderConfig, LogConfig,
        ExperimentConfig, LearningRateConfig, LossWeightConfig, PipelineConfig,
        StageLmkInitRigidConfig, StageLmkInitAllConfig,
        StageLmkSequentialTrackingConfig, StageLmkGlobalTrackingConfig,
        StageRgbInitTextureConfig, StageRgbInitAllConfig,
        StageRgbInitOffsetConfig, StageRgbSequentialTrackingConfig,
        StageRgbGlobalTrackingConfig,
    )
    from vhap.model.tracker import GlobalTracker

    tracking_output = os.path.join(working_dir, "video_tracking", "tracking")
    pipeline = PipelineConfig(
        lmk_init_rigid=StageLmkInitRigidConfig(),
        lmk_init_all=StageLmkInitAllConfig(),
        lmk_sequential_tracking=StageLmkSequentialTrackingConfig(),
        lmk_global_tracking=StageLmkGlobalTrackingConfig(),
        rgb_init_texture=StageRgbInitTextureConfig(),
        rgb_init_all=StageRgbInitAllConfig(),
        rgb_init_offset=StageRgbInitOffsetConfig(),
        rgb_sequential_tracking=StageRgbSequentialTrackingConfig(),
        rgb_global_tracking=StageRgbGlobalTrackingConfig(),
    )

    vhap_cfg = BaseTrackingConfig(
        data=DataConfig(
            root_folder=Path(frames_root), sequence=sequence_name, landmark_source="star",
        ),
        model=ModelConfig(), render=RenderConfig(), log=LogConfig(),
        exp=ExperimentConfig(output_folder=Path(tracking_output), photometric=True),
        lr=LearningRateConfig(), w=LossWeightConfig(), pipeline=pipeline,
    )

    tracker = GlobalTracker(vhap_cfg)
    tracker.optimize()
    torch.cuda.empty_cache()

    report("  Exporting motion sequence...")
    from vhap.export_as_nerf_dataset import (
        NeRFDatasetWriter, TrackedFLAMEDatasetWriter, split_json, load_config,
    )

    export_dir = os.path.join(working_dir, "video_tracking", "export", sequence_name)
    export_path = Path(export_dir)
    src_folder, cfg_loaded = load_config(Path(tracking_output))
    NeRFDatasetWriter(cfg_loaded.data, export_path, None, None, "white").write()
    TrackedFLAMEDatasetWriter(cfg_loaded.model, src_folder, export_path, mode="param", epoch=-1).write()
    split_json(export_path)

    return os.path.join(export_dir, "flame_param")


def _generate_concierge_zip(image_path, video_path, cfg, lam, flametracking, motion_name=None):
    """Full pipeline: image + video -> concierge.zip"""
    import torch
    import numpy as np
    import subprocess
    import zipfile
    import shutil
    import tempfile
    import json
    from pathlib import Path
    from PIL import Image
    from glob import glob
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
    
    # --- MAJOR FIX: USE OFFICIAL TOOL INSTEAD OF INLINE SCRIPT ---
    from tools.generateARKITGLBWithBlender import generate_glb

    from datetime import datetime as _dt
    working_dir = tempfile.mkdtemp(prefix="concierge_")
    base_iid = "chatting_avatar_" + _dt.now().strftime("%Y%m%d%H%M%S")
    
    try:
        # Step 0: Clean stale FLAME tracking data
        tracking_root = os.path.join(os.getcwd(), "output", "tracking")
        if os.path.isdir(tracking_root):
            for subdir in ["preprocess", "tracking", "export"]:
                stale = os.path.join(tracking_root, subdir)
                if os.path.isdir(stale):
                    shutil.rmtree(stale)

        # Step 1: Source image FLAME tracking
        yield "Step 1: FLAME tracking on source image...", None, None, None, None

        image_raw = os.path.join(working_dir, "raw.png")
        with Image.open(image_path).convert("RGB") as img:
            img.save(image_raw)

        ret = flametracking.preprocess(image_raw)
        assert ret == 0, "FLAME preprocess failed"
        ret = flametracking.optimize()
        assert ret == 0, "FLAME optimize failed"
        ret, output_dir = flametracking.export()
        assert ret == 0, "FLAME export failed"

        tracked_image = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        yield f"Step 1 done: check tracked face -->", None, None, tracked_image, None

        # Step 2: Motion sequence
        if video_path and os.path.isfile(video_path):
            total_steps = 6
            yield f"Step 2/{total_steps}: Processing custom motion video...", None, None, None, None
            flame_params_dir = _track_video_to_motion(video_path, flametracking, working_dir)
            motion_source = "custom video"
        else:
            total_steps = 5
            sample_motions = glob("./model_zoo/sample_motion/export/*/flame_param")
            if not sample_motions:
                raise RuntimeError("No motion sequences available.")
            flame_params_dir = sample_motions[0]
            if motion_name:
                for sp in sample_motions:
                    if os.path.basename(os.path.dirname(sp)) == motion_name:
                        flame_params_dir = sp
                        break
            motion_source = f"sample '{os.path.basename(os.path.dirname(flame_params_dir))}'"

        yield f"Step 3/{total_steps}: Preparing LAM inference...", None, None, None, None

        # Step 3: LAM inference
        image_tensor, _, _, shape_param = preprocess_image(
            tracked_image, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1.0, 
            max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=cfg.source_size, multiply=14, need_mask=True, get_shape_param=True,
        )

        preproc_vis_path = os.path.join(working_dir, "preprocessed_input.png")
        vis_img = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_img).save(preproc_vis_path)

        src_name = os.path.splitext(os.path.basename(image_path))[0]
        driven_name = os.path.basename(os.path.dirname(flame_params_dir))
        
        motion_seq = prepare_motion_seqs(
            flame_params_dir, None, save_root=working_dir, fps=30,
            bg_color=1.0, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_image_res=cfg.render_size, multiply=16,
            need_mask=False, vis_motion=False, shape_param=shape_param, test_sample=False,
            cross_id=False, src_driven=[src_name, driven_name],
            max_squen_length=300,
        )

        yield f"Step 4/{total_steps}: Running LAM inference...", None, None, None, preproc_vis_path

        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device = "cuda"
        with torch.no_grad():
            res = lam.infer_single_view(
                image_tensor.unsqueeze(0).to(device, torch.float32),
                None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
            )

        # Step 4: Generate GLB + ZIP
        yield f"Step 5/{total_steps}: Generating 3D avatar (Blender GLB)...", None, None, None, preproc_vis_path

        oac_dir = os.path.join(working_dir, "oac_export", base_iid)
        os.makedirs(oac_dir, exist_ok=True)

        saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir,
        )

        # Save offset.ply BEFORE generate_glb (matches official app.py order)
        res["cano_gs_lst"][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True,
        )

        # --- USE OFFICIAL GLB EXPORT ---
        # generate_glb internally runs: update_flame_shape → convert_ascii_to_binary
        # → convertFBX2GLB.py (skin.glb) → generateVertexIndices.py (vertex_order.json)
        generate_glb(
            input_mesh=Path(saved_head_path),
            template_fbx=Path("./model_zoo/sample_oac/template_file.fbx"),
            output_glb=Path(os.path.join(oac_dir, "skin.glb")),
            blender_exec=Path("/usr/local/bin/blender")
        )

        shutil.copy(
            src="./model_zoo/sample_oac/animation.glb",
            dst=os.path.join(oac_dir, "animation.glb"),
        )
        if os.path.exists(saved_head_path):
            os.remove(saved_head_path)

        # Create ZIP
        step_label = f"Step {total_steps}/{total_steps}"
        yield f"{step_label}: Creating concierge.zip...", None, None, None, preproc_vis_path

        output_zip = os.path.join(working_dir, "concierge.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            dir_info = zipfile.ZipInfo(os.path.basename(oac_dir) + "/")
            zf.writestr(dir_info, "")
            for root, _, files in os.walk(oac_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, os.path.dirname(oac_dir))
                    zf.write(fpath, arcname)

        # Preview video
        preview_path = os.path.join(working_dir, "preview.mp4")
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        
        from app_lam import save_images2video, add_audio_to_video
        save_images2video(rgb, preview_path, 30)

        # Re-encode for browser
        preview_browser = os.path.join(working_dir, "preview_browser.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", preview_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "faststart", preview_browser])
        if os.path.isfile(preview_browser) and os.path.getsize(preview_browser) > 0:
            os.replace(preview_browser, preview_path)

        final_preview = preview_path
        if video_path and os.path.isfile(video_path):
            try:
                preview_with_audio = os.path.join(working_dir, "preview_audio.mp4")
                add_audio_to_video(preview_path, preview_with_audio, video_path)
                preview_audio_browser = os.path.join(working_dir, "preview_audio_browser.mp4")
                subprocess.run(["ffmpeg", "-y", "-i", preview_with_audio, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "faststart", preview_audio_browser])
                if os.path.isfile(preview_audio_browser) and os.path.getsize(preview_audio_browser) > 0:
                    os.replace(preview_audio_browser, preview_with_audio)
                final_preview = preview_with_audio
            except Exception:
                pass

        # Save to volume
        vol_dir = OUTPUT_VOL_PATH
        os.makedirs(vol_dir, exist_ok=True)
        shutil.copy2(output_zip, os.path.join(vol_dir, "concierge.zip"))
        shutil.copy2(final_preview, os.path.join(vol_dir, "preview.mp4"))
        output_vol.commit()

        yield (
            f"concierge.zip generated ({os.path.getsize(output_zip)/(1024*1024):.1f} MB)",
            output_zip, final_preview, None, preproc_vis_path,
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n_generate_concierge_zip ERROR:\n{tb}", flush=True)
        yield f"Error: {str(e)}\n\nTraceback:\n{tb}", None, None, None, None


@app.cls(gpu="L4", image=image, volumes={OUTPUT_VOL_PATH: output_vol, STORAGE_VOL_PATH: storage_vol}, timeout=600, scaledown_window=300, min_containers=0, max_containers=1)
class Generator:
    @modal.enter()
    def setup(self):
        import shutil
        os.chdir("/root/LAM")
        sys.path.insert(0, "/root/LAM")

        print("Initializing LAM pipeline on GPU...")
        self.cfg, self.lam, self.flametracking = _init_lam_pipeline()
        print("GPU pipeline ready.")

    @modal.method()
    def generate(self, image_bytes: bytes, video_bytes: bytes, motion_name: str, job_id: str):
        import shutil
        import tempfile
        import json

        upload_dir = tempfile.mkdtemp(prefix="gpu_upload_")
        image_path = os.path.join(upload_dir, "input.png")
        with open(image_path, "wb") as f: f.write(image_bytes)

        video_path = None
        if video_bytes:
            video_path = os.path.join(upload_dir, "input_video.mp4")
            with open(video_path, "wb") as f: f.write(video_bytes)

        effective_video = video_path if motion_name == "custom" else None
        selected_motion = motion_name if motion_name != "custom" else None

        vol_dir = OUTPUT_VOL_PATH
        status_file = os.path.join(vol_dir, f"status_{job_id}.json")

        try:
            final_msg = "Processing..."
            for status, _, _, tracked_img, preproc_img in _generate_concierge_zip(
                image_path, effective_video, self.cfg, self.lam, self.flametracking, motion_name=selected_motion,
            ):
                final_msg = status
                if tracked_img and os.path.isfile(tracked_img):
                    shutil.copy2(tracked_img, os.path.join(vol_dir, "tracked_face.png"))
                    output_vol.commit()
                if preproc_img and os.path.isfile(preproc_img):
                    shutil.copy2(preproc_img, os.path.join(vol_dir, "preproc_input.png"))
                    output_vol.commit()

            with open(status_file, "w") as f: json.dump({"type": "done", "msg": final_msg}, f)
            output_vol.commit()

        except Exception as e:
            try:
                with open(status_file, "w") as f: json.dump({"type": "error", "msg": str(e)}, f)
                output_vol.commit()
            except Exception: pass
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)


@app.function(image=ui_image, timeout=7200, volumes={OUTPUT_VOL_PATH: output_vol, STORAGE_VOL_PATH: storage_vol})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from glob import glob
    import gradio_client.utils as _gc_utils
    
    _orig_jst = _gc_utils._json_schema_to_python_type
    def _safe_jst(schema, defs=None): return "Any" if isinstance(schema, bool) else _orig_jst(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_jst

    lam_root = "/root/LAM"
    vol_lam = os.path.join(STORAGE_VOL_PATH, "LAM")
    # Scan sample motions from the storage volume
    sample_motions = sorted(
        glob(f"{vol_lam}/model_zoo/sample_motion/export/*/*.mp4") +
        glob(f"{vol_lam}/assets/sample_motion/export/*/*.mp4")
    )

    def process(image_path, video_path, motion_choice):
        import time, json, threading, uuid
        
        # --- FIX FOR FileNotFound ERROR ---
        # The 'image_path' from Gradio is a temp path that might not be accessible or persistent.
        # We read it immediately into bytes within this context.
        if image_path is None:
            yield "Error: Please upload a face image", None, None, None, None
            return

        try:
            with open(image_path, "rb") as f: image_bytes = f.read()
        except FileNotFoundError:
            yield "Error: Image file lost during upload. Please try again.", None, None, None, None
            return

        video_bytes = b""
        if motion_choice == "custom" and video_path:
            try:
                with open(video_path, "rb") as f: video_bytes = f.read()
            except FileNotFoundError:
                yield "Error: Video file lost during upload. Please try again.", None, None, None, None
                return

        job_id = uuid.uuid4().hex[:8]
        status_file = os.path.join(OUTPUT_VOL_PATH, f"status_{job_id}.json")
        
        def _call_gpu():
            try:
                gen = Generator()
                # Pass bytes, not paths, to the remote GPU function
                gen.generate.remote(image_bytes, video_bytes, motion_choice or "custom", job_id)
            except Exception as e:
                print(f"GPU Launch Error: {e}")
        
        threading.Thread(target=_call_gpu, daemon=True).start()
        yield "Starting GPU worker...", None, None, None, None

        start = time.time()
        while True:
            time.sleep(5)
            output_vol.reload()
            
            tracked_p = os.path.join(OUTPUT_VOL_PATH, "tracked_face.png")
            preproc_p = os.path.join(OUTPUT_VOL_PATH, "preproc_input.png")
            last_tracked = tracked_p if os.path.isfile(tracked_p) else None
            last_preproc = preproc_p if os.path.isfile(preproc_p) else None

            if os.path.isfile(status_file):
                with open(status_file) as f: result = json.load(f)
                try: os.remove(status_file); output_vol.commit()
                except: pass
                
                if result["type"] == "error":
                    yield f"Error: {result['msg']}", None, None, None, None
                    return
                
                zip_p = os.path.join(OUTPUT_VOL_PATH, "concierge.zip")
                preview_p = os.path.join(OUTPUT_VOL_PATH, "preview.mp4")
                yield result["msg"], zip_p if os.path.isfile(zip_p) else None, preview_p if os.path.isfile(preview_p) else None, last_tracked, last_preproc
                return

            elapsed = int(time.time() - start)
            if elapsed > 1800: # 30 min timeout for UI polling
                 yield "Error: UI polling timed out", None, None, None, None
                 return

            mins, secs = divmod(elapsed, 60)
            yield f"Processing... ({mins}m{secs:02d}s)", None, None, last_tracked, last_preproc

    with gr.Blocks(title="Concierge ZIP Generator") as demo:
        gr.Markdown("# Concierge ZIP Generator (Final Fixed Version)")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Face Image", type="filepath")
                motion_choice = gr.Radio(label="Motion", choices=["custom"] + [os.path.basename(os.path.dirname(m)) for m in sample_motions], value="custom")
                input_video = gr.Video(label="Custom Video")
                btn = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status")
            with gr.Column():
                with gr.Row():
                    tracked = gr.Image(label="Tracked Face", height=200)
                    preproc = gr.Image(label="Model Input", height=200)
                preview = gr.Video(label="Preview")
                dl = gr.File(label="Download ZIP")

        btn.click(process, [input_image, input_video, motion_choice], [status, dl, preview, tracked, preproc])

    web_app = FastAPI()
    @web_app.get("/download-zip")
    async def download_zip():
        output_vol.reload()
        p = os.path.join(OUTPUT_VOL_PATH, "concierge.zip")
        return FileResponse(p, filename="concierge.zip") if os.path.isfile(p) else {"error": "Not found"}

    import mimetypes
    @web_app.api_route("/file={file_path:path}", methods=["GET", "HEAD"])
    async def serve_file(file_path: str):
        abs_path = "/" + file_path if not file_path.startswith("/") else file_path
        if (abs_path.startswith("/tmp/") or abs_path.startswith(OUTPUT_VOL_PATH)) and os.path.isfile(abs_path):
            return FileResponse(abs_path, media_type=mimetypes.guess_type(abs_path)[0])
        return {"error": "Not found"}

    return gr.mount_gradio_app(web_app, demo, path="/", allowed_paths=["/tmp/", OUTPUT_VOL_PATH])

@app.function(image=dl_image, volumes={OUTPUT_VOL_PATH: output_vol})
@modal.asgi_app()
def download():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    dl_app = FastAPI()
    @dl_app.get("/concierge.zip")
    async def get_zip():
        output_vol.reload()
        p = os.path.join(OUTPUT_VOL_PATH, "concierge.zip")
        return FileResponse(p, filename="concierge.zip") if os.path.isfile(p) else {"error": "Wait"}
    return dl_app
