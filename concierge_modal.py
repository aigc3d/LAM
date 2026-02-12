"""
concierge_modal.py - Concierge ZIP Generator on Modal
=====================================================

Gradio UI for generating concierge.zip with CUSTOM image + CUSTOM motion video.

The official LAM UI only allows selecting from pre-set sample motion videos.
This script enables uploading your own motion video, which is processed through
VHAP FLAME tracking to extract per-frame expression/pose parameters, then used
with LAM inference to generate a high-quality concierge.zip.

Usage:
  modal serve concierge_modal.py   # Development mode (hot reload)
  modal deploy concierge_modal.py  # Production deployment

Pipeline:
  1. Source Image  → FlameTrackingSingleImage → shape parameters
  2. Motion Video  → VHAP GlobalTracker → per-frame FLAME parameters
  3. Shape + Motion → LAM inference → 3D Gaussian avatar
  4. Avatar data   → Blender GLB export → concierge.zip

Prerequisites:
  - Modal account with GPU access (A10G)
  - Run from your LAM repo root where model files are already downloaded
  - Required local directories:
      ./model_zoo/   (LAM weights, flame tracking models)
      ./assets/      (human_parametric_models, sample_oac, sample_motion)
"""

import os
import sys
import modal

app = modal.App("concierge-zip-generator")

# Detect which local directories contain model files.
# These are mounted into the container at build time.
_has_model_zoo = os.path.isdir("./model_zoo")
_has_assets = os.path.isdir("./assets")

if not _has_model_zoo and not _has_assets:
    print(
        "WARNING: Neither ./model_zoo/ nor ./assets/ found.\n"
        "Run `modal serve concierge_modal.py` from your LAM repo root."
    )

# ============================================================
# Modal Image Build
# ============================================================
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "tree",
        "libusb-1.0-0", "build-essential", "ninja-build",
        "clang", "llvm", "libclang-dev",
        # Blender runtime deps
        "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libxkbcommon0", "libsm6",
    )
    # Base Python
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy==1.23.5'",
    )
    # PyTorch 2.2.0 + CUDA 11.8
    .run_commands(
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    # CUDA build environment
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "CC": "clang",
        "CXX": "clang++",
    })
    # CUDA extensions (require no-build-isolation)
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7 --no-build-isolation",
    )
    # Python dependencies
    .pip_install(
        # --- Gradio 4.x (ASGI-native, no patching needed) ---
        # Pin to 4.44.0 to avoid json_schema_to_python_type bug in later versions
        "gradio==4.44.0",
        "gradio_client==1.3.0",
        "fastapi",
        # --- LAM dependencies ---
        "omegaconf==2.3.0",
        "pandas",
        "scipy<1.14.0",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy==1.0.3",
        "rembg[gpu]",
        "scikit-image",
        "pillow",
        "onnxruntime-gpu",
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
        "numpy==1.23.5",
    )
    # More CUDA extensions
    .run_commands(
        "pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git --no-build-isolation",
        "pip install git+https://github.com/ShenhanQian/nvdiffrast.git@backface-culling --no-build-isolation",
    )
    # FBX SDK Python bindings (needed for OBJ → FBX → GLB avatar export)
    .run_commands(
        "pip install https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl",
    )
    # Blender 4.2 LTS (needed for GLB generation)
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
        # Build cpu_nms for face detection
        "cd /root/LAM/external/landmark_detection/FaceBoxesV2/utils/nms && "
        "python -c \""
        "from setuptools import setup, Extension; "
        "from Cython.Build import cythonize; "
        "import numpy; "
        "setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), "
        "include_dirs=[numpy.get_include()])\" "
        "build_ext --inplace",
    )
)


# Download model weights not included locally (cached in image layer).
# This runs BEFORE add_local_dir so Modal ordering is satisfied.
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

    # FLAME tracking models (face detection, landmark, VGGHead, matting)
    # These are CRITICAL for FlameTrackingSingleImage to work correctly.
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
        # Verify extraction
        for f in ["FaceBoxesV2.pth", "68_keypoints_model.pkl",
                   "vgghead/vgg_heads_l.trcd", "matting/stylematte_synth.pt"]:
            path = f"/root/LAM/model_zoo/flame_tracking_models/{f}"
            if os.path.isfile(path):
                print(f"  OK: {path}")
            else:
                print(f"  WARNING: missing after extraction: {path}")

    # FLAME parametric model (flame2023.pkl, head mesh, etc.)
    # LAM_human_model.tar extracts to assets/human_parametric_models/.
    # IMPORTANT: add_local_dir("./assets") later overwrites /root/LAM/assets/
    # with the (sparse) local directory. We must copy to model_zoo/ to survive.
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
        # Copy to model_zoo/ so it survives the add_local_dir mount of assets/
        src = "/root/LAM/assets/human_parametric_models"
        dst = "/root/LAM/model_zoo/human_parametric_models"
        if os.path.isdir(src) and not os.path.exists(dst):
            subprocess.run(["cp", "-r", src, dst], check=True)
            print(f"  Copied assets/human_parametric_models -> model_zoo/")
        if os.path.isfile(f"{dst}/flame_assets/flame/flame2023.pkl"):
            print("  OK: flame2023.pkl extracted and copied")
        else:
            print("  WARNING: flame2023.pkl not found after extraction")

    # LAM assets (sample motions, parametric models, etc.)
    # Use official 3DAIGC/LAM-assets repo.
    # Extract to model_zoo/ so they survive the add_local_dir mount of assets/
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
        # Move extracted assets into model_zoo/ to avoid being
        # overwritten by the add_local_dir mount of assets/
        for subdir in ["sample_oac", "sample_motion"]:
            src = f"/root/LAM/assets/{subdir}"
            dst = f"/root/LAM/model_zoo/{subdir}"
            if os.path.isdir(src) and not os.path.exists(dst):
                subprocess.run(["cp", "-r", src, dst], check=True)
                print(f"  Copied assets/{subdir} -> model_zoo/{subdir}")

    # sample_oac (template_file.fbx, animation.glb) - separate download
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
        print("  Extracted sample_oac -> model_zoo/sample_oac")

    print("Model downloads complete.")


image = image.run_function(_download_missing_models)

# Mount local model files into the container (must be LAST in image build).
# modal serve must be run from the LAM repo root.
if _has_model_zoo:
    image = image.add_local_dir("./model_zoo", remote_path="/root/LAM/model_zoo")
if _has_assets:
    image = image.add_local_dir("./assets", remote_path="/root/LAM/assets")
# Mount tools/ so Blender subprocess scripts (convertFBX2GLB.py, generateVertexIndices.py)
# are available even if the git-cloned upstream version differs from our local copy.
if os.path.isdir("./tools"):
    image = image.add_local_dir("./tools", remote_path="/root/LAM/tools")


# ============================================================
# Pipeline Functions (run inside container)
# ============================================================

def _setup_model_paths():
    """Create symlinks to bridge local directory layout to what LAM code expects.

    The user may have all models under assets/ instead of model_zoo/.
    This function bridges the gap so LAM code (which expects model_zoo/) works.

    Called once at container startup (not during image build).
    """
    import subprocess

    model_zoo = "/root/LAM/model_zoo"
    assets = "/root/LAM/assets"

    # If model_zoo/ doesn't exist at all, symlink it to assets/
    # (user keeps everything under assets/ instead of model_zoo/)
    if not os.path.exists(model_zoo) and os.path.isdir(assets):
        os.symlink(assets, model_zoo)
        print(f"Symlink: model_zoo -> assets (unified layout)")
    elif os.path.isdir(model_zoo) and os.path.isdir(assets):
        # Both exist - bridge missing subdirectories from assets/ into model_zoo/
        for subdir in os.listdir(assets):
            src = os.path.join(assets, subdir)
            dst = os.path.join(model_zoo, subdir)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"Symlink: model_zoo/{subdir} -> assets/{subdir}")

    # Resolve human_parametric_models path
    hpm = os.path.join(model_zoo, "human_parametric_models")

    # If flame_assets/ has no flame/ subdirectory but files sit directly inside,
    # create flame/ as a symlink to flame_assets/ itself.
    if os.path.isdir(hpm):
        flame_subdir = os.path.join(hpm, "flame_assets", "flame")
        flame_assets_dir = os.path.join(hpm, "flame_assets")
        if os.path.isdir(flame_assets_dir) and not os.path.exists(flame_subdir):
            if os.path.isfile(os.path.join(flame_assets_dir, "flame2023.pkl")):
                os.symlink(flame_assets_dir, flame_subdir)
                print("Symlink: flame_assets/flame -> flame_assets/ (flat layout)")

        # VHAP expects flame_vhap/; LAM provides flame_assets/flame/
        flame_vhap = os.path.join(hpm, "flame_vhap")
        if not os.path.exists(flame_vhap):
            for candidate in [flame_subdir, flame_assets_dir]:
                if os.path.isdir(candidate):
                    os.symlink(candidate, flame_vhap)
                    print(f"Symlink: flame_vhap -> {os.path.basename(candidate)}")
                    break

    # Verify critical files
    print("\n=== Model file verification ===")
    search_dirs = [d for d in [model_zoo, assets] if os.path.isdir(d)]
    for name in [
        "flame2023.pkl", "FaceBoxesV2.pth", "68_keypoints_model.pkl",
        "vgg_heads_l.trcd", "stylematte_synth.pt",
        "model.safetensors",
        "template_file.fbx", "animation.glb",
    ]:
        found = False
        for d in search_dirs:
            result = subprocess.run(
                ["find", d, "-name", name],
                capture_output=True, text=True,
            )
            paths = result.stdout.strip()
            if paths:
                found = True
                for p in paths.split("\n"):
                    print(f"  OK: {p}")
        if not found:
            print(f"  MISSING: {name}")


def _init_lam_pipeline():
    """Initialize FLAME tracking and LAM model. Called once per container."""
    import torch
    import io
    from contextlib import redirect_stdout

    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")

    # Setup symlinks for model paths (runs once at container startup)
    _setup_model_paths()

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
        "APP_TYPE": "infer.lam",
        "NUMBA_THREADING_LAYER": "omp",
    })

    # Parse config
    from app_lam import parse_configs, _build_model
    cfg, _ = parse_configs()

    # Build and load LAM model - capture warnings from weight loading
    print("Loading LAM model...")
    build_log = io.StringIO()
    with redirect_stdout(build_log):
        lam = _build_model(cfg)
    build_output = build_log.getvalue()
    # Print captured output and flag any warnings
    print(build_output)
    warn_lines = [l for l in build_output.splitlines() if "WARN" in l]
    if warn_lines:
        print(f"\n!!! MODEL LOADING: {len(warn_lines)} weight warnings detected !!!")
        for w in warn_lines:
            print(f"  {w}")
    else:
        print("Model loading: all weights matched successfully.")

    lam.to("cuda")
    lam.eval()
    print("LAM model loaded.")

    # Store warnings for diagnostics
    lam._build_warnings = warn_lines

    # Initialize FLAME tracking (reused for both image and video frames)
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
    print("FLAME tracking initialized.")

    return cfg, lam, flametracking


def _track_video_to_motion(video_path, flametracking, working_dir, status_callback=None):
    """
    Process a custom motion video through VHAP FLAME tracking.

    Pipeline:
      video.mp4 → extract frames → per-frame preprocessing (face detect, matting, landmarks)
      → VHAP GlobalTracker → tracked_flame_params.npz → export as NeRF dataset
      → motion_dir/ with transforms.json + flame_param/*.npz

    Args:
        video_path: Path to the uploaded video file
        flametracking: FlameTrackingSingleImage instance (reuse detection models)
        working_dir: Temporary working directory
        status_callback: Optional function to report progress

    Returns:
        flame_params_dir: Path to flame_param/ directory for LAM inference
    """
    import cv2
    import numpy as np
    import torch
    import torchvision
    from pathlib import Path

    def report(msg):
        if status_callback:
            status_callback(msg)
        print(msg)

    # --- Step A: Extract frames from video ---
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample at 30fps or video's native fps, whichever is lower
    target_fps = min(30, video_fps) if video_fps > 0 else 30
    frame_interval = max(1, int(round(video_fps / target_fps)))
    max_frames = 300  # Cap at 10 seconds at 30fps to keep processing manageable

    report(f"  Video: {total_frames} frames at {video_fps:.1f}fps, sampling every {frame_interval} frame(s)")

    # --- Step B: Per-frame preprocessing ---
    report("  Processing frames (face detection, matting, landmarks)...")
    all_landmarks = []
    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        if processed_count >= max_frames:
            break

        # Convert to RGB tensor for VGGHead detection
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # [3, H, W]

        # Face detection via VGGHead
        try:
            from tools.flame_tracking_single_image import expand_bbox
            _, bbox, _ = flametracking.vgghead_encoder(frame_tensor, processed_count)
            if bbox is None:
                frame_idx += 1
                continue
        except Exception:
            frame_idx += 1
            continue

        # Expand bbox and crop
        bbox = expand_bbox(bbox, scale=1.65).long()
        cropped = torchvision.transforms.functional.crop(
            frame_tensor, top=bbox[1], left=bbox[0],
            height=bbox[3] - bbox[1], width=bbox[2] - bbox[0],
        )
        cropped = torchvision.transforms.functional.resize(
            cropped, (1024, 1024), antialias=True,
        )

        # Matting (background removal)
        cropped_matted, mask = flametracking.matting_engine(
            cropped / 255.0, return_type="matting", background_rgb=1.0,
        )
        cropped_matted = cropped_matted.cpu() * 255.0
        saved_image = np.round(
            cropped_matted.permute(1, 2, 0).numpy()
        ).astype(np.uint8)[:, :, ::-1]  # RGB -> BGR for cv2

        # Save image and alpha map
        fname = f"{processed_count:05d}.png"
        cv2.imwrite(os.path.join(images_dir, fname), saved_image)
        cv2.imwrite(
            os.path.join(alpha_dir, fname.replace(".png", ".jpg")),
            (np.ones_like(saved_image) * 255).astype(np.uint8),
        )

        # Landmark detection
        saved_image_rgb = saved_image[:, :, ::-1]  # BGR -> RGB for alignment
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

        if processed_count % 30 == 0:
            report(f"  Processed {processed_count} frames...")

        frame_idx += 1

    cap.release()
    torch.cuda.empty_cache()

    if processed_count == 0:
        raise RuntimeError("No valid face frames found in video")

    report(f"  Preprocessed {processed_count} frames")

    # Save landmarks in the format VHAP expects
    # landmarks.npz: bounding_box=[], face_landmark_2d=[N, num_lmks, 3]
    stacked_landmarks = np.stack(all_landmarks, axis=0)  # [N, 68, 3]
    np.savez(
        os.path.join(landmark_dir, "landmarks.npz"),
        bounding_box=[],
        face_landmark_2d=stacked_landmarks,
    )

    # --- Step C: VHAP Tracking ---
    report("  Running VHAP FLAME tracking (this may take several minutes)...")

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

    # Build VHAP config programmatically (no YAML dependency)
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
            root_folder=Path(frames_root),
            sequence=sequence_name,
            landmark_source="star",
        ),
        model=ModelConfig(),
        render=RenderConfig(),
        log=LogConfig(),
        exp=ExperimentConfig(
            output_folder=Path(tracking_output),
            photometric=True,
        ),
        lr=LearningRateConfig(),
        w=LossWeightConfig(),
        pipeline=pipeline,
    )

    tracker = GlobalTracker(vhap_cfg)
    tracker.optimize()
    torch.cuda.empty_cache()

    report("  VHAP tracking complete")

    # --- Step D: Export to NeRF dataset format ---
    report("  Exporting motion sequence...")

    from vhap.export_as_nerf_dataset import (
        NeRFDatasetWriter, TrackedFLAMEDatasetWriter, split_json, load_config,
    )

    export_dir = os.path.join(working_dir, "video_tracking", "export", sequence_name)
    export_path = Path(export_dir)

    src_folder, cfg_loaded = load_config(Path(tracking_output))

    nerf_writer = NeRFDatasetWriter(cfg_loaded.data, export_path, None, None, "white")
    nerf_writer.write()

    flame_writer = TrackedFLAMEDatasetWriter(
        cfg_loaded.model, src_folder, export_path, mode="param", epoch=-1,
    )
    flame_writer.write()

    split_json(export_path)

    flame_params_dir = os.path.join(export_dir, "flame_param")
    report(f"  Motion sequence exported: {len(os.listdir(flame_params_dir))} frames")

    return flame_params_dir


def _log_tensor(name, t):
    """Log tensor statistics for debugging."""
    import numpy as _np
    if t is None:
        return f"  {name}: None"
    if isinstance(t, _np.ndarray):
        return f"  {name}: shape={t.shape} dtype={t.dtype} min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}"
    return f"  {name}: shape={list(t.shape)} dtype={t.dtype} min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}"


def _generate_concierge_zip(image_path, video_path, cfg, lam, flametracking,
                            motion_name=None):
    """
    Full pipeline: image + video -> concierge.zip

    If video_path is provided, extract custom motion via VHAP tracking.
    Otherwise, use the selected sample motion (or first available).

    Yields (status_msg, zip_path, preview_video_path, tracked_image_path, preproc_image_path) tuples.
    """
    import torch
    import numpy as np
    import zipfile
    import shutil
    import tempfile
    from pathlib import Path
    from PIL import Image
    from glob import glob

    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

    working_dir = tempfile.mkdtemp(prefix="concierge_")
    base_iid = "concierge"
    diag = []  # Collect diagnostic messages

    # Report model loading warnings
    build_warnings = getattr(lam, "_build_warnings", [])
    if build_warnings:
        diag.append(f"[MODEL] {len(build_warnings)} weight warnings:")
        for w in build_warnings[:5]:
            diag.append(f"  {w}")
    else:
        diag.append("[MODEL] All weights loaded OK")

    try:
        # ============================================================
        # Step 0: Clean stale FLAME tracking data
        # ============================================================
        # FlameTrackingSingleImage uses fixed output dirs under output/tracking/.
        # Stale data from a previous run can corrupt results. Clean everything.
        tracking_root = os.path.join(os.getcwd(), "output", "tracking")
        if os.path.isdir(tracking_root):
            for subdir in ["preprocess", "tracking", "export"]:
                stale = os.path.join(tracking_root, subdir)
                if os.path.isdir(stale):
                    shutil.rmtree(stale)
                    print(f"[DIAG] Cleaned stale tracking dir: {stale}")
            diag.append("[CLEAN] Removed stale tracking data")
        else:
            diag.append("[CLEAN] No stale tracking data found")

        # ============================================================
        # Step 1: Source image FLAME tracking
        # ============================================================
        yield "Step 1: FLAME tracking on source image...", None, None, None, None

        image_raw = os.path.join(working_dir, "raw.png")
        with Image.open(image_path).convert("RGB") as img:
            diag.append(f"[INPUT] Image size: {img.size}, mode: {img.mode}")
            img.save(image_raw)

        ret = flametracking.preprocess(image_raw)
        assert ret == 0, "FLAME preprocess failed - could not detect face in image"
        ret = flametracking.optimize()
        assert ret == 0, "FLAME optimize failed"
        ret, output_dir = flametracking.export()
        assert ret == 0, "FLAME export failed"

        tracked_image = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        # Verify tracked outputs exist and log stats
        diag.append(f"[FLAME] output_dir: {output_dir}")
        diag.append(f"[FLAME] tracked_image exists: {os.path.isfile(tracked_image)}")
        diag.append(f"[FLAME] mask exists: {os.path.isfile(mask_path)}")
        flame_npz = os.path.join(output_dir, "canonical_flame_param.npz")
        if os.path.isfile(flame_npz):
            fp = np.load(flame_npz)
            diag.append(f"[FLAME] canonical_flame_param keys: {list(fp.keys())}")
            if "shape" in fp:
                sp = fp["shape"]
                diag.append(f"[FLAME] shape_param: shape={sp.shape} min={sp.min():.4f} max={sp.max():.4f} mean={sp.mean():.4f}")
                # Flag if shape params look suspicious (near-zero or extreme)
                if np.abs(sp).max() < 0.01:
                    diag.append("[FLAME] WARNING: shape params near-zero!")
                if np.abs(sp).max() > 10:
                    diag.append("[FLAME] WARNING: shape params extremely large!")
        else:
            diag.append(f"[FLAME] WARNING: canonical_flame_param.npz NOT FOUND at {flame_npz}")

        # Show tracked face so user can verify FLAME tracking quality
        yield f"Step 1 done: check tracked face -->", None, None, tracked_image, None

        # ============================================================
        # Step 2: Motion sequence preparation
        # ============================================================
        if video_path and os.path.isfile(video_path):
            # --- Custom video: VHAP tracking ---
            total_steps = 6
            yield f"Step 2/{total_steps}: Processing custom motion video (VHAP tracking)...", None, None, None, None

            def video_status(msg):
                # Forward sub-status through generator isn't easy,
                # so we just print to container logs
                print(f"  [Video Tracking] {msg}")

            flame_params_dir = _track_video_to_motion(
                video_path, flametracking, working_dir,
                status_callback=video_status,
            )
            motion_source = "custom video"
        else:
            # --- Sample motion ---
            total_steps = 5
            # Find available sample motions
            sample_motions = glob("./model_zoo/sample_motion/export/*/flame_param")
            if not sample_motions:
                raise RuntimeError(
                    "No motion sequences available. "
                    "Please upload a custom motion video."
                )

            # Use the motion selected by the user (if it matches)
            flame_params_dir = sample_motions[0]  # default fallback
            if motion_name:
                for sp in sample_motions:
                    if os.path.basename(os.path.dirname(sp)) == motion_name:
                        flame_params_dir = sp
                        break

            resolved_name = os.path.basename(os.path.dirname(flame_params_dir))
            motion_source = f"sample '{resolved_name}'"

        yield f"Step 3/{total_steps}: Preparing LAM inference (motion: {motion_source})...", None, None, None, None

        # ============================================================
        # Step 3: LAM inference
        # ============================================================
        source_size = cfg.source_size
        render_size = cfg.render_size

        image_tensor, _, _, shape_param = preprocess_image(
            tracked_image, mask_path=mask_path, intr=None,
            pad_ratio=0, bg_color=1.0, max_tgt_size=None,
            aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14,
            need_mask=True, get_shape_param=True,
        )

        # --- Diagnostics: preprocessed image tensor ---
        diag.append(f"[PREPROCESS] source_size={source_size}, render_size={render_size}")
        diag.append(_log_tensor("image_tensor", image_tensor))
        diag.append(_log_tensor("shape_param", shape_param))
        # Save preprocessed image for visual inspection
        preproc_vis_path = os.path.join(working_dir, "preprocessed_input.png")
        vis_img = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_img).save(preproc_vis_path)
        diag.append(f"[PREPROCESS] Saved preprocessed input to: {preproc_vis_path}")

        src = tracked_image.split("/")[-3]
        driven = flame_params_dir.split("/")[-2]
        motion_seq = prepare_motion_seqs(
            flame_params_dir, None, save_root=working_dir, fps=30,
            bg_color=1.0, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16,
            need_mask=False, vis_motion=False,
            shape_param=shape_param, test_sample=False,
            cross_id=False, src_driven=[src, driven],
        )

        # --- Diagnostics: motion sequence ---
        diag.append(f"[MOTION] flame_params_dir: {flame_params_dir}")
        diag.append(f"[MOTION] num_frames: {motion_seq['render_c2ws'].shape[1]}")
        diag.append(_log_tensor("render_c2ws", motion_seq["render_c2ws"]))
        diag.append(_log_tensor("render_intrs", motion_seq["render_intrs"]))
        for pk, pv in motion_seq["flame_params"].items():
            diag.append(_log_tensor(f"flame.{pk}", pv))

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
                flame_params={
                    k: v.to(device) for k, v in motion_seq["flame_params"].items()
                },
            )

        # --- Diagnostics: model output ---
        diag.append(_log_tensor("comp_rgb", res["comp_rgb"]))
        diag.append(_log_tensor("comp_mask", res["comp_mask"]))

        # ============================================================
        # Step 4: Generate GLB + ZIP
        # ============================================================
        yield f"Step 5/{total_steps}: Generating 3D avatar (Blender GLB)...", None, None, None, preproc_vis_path

        oac_dir = os.path.join(working_dir, "oac_export", base_iid)
        os.makedirs(oac_dir, exist_ok=True)

        # Save shaped mesh + Gaussian offset
        saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir,
        )
        assert os.path.isfile(saved_head_path), f"save_shaped_mesh failed: {saved_head_path}"

        res["cano_gs_lst"][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True,
        )

        # Generate GLB via Blender (with per-request temp dir to avoid conflicts)
        from tools.generateARKITGLBWithBlender import (
            update_flame_shape,
            convert_ascii_to_binary,
            convert_with_blender,
            gen_vertex_order_with_blender,
        )

        skin_glb_path = Path(os.path.join(oac_dir, "skin.glb"))
        vertex_order_path = Path(os.path.join(oac_dir, "vertex_order.json"))
        template_fbx = Path("./model_zoo/sample_oac/template_file.fbx")
        blender_exec = Path("/usr/local/bin/blender")

        # Validate prerequisites before starting the pipeline
        diag.append(f"[GLB] CWD={os.getcwd()}")
        diag.append(f"[GLB] blender exists: {blender_exec.exists()}")
        diag.append(f"[GLB] template_fbx exists: {template_fbx.exists()}")
        diag.append(f"[GLB] saved_head_path exists: {os.path.isfile(saved_head_path)}")
        script_dir = Path(__file__).resolve().parent / "tools"
        diag.append(f"[GLB] convertFBX2GLB.py exists: {(script_dir / 'convertFBX2GLB.py').exists()}")
        # Also check fallback path under CWD
        diag.append(f"[GLB] tools/convertFBX2GLB.py (CWD): {Path('tools/convertFBX2GLB.py').exists()}")

        # Use working_dir for temp files (avoid CWD collision across requests)
        temp_ascii = Path(os.path.join(working_dir, "temp_ascii.fbx"))
        temp_binary = Path(os.path.join(working_dir, "temp_bin.fbx"))

        try:
            update_flame_shape(Path(saved_head_path), temp_ascii, template_fbx)
            assert temp_ascii.exists(), f"update_flame_shape produced no output: {temp_ascii}"
            diag.append(f"[GLB] ASCII FBX size: {temp_ascii.stat().st_size} bytes")

            convert_ascii_to_binary(temp_ascii, temp_binary)
            assert temp_binary.exists(), f"convert_ascii_to_binary produced no output: {temp_binary}"
            diag.append(f"[GLB] Binary FBX size: {temp_binary.stat().st_size} bytes")

            convert_with_blender(temp_binary, skin_glb_path, blender_exec)
            diag.append(f"[GLB] skin.glb size: {skin_glb_path.stat().st_size} bytes")

            gen_vertex_order_with_blender(
                Path(saved_head_path), vertex_order_path, blender_exec,
            )
            diag.append(f"[GLB] vertex_order.json size: {vertex_order_path.stat().st_size} bytes")
        finally:
            for f in [temp_ascii, temp_binary]:
                if f.exists():
                    f.unlink()

        # Copy template animation
        shutil.copy(
            src="./model_zoo/sample_oac/animation.glb",
            dst=os.path.join(oac_dir, "animation.glb"),
        )

        # Clean up intermediate mesh (nature.obj)
        if os.path.exists(saved_head_path):
            os.remove(saved_head_path)

        # Verify all required files before creating ZIP
        required_files = ["offset.ply", "skin.glb", "vertex_order.json", "animation.glb"]
        missing = [f for f in required_files if not os.path.isfile(os.path.join(oac_dir, f))]
        if missing:
            raise RuntimeError(f"OAC export incomplete - missing: {', '.join(missing)}")

        # ============================================================
        # Step 5: Create ZIP + preview
        # ============================================================
        step_label = f"Step {total_steps}/{total_steps}"
        yield f"{step_label}: Creating concierge.zip...", None, None, None, preproc_vis_path

        output_zip = os.path.join(working_dir, "concierge.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(oac_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, os.path.dirname(oac_dir))
                    zf.write(fpath, arcname)

        # Generate preview video
        preview_path = os.path.join(working_dir, "preview.mp4")
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        from app_lam import save_images2video
        save_images2video(rgb, preview_path, 30)

        # Add audio if available (from custom video)
        final_preview = preview_path
        if video_path and os.path.isfile(video_path):
            try:
                from app_lam import add_audio_to_video
                preview_with_audio = os.path.join(working_dir, "preview_audio.mp4")
                add_audio_to_video(preview_path, preview_with_audio, video_path)
                final_preview = preview_with_audio
            except Exception:
                pass  # Fallback to video without audio

        zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
        num_motion_frames = len(os.listdir(flame_params_dir))

        # Save diagnostics to file for download
        diag_path = os.path.join(working_dir, "diagnostics.txt")
        with open(diag_path, "w") as f:
            f.write("\n".join(diag))
        print("\n=== DIAGNOSTICS ===\n" + "\n".join(diag) + "\n=== END ===\n")

        yield (
            f"concierge.zip generated ({zip_size_mb:.1f} MB) | "
            f"Motion: {motion_source} ({num_motion_frames} frames)",
            output_zip,
            final_preview,
            None,
            preproc_vis_path,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Include diagnostics in error message
        diag_summary = "\n".join(diag[-10:]) if diag else "No diagnostics"
        print("\n=== DIAGNOSTICS (error) ===\n" + "\n".join(diag) + "\n=== END ===\n")
        yield f"Error: {str(e)}\n\nDiagnostics:\n{diag_summary}", None, None, None, None


# ============================================================
# Gradio UI + Modal ASGI App
# ============================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
)
# Gradio needs all requests (uploads, queue, SSE) on the SAME container.
# Default concurrency is 1, which forces Modal to spin up new containers
# per request, breaking Gradio's in-memory file storage and queue state.
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    """Gradio UI served via ASGI (no subprocess, no patching)."""
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from glob import glob

    # Monkey-patch gradio_client bug: additionalProperties can be a bool,
    # but _json_schema_to_python_type assumes it's always a dict/schema.
    import gradio_client.utils as _gc_utils
    _orig_jst = _gc_utils._json_schema_to_python_type
    def _safe_jst(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_jst(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_jst

    # --- Initialize pipeline (once per container) ---
    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")

    # Monkey-patch torch.utils.cpp_extension.load to inject
    # -Wno-c++11-narrowing, fixing nvdiffrast JIT build with clang.
    import torch.utils.cpp_extension as _cext
    _orig_load = _cext.load
    def _patched_load(*args, **kwargs):
        cflags = list(kwargs.get("extra_cflags", []) or [])
        if "-Wno-c++11-narrowing" not in cflags:
            cflags.append("-Wno-c++11-narrowing")
        kwargs["extra_cflags"] = cflags
        return _orig_load(*args, **kwargs)
    _cext.load = _patched_load

    # Suppress torch._dynamo errors so it falls back to eager mode
    # instead of crashing on unsupported operations.
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    print("Initializing LAM pipeline...")
    cfg, lam, flametracking = _init_lam_pipeline()
    print("Pipeline ready. Starting Gradio UI...")

    # Discover available sample motions (if any)
    sample_motions = sorted(glob("./model_zoo/sample_motion/export/*/*.mp4"))

    # Track latest ZIP for direct download endpoint
    _latest_zip = {"path": None}

    # --- Processing function ---
    def process(image_path, video_path, motion_choice):
        if image_path is None:
            yield "Error: Please upload a face image", None, None, None, None
            return

        # Determine motion source
        effective_video = None
        selected_motion = None
        if motion_choice == "custom" and video_path:
            effective_video = video_path
        elif motion_choice and motion_choice != "custom":
            # Using a sample motion - pass name so it's correctly resolved
            effective_video = None
            selected_motion = motion_choice

        for status, zip_path, preview, tracked_img, preproc_img in _generate_concierge_zip(
            image_path, effective_video, cfg, lam, flametracking,
            motion_name=selected_motion,
        ):
            if zip_path:
                _latest_zip["path"] = zip_path
            yield status, zip_path, preview, tracked_img, preproc_img

    # --- Build Gradio Blocks ---
    with gr.Blocks(
        title="Concierge ZIP Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; font-size: 0.95em; margin-bottom: 1.5em; }
        footer { display: none !important; }
        .tip-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px;
                    padding: 12px 16px; margin-top: 8px; font-size: 0.9em; color: #0369a1; }
        """,
    ) as demo:
        gr.HTML('<h1 class="main-title">Concierge ZIP Generator</h1>')
        gr.HTML(
            '<p class="subtitle">'
            "Upload your face image + custom motion video to generate "
            "a high-quality concierge.zip for LAMAvatar"
            "</p>"
        )

        with gr.Row():
            # ---- Left: Inputs ----
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="1. Source Face Image",
                    type="filepath",
                    height=300,
                )

                motion_choice = gr.Radio(
                    label="2. Motion Source",
                    choices=["custom"] + [
                        os.path.basename(os.path.dirname(m))
                        for m in sample_motions
                    ] if sample_motions else ["custom"],
                    value="custom",
                    info="Select 'custom' to upload your own video, or choose a sample",
                )

                input_video = gr.Video(
                    label="3. Custom Motion Video",
                    height=200,
                )

                gr.HTML(
                    '<div class="tip-box">'
                    "<b>IMPORTANT - input image requirements:</b><br>"
                    "- Must be a <b>real photograph</b> (not illustration/AI art/anime)<br>"
                    "- Front-facing, good lighting, neutral expression<br>"
                    "- FLAME face tracking only works on real human faces<br>"
                    "<br>"
                    "<b>Motion video tips:</b><br>"
                    "- Clear face, consistent lighting, 3-10 seconds<br>"
                    "- The motion video's expressions drive the avatar animation"
                    "</div>"
                )

                generate_btn = gr.Button(
                    "Generate concierge.zip",
                    variant="primary",
                    size="lg",
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload image + video, then click Generate...",
                    lines=2,
                )

            # ---- Right: Outputs ----
            with gr.Column(scale=1):
                with gr.Row():
                    tracked_face = gr.Image(
                        label="Tracked Face (FLAME output)",
                        height=200,
                    )
                    preproc_image = gr.Image(
                        label="Model Input (what LAM actually sees)",
                        height=200,
                    )
                preview_video = gr.Video(
                    label="Avatar Preview",
                    height=350,
                    autoplay=True,
                )
                output_file = gr.File(
                    label="Download concierge.zip",
                )
                gr.Markdown(
                    "If the download button above doesn't work, use the "
                    "direct link: **[/download-zip](/download-zip)**"
                )
                gr.Markdown(
                    "**Usage:** Place the downloaded `concierge.zip` at "
                    "`gourmet-sp/public/avatar/concierge.zip` for the "
                    "LAMAvatar component."
                )

        # Wire up the generate button
        generate_btn.click(
            fn=process,
            inputs=[input_image, input_video, motion_choice],
            outputs=[status_text, output_file, preview_video, tracked_face, preproc_image],
        )

    # --- Mount Gradio on FastAPI (proper ASGI serving) ---
    web_app = FastAPI()

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": "LAM-20K", "blender": "4.2.0"}

    @web_app.get("/download-zip")
    async def download_zip():
        p = _latest_zip.get("path")
        if p and os.path.isfile(p):
            return FileResponse(
                p, media_type="application/zip", filename="concierge.zip",
            )
        return {"error": "No ZIP available yet. Run Generate first."}

    return gr.mount_gradio_app(web_app, demo, path="/")


# ============================================================
# Local entry point
# ============================================================
if __name__ == "__main__":
    print("Concierge ZIP Generator - Modal Deployment")
    print("=" * 50)
    print()
    print("Usage:")
    print("  modal serve concierge_modal.py   # Dev mode (hot reload)")
    print("  modal deploy concierge_modal.py  # Production")
    print()
    print("The Gradio UI will be available at the URL shown by Modal.")
    print()
    print("Pipeline:")
    print("  1. Upload source face image")
    print("  2. Upload custom motion video (or select sample)")
    print("  3. Click Generate")
    print("  4. Download concierge.zip")
