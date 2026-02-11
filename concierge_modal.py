"""
concierge_modal.py - Concierge ZIP Generator on Modal
=====================================================

Gradio UI for generating concierge.zip (3D avatar data for LAMAvatar/gourmet-sp).

Usage:
  modal serve concierge_modal.py   # Development mode (hot reload)
  modal deploy concierge_modal.py  # Production deployment

Prerequisites:
  - ./assets/human_parametric_models/ directory with FLAME model assets
  - Modal account with GPU access (A10G)

Architecture:
  - @modal.asgi_app() + Gradio 4.x (no subprocess, no patching)
  - Direct Python integration with LAM pipeline
  - Blender 4.2 for GLB generation (OpenAvatarChat format)
"""

import os
import sys
import modal

app = modal.App("concierge-zip-generator")

# ============================================================
# Local asset check
# ============================================================
REQUIRED_ASSET = "./assets/human_parametric_models/flame_assets/flame/flame2023.pkl"
if __name__ == "__main__":
    if not os.path.exists(REQUIRED_ASSET):
        print(f"ERROR: Required FLAME asset not found: {REQUIRED_ASSET}")
        print("Please ensure ./assets/human_parametric_models/ exists with FLAME models.")
        sys.exit(1)

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
        "gradio>=4.0,<5.0",
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
        "numpy==1.23.5",
    )
    # More CUDA extensions
    .run_commands(
        "pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git --no-build-isolation",
        "pip install git+https://github.com/ShenhanQian/nvdiffrast.git@backface-culling --no-build-isolation",
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


# Download model checkpoints during image build
def _download_models():
    from huggingface_hub import snapshot_download
    print("Downloading LAM-20K checkpoints from HuggingFace...")
    snapshot_download(
        repo_id="3DAIGC/LAM-20K",
        local_dir="/root/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500",
        local_dir_use_symlinks=False,
    )
    print("Checkpoints downloaded successfully.")


image = (
    image
    .run_function(_download_models)
    # Copy local FLAME assets into the container
    .add_local_dir("./assets", remote_path="/root/LAM/model_zoo", copy=True)
)


# ============================================================
# Pipeline Functions (run inside container)
# ============================================================

def _init_lam_pipeline():
    """Initialize FLAME tracking and LAM model. Called once per container."""
    import torch
    from omegaconf import OmegaConf

    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")

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

    # Build and load LAM model
    print("Loading LAM model...")
    lam = _build_model(cfg)
    lam.to("cuda")
    lam.eval()
    print("LAM model loaded.")

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
    print("FLAME tracking initialized.")

    return cfg, lam, flametracking


def _generate_concierge_zip(image_path: str, cfg, lam, flametracking):
    """
    Full pipeline: image -> FLAME tracking -> LAM inference -> Blender GLB -> ZIP.

    Returns (status_msg, zip_path, preview_video_path).
    """
    import torch
    import numpy as np
    import zipfile
    import shutil
    import tempfile
    from pathlib import Path
    from PIL import Image

    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

    working_dir = tempfile.mkdtemp(prefix="concierge_")
    base_iid = "concierge"
    # Use first available motion sequence
    default_motion = "nice"
    flame_params_dir = f"./assets/sample_motion/export/{default_motion}/flame_param"

    try:
        # --- Step 1: Preprocess image ---
        yield "Step 1/5: Preprocessing image...", None, None

        image_raw = os.path.join(working_dir, "raw.png")
        with Image.open(image_path).convert("RGB") as img:
            img.save(image_raw)

        # FLAME tracking
        yield "Step 2/5: FLAME face tracking...", None, None
        ret = flametracking.preprocess(image_raw)
        assert ret == 0, "FLAME preprocess failed"
        ret = flametracking.optimize()
        assert ret == 0, "FLAME optimize failed"
        ret, output_dir = flametracking.export()
        assert ret == 0, "FLAME export failed"

        tracked_image = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        # --- Step 2: Prepare inputs ---
        yield "Step 3/5: LAM inference...", None, None

        source_size = cfg.source_size
        render_size = cfg.render_size

        image_tensor, _, _, shape_param = preprocess_image(
            tracked_image, mask_path=mask_path, intr=None,
            pad_ratio=0, bg_color=1.0, max_tgt_size=None,
            aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14,
            need_mask=True, get_shape_param=True,
        )

        # Prepare motion sequence
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

        # --- Step 3: LAM inference ---
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

        # --- Step 4: Generate GLB + export ---
        yield "Step 4/5: Generating 3D avatar (Blender GLB)...", None, None

        oac_dir = os.path.join(working_dir, "oac_export", base_iid)
        os.makedirs(oac_dir, exist_ok=True)

        # Save shaped mesh + Gaussian offset
        saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir
        )
        res["cano_gs_lst"][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True
        )

        # Generate GLB via Blender
        from tools.generateARKITGLBWithBlender import generate_glb
        generate_glb(
            input_mesh=Path(saved_head_path),
            template_fbx=Path("./assets/sample_oac/template_file.fbx"),
            output_glb=Path(os.path.join(oac_dir, "skin.glb")),
            blender_exec=Path("/usr/local/bin/blender"),
        )

        # Copy template animation
        shutil.copy(
            src="./assets/sample_oac/animation.glb",
            dst=os.path.join(oac_dir, "animation.glb"),
        )

        # Clean up intermediate mesh
        if os.path.exists(saved_head_path):
            os.remove(saved_head_path)

        # --- Step 5: Create ZIP ---
        yield "Step 5/5: Creating concierge.zip...", None, None

        output_zip = os.path.join(working_dir, "concierge.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(oac_dir):
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

        zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
        yield (
            f"concierge.zip generated successfully ({zip_size_mb:.1f} MB)",
            output_zip,
            preview_path,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"Error: {str(e)}", None, None


# ============================================================
# Gradio UI + Modal ASGI App
# ============================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    allow_concurrent_inputs=1,
)
@modal.asgi_app()
def web():
    """Gradio UI served via ASGI (no subprocess, no patching)."""
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    # --- Initialize pipeline (once per container) ---
    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")

    print("Initializing LAM pipeline...")
    cfg, lam, flametracking = _init_lam_pipeline()
    print("Pipeline ready. Starting Gradio UI...")

    # --- Processing function ---
    def process_image(image_path):
        if image_path is None:
            yield "Error: No image uploaded", None, None
            return
        yield from _generate_concierge_zip(image_path, cfg, lam, flametracking)

    # --- Build Gradio Blocks ---
    with gr.Blocks(
        title="Concierge ZIP Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.HTML('<h1 class="main-title">Concierge ZIP Generator</h1>')
        gr.HTML(
            '<p class="subtitle">'
            "Upload a face image to generate concierge.zip "
            "(3D Gaussian avatar for LAMAvatar)"
            "</p>"
        )

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Face Image",
                    type="filepath",
                    height=400,
                )
                generate_btn = gr.Button(
                    "Generate concierge.zip",
                    variant="primary",
                    size="lg",
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload an image and click Generate...",
                )

            # Right column: Output
            with gr.Column(scale=1):
                preview_video = gr.Video(
                    label="Avatar Preview",
                    height=400,
                    autoplay=True,
                )
                output_file = gr.File(
                    label="Download concierge.zip",
                )

        gr.Markdown(
            "---\n"
            "**How it works:** Image → FLAME Tracking → LAM Inference → "
            "Blender GLB Export → concierge.zip\n\n"
            "The generated zip can be placed at `/avatar/concierge.zip` "
            "in gourmet-sp for the LAMAvatar component."
        )

        # Wire up the generate button
        generate_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[status_text, output_file, preview_video],
        )

    # --- Mount Gradio on FastAPI (proper ASGI serving) ---
    web_app = FastAPI()

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": "LAM-20K", "blender": "4.2.0"}

    return gr.mount_gradio_app(web_app, demo, path="/")


# ============================================================
# Local entry point (for testing without Modal)
# ============================================================
if __name__ == "__main__":
    print("This script is designed to run on Modal.")
    print("  modal serve concierge_modal.py   # Dev mode")
    print("  modal deploy concierge_modal.py  # Production")
    print()
    print("To test locally (requires GPU + all dependencies):")
    print("  python -c 'from concierge_modal import web; web()'")
