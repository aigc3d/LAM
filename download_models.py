"""Download model weights for LAM concierge.

Run during Docker build to cache weights in the image layer.
Extracted from concierge_modal.py's _download_missing_models().
"""

import os
import subprocess
from huggingface_hub import snapshot_download, hf_hub_download

os.chdir("/app/LAM")

# 1. LAM-20K model weights
target = "/app/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500"
if not os.path.isfile(os.path.join(target, "model.safetensors")):
    print("[1/5] Downloading LAM-20K model weights...")
    snapshot_download(
        repo_id="3DAIGC/LAM-20K",
        local_dir=target,
        local_dir_use_symlinks=False,
    )

# 2. FLAME tracking models
if not os.path.isfile("/app/LAM/model_zoo/flame_tracking_models/FaceBoxesV2.pth"):
    print("[2/5] Downloading FLAME tracking models...")
    hf_hub_download(
        repo_id="3DAIGC/LAM-assets",
        repo_type="model",
        filename="thirdparty_models.tar",
        local_dir="/app/LAM/",
    )
    subprocess.run(
        "tar -xf thirdparty_models.tar && rm thirdparty_models.tar",
        shell=True, cwd="/app/LAM", check=True,
    )

# 3. FLAME parametric model (flame2023.pkl etc.)
if not os.path.isfile("/app/LAM/model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl"):
    print("[3/5] Downloading FLAME parametric model...")
    hf_hub_download(
        repo_id="3DAIGC/LAM-assets",
        repo_type="model",
        filename="LAM_human_model.tar",
        local_dir="/app/LAM/",
    )
    subprocess.run(
        "tar -xf LAM_human_model.tar && rm LAM_human_model.tar",
        shell=True, cwd="/app/LAM", check=True,
    )
    # Copy to model_zoo/ (LAM code expects this path)
    src = "/app/LAM/assets/human_parametric_models"
    dst = "/app/LAM/model_zoo/human_parametric_models"
    if os.path.isdir(src) and not os.path.exists(dst):
        subprocess.run(["cp", "-r", src, dst], check=True)
        print("  Copied assets/human_parametric_models -> model_zoo/")

# 4. LAM assets (sample motions, sample_oac)
if not os.path.isfile("/app/LAM/model_zoo/sample_motion/export/talk/flame_param/00000.npz"):
    print("[4/5] Downloading LAM assets (sample motions)...")
    hf_hub_download(
        repo_id="3DAIGC/LAM-assets",
        repo_type="model",
        filename="LAM_assets.tar",
        local_dir="/app/LAM/",
    )
    subprocess.run(
        "tar -xf LAM_assets.tar && rm LAM_assets.tar",
        shell=True, cwd="/app/LAM", check=True,
    )
    for subdir in ["sample_oac", "sample_motion"]:
        src = f"/app/LAM/assets/{subdir}"
        dst = f"/app/LAM/model_zoo/{subdir}"
        if os.path.isdir(src) and not os.path.exists(dst):
            subprocess.run(["cp", "-r", src, dst], check=True)

# 5. sample_oac templates
if not os.path.isfile("/app/LAM/model_zoo/sample_oac/template_file.fbx"):
    print("[5/5] Downloading sample_oac (FBX/GLB templates)...")
    subprocess.run(
        "wget -q https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar"
        " -O /app/LAM/sample_oac.tar",
        shell=True, check=True,
    )
    subprocess.run(
        "mkdir -p /app/LAM/model_zoo/sample_oac && "
        "tar -xf /app/LAM/sample_oac.tar -C /app/LAM/model_zoo/ && "
        "rm /app/LAM/sample_oac.tar",
        shell=True, check=True,
    )

print("All model downloads complete.")
