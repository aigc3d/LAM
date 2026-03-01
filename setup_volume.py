"""
setup_volume.py — Download and extract all required LAM model files into Modal volume.

Replicates the official app.py launch_pretrained() function:
  1. FLAME tracking models       (yuandong513/flametracking_model)
  2. Human model files (FLAME)   (3DAIGC/LAM-assets → LAM_human_model.tar)
  3. LAM-20K weights             (3DAIGC/LAM-20K → config.json + model.safetensors)
  4. LAM assets (motions, OAC)   (3DAIGC/LAM-assets → LAM_assets.tar)

Also extracts any unextracted tar files already on the volume (tmp_assets/).

Usage:
  modal run setup_volume.py
"""
import os
import modal

app = modal.App("lam-setup-volume")

# Use a basic image with huggingface_hub for downloads
setup_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub")
)

storage_vol = modal.Volume.from_name("lam-storage", create_if_missing=True)
STORAGE_VOL_PATH = "/vol/lam-storage"


@app.function(
    image=setup_image,
    volumes={STORAGE_VOL_PATH: storage_vol},
    timeout=3600,
)
def setup():
    from huggingface_hub import hf_hub_download
    import subprocess

    lam_root = os.path.join(STORAGE_VOL_PATH, "LAM")
    os.makedirs(lam_root, exist_ok=True)
    os.chdir(lam_root)

    print("=" * 80)
    print("LAM Volume Setup — replicating official launch_pretrained()")
    print(f"Working directory: {lam_root}")
    print("=" * 80)

    # ================================================================
    # 1. FLAME tracking pretrained models
    # ================================================================
    flame_marker = os.path.join(lam_root, "model_zoo", "flame_tracking_models", "FaceBoxesV2.pth")
    if os.path.isfile(flame_marker):
        print("\n[1/4] FLAME tracking models: already present, skipping.")
    else:
        print("\n[1/4] Downloading FLAME tracking models...")
        hf_hub_download(
            repo_id="yuandong513/flametracking_model",
            repo_type="model",
            filename="pretrain_model.tar",
            local_dir="./",
        )
        print("  Extracting pretrain_model.tar...")
        subprocess.run(["tar", "-xf", "pretrain_model.tar"], check=True)
        os.remove("pretrain_model.tar")
        print("  Done. Listing extracted structure:")
        # Show what was extracted
        for root, dirs, files in os.walk("./model_zoo/flame_tracking_models"):
            for f in files:
                print(f"    {os.path.join(root, f)}")
            break  # only top level

    # ================================================================
    # 2. Human model files (FLAME head model — flame2023.pkl etc.)
    # ================================================================
    flame2023_candidates = [
        os.path.join(lam_root, "pretrained_models", "human_model_files", "flame_assets", "flame", "flame2023.pkl"),
        os.path.join(lam_root, "model_zoo", "human_parametric_models", "flame_assets", "flame", "flame2023.pkl"),
    ]
    found_flame = any(os.path.isfile(c) for c in flame2023_candidates)

    if found_flame:
        print("\n[2/4] Human model files (flame2023.pkl): already present, skipping.")
    else:
        print("\n[2/4] Downloading human model files (LAM_human_model.tar)...")
        hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type="model",
            filename="LAM_human_model.tar",
            local_dir="./",
        )
        print("  Extracting LAM_human_model.tar...")
        subprocess.run(["tar", "-xf", "LAM_human_model.tar"], check=True)
        os.remove("LAM_human_model.tar")

        # Show what was extracted (find flame2023.pkl)
        result = subprocess.run(
            ["find", ".", "-name", "flame2023.pkl"],
            capture_output=True, text=True,
        )
        print(f"  flame2023.pkl extracted to: {result.stdout.strip()}")

    # ================================================================
    # 3. LAM-20K model weights (config.json + model.safetensors)
    # ================================================================
    # Official path: ./exps/releases/lam/lam-20k/step_045500/
    # Our path:      ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
    weights_dir = os.path.join(lam_root, "model_zoo", "lam_models", "releases", "lam", "lam-20k", "step_045500")
    weights_safetensors = os.path.join(weights_dir, "model.safetensors")

    # Also check official path
    official_weights_dir = os.path.join(lam_root, "exps", "releases", "lam", "lam-20k", "step_045500")
    official_safetensors = os.path.join(official_weights_dir, "model.safetensors")

    if os.path.isfile(weights_safetensors) or os.path.isfile(official_safetensors):
        print("\n[3/4] LAM-20K weights: already present, skipping.")
    else:
        print("\n[3/4] Downloading LAM-20K model weights...")
        os.makedirs(weights_dir, exist_ok=True)

        for fname in ["config.json", "model.safetensors", "README.md"]:
            print(f"  Downloading {fname}...")
            hf_hub_download(
                repo_id="3DAIGC/LAM-20K",
                repo_type="model",
                filename=fname,
                local_dir=weights_dir,
            )

        # Verify
        if os.path.isfile(os.path.join(weights_dir, "model.safetensors")):
            size_mb = os.path.getsize(os.path.join(weights_dir, "model.safetensors")) / (1024 * 1024)
            print(f"  model.safetensors: {size_mb:.0f} MB")
        else:
            print("  WARNING: model.safetensors not found after download!")

    # ================================================================
    # 4. LAM assets (sample motions, OAC templates)
    # ================================================================
    motion_marker = os.path.join(lam_root, "assets", "sample_motion", "export", "GEM", "flame_param")
    oac_marker = os.path.join(lam_root, "assets", "sample_oac", "template_file.fbx")

    if os.path.isdir(motion_marker) and os.path.isfile(oac_marker):
        print("\n[4/4] LAM assets (motions + OAC): already present, skipping.")
    else:
        print("\n[4/4] Downloading LAM assets...")

        # Check if LAM_assets.tar already exists in tmp_assets
        tar_path = os.path.join(lam_root, "tmp_assets", "LAM_assets.tar")
        if os.path.isfile(tar_path):
            print(f"  Found existing tar: {tar_path}")
        else:
            print("  Downloading LAM_assets.tar from HuggingFace...")
            hf_hub_download(
                repo_id="3DAIGC/LAM-assets",
                repo_type="model",
                filename="LAM_assets.tar",
                local_dir="./",
            )
            tar_path = os.path.join(lam_root, "LAM_assets.tar")

        print(f"  Extracting {tar_path}...")
        subprocess.run(["tar", "-xf", tar_path], check=True)
        # Don't delete — might be in tmp_assets shared location

        # Show motion dirs
        motion_export = os.path.join(lam_root, "assets", "sample_motion", "export")
        if os.path.isdir(motion_export):
            motions = os.listdir(motion_export)
            print(f"  Motions extracted: {motions}")
        else:
            print("  WARNING: assets/sample_motion/export not found after extraction!")

    # ================================================================
    # Also extract thirdparty_models.tar if present
    # ================================================================
    thirdparty_tar = os.path.join(lam_root, "tmp_assets", "thirdparty_models.tar")
    if os.path.isfile(thirdparty_tar):
        print("\n[Extra] Extracting thirdparty_models.tar...")
        subprocess.run(["tar", "-xf", thirdparty_tar], check=True)
        print("  Done.")

    # ================================================================
    # Final verification
    # ================================================================
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    checks = {
        "flame2023.pkl": [
            "pretrained_models/human_model_files/flame_assets/flame/flame2023.pkl",
            "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl",
        ],
        "model.safetensors": [
            "model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors",
            "exps/releases/lam/lam-20k/step_045500/model.safetensors",
        ],
        "FaceBoxesV2.pth": [
            "model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        ],
        "GEM motion": [
            "assets/sample_motion/export/GEM/flame_param",
        ],
        "template_file.fbx": [
            "assets/sample_oac/template_file.fbx",
        ],
    }

    all_ok = True
    for name, paths in checks.items():
        found = False
        for p in paths:
            full = os.path.join(lam_root, p)
            if os.path.exists(full):
                size_info = ""
                if os.path.isfile(full):
                    size_mb = os.path.getsize(full) / (1024 * 1024)
                    size_info = f" ({size_mb:.1f} MB)"
                print(f"  OK: {name} -> {p}{size_info}")
                found = True
                break
        if not found:
            print(f"  MISSING: {name}")
            all_ok = False

    # Commit volume changes
    storage_vol.commit()
    print(f"\nVolume committed.")

    if all_ok:
        print("\nAll required files are present. Ready to run LAM pipeline!")
    else:
        print("\nSome files are still missing. Check the output above.")

    return all_ok


@app.local_entrypoint()
def main():
    result = setup.remote()
    if result:
        print("\nSetup complete! You can now run:")
        print("  modal run lam_avatar_batch.py --image-path ./input/face.jpg")
    else:
        print("\nSetup incomplete. Some files could not be downloaded.")
