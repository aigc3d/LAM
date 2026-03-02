"""Diagnostic: list volume contents to understand directory structure."""
import os
import modal

app = modal.App("lam-diag")

storage_vol = modal.Volume.from_name("lam-storage", create_if_missing=False)
STORAGE_VOL_PATH = "/vol/lam-storage"


@app.function(
    volumes={STORAGE_VOL_PATH: storage_vol},
    timeout=120,
)
def list_volume():
    vol_root = STORAGE_VOL_PATH

    print("=" * 80)
    print(f"Volume root: {vol_root}")
    print("=" * 80)

    # Top-level
    for entry in sorted(os.listdir(vol_root)):
        full = os.path.join(vol_root, entry)
        kind = "DIR" if os.path.isdir(full) else "FILE"
        print(f"  {kind}: {entry}")

    # LAM subdir (depth 3)
    lam_root = os.path.join(vol_root, "LAM")
    if os.path.isdir(lam_root):
        print(f"\n{'=' * 80}")
        print(f"LAM directory tree (depth 3):")
        print(f"{'=' * 80}")
        for root, dirs, files in os.walk(lam_root):
            depth = root.replace(lam_root, "").count(os.sep)
            if depth >= 3:
                dirs.clear()
                continue
            indent = "  " * (depth + 1)
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files)[:10]:
                print(f"{indent}  {f}")
            if len(files) > 10:
                print(f"{indent}  ... and {len(files) - 10} more files")

    # Specifically search for flame2023.pkl
    print(f"\n{'=' * 80}")
    print("Searching for flame2023.pkl...")
    print("=" * 80)
    import subprocess
    result = subprocess.run(
        ["find", vol_root, "-name", "flame2023.pkl"],
        capture_output=True, text=True, timeout=30,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    else:
        print("  NOT FOUND anywhere on volume!")

    # Also search for key files
    for name in ["flame2023.pkl", "model.safetensors", "config.json",
                 "FaceBoxesV2.pth", "flame_param", "template_file.fbx",
                 "animation.glb"]:
        result = subprocess.run(
            ["find", vol_root, "-name", name],
            capture_output=True, text=True, timeout=30,
        )
        found = result.stdout.strip()
        if found:
            for line in found.split("\n")[:3]:
                print(f"  {name}: {line}")
        else:
            print(f"  {name}: NOT FOUND")


@app.local_entrypoint()
def main():
    list_volume.remote()
