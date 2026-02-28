"""
lam_avatar_batch.py - LAM Avatar Batch Generator (No UI)
=========================================================

Single-shot GPU batch execution for LAM ZIP model file generation.
Reuses the proven image definition from concierge_modal.py.

Design decisions (from ChatGPT consultation 2026-02-26):
- No Gradio / No Web UI / No keep_warm -> minimal Modal credit consumption
- Input: image (bytes) + params (JSON dict) via CLI
- Output: ZIP (skin.glb + offset.ply + animation.glb) + preview PNG + comparison PNG
- shape_param guard to detect "bird monster" (vertex explosion) artifacts
- concierge_modal.py image reuse for proven dependency stability

Usage:
  modal run lam_avatar_batch.py --image-path ./input/input.jpg --param-json-path ./input/params.json
  modal run lam_avatar_batch.py --image-path ./input/input.jpg  # default params
"""

import os
import sys
import json
import modal

app = modal.App("lam-avatar-batch")

# Reuse the proven image definition from concierge_modal.py
from concierge_modal import image as concierge_image

# Output volume for results
output_vol = modal.Volume.from_name("lam-batch-output", create_if_missing=True)
OUTPUT_VOL_PATH = "/vol/batch_output"


def _shape_guard(shape_param):
    """
    Detect 'bird monster' (vertex explosion) artifacts.
    shape_param in FLAME PCA space should be within [-3, +3] for normal faces.
    NaN or abs > 5.0 indicates FLAME tracking failure.
    """
    import numpy as np

    arr = shape_param.detach().cpu().numpy() if hasattr(shape_param, 'detach') else np.array(shape_param)

    if np.isnan(arr).any():
        raise RuntimeError(
            "shape_param contains NaN -- FLAME tracking completely failed. "
            "Check input image quality (frontal face, good lighting)."
        )

    max_abs = np.abs(arr).max()
    if max_abs > 5.0:
        raise RuntimeError(
            f"shape_param exploded (max abs = {max_abs:.2f}) -- "
            "FLAME tracking produced abnormal values. "
            "This typically causes 'bird monster' mesh artifacts. "
            "Check input image or tracking configuration."
        )

    print(f"[shape_guard] OK: range [{arr.min():.3f}, {arr.max():.3f}]")


@app.function(
    gpu="L4",
    image=concierge_image,
    volumes={OUTPUT_VOL_PATH: output_vol},
    timeout=7200,
)
def generate_avatar_batch(image_bytes: bytes, params: dict):
    """
    Main batch inference function.

    Args:
        image_bytes: Raw bytes of input face image (PNG/JPG)
        params: Dict with optional keys:
            - shape_scale (float): Scale factor for shape_param identity emphasis (default 1.0)
            - motion_name (str): Name of sample motion folder (default "talk")
    """
    import tempfile
    import shutil
    import zipfile
    import numpy as np
    import torch
    import torch._dynamo
    from pathlib import Path
    from PIL import Image
    from glob import glob

    # ==========================================
    # BIRD-MONSTER FIX v3: Disable torch.compile BEFORE any imports.
    # The @torch.compile decorators on forward_latent_points and
    # Dinov2FusionWrapper.forward are evaluated at import time.
    # The monkey-patch in _init_lam_pipeline now runs BEFORE imports.
    # These env vars provide an additional safety layer.
    # ==========================================
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()

    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")

    from concierge_modal import _init_lam_pipeline
    # NOTE: _setup_model_paths() is called inside _init_lam_pipeline() — no need to call here

    # Clean stale FLAME tracking data from previous runs
    tracking_root = os.path.join(os.getcwd(), "output", "tracking")
    if os.path.isdir(tracking_root):
        for subdir in ["preprocess", "tracking", "export"]:
            stale = os.path.join(tracking_root, subdir)
            if os.path.isdir(stale):
                shutil.rmtree(stale)

    # Parse params
    shape_scale = params.get("shape_scale", 1.0)
    motion_name = params.get("motion_name", "talk")

    # Save input image to temp file
    tmpdir = tempfile.mkdtemp(prefix="lam_batch_")
    image_path = os.path.join(tmpdir, "input.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    print(f"Input image saved: {image_path} ({len(image_bytes)} bytes)")
    print(f"Params: shape_scale={shape_scale}, motion_name={motion_name}")

    # Initialize LAM pipeline
    print("=" * 80)
    print("Initializing LAM pipeline...")
    cfg, lam, flametracking = _init_lam_pipeline()

    # Verify model state
    total_params = sum(1 for _ in lam.state_dict())
    print(f"[DIAG] Model total state_dict keys: {total_params}")
    print(f"[DIAG] Model device: {next(lam.parameters()).device}")
    print(f"[DIAG] Encoder type: {type(lam.encoder).__name__}")
    print(f"[DIAG] forward_latent_points type: {type(lam.forward_latent_points).__name__}")

    # Quick encoder sanity check: feed a dummy image and verify output is not garbage
    print("[DIAG] Running encoder sanity check...")
    with torch.no_grad():
        dummy = torch.randn(1, 3, 504, 504, device="cuda", dtype=torch.float32) * 0.5 + 0.5
        enc_out = lam.forward_encode_image(dummy)
        print(f"  Encoder output: shape={enc_out.shape}, "
              f"min={enc_out.min():.4f}, max={enc_out.max():.4f}, "
              f"mean={enc_out.mean():.4f}, std={enc_out.std():.4f}")
        if enc_out.std() < 0.001:
            print("  [CRITICAL] Encoder output has near-zero variance - weights may not be loaded!")
        if torch.isnan(enc_out).any():
            print("  [CRITICAL] Encoder output contains NaN!")
        del dummy, enc_out
        torch.cuda.empty_cache()

    print("LAM pipeline ready.")
    print("=" * 80)

    try:
        # Step 1: FLAME tracking on source image
        print("[Step 1/5] FLAME tracking on source image...")
        image_raw = os.path.join(tmpdir, "raw.png")
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
        print(f"  Tracked image: {tracked_image}")

        # Step 2: Prepare motion sequence
        print(f"[Step 2/5] Preparing motion sequence: {motion_name}...")
        sample_motions = glob("./model_zoo/sample_motion/export/*/flame_param")
        if not sample_motions:
            raise RuntimeError("No motion sequences found in model_zoo/sample_motion/export/")

        flame_params_dir = sample_motions[0]  # default
        for sp in sample_motions:
            if os.path.basename(os.path.dirname(sp)) == motion_name:
                flame_params_dir = sp
                break
        print(f"  Using motion: {os.path.dirname(flame_params_dir)}")

        # Step 3: Preprocess image and prepare inference inputs
        print("[Step 3/5] Preprocessing image for LAM inference...")
        from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

        image_tensor, _, _, shape_param = preprocess_image(
            tracked_image, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1.0,
            max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=cfg.source_size, multiply=14, need_mask=True, get_shape_param=True,
        )

        # Shape guard: detect bird monster
        _shape_guard(shape_param)

        # Apply shape_scale for identity emphasis
        if shape_scale != 1.0:
            print(f"  Applying shape_scale={shape_scale} (identity emphasis)")
            shape_param = shape_param * shape_scale
            _shape_guard(shape_param)  # re-check after scaling

        # Save preprocessed visualization
        preproc_vis_path = os.path.join(tmpdir, "preprocessed_input.png")
        vis_img = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_img).save(preproc_vis_path)

        src_name = os.path.splitext(os.path.basename(image_path))[0]
        driven_name = os.path.basename(os.path.dirname(flame_params_dir))

        motion_seq = prepare_motion_seqs(
            flame_params_dir, None, save_root=tmpdir, fps=30,
            bg_color=1.0, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_image_res=cfg.render_size, multiply=16,
            need_mask=False, vis_motion=False, shape_param=shape_param, test_sample=False,
            cross_id=False, src_driven=[src_name, driven_name],
            max_squen_length=300,  # Match official ModelScope app.py
        )

        # Step 4: LAM inference
        print("[Step 4/5] Running LAM inference...")
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device = "cuda"

        # Diagnostic: input tensor stats
        inp = image_tensor.unsqueeze(0).to(device, torch.float32)
        print(f"  [DIAG] Input tensor: shape={inp.shape}, "
              f"min={inp.min():.4f}, max={inp.max():.4f}, mean={inp.mean():.4f}")
        print(f"  [DIAG] shape_param: shape={shape_param.shape}, "
              f"min={shape_param.min():.4f}, max={shape_param.max():.4f}")
        print(f"  [DIAG] render_c2ws: shape={motion_seq['render_c2ws'].shape}")
        print(f"  [DIAG] render_intrs: shape={motion_seq['render_intrs'].shape}")
        print(f"  [DIAG] betas: shape={motion_seq['flame_params']['betas'].shape}")

        with torch.no_grad():
            res = lam.infer_single_view(
                inp,
                None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
            )

        # Diagnostic: output tensor stats
        comp_rgb = res["comp_rgb"]
        comp_mask = res["comp_mask"]
        print(f"  [DIAG] comp_rgb: shape={comp_rgb.shape}, "
              f"min={comp_rgb.min():.4f}, max={comp_rgb.max():.4f}, mean={comp_rgb.mean():.4f}")
        print(f"  [DIAG] comp_mask: shape={comp_mask.shape}, "
              f"min={comp_mask.min():.4f}, max={comp_mask.max():.4f}, mean={comp_mask.mean():.4f}")
        if comp_rgb.mean() < 0.01 or comp_rgb.max() < 0.1:
            print("  [WARN] comp_rgb is nearly black -- model may not have loaded weights correctly")
        if torch.isnan(comp_rgb).any():
            print("  [CRITICAL] comp_rgb contains NaN!")

        # Diagnostic: Gaussian splat position stats (canonical model)
        if "cano_gs_lst" in res and len(res["cano_gs_lst"]) > 0:
            gs_model = res["cano_gs_lst"][0]
            gs_xyz = gs_model.xyz
            gs_offset = gs_model.offset if hasattr(gs_model, 'offset') and gs_model.offset is not None else None
            print(f"  [DIAG] Gaussian XYZ: shape={gs_xyz.shape}, "
                  f"min=[{gs_xyz.min(0).values[0]:.4f},{gs_xyz.min(0).values[1]:.4f},{gs_xyz.min(0).values[2]:.4f}], "
                  f"max=[{gs_xyz.max(0).values[0]:.4f},{gs_xyz.max(0).values[1]:.4f},{gs_xyz.max(0).values[2]:.4f}]")
            if gs_offset is not None:
                print(f"  [DIAG] Gaussian offset: shape={gs_offset.shape}, "
                      f"abs_mean={gs_offset.abs().mean():.6f}, abs_max={gs_offset.abs().max():.6f}")
            gs_scaling = gs_model.scaling
            if gs_scaling is not None:
                print(f"  [DIAG] Gaussian scaling: mean={gs_scaling.mean():.6f}, max={gs_scaling.max():.6f}")

        # Diagnostic: Per-frame stats for first 3 frames
        n_frames = min(3, comp_rgb.shape[0])
        for fi in range(n_frames):
            frame = comp_rgb[fi]
            mask_f = comp_mask[fi]
            fg_pixels = (mask_f > 0.5).sum().item()
            total_pixels = mask_f.numel()
            print(f"  [DIAG] Frame {fi}: mean={frame.mean():.4f}, "
                  f"fg_pixels={fg_pixels}/{total_pixels} ({100*fg_pixels/total_pixels:.1f}%)")

        # Save first 3 frames as individual diagnostic PNGs
        diag_dir = os.path.join(tmpdir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        rgb_np = comp_rgb.detach().cpu().numpy()
        mask_np = comp_mask.detach().cpu().numpy()
        mask_np[mask_np < 0.5] = 0.0
        for fi in range(n_frames):
            frame_rgb = rgb_np[fi] * mask_np[fi] + (1 - mask_np[fi]) * 1
            frame_rgb = (np.clip(frame_rgb, 0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(frame_rgb).save(os.path.join(diag_dir, f"frame_{fi:03d}.png"))
        print(f"  [DIAG] Saved {n_frames} diagnostic frames to {diag_dir}")
        print("  Inference complete.")

        # Step 5: Generate GLB + ZIP
        print("[Step 5/5] Generating 3D avatar (GLB + ZIP)...")
        from tools.generateARKITGLBWithBlender import generate_glb

        oac_dir = os.path.join(tmpdir, "oac_export", "avatar")
        os.makedirs(oac_dir, exist_ok=True)

        # Save shaped mesh -> GLB via Blender
        saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir,
        )

        generate_glb(
            input_mesh=Path(saved_head_path),
            template_fbx=Path("./model_zoo/sample_oac/template_file.fbx"),
            output_glb=Path(os.path.join(oac_dir, "skin.glb")),
            blender_exec=Path("/usr/local/bin/blender"),
        )

        # Save offset PLY (Gaussian splatting)
        res["cano_gs_lst"][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True,
        )

        # Copy animation template
        shutil.copy(
            src="./model_zoo/sample_oac/animation.glb",
            dst=os.path.join(oac_dir, "animation.glb"),
        )

        # vertex_order.json is already generated by generate_glb step 4
        # (gen_vertex_order_with_blender) which correctly sorts vertices by Z
        # coordinate after 90-degree rotation. This matches the downstream
        # viewer's expectations. DO NOT overwrite it with sequential indices.

        # Clean up temp mesh (nature.obj)
        if os.path.exists(saved_head_path):
            os.remove(saved_head_path)

        # Create ZIP
        output_zip = os.path.join(tmpdir, "avatar.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            dir_info = zipfile.ZipInfo("avatar/")
            zf.writestr(dir_info, "")
            for root, _, files in os.walk(oac_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, os.path.dirname(oac_dir))
                    zf.write(fpath, arcname)

        zip_size = os.path.getsize(output_zip) / (1024 * 1024)
        print(f"  ZIP created: {output_zip} ({zip_size:.1f} MB)")

        # Generate preview PNG (first frame)
        preview_path = os.path.join(tmpdir, "preview.png")
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(rgb[0]).save(preview_path)
        print(f"  Preview: {preview_path}")

        # Generate comparison image (input vs output side-by-side)
        compare_path = os.path.join(tmpdir, "compare.png")
        img_in = Image.open(image_path).convert("RGB").resize((256, 256))
        img_out = Image.open(preview_path).convert("RGB").resize((256, 256))
        canvas = Image.new("RGB", (512, 256), (255, 255, 255))
        canvas.paste(img_in, (0, 0))
        canvas.paste(img_out, (256, 0))
        canvas.save(compare_path)
        print(f"  Comparison: {compare_path}")

        # Save results to volume
        vol_out = OUTPUT_VOL_PATH
        os.makedirs(vol_out, exist_ok=True)

        shutil.copy2(output_zip, os.path.join(vol_out, "avatar.zip"))
        shutil.copy2(preview_path, os.path.join(vol_out, "preview.png"))
        shutil.copy2(compare_path, os.path.join(vol_out, "compare.png"))
        shutil.copy2(preproc_vis_path, os.path.join(vol_out, "preprocessed_input.png"))

        # Save diagnostic frames
        for fi in range(n_frames):
            src_diag = os.path.join(diag_dir, f"frame_{fi:03d}.png")
            if os.path.exists(src_diag):
                shutil.copy2(src_diag, os.path.join(vol_out, f"frame_{fi:03d}.png"))

        # Save params for experiment tracking (with diagnostics)
        result_meta = {
            "params": params,
            "shape_param_range": [float(shape_param.min()), float(shape_param.max())],
            "shape_param_dim": int(shape_param.shape[-1]),
            "zip_size_mb": round(zip_size, 2),
            "comp_rgb_stats": {
                "mean": float(res["comp_rgb"].mean()),
                "min": float(res["comp_rgb"].min()),
                "max": float(res["comp_rgb"].max()),
                "has_nan": bool(torch.isnan(res["comp_rgb"]).any()),
            },
            "bird_fix_applied": True,
        }
        with open(os.path.join(vol_out, "result_meta.json"), "w") as f:
            json.dump(result_meta, f, indent=2)

        output_vol.commit()

        print("=" * 80)
        print("BATCH GENERATION COMPLETE")
        print(f"  ZIP: {zip_size:.1f} MB")
        print(f"  shape_param range: [{shape_param.min():.3f}, {shape_param.max():.3f}]")
        print(f"  Results saved to volume: {vol_out}")
        print("=" * 80)

        return result_meta

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\nBATCH GENERATION ERROR:\n{tb}", flush=True)
        raise

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.local_entrypoint()
def main(
    image_path: str,
    param_json_path: str = "",
    output_dir: str = "./output",
):
    """
    Local entrypoint for CLI execution.

    Args:
        image_path: Path to input face image (PNG/JPG)
        param_json_path: Path to params JSON file (optional)
        output_dir: Local directory to download results (default: ./output)
    """
    # Read image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    print(f"Read image: {image_path} ({len(image_bytes)} bytes)")

    # Read params or use defaults
    if param_json_path and os.path.isfile(param_json_path):
        with open(param_json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"Read params: {param_json_path} -> {params}")
    else:
        params = {"shape_scale": 1.0, "motion_name": "talk"}
        print(f"Using default params: {params}")

    # Execute on remote GPU
    result = generate_avatar_batch.remote(image_bytes, params)
    print(f"\nResult: {json.dumps(result, indent=2)}")

    # Download results from Modal Volume to local machine
    os.makedirs(output_dir, exist_ok=True)
    download_files = [
        "avatar.zip", "preview.png", "compare.png", "preprocessed_input.png", "result_meta.json",
        "frame_000.png", "frame_001.png", "frame_002.png",
    ]
    print(f"\nDownloading results to {output_dir}/...")

    for fname in download_files:
        try:
            data = b""
            for chunk in output_vol.read_file(fname):
                data += chunk
            local_path = os.path.join(output_dir, fname)
            with open(local_path, "wb") as f:
                f.write(data)
            size_str = f"{len(data) / (1024*1024):.1f} MB" if len(data) > 1024*1024 else f"{len(data) / 1024:.0f} KB"
            print(f"  Downloaded: {fname} ({size_str})")
        except Exception as e:
            print(f"  Skip: {fname} ({e})")

    print(f"\nDone. Results in: {os.path.abspath(output_dir)}/")
    print(f"  avatar.zip  -- ZIPモデルファイル (skin.glb + offset.ply + animation.glb)")
    print(f"  compare.png -- 入力 vs 出力 比較画像")
