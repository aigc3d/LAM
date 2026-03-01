"""
lam_avatar_batch.py - LAM Avatar Batch Generator
=================================================

Clean pipeline based on official ModelScope app.py core_fn.
No dependency on app_lam.py.

Usage:
  modal run lam_avatar_batch.py --image-path ./input/input.jpg --param-json-path ./input/params.json
  modal run lam_avatar_batch.py --image-path ./input/input.jpg  # default: motion_name=GEM
"""

import os
import sys
import json
import modal

app = modal.App("lam-avatar-batch")

# Modal image + volume definitions from concierge_modal.py
from concierge_modal import image as concierge_image
from concierge_modal import storage_vol, STORAGE_VOL_PATH

# Output volume for results
output_vol = modal.Volume.from_name("lam-batch-output", create_if_missing=True)
OUTPUT_VOL_PATH = "/vol/batch_output"


def _shape_guard(shape_param):
    """
    Detect 'bird monster' (vertex explosion) artifacts.
    Safety net not present in official app.py but harmless.
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


# ============================================================
# Video helpers (matching official app.py save_imgs_2_video / add_audio_to_video)
# ============================================================

def _save_imgs_2_video(imgs, v_pth, fps=30):
    import numpy as np
    from moviepy.editor import ImageSequenceClip

    images = [image.astype(np.uint8) for image in imgs]
    clip = ImageSequenceClip(images, fps=fps)
    clip = clip.subclip(0, len(images) / fps)
    clip.write_videofile(v_pth, codec='libx264')
    print(f"Video saved successfully at {v_pth}")


def _add_audio_to_video(video_path, out_path, audio_path, fps=30):
    from moviepy.editor import VideoFileClip, AudioFileClip

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    if audio_clip.duration > 10:
        audio_clip = audio_clip.subclip(0, 10)

    video_clip_with_audio = video_clip.set_audio(audio_clip)
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac', fps=fps)
    print(f"Audio added successfully at {out_path}")


@app.function(
    gpu="L4",
    image=concierge_image,
    volumes={OUTPUT_VOL_PATH: output_vol, STORAGE_VOL_PATH: storage_vol},
    timeout=7200,
)
def generate_avatar_batch(image_bytes: bytes, params: dict):
    """
    Main batch inference function based on official ModelScope app.py core_fn.

    Args:
        image_bytes: Raw bytes of input face image (PNG/JPG)
        params: Dict with optional keys:
            - motion_name (str): Name of sample motion folder (default "GEM")
    """
    import tempfile
    import shutil
    import numpy as np
    import torch
    from pathlib import Path
    from PIL import Image
    from datetime import datetime

    from concierge_modal import _init_lam_pipeline

    # Parse params
    motion_name = params.get("motion_name", "GEM")

    # Save input image to temp file
    tmpdir = tempfile.mkdtemp(prefix="lam_batch_")
    image_path = os.path.join(tmpdir, "input.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    print(f"Input image saved: {image_path} ({len(image_bytes)} bytes)")
    print(f"Params: motion_name={motion_name}")

    # Clean stale FLAME tracking data
    tracking_root = os.path.join("/root/LAM", "output", "tracking")
    if os.path.isdir(tracking_root):
        for subdir in ["preprocess", "tracking", "export"]:
            stale = os.path.join(tracking_root, subdir)
            if os.path.isdir(stale):
                shutil.rmtree(stale)

    # Initialize pipeline (official from_pretrained, no app_lam dependency)
    print("=" * 80)
    print("Initializing LAM pipeline...")
    cfg, lam, flametracking = _init_lam_pipeline()
    print("LAM pipeline ready.")
    print("=" * 80)

    try:
        # ============================================================
        # Official core_fn flow (from ModelScope app.py)
        # ============================================================

        # Step 1: Save raw image (matching official)
        print("[Step 1/6] FLAME tracking on source image...")
        image_raw = os.path.join(tmpdir, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)

        # Step 2: Resolve motion sequence (matching official)
        # Official: base_vid = os.path.basename(video_params).split(".")[0]
        #           flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        print(f"[Step 2/6] Resolving motion: {motion_name}...")
        flame_params_dir = os.path.join("./assets/sample_motion/export", motion_name, "flame_param")
        if not os.path.isdir(flame_params_dir):
            flame_params_dir = os.path.join("./model_zoo/sample_motion/export", motion_name, "flame_param")
        if not os.path.isdir(flame_params_dir):
            # List available motions for error message
            from glob import glob
            available = (
                glob("./assets/sample_motion/export/*/flame_param") +
                glob("./model_zoo/sample_motion/export/*/flame_param")
            )
            names = [os.path.basename(os.path.dirname(p)) for p in available]
            raise RuntimeError(
                f"Motion '{motion_name}' not found. Available: {names}"
            )

        motion_seqs_dir = flame_params_dir
        print(f"  Using: {flame_params_dir}")

        # Step 3: FLAME tracking (matching official preprocess → optimize → export)
        return_code = flametracking.preprocess(image_raw)
        assert return_code == 0, "flametracking preprocess failed!"
        return_code = flametracking.optimize()
        assert return_code == 0, "flametracking optimize failed!"
        return_code, output_dir = flametracking.export()
        assert return_code == 0, "flametracking export failed!"

        tracked_image = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        print(f"  image_path: {tracked_image}")
        print(f"  mask_path: {mask_path}")

        # Step 4: Preprocess + prepare motion (matching official)
        print("[Step 3/6] Preprocessing image for LAM inference...")
        from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

        aspect_standard = 1.0 / 1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30

        motion_img_need_mask = cfg.get("motion_img_need_mask", False)
        vis_motion = cfg.get("vis_motion", False)

        # preprocess_image (matching official exactly)
        image_tensor, _, _, shape_param = preprocess_image(
            tracked_image, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1.,
            max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14, need_mask=True, get_shape_param=True,
        )

        # Shape guard (safety net, not in official)
        _shape_guard(shape_param)

        # Save masked image for visualization (matching official)
        save_ref_img_path = os.path.join(tmpdir, "output.png")
        vis_ref_img = (image_tensor[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # prepare_motion_seqs (matching official)
        # Official: src = image_path.split('/')[-3], driven = motion_seqs_dir.split('/')[-2]
        src = tracked_image.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]

        motion_seq = prepare_motion_seqs(
            motion_seqs_dir, None, save_root=tmpdir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16,
            need_mask=motion_img_need_mask, vis_motion=vis_motion,
            shape_param=shape_param, test_sample=False, cross_id=False,
            src_driven=src_driven,
        )

        # Step 5: LAM inference (matching official)
        print("[Step 4/6] Running LAM inference...")
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32

        print("start to inference...................")
        with torch.no_grad():
            res = lam.infer_single_view(
                image_tensor.unsqueeze(0).to(device, dtype), None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
            )

        print("  Inference complete.")

        # Step 6: OAC ZIP generation (matching official enable_oac_file block)
        print("[Step 5/6] Generating OAC ZIP (skin.glb + offset.ply + animation.glb)...")

        from tools.generateARKITGLBWithBlender import generate_glb

        base_iid = 'chatting_avatar_' + datetime.now().strftime("%Y%m%d%H%M%S")
        oac_dir = os.path.join('./', base_iid)
        os.makedirs(oac_dir, exist_ok=True)

        # save_shaped_mesh (matching official)
        saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir,
        )

        # offset.ply (matching official)
        res['cano_gs_lst'][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True,
        )

        # generate_glb (matching official)
        # Official: template_fbx=Path("./assets/sample_oac/template_file.fbx")
        template_fbx = Path("./assets/sample_oac/template_file.fbx")
        if not template_fbx.exists():
            template_fbx = Path("./model_zoo/sample_oac/template_file.fbx")

        generate_glb(
            input_mesh=Path(saved_head_path),
            template_fbx=template_fbx,
            output_glb=Path(os.path.join(oac_dir, "skin.glb")),
            blender_exec=Path("/usr/local/bin/blender"),
        )

        # animation.glb (matching official)
        animation_src = './assets/sample_oac/animation.glb'
        if not os.path.isfile(animation_src):
            animation_src = './model_zoo/sample_oac/animation.glb'
        shutil.copy(src=animation_src, dst=os.path.join(oac_dir, 'animation.glb'))

        # Clean up temp OBJ (matching official: os.remove(saved_head_path))
        if os.path.exists(saved_head_path):
            os.remove(saved_head_path)

        # Create ZIP (matching official: zip -r)
        output_zip = os.path.join('./', base_iid + '.zip')
        if os.path.exists(output_zip):
            os.remove(output_zip)
        os.system('zip -r {} {}'.format(output_zip, oac_dir))
        shutil.rmtree(oac_dir)

        zip_size = os.path.getsize(output_zip) / (1024 * 1024)
        print(f"  ZIP created: {output_zip} ({zip_size:.1f} MB)")

        # Step 7: Video generation (matching official)
        print("[Step 6/6] Generating video and preview images...")

        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask = res["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        # save_imgs_2_video (matching official)
        dump_video_path = os.path.join(tmpdir, "output.mp4")
        _save_imgs_2_video(rgb, dump_video_path, render_fps)

        # add_audio_to_video (matching official)
        # Official: audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid + ".wav")
        audio_path = os.path.join("./assets/sample_motion/export", motion_name, motion_name + ".wav")
        if not os.path.isfile(audio_path):
            audio_path = os.path.join("./model_zoo/sample_motion/export", motion_name, motion_name + ".wav")

        final_video_path = dump_video_path
        if os.path.isfile(audio_path):
            dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
            _add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path, render_fps)
            final_video_path = dump_video_path_wa

        # Preview image (first frame)
        preview_path = os.path.join(tmpdir, "preview.png")
        Image.fromarray(rgb[0]).save(preview_path)

        # Comparison image (input vs output side-by-side)
        compare_path = os.path.join(tmpdir, "compare.png")
        img_in = Image.open(image_path).convert("RGB").resize((256, 256))
        img_out = Image.open(preview_path).convert("RGB").resize((256, 256))
        canvas = Image.new("RGB", (512, 256), (255, 255, 255))
        canvas.paste(img_in, (0, 0))
        canvas.paste(img_out, (256, 0))
        canvas.save(compare_path)

        # ============================================================
        # Save results to volume
        # ============================================================
        vol_out = OUTPUT_VOL_PATH
        os.makedirs(vol_out, exist_ok=True)

        shutil.copy2(output_zip, os.path.join(vol_out, "avatar.zip"))
        shutil.copy2(preview_path, os.path.join(vol_out, "preview.png"))
        shutil.copy2(compare_path, os.path.join(vol_out, "compare.png"))
        shutil.copy2(save_ref_img_path, os.path.join(vol_out, "preprocessed_input.png"))
        if os.path.isfile(final_video_path):
            shutil.copy2(final_video_path, os.path.join(vol_out, "output.mp4"))

        # Result metadata
        result_meta = {
            "params": params,
            "motion_name": motion_name,
            "shape_param_range": [float(shape_param.min()), float(shape_param.max())],
            "zip_size_mb": round(zip_size, 2),
        }
        with open(os.path.join(vol_out, "result_meta.json"), "w") as f:
            json.dump(result_meta, f, indent=2)

        output_vol.commit()

        # Clean up ZIP from working dir
        if os.path.exists(output_zip):
            os.remove(output_zip)

        print("=" * 80)
        print("BATCH GENERATION COMPLETE")
        print(f"  motion: {motion_name}")
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
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    print(f"Read image: {image_path} ({len(image_bytes)} bytes)")

    if param_json_path and os.path.isfile(param_json_path):
        with open(param_json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"Read params: {param_json_path} -> {params}")
    else:
        params = {"motion_name": "GEM"}
        print(f"Using default params: {params}")

    result = generate_avatar_batch.remote(image_bytes, params)
    print(f"\nResult: {json.dumps(result, indent=2)}")

    os.makedirs(output_dir, exist_ok=True)
    download_files = [
        "avatar.zip", "preview.png", "compare.png", "preprocessed_input.png",
        "result_meta.json", "output.mp4",
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
    print(f"  avatar.zip  -- OAC ZIP (skin.glb + offset.ply + animation.glb + vertex_order.json)")
    print(f"  output.mp4  -- Rendered animation video with audio")
    print(f"  compare.png -- Input vs output comparison")
