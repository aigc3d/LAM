# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
#
# LAM Avatar Batch Generator — Modal CLI
# Aligned with the official ModelScope app.py (app_lam.py) pipeline.
#
# Usage:
#   modal run lam_avatar_batch.py --image-path ./input/face.jpg
#   modal run lam_avatar_batch.py --image-path ./input/face.jpg --param-json-path params.json
#   modal run lam_avatar_batch.py --image-path ./input/face.jpg --output-dir ./my_output

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import modal

from concierge_modal import image as concierge_image
from concierge_modal import vol

MINUTES = 60
OUTPUT_DIR = "/tmp/lam_output"

app = modal.App("lam-avatar-batch")


@app.function(
    image=concierge_image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    timeout=30 * MINUTES,
    volumes={"/root/LAM/model_zoo": vol},
)
def generate_avatar(image_bytes: bytes, image_filename: str, params: dict) -> dict:
    """
    Generate a 3D avatar from a face image.

    This function mirrors the official app_lam.py core_fn() logic exactly:
      1. FLAME tracking (preprocess → optimize → export)
      2. Preprocess image (crop, mask, extract shape_param)
      3. Prepare motion sequence
      4. LAM inference (infer_single_view)
      5. Render video + add audio
      6. OAC ZIP generation (save_shaped_mesh → offset.ply → generate_glb → animation.glb)

    Args:
        image_bytes: Raw bytes of the input face image
        image_filename: Original filename (for extension)
        params: {"motion_name": "talk"} — selects which motion sequence to use

    Returns:
        dict with keys: zip_bytes, video_bytes, preview_bytes, compare_bytes, meta
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from concierge_modal import _init_lam_pipeline

    os.chdir("/root/LAM")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    cfg, lam, flametracking = _init_lam_pipeline()

    motion_name = params.get("motion_name", "talk")
    working_dir = tempfile.mkdtemp(prefix="lam_batch_")

    # Save input image
    ext = os.path.splitext(image_filename)[1] or ".png"
    image_raw = os.path.join(working_dir, f"raw{ext}")
    with open(image_raw, "wb") as f:
        f.write(image_bytes)

    # Derive identifiers (match official app_lam.py naming)
    base_iid = os.path.splitext(os.path.basename(image_filename))[0]
    base_vid = motion_name

    # Motion paths (official structure: assets/sample_motion/export/{motion_name}/...)
    flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
    audio_path = os.path.join("./assets/sample_motion/export", base_vid, f"{base_vid}.wav")

    # Output paths
    dump_video_path = os.path.join(working_dir, "output.mp4")
    dump_image_path = os.path.join(working_dir, "output.png")

    print(f"[LAM] motion={motion_name}, image={image_filename}")
    print(f"[LAM] flame_params_dir={flame_params_dir}")

    # ------------------------------------------------------------------
    # Step 1: FLAME Tracking (official: flametracking.preprocess/optimize/export)
    # ------------------------------------------------------------------
    print("[LAM] Step 1/6: FLAME tracking — preprocess")
    return_code = flametracking.preprocess(image_raw)
    assert return_code == 0, "flametracking.preprocess() failed"

    print("[LAM] Step 2/6: FLAME tracking — optimize")
    return_code = flametracking.optimize()
    assert return_code == 0, "flametracking.optimize() failed"

    print("[LAM] Step 3/6: FLAME tracking — export")
    return_code, output_dir = flametracking.export()
    assert return_code == 0, "flametracking.export() failed"

    image_path = os.path.join(output_dir, "images/00000_00.png")
    mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
    print(f"[LAM] tracked image: {image_path}")
    print(f"[LAM] tracked mask:  {mask_path}")

    # ------------------------------------------------------------------
    # Step 2: Preprocess image (official: preprocess_image with exact params)
    # ------------------------------------------------------------------
    print("[LAM] Step 4/6: Preprocess image + prepare motion")
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

    aspect_standard = 1.0 / 1.0
    source_size = cfg.source_size   # 512
    render_size = cfg.render_size   # 512
    render_fps = 30

    # preprocess_image — exact params from official app_lam.py L264-266
    image, _, _, shape_param = preprocess_image(
        image_path,
        mask_path=mask_path,
        intr=None,
        pad_ratio=0,
        bg_color=1.0,
        max_tgt_size=None,
        aspect_standard=aspect_standard,
        enlarge_ratio=[1.0, 1.0],
        render_tgt_size=source_size,
        multiply=14,
        need_mask=True,
        get_shape_param=True,
    )

    # Save masked reference image for comparison (official L269-271)
    vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    Image.fromarray(vis_ref_img).save(dump_image_path)

    # ------------------------------------------------------------------
    # Step 3: Prepare motion sequence (official params from app_lam.py L274-281)
    # ------------------------------------------------------------------
    motion_img_need_mask = cfg.get("motion_img_need_mask", False)
    vis_motion = cfg.get("vis_motion", False)

    src = image_path.split("/")[-3]
    driven = flame_params_dir.split("/")[-2]
    src_driven = [src, driven]

    # NOTE: enlarge_ratio=[1.0, 1, 0] is the official code (typo in app_lam.py L278)
    motion_seq = prepare_motion_seqs(
        flame_params_dir,
        None,
        save_root=working_dir,
        fps=render_fps,
        bg_color=1.0,
        aspect_standard=aspect_standard,
        enlarge_ratio=[1.0, 1, 0],      # official typo preserved
        render_image_res=render_size,
        multiply=16,
        need_mask=motion_img_need_mask,
        vis_motion=vis_motion,
        shape_param=shape_param,
        test_sample=False,
        cross_id=False,
        src_driven=src_driven,
    )

    # ------------------------------------------------------------------
    # Step 4: LAM inference (official L283-293)
    # ------------------------------------------------------------------
    print("[LAM] Step 5/6: LAM inference")
    motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
    device, dtype = "cuda", torch.float32

    with torch.no_grad():
        res = lam.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
        )

    # ------------------------------------------------------------------
    # Step 5: OAC ZIP generation (official L304-341, enable_oac_file block)
    # ------------------------------------------------------------------
    print("[LAM] Step 6/6: OAC ZIP + video generation")

    from tools.generateARKITGLBWithBlender import generate_glb

    oac_dir = os.path.join(working_dir, "oac_export", base_iid)
    os.makedirs(oac_dir, exist_ok=True)

    # 5a. save_shaped_mesh → OBJ (official L312)
    saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
        shape_param.unsqueeze(0).cuda(), fd=oac_dir,
    )
    assert os.path.isfile(saved_head_path), f"save_shaped_mesh failed: {saved_head_path}"

    # 5b. offset.ply (official L313)
    res["cano_gs_lst"][0].save_ply(
        os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True,
    )

    # 5c. generate_glb → skin.glb + vertex_order.json (official L314-319)
    template_fbx = Path("./assets/sample_oac/template_file.fbx")
    blender_exec = Path(cfg.blender_path)
    generate_glb(
        input_mesh=Path(saved_head_path),
        template_fbx=template_fbx,
        output_glb=Path(os.path.join(oac_dir, "skin.glb")),
        blender_exec=blender_exec,
    )

    # 5d. animation.glb (official L320-323)
    shutil.copy(
        src="./assets/sample_oac/animation.glb",
        dst=os.path.join(oac_dir, "animation.glb"),
    )

    # 5e. Remove temporary OBJ (official L324)
    if os.path.exists(saved_head_path):
        os.remove(saved_head_path)

    # 5f. Create ZIP with folder structure (official L326-341, uses os.system('zip -r'))
    output_zip_path = os.path.join(working_dir, f"{base_iid}.zip")
    if os.path.exists(output_zip_path):
        os.remove(output_zip_path)
    original_cwd = os.getcwd()
    oac_parent_dir = os.path.dirname(oac_dir)
    base_iid_dir = os.path.basename(oac_dir)
    os.chdir(oac_parent_dir)
    try:
        os.system(f"zip -r {os.path.abspath(output_zip_path)} {base_iid_dir}")
    finally:
        os.chdir(original_cwd)

    # Verify ZIP
    assert os.path.isfile(output_zip_path), f"ZIP creation failed: {output_zip_path}"

    # ------------------------------------------------------------------
    # Step 6: Render video + add audio (official L346-363)
    # ------------------------------------------------------------------
    rgb = res["comp_rgb"].detach().cpu().numpy()   # [Nv, H, W, 3], 0-1
    mask = res["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
    mask[mask < 0.5] = 0.0
    rgb = rgb * mask + (1 - mask) * 1
    rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

    if vis_motion:
        vis_ref_img_tiled = np.tile(
            cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]),
                       interpolation=cv2.INTER_AREA)[None, :, :, :],
            (rgb.shape[0], 1, 1, 1),
        )
        rgb = np.concatenate(
            [vis_ref_img_tiled, rgb, motion_seq["vis_motion_render"]], axis=2,
        )

    # save_images2video (official save_images2video L94-105)
    from moviepy.editor import ImageSequenceClip
    images_u8 = [frame.astype(np.uint8) for frame in rgb]
    clip = ImageSequenceClip(images_u8, fps=render_fps)
    clip.write_videofile(dump_video_path, codec="libx264")
    print(f"[LAM] Video saved: {dump_video_path}")

    # add_audio_to_video (official L108-124)
    dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
    if os.path.isfile(audio_path):
        from moviepy.editor import AudioFileClip, VideoFileClip
        video_clip = VideoFileClip(dump_video_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        video_clip_with_audio.write_videofile(dump_video_path_wa, codec="libx264", audio_codec="aac")
        print(f"[LAM] Audio-video saved: {dump_video_path_wa}")
        final_video_path = dump_video_path_wa
    else:
        print(f"[LAM] Audio not found at {audio_path}, skipping audio merge")
        final_video_path = dump_video_path

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------
    # Preview: first rendered frame
    preview_path = os.path.join(working_dir, "preview.png")
    Image.fromarray(rgb[0]).save(preview_path)

    # Compare: input vs output side by side
    compare_path = os.path.join(working_dir, "compare.png")
    input_img = np.array(Image.open(image_raw).convert("RGB").resize(
        (rgb[0].shape[1], rgb[0].shape[0])))
    compare = np.concatenate([input_img, rgb[0]], axis=1)
    Image.fromarray(compare).save(compare_path)

    # Metadata
    meta = {
        "motion_name": motion_name,
        "source_size": cfg.source_size,
        "render_size": cfg.render_size,
        "render_fps": render_fps,
        "num_frames": len(rgb),
        "oac_files": os.listdir(oac_dir) if os.path.isdir(oac_dir) else [],
    }

    # Read result files
    def read_bytes(p):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
        return None

    result = {
        "zip_bytes": read_bytes(output_zip_path),
        "video_bytes": read_bytes(final_video_path),
        "preview_bytes": read_bytes(preview_path),
        "compare_bytes": read_bytes(compare_path),
        "preprocessed_bytes": read_bytes(dump_image_path),
        "meta": meta,
    }

    # Cleanup
    shutil.rmtree(working_dir, ignore_errors=True)

    return result


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    image_path: str,
    param_json_path: str = "",
    output_dir: str = "./output",
):
    """
    Modal CLI entry point.

    Args:
        image_path: Path to input face image (PNG/JPG)
        param_json_path: Optional JSON file with {"motion_name": "..."}
        output_dir: Local directory for output files (default: ./output)
    """
    # Read input image
    image_path = os.path.abspath(image_path)
    assert os.path.isfile(image_path), f"Image not found: {image_path}"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_filename = os.path.basename(image_path)

    # Read params
    params = {}
    if param_json_path and os.path.isfile(param_json_path):
        with open(param_json_path) as f:
            params = json.load(f)

    print(f"[LOCAL] Input:  {image_path}")
    print(f"[LOCAL] Params: {params}")
    print(f"[LOCAL] Output: {output_dir}")

    # Run remote
    t0 = time.time()
    result = generate_avatar.remote(image_bytes, image_filename, params)
    elapsed = time.time() - t0
    print(f"[LOCAL] Remote execution took {elapsed:.1f}s")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(image_filename)[0]

    file_map = {
        f"{base}_avatar.zip": result.get("zip_bytes"),
        f"{base}_output.mp4": result.get("video_bytes"),
        f"{base}_preview.png": result.get("preview_bytes"),
        f"{base}_compare.png": result.get("compare_bytes"),
        f"{base}_preprocessed.png": result.get("preprocessed_bytes"),
    }

    for fname, data in file_map.items():
        if data:
            out_path = os.path.join(output_dir, fname)
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"[LOCAL] Saved: {out_path} ({len(data)} bytes)")

    # Save metadata
    meta_path = os.path.join(output_dir, f"{base}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(result.get("meta", {}), f, indent=2)
    print(f"[LOCAL] Saved: {meta_path}")

    print(f"[LOCAL] Done! Output in {output_dir}/")
