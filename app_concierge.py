"""
app_concierge.py - Concierge ZIP Generator (HF Spaces / Docker)
================================================================

Modal-free Gradio app for generating concierge.zip.
Inference logic is taken directly from concierge_modal.py (verified working).

Usage:
  python app_concierge.py                    # Run locally with GPU
  docker run --gpus all -p 7860:7860 image   # Docker
  # Or deploy as HF Space with Docker SDK

Pipeline:
  1. Source Image  -> FlameTrackingSingleImage -> shape parameters
  2. Motion Video  -> VHAP GlobalTracker -> per-frame FLAME parameters
  3. Shape + Motion -> LAM inference -> 3D Gaussian avatar
  4. Avatar data   -> Blender GLB export -> concierge.zip
"""

import os
import sys
import shutil
import tempfile
import subprocess
import zipfile
import json
import traceback
from pathlib import Path
from glob import glob

import numpy as np
import torch
import gradio as gr
from PIL import Image

# ============================================================
# Setup paths
# ============================================================
# Support both /app/LAM (Docker) and local repo root
LAM_ROOT = "/app/LAM" if os.path.isdir("/app/LAM") else os.path.dirname(os.path.abspath(__file__))
os.chdir(LAM_ROOT)
sys.path.insert(0, LAM_ROOT)

OUTPUT_DIR = os.path.join(LAM_ROOT, "output", "concierge_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Model path setup (symlinks to bridge layout differences)
# ============================================================
def setup_model_paths():
    """Create symlinks to bridge local directory layout to what LAM code expects.
    Taken from concierge_modal.py _setup_model_paths().
    """
    model_zoo = os.path.join(LAM_ROOT, "model_zoo")
    assets = os.path.join(LAM_ROOT, "assets")

    if not os.path.exists(model_zoo) and os.path.isdir(assets):
        os.symlink(assets, model_zoo)
        print(f"Symlink: model_zoo -> assets")
    elif os.path.isdir(model_zoo) and os.path.isdir(assets):
        for subdir in os.listdir(assets):
            src = os.path.join(assets, subdir)
            dst = os.path.join(model_zoo, subdir)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"Symlink: model_zoo/{subdir} -> assets/{subdir}")

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

    # Verify critical files
    print("\n=== Model file verification ===")
    for name in [
        "flame2023.pkl", "FaceBoxesV2.pth", "68_keypoints_model.pkl",
        "vgg_heads_l.trcd", "stylematte_synth.pt",
        "model.safetensors",
        "template_file.fbx", "animation.glb",
    ]:
        result = subprocess.run(
            ["find", model_zoo, "-name", name],
            capture_output=True, text=True,
        )
        paths = result.stdout.strip()
        if paths:
            for p in paths.split("\n"):
                print(f"  OK: {p}")
        else:
            print(f"  MISSING: {name}")


# ============================================================
# Initialize pipeline (called once at startup)
# ============================================================
def init_pipeline():
    """Initialize FLAME tracking and LAM model.
    Taken from concierge_modal.py _init_lam_pipeline().
    """
    setup_model_paths()

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
        "APP_TYPE": "infer.lam",
        "NUMBA_THREADING_LAYER": "omp",
    })

    # Verify xformers
    try:
        import xformers.ops
        print(f"xformers {xformers.__version__} available - "
              f"DINOv2 will use memory_efficient_attention")
    except ImportError:
        print("!!! CRITICAL: xformers NOT installed !!!")
        print("DINOv2 will fall back to standard attention, producing wrong output.")

    # Disable torch.compile / dynamo
    import torch._dynamo
    torch._dynamo.config.disable = True

    # Parse config
    from app_lam import parse_configs
    cfg, _ = parse_configs()

    # Build and load LAM model
    print("Loading LAM model...")
    from lam.models import ModelLAM
    from safetensors.torch import load_file as _load_safetensors

    model_cfg = cfg.model
    lam = ModelLAM(**model_cfg)

    ckpt_path = os.path.join(cfg.model_name, "model.safetensors")
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = _load_safetensors(ckpt_path, device="cpu")

    missing_keys, unexpected_keys = lam.load_state_dict(ckpt, strict=False)

    flame_missing = [k for k in missing_keys if "flame_model" in k]
    real_missing = [k for k in missing_keys if "flame_model" not in k]
    print(f"Checkpoint keys: {len(ckpt)}")
    print(f"Model     keys: {len(lam.state_dict())}")
    print(f"Missing   keys: {len(missing_keys)} ({len(flame_missing)} FLAME buffers, {len(real_missing)} real)")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    if real_missing:
        print(f"\n!!! {len(real_missing)} CRITICAL MISSING KEYS !!!")
        for k in real_missing:
            print(f"  MISSING: {k}")
    if unexpected_keys:
        print(f"\n!!! {len(unexpected_keys)} UNEXPECTED KEYS !!!")
        for k in unexpected_keys:
            print(f"  UNEXPECTED: {k}")

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


# ============================================================
# VHAP video tracking (custom motion video -> FLAME params)
# ============================================================
def track_video_to_motion(video_path, flametracking, working_dir, status_callback=None):
    """Process a custom motion video through VHAP FLAME tracking.
    Taken from concierge_modal.py _track_video_to_motion().
    """
    import cv2
    import torchvision

    def report(msg):
        if status_callback:
            status_callback(msg)
        print(msg)

    # Extract frames
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
    target_fps = min(30, video_fps) if video_fps > 0 else 30
    frame_interval = max(1, int(round(video_fps / target_fps)))
    max_frames = 300

    report(f"  Video: {total_frames} frames at {video_fps:.1f}fps, "
           f"sampling every {frame_interval} frame(s)")

    # Per-frame preprocessing
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
        cropped = torchvision.transforms.functional.resize(
            cropped, (1024, 1024), antialias=True,
        )

        cropped_matted, mask = flametracking.matting_engine(
            cropped / 255.0, return_type="matting", background_rgb=1.0,
        )
        cropped_matted = cropped_matted.cpu() * 255.0
        saved_image = np.round(
            cropped_matted.permute(1, 2, 0).numpy()
        ).astype(np.uint8)[:, :, ::-1]

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

        if processed_count % 30 == 0:
            report(f"  Processed {processed_count} frames...")

        frame_idx += 1

    cap.release()
    torch.cuda.empty_cache()

    if processed_count == 0:
        raise RuntimeError("No valid face frames found in video")

    report(f"  Preprocessed {processed_count} frames")

    stacked_landmarks = np.stack(all_landmarks, axis=0)
    np.savez(
        os.path.join(landmark_dir, "landmarks.npz"),
        bounding_box=[],
        face_landmark_2d=stacked_landmarks,
    )

    # VHAP Tracking
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

    # Export to NeRF dataset format
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


# ============================================================
# Full generation pipeline
# ============================================================
def generate_concierge_zip(image_path, video_path, cfg, lam, flametracking,
                           motion_name=None):
    """Full pipeline: image + video -> concierge.zip
    Taken from concierge_modal.py _generate_concierge_zip().

    Yields (status_msg, zip_path, preview_video_path, tracked_image_path, preproc_image_path).
    """
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
    from tools.generateARKITGLBWithBlender import update_flame_shape, convert_ascii_to_binary

    working_dir = tempfile.mkdtemp(prefix="concierge_")
    base_iid = "concierge"

    try:
        # Clean stale FLAME tracking data
        tracking_root = os.path.join(os.getcwd(), "output", "tracking")
        if os.path.isdir(tracking_root):
            for subdir in ["preprocess", "tracking", "export"]:
                stale = os.path.join(tracking_root, subdir)
                if os.path.isdir(stale):
                    shutil.rmtree(stale)

        # === Step 1: Source image FLAME tracking ===
        yield "Step 1/5: FLAME tracking on source image...", None, None, None, None

        image_raw = os.path.join(working_dir, "raw.png")
        with Image.open(image_path).convert("RGB") as img:
            img.save(image_raw)

        ret = flametracking.preprocess(image_raw)
        assert ret == 0, "FLAME preprocess failed - could not detect face in image"
        ret = flametracking.optimize()
        assert ret == 0, "FLAME optimize failed"
        ret, output_dir = flametracking.export()
        assert ret == 0, "FLAME export failed"

        tracked_image = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        yield "Step 1 done: check tracked face -->", None, None, tracked_image, None

        # === Step 2: Motion sequence preparation ===
        if video_path and os.path.isfile(video_path):
            total_steps = 6
            yield f"Step 2/{total_steps}: Processing custom motion video...", None, None, tracked_image, None
            flame_params_dir = track_video_to_motion(
                video_path, flametracking, working_dir,
                status_callback=lambda msg: print(f"  [Video] {msg}"),
            )
            motion_source = "custom video"
        else:
            total_steps = 5
            sample_motions = sorted(glob("./model_zoo/sample_motion/export/*/flame_param"))
            if not sample_motions:
                # Try assets/ fallback
                sample_motions = sorted(glob("./assets/sample_motion/export/*/flame_param"))
            if not sample_motions:
                raise RuntimeError("No motion sequences available. Upload a custom video.")

            flame_params_dir = sample_motions[0]
            if motion_name:
                for sp in sample_motions:
                    if os.path.basename(os.path.dirname(sp)) == motion_name:
                        flame_params_dir = sp
                        break

            resolved_name = os.path.basename(os.path.dirname(flame_params_dir))
            motion_source = f"sample '{resolved_name}'"

        # === Step 3: LAM inference ===
        yield f"Step 3/{total_steps}: Preparing LAM inference (motion: {motion_source})...", None, None, tracked_image, None

        source_size = cfg.source_size
        render_size = cfg.render_size

        image_tensor, _, _, shape_param = preprocess_image(
            tracked_image, mask_path=mask_path, intr=None,
            pad_ratio=0, bg_color=1.0, max_tgt_size=None,
            aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14,
            need_mask=True, get_shape_param=True,
        )

        preproc_vis_path = os.path.join(working_dir, "preprocessed_input.png")
        vis_img = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_img).save(preproc_vis_path)

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

        yield f"Step 4/{total_steps}: Running LAM inference...", None, None, tracked_image, preproc_vis_path

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

        # === Step 4: Generate GLB + ZIP ===
        yield f"Step 5/{total_steps}: Generating 3D avatar (Blender GLB)...", None, None, tracked_image, preproc_vis_path

        oac_dir = os.path.join(working_dir, "oac_export", base_iid)
        os.makedirs(oac_dir, exist_ok=True)

        saved_head_path = lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir,
        )
        assert os.path.isfile(saved_head_path), f"save_shaped_mesh failed: {saved_head_path}"

        skin_glb_path = Path(os.path.join(oac_dir, "skin.glb"))
        vertex_order_path = Path(os.path.join(oac_dir, "vertex_order.json"))
        template_fbx = Path("./model_zoo/sample_oac/template_file.fbx")
        blender_exec = Path("/usr/local/bin/blender")

        # If Blender not at /usr/local/bin, try PATH
        if not blender_exec.exists():
            blender_which = shutil.which("blender")
            if blender_which:
                blender_exec = Path(blender_which)

        # Write combined Blender script (GLB + vertex_order in one session)
        convert_script = Path(os.path.join(working_dir, "convert_and_order.py"))
        convert_script.write_text('''\
import bpy, sys, json
from pathlib import Path

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in [bpy.data.meshes, bpy.data.materials, bpy.data.textures]:
        for item in c:
            c.remove(item)

def strip_materials():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for tex in list(bpy.data.textures):
        bpy.data.textures.remove(tex)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)

argv = sys.argv[sys.argv.index("--") + 1:]
input_fbx = Path(argv[0])
output_glb = Path(argv[1])
output_vertex_order = Path(argv[2])

clean_scene()
bpy.ops.import_scene.fbx(filepath=str(input_fbx))

mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if len(mesh_objects) != 1:
    raise ValueError(f"Expected 1 mesh, found {len(mesh_objects)}")
mesh_obj = mesh_objects[0]

world_matrix = mesh_obj.matrix_world
vertices = [(i, (world_matrix @ v.co).z) for i, v in enumerate(mesh_obj.data.vertices)]
sorted_vertices = sorted(vertices, key=lambda x: x[1])
sorted_vertex_indices = [idx for idx, z in sorted_vertices]

with open(str(output_vertex_order), "w") as f:
    json.dump(sorted_vertex_indices, f)
print(f"vertex_order.json: {len(sorted_vertex_indices)} vertices")

strip_materials()
bpy.ops.export_scene.gltf(
    filepath=str(output_glb),
    export_format='GLB',
    export_skins=True,
    export_materials='NONE',
    export_normals=False,
    export_texcoords=False,
    export_morph_normal=False,
)
print("GLB + vertex_order export completed successfully")
''')

        temp_ascii = Path(os.path.join(working_dir, "temp_ascii.fbx"))
        temp_binary = Path(os.path.join(working_dir, "temp_bin.fbx"))

        try:
            update_flame_shape(Path(saved_head_path), temp_ascii, template_fbx)
            assert temp_ascii.exists(), f"update_flame_shape produced no output"

            convert_ascii_to_binary(temp_ascii, temp_binary)
            assert temp_binary.exists(), f"convert_ascii_to_binary produced no output"

            # Blender: FBX -> GLB + vertex_order.json
            cmd = [
                str(blender_exec), "--background",
                "--python", str(convert_script), "--",
                str(temp_binary), str(skin_glb_path), str(vertex_order_path),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                raise RuntimeError(
                    f"Blender exited with code {r.returncode}\n"
                    f"stdout: {r.stdout[-1000:]}\nstderr: {r.stderr[-1000:]}"
                )
            assert skin_glb_path.exists(), "skin.glb not created"
            assert vertex_order_path.exists(), "vertex_order.json not created"
        finally:
            for f in [temp_ascii, temp_binary]:
                if f.exists():
                    f.unlink()

        # Save PLY (FLAME vertex order, direct 1:1 mapping with GLB)
        res["cano_gs_lst"][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True,
        )

        # Copy template animation
        animation_src = "./model_zoo/sample_oac/animation.glb"
        if not os.path.isfile(animation_src):
            animation_src = "./assets/sample_oac/animation.glb"
        shutil.copy(src=animation_src, dst=os.path.join(oac_dir, "animation.glb"))

        if os.path.exists(saved_head_path):
            os.remove(saved_head_path)

        # Verify all required files
        required_files = ["offset.ply", "skin.glb", "vertex_order.json", "animation.glb"]
        missing = [f for f in required_files if not os.path.isfile(os.path.join(oac_dir, f))]
        if missing:
            raise RuntimeError(f"OAC export incomplete - missing: {', '.join(missing)}")

        # === Step 5: Create ZIP + preview ===
        yield f"Step {total_steps}/{total_steps}: Creating concierge.zip...", None, None, tracked_image, preproc_vis_path

        output_zip = os.path.join(OUTPUT_DIR, "concierge.zip")
        folder_name = os.path.basename(oac_dir)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            dir_info = zipfile.ZipInfo(folder_name + "/")
            zf.writestr(dir_info, "")
            for root, _dirs, files in os.walk(oac_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, os.path.dirname(oac_dir))
                    zf.write(fpath, arcname)

        # Generate preview video
        preview_path = os.path.join(OUTPUT_DIR, "preview.mp4")
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        from app_lam import save_images2video
        save_images2video(rgb, preview_path, 30)

        # Re-encode for browser compatibility
        preview_browser = os.path.join(OUTPUT_DIR, "preview_browser.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", preview_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "faststart", preview_browser],
            capture_output=True,
        )
        if os.path.isfile(preview_browser) and os.path.getsize(preview_browser) > 0:
            os.replace(preview_browser, preview_path)

        # Add audio if available
        final_preview = preview_path
        if video_path and os.path.isfile(video_path):
            try:
                from app_lam import add_audio_to_video
                preview_with_audio = os.path.join(OUTPUT_DIR, "preview_audio.mp4")
                add_audio_to_video(preview_path, preview_with_audio, video_path)
                preview_audio_browser = os.path.join(OUTPUT_DIR, "preview_audio_browser.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", preview_with_audio,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-movflags", "faststart",
                     preview_audio_browser],
                    capture_output=True,
                )
                if os.path.isfile(preview_audio_browser) and os.path.getsize(preview_audio_browser) > 0:
                    os.replace(preview_audio_browser, preview_with_audio)
                final_preview = preview_with_audio
            except Exception:
                pass

        zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
        num_motion_frames = len(os.listdir(flame_params_dir))

        yield (
            f"Done! concierge.zip ({zip_size_mb:.1f} MB) | "
            f"Motion: {motion_source} ({num_motion_frames} frames)",
            output_zip,
            final_preview,
            tracked_image,
            preproc_vis_path,
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n{'='*60}\nERROR\n{'='*60}\n{tb}\n{'='*60}", flush=True)
        yield f"Error: {str(e)}\n\nTraceback:\n{tb}", None, None, None, None


# ============================================================
# Gradio UI
# ============================================================
def build_ui(cfg, lam, flametracking):
    """Build the Gradio interface."""

    # Discover sample motions
    sample_motions = sorted(glob("./model_zoo/sample_motion/export/*/*.mp4"))
    if not sample_motions:
        sample_motions = sorted(glob("./assets/sample_motion/export/*/*.mp4"))

    def process(image_path, video_path, motion_choice):
        if image_path is None:
            yield "Error: Please upload a face image", None, None, None, None
            return

        effective_video = video_path if motion_choice == "custom" else None
        selected_motion = motion_choice if motion_choice != "custom" else None

        for status, zip_path, preview, tracked_img, preproc_img in generate_concierge_zip(
            image_path, effective_video, cfg, lam, flametracking,
            motion_name=selected_motion,
        ):
            yield status, zip_path, preview, tracked_img, preproc_img

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
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="1. Source Face Image",
                    type="filepath",
                    height=300,
                )

                motion_choices = ["custom"] + [
                    os.path.basename(os.path.dirname(m))
                    for m in sample_motions
                ]
                motion_choice = gr.Radio(
                    label="2. Motion Source",
                    choices=motion_choices,
                    value="custom",
                    info="Select 'custom' to upload your own video, or choose a sample",
                )

                input_video = gr.Video(
                    label="3. Custom Motion Video",
                    height=200,
                )

                gr.HTML(
                    '<div class="tip-box">'
                    "<b>Input image requirements:</b><br>"
                    "- Must be a <b>real photograph</b> (not illustration/AI art)<br>"
                    "- Front-facing, good lighting, neutral expression<br>"
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
                    lines=3,
                )

            with gr.Column(scale=1):
                with gr.Row():
                    tracked_face = gr.Image(
                        label="Tracked Face (FLAME output)",
                        height=200,
                    )
                    preproc_image = gr.Image(
                        label="Model Input (what LAM sees)",
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
                    "**Usage:** Place the downloaded `concierge.zip` at "
                    "`gourmet-sp/public/avatar/concierge.zip` for LAMAvatar."
                )

        generate_btn.click(
            fn=process,
            inputs=[input_image, input_video, motion_choice],
            outputs=[status_text, output_file, preview_video, tracked_face, preproc_image],
        )

    return demo


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Monkey-patch torch.utils.cpp_extension.load for nvdiffrast JIT
    import torch.utils.cpp_extension as _cext
    _orig_load = _cext.load
    def _patched_load(*args, **kwargs):
        cflags = list(kwargs.get("extra_cflags", []) or [])
        if "-Wno-c++11-narrowing" not in cflags:
            cflags.append("-Wno-c++11-narrowing")
        kwargs["extra_cflags"] = cflags
        return _orig_load(*args, **kwargs)
    _cext.load = _patched_load

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    print("=" * 60)
    print("Concierge ZIP Generator (HF Spaces / Docker)")
    print("=" * 60)

    print("\nInitializing pipeline...")
    cfg, lam, flametracking = init_pipeline()

    print("\nBuilding Gradio UI...")
    demo = build_ui(cfg, lam, flametracking)

    print("\nLaunching server on 0.0.0.0:7860...")
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
