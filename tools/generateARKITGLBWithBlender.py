"""
Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.

FLAME Model FBX/GLB Converter
A pipeline for processing FLAME 3D models including:
1. Shape parameter injection into FBX templates
2. FBX format conversion (ASCII <-> Binary)
3. GLB export via Blender
"""


import os.path

import numpy as np
import logging
import subprocess
from pathlib import Path
import trimesh

try:
    import fbx
except ImportError:
    raise RuntimeError(
        "FBX SDK required: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-2")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def update_flame_shape(
        input_mesh: Path,
        output_ascii_fbx: Path,
        template_fbx: Path
) -> None:
    """
    Injects FLAME shape parameters into FBX template

    Args:
        input_mesh: Path to FLAME mesh (OBJ format)
        output_ascii_fbx: Output path for modified ASCII FBX
        template_fbx: Template FBX with FLAME structure

    Raises:
        FileNotFoundError: If input files are missing
        ValueError: If template format mismatch
    """
    logger.info(f"Updating FLAME shape in {template_fbx}")

    # Validate inputs
    if not all([input_mesh.exists(), template_fbx.exists()]):
        raise FileNotFoundError("Missing input file(s)")

    # Load and process FLAME mesh
    mesh = trimesh.load(input_mesh)
    bs_verts = np.array(mesh.vertices).flatten()
    verts_csv = ",".join([f"{v:.6f}" for v in bs_verts]) + ","

    # Read template FBX
    with template_fbx.open('r',encoding='utf-8') as f:
        template_lines = f.readlines()
    f.close()
    # Replace vertex data section
    output_lines = []
    vertex_section = False
    VERTEX_HEADER = "Vertices: *60054 {"  # FLAME-specific vertex count

    for line in template_lines:
        if VERTEX_HEADER in line:
            vertex_section = True
            output_lines.append(line)
            # Inject new vertex data
            output_lines.extend([f"        {v}\n" for v in verts_csv.split(",") if v])
            continue

        if vertex_section:
            if '}' in line:
                vertex_section = False
                output_lines.append(line)
            continue

        output_lines.append(line)

    # Write modified FBX
    with output_ascii_fbx.open('w',encoding='utf-8') as f:
        f.writelines(output_lines)
    f.close()
    logger.info(f"Generated updated ASCII FBX: {output_ascii_fbx}")


def convert_ascii_to_binary(
        input_ascii: Path,
        output_binary: Path
) -> None:
    """
    Converts FBX between ASCII and Binary formats

    Args:
        input_ascii: Path to ASCII FBX
        output_binary: Output path for binary FBX

    Raises:
        RuntimeError: If conversion fails
    """
    logger.info(f"Converting {input_ascii} to binary FBX")

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    try:
        # Initialize scene
        scene = fbx.FbxScene.Create(manager, "ConversionScene")

        # Import ASCII
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(str(input_ascii), -1, manager.GetIOSettings()):
            raise RuntimeError(f"FBX import failed: {importer.GetStatus().GetErrorString()}")
        importer.Import(scene)

        # Export Binary
        exporter = fbx.FbxExporter.Create(manager, "")
        if not exporter.Initialize(str(output_binary), 0, manager.GetIOSettings()):
            raise RuntimeError(f"FBX export failed: {exporter.GetStatus().GetErrorString()}")
        exporter.Export(scene)

    finally:
        # Cleanup FBX SDK resources
        scene.Destroy()
        importer.Destroy()
        exporter.Destroy()
        manager.Destroy()

    logger.info(f"Binary FBX saved to {output_binary}")


def convert_with_blender(
        input_fbx: Path,
        output_glb: Path,
        blender_exec: Path = Path("blender"),
        input_mesh:  Path = Path("input_mesh.obj"),
) -> None:
    """
    Converts FBX to GLB using Blender

    Args:
        input_fbx: Path to input FBX
        output_glb: Output GLB path
        blender_exec: Path to Blender executable

    Raises:
        CalledProcessError: If Blender conversion fails
    """
    logger.info(f"Starting Blender conversion to GLB")

    cmd = [
        str(blender_exec),
        "--background",
        "--python", "tools/convertFBX2GLB.py",  # Path to conversion script
        "--", str(input_fbx), str(output_glb)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

    except subprocess.CalledProcessError as e:
        logger.error(f"Blender conversion failed: {e.stderr}")
        raise
    logger.info(f"GLB output saved to {output_glb}")

def gen_vertex_order_with_blender(
        input_mesh: Path,
        output_json: Path,
        blender_exec: Path = Path("blender"),
) -> None:
    """
    Args:
        input_mesh: Path to input mesh
        output_json: Output json path
        blender_exec: Path to Blender executable

    Raises:
        CalledProcessError: If Blender conversion fails
    """
    logger.info(f"Starting Generation Vertex Order")

    cmd = [
        str(blender_exec),
        "--background",
        "--python", "tools/generateVertexIndices.py",  # Path to conversion script
        "--", str(input_mesh), str(output_json)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(f"Blender conversion failed: {e.stderr}")
        raise
    logger.info(f"Vertex Order output saved to {output_json}")


def generate_glb(
        input_mesh: Path,
        template_fbx: Path,
        output_glb: Path,
        blender_exec: Path = Path("blender"),
        cleanup: bool = True
) -> None:
    """
    Complete pipeline for FLAME GLB generation

    Args:
        input_mesh: Input FLAME mesh (OBJ)
        template_fbx: Template FBX file
        output_glb: Final GLB output
        blender_exec: Blender executable path
        cleanup: Remove temporary files
    """
    temp_files = {
        "ascii": Path("./temp_ascii.fbx"),
        "binary": Path("./temp_bin.fbx")
    }

    try:
        # Step 1: Shape parameter injection
        update_flame_shape(input_mesh, temp_files["ascii"], template_fbx)

        # Step 2: FBX format conversion
        convert_ascii_to_binary(temp_files["ascii"], temp_files["binary"])

        # Step 3: Blender conversion
        convert_with_blender(temp_files["binary"], output_glb, blender_exec)

        # Step 4: Vertex Order Generation
        gen_vertex_order_with_blender(input_mesh,
                                      Path(os.path.join(os.path.dirname(output_glb),'vertex_order.json')),
                                      blender_exec)

    finally:
        # Cleanup temporary files
        if cleanup:
            for f in temp_files.values():
                if f.exists():
                    f.unlink()
            logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    # Example usage
    generate_glb(
        input_mesh=Path("./asserts/sample_oac/nature.obj"),
        template_fbx=Path("./asserts/sample_oac/template_file.fbx"),
        output_glb=Path("./asserts/sample_oac/skin.glb"),
        blender_exec=Path("./blender-4.0.0-linux-x64/blender")
    )