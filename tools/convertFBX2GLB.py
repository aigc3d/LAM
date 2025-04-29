"""
Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.

Blender FBX to GLB Converter
Converts 3D models from FBX to glTF Binary (GLB) format with optimized settings.
Requires Blender to run in background mode.
"""

import bpy
import sys
from pathlib import Path

def clean_scene():
    """Clear all objects and data from the current Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures]:
        for item in collection:
            collection.remove(item)


def main():
    try:
        # Parse command line arguments after "--"
        argv = sys.argv[sys.argv.index("--") + 1:]
        input_fbx = Path(argv[0])
        output_glb = Path(argv[1])

        # Validate input file
        if not input_fbx.exists():
            raise FileNotFoundError(f"Input FBX file not found: {input_fbx}")

        # Prepare scene
        clean_scene()

        # Import FBX with default settings
        print(f"Importing {input_fbx}...")
        bpy.ops.import_scene.fbx(filepath=str(input_fbx))

        # Export optimized GLB
        print(f"Exporting to {output_glb}...")
        bpy.ops.export_scene.gltf(
            filepath=str(output_glb),
            export_format='GLB',          # Binary format
            export_skins=True,            # Keep skinning data
            export_texcoords=False,       # Reduce file size
            export_normals=False,         # Reduce file size
            export_colors=False,          # Reduce file size
        )

        print("Conversion completed successfully")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
