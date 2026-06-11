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


def strip_materials():
    """Remove all materials, textures, and images after FBX import.

    The OAC renderer only uses mesh geometry and bone weights.
    Embedded FBX textures bloat the GLB from ~3.6MB to ~43.5MB.
    """
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for tex in list(bpy.data.textures):
        bpy.data.textures.remove(tex)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


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

        # Strip materials/textures — OAC renderer only needs geometry + skins.
        # FBX templates embed textures that bloat GLB from ~3.6MB to ~43.5MB.
        strip_materials()

        # Export optimized GLB — OAC renderer only needs positions + skin weights.
        # NOTE: Blender 4.2 renamed export_colors → export_vertex_color but
        # export_normals and export_texcoords are still valid.
        # CRITICAL: export_morph_normal defaults to True and exports normals
        # for every morph target (blend shape). With 100+ FLAME blend shapes
        # this adds ~48MB. Setting it to False is the primary size fix.
        print(f"Exporting to {output_glb}...")
        bpy.ops.export_scene.gltf(
            filepath=str(output_glb),
            export_format='GLB',          # Binary format
            export_skins=True,            # Keep skinning data
            export_materials='NONE',      # No materials/textures
            export_normals=False,         # OAC renderer doesn't use normals
            export_texcoords=False,       # No UV maps needed
            export_morph_normal=False,    # Morph target normals cause massive bloat
        )

        print("Conversion completed successfully")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
