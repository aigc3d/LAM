"""
Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.

Blender FBX to GLB Converter
Converts 3D models from FBX to glTF Binary (GLB) format with optimized settings.
Requires Blender to run in background mode.
"""

import bpy
import sys
import os
import json
from pathlib import Path

def import_obj(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在：{filepath}")
    bpy.ops.wm.obj_import(filepath=filepath)
    print(f"成功导入：{filepath}")


def clean_scene():
    """Clear all objects and data from the current Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures]:
        for item in collection:
            collection.remove(item)

def apply_rotation(obj):
    obj.rotation_euler = (1.5708, 0, 0)
    bpy.context.view_layer.update()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)  # 应用旋转
    print(f"Applied 90-degree rotation to object: {obj.name}")

def main():
    try:
        # Parse command line arguments after "--"
        argv = sys.argv[sys.argv.index("--") + 1:]
        input_mesh = Path(argv[0])
        output_vertex_order_file  = argv[1]

        # Validate input file
        if not input_mesh.exists():
            raise FileNotFoundError(f"Input FBX file not found: {input_mesh}")

        # Prepare scene
        clean_scene()

        # Import FBX with default settings
        print(f"Importing {input_mesh}...")
        import_obj(str(input_mesh))
        base_obj = bpy.context.view_layer.objects.active

        apply_rotation(base_obj)

        bpy.context.view_layer.objects.active = base_obj
        base_obj.select_set(True)
        bpy.ops.object.mode_set(mode='OBJECT')

        base_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if len(base_objects) != 1:
            raise ValueError("Scene should contain exactly one base mesh object.")
        base_obj = base_objects[0]

        vertices = [(i, v.co.z) for i, v in enumerate(base_obj.data.vertices)]

        sorted_vertices = sorted(vertices, key=lambda x: x[1])  # 按 Z 坐标从小到大排序
        sorted_vertex_indices = [idx for idx, z in sorted_vertices]

        with open(str(output_vertex_order_file), "w") as f:
            json.dump(sorted_vertex_indices, f, indent=4)  # 保存为 JSON 数组
        print(f"Exported vertex order to: {str(output_vertex_order_file)}")


    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
