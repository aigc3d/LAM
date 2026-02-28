import json
import bpy
import os

def import_obj(filepath):
    """导入OBJ文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在：{filepath}")
    bpy.ops.wm.obj_import(filepath=filepath)
    print(f"成功导入：{filepath}")

def create_armature_from_bone_tree(obj, bone_tree_path):
    """根据骨骼树JSON创建骨骼系统并绑定到模型"""
    with open(bone_tree_path, 'r') as f:
        bones_data = json.load(f)['bones'][0]

    # 创建新骨骼
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "ModelArmature"
    edit_bones = armature.data.edit_bones

    # 创建骨骼树的递归函数
    # root_bone = edit_bones.new(bones_data['name'])
    # print(bones_data['name'])
    def recursive_create_bones(parent_bone, bone_list):
        for bone_data in bone_list:
            new_bone = edit_bones.new(bone_data['name'])
            print(new_bone.name)
            new_bone.head = bone_data['position']
            new_bone.tail = [*new_bone.head[:2], new_bone.head[2]+1]  # 设置轴向长度
            if parent_bone:
                new_bone.parent = parent_bone
            if 'children' in bone_data:
                recursive_create_bones(new_bone, bone_data['children'])
            # recursive_create_bones(new_bone, bone_list=bone_data.get('children', []))

    # 从根骨骼开始构建
    # print(bones_data.get('children', []))
    recursive_create_bones(None, [bones_data])  

    # 设置骨骼与模型绑定
    obj.parent = armature
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = armature
    mod.use_vertex_groups = True

    # 创建顶点组（与骨骼同名）
    # print(armature.data.edit_bones)
    for bone in armature.data.edit_bones:
        # print(bone.name)
        if bone.name not in obj.vertex_groups:
            obj.vertex_groups.new(name=bone.name)
        else:
            print(f"顶点组 {bone.name} 已存在")

    
    # 返回创建的骨骼对象用于后续操作
    bpy.ops.object.mode_set(mode='OBJECT')
    return armature

def apply_vertex_weights(obj, weight_file):
    """根据权重JSON设置顶点权重"""
    with open(weight_file, 'r') as f:
        js_data = json.load(f)
        try:
            weights = js_data.get('vertex_weights', {})
        except:
            weights = js_data

    # 获取所有顶点组名称
    bpy.ops.object.mode_set(mode='OBJECT')
    existing_vertex_groups = {vg.name: vg for vg in obj.vertex_groups}

    # print(existing_vertex_groups)
    # 遍历每个顶点的权重
    for vert_idx, weight_dict in enumerate(weights):
        if vert_idx >= len(obj.data.vertices):
            print(f"顶点索引 {vert_idx} 超出范围")
            continue

        # 清除当前顶点的所有权重
        for group in obj.vertex_groups:
            if vert_idx < len(obj.data.vertices):
                group.remove([vert_idx])

        # 重新分配权重
        bones = ['root', 'neck', 'jaw', 'leftEye', 'rightEye']
        #  for vert_idx, weight_dict in enumerate(weights):
        
        # print(obj.vertex_groups)
        for bone_idx, weight in enumerate(weight_dict):
            bone_name = bones[bone_idx]
            # print(bone_name)
            if bone_name not in existing_vertex_groups:
                print(f"顶点组 {bone_name} 不存在，跳过")
                continue

            vgroup = existing_vertex_groups[bone_name]
            if vgroup and vert_idx < len(obj.data.vertices):
                vgroup.add([vert_idx], weight, 'REPLACE')
            else:
                print(f"顶点索引 {vert_idx} 或顶点组 {bone_name} 无效")

def add_shape_keys(base_obj, bs_obj_files):
    """添加多个Shape Keys（表情文件）"""
    if not base_obj.data.shape_keys:
        base_obj.shape_key_add(name="Basis")
    
    for idx, path in enumerate(bs_obj_files):
        if not os.path.exists(path):
            print(f"表情文件缺失：{path}")
            continue
        # 导入表情模型（需保持基础模型选中）
        bpy.ops.wm.obj_import(filepath=path)
        imported_obj = [obj for obj in bpy.data.objects if obj.select_get()][0]
        
        # 创建新的Shape Key
        new_sk_name = os.path.basename(path).split('.')[0]
        new_sk = base_obj.shape_key_add(name=new_sk_name)
        
        # 复制顶点位置到Shape Key
        for v in base_obj.data.vertices:
            if v.index < len(imported_obj.data.vertices):
                new_sk.data[v.index].co = imported_obj.data.vertices[v.index].co
        
        # 清理临时对象
        bpy.data.objects.remove(imported_obj)

def layout_bones_pose(armature, pose_config):
    """设置骨骼的初始姿势（可选）"""
    if pose_config:
        with open(pose_config, 'r') as f:
            pose_data = json.load(f)
        for bone in armature.pose.bones:
            if bone.name in pose_data:
                bone.rotation_euler = tuple(pose_data[bone.name]['rotation'])
                bone.location = tuple(pose_data[bone.name]['location'])

def apply_rotation(obj):
    """手动应用 90 度旋转（绕 X 轴）并将变换应用到模型"""
    obj.rotation_euler = (1.5708, 0, 0)  # 90 度旋转（弧度制：1.5708 = π/2）
    bpy.context.view_layer.update()      # 更新场景以应用旋转
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)  # 应用旋转
    print(f"Applied 90-degree rotation to object: {obj.name}")

def export_as_glb(obj, output_path, output_vertex_order_file):
    """导出为GLB格式，确保包含骨骼信息"""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # 切换到对象模式
    bpy.ops.object.mode_set(mode='OBJECT') 

    # 获取基础模型
    base_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if len(base_objects) != 1:
        raise ValueError("Scene should contain exactly one base mesh object.")
    base_obj = base_objects[0]

    # 获取顶点数据
    vertices = [(i, v.co.z) for i, v in enumerate(base_obj.data.vertices)]

    # 根据 Z 轴坐标排序
    sorted_vertices = sorted(vertices, key=lambda x: x[1])  # 按 Z 坐标从小到大排序
    sorted_vertex_indices = [idx for idx, z in sorted_vertices]

    # 输出顶点顺序到文件
    with open(output_vertex_order_file, "w") as f:
        json.dump(sorted_vertex_indices, f, indent=4)  # 保存为 JSON 数组
    print(f"Exported vertex order to: {output_vertex_order_file}")
    
    # 执行导出
    bpy.ops.export_scene.gltf(filepath=output_path, 
                            export_format='GLB',
                            export_skins=True,
                            export_texcoords=False,         # 不导出 UV 数据
                            export_normals=False            # 不导出法线数据
                            )
    print(f"导出成功：{output_path}")

def main():
    base_model_path = "runtime_data/nature.obj"  # 基础模型路径
    expression_dir = "runtime_data/bs"  # 表情文件夹
    bone_tree_path = "runtime_data/bone_tree.json"  # 骨骼结构配置
    weight_data_path = "runtime_data/lbs_weight_20k.json"  # 权重数据
    output_glb_path = "runtime_data/skin.glb"
    output_vertex_order_file = "runtime_data/vertex_order.json"  # 输出顶点顺序文件

    # 清空场景
    bpy.ops.wm.read_homefile(use_empty=True)

    # 导入基础模型
    import_obj(base_model_path)
    base_obj = bpy.context.view_layer.objects.active

    # 创建骨骼系统
    armature = create_armature_from_bone_tree(base_obj, bone_tree_path)
    
    # 设置骨骼姿势（如果需要）
    # layout_bones_pose(armature, "pose_config.json") 

    # 应用顶点权重
    apply_vertex_weights(base_obj, weight_data_path)

    # 加载所有Shape Keys（表情）
    expression_files = [
        os.path.join(expression_dir, f)
        for f in os.listdir(expression_dir) 
        if f.endswith(('.obj', '.OBJ'))
    ]
    add_shape_keys(base_obj, expression_files)
    apply_rotation(base_obj)

    # 导出为GLB格式
    export_as_glb(base_obj, output_glb_path, output_vertex_order_file)

if __name__ == "__main__":
    main()
