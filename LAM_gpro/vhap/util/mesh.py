# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import torch


def get_mtl_content(tex_fname):
    return f'newmtl Material\nmap_Kd {tex_fname}\n'

def get_obj_content(vertices, faces, uv_coordinates=None, uv_indices=None, mtl_fname=None):
    obj = ('# Generated with multi-view-head-tracker\n')

    if mtl_fname is not None:
        obj += f'mtllib {mtl_fname}\n'
        obj += 'usemtl Material\n'

    # Write the vertices
    for vertex in vertices:
        obj += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"

    # Write the UV coordinates
    if uv_coordinates is not None:
        for uv in uv_coordinates:
            obj += f"vt {uv[0]} {uv[1]}\n"

    # Write the faces with UV indices
    if uv_indices is not None:
        for face, uv_indices in zip(faces, uv_indices):
            obj += f"f {face[0]+1}/{uv_indices[0]+1} {face[1]+1}/{uv_indices[1]+1} {face[2]+1}/{uv_indices[2]+1}\n"
    else:
        for face in faces:
            obj += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    return obj

def normalize_image_points(u, v, resolution):
    """
    normalizes u, v coordinates from [0 ,image_size] to [-1, 1]
    :param u:
    :param v:
    :param resolution:
    :return:
    """
    u = 2 * (u - resolution[1] / 2.0) / resolution[1]
    v = 2 * (v - resolution[0] / 2.0) / resolution[0]
    return u, v


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

