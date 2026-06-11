import torch
import numpy as np
import math
import torch.nn as nn

from pytorch3d.structures       import Meshes
from pytorch3d.io               import load_obj
from pytorch3d.renderer.mesh    import rasterize_meshes
from pytorch3d.ops              import mesh_face_areas_normals

#-------------------------------------------------------------------------------#

def gen_tritex(vt: np.ndarray, vi: np.ndarray, vti: np.ndarray, texsize: int):
    """
    Copied from MVP
    Create 3 texture maps containing the vertex indices, texture vertex
    indices, and barycentric coordinates

    Parameters
    ----------
        vt: uv coordinates of texels
        vi: triangle list mapping into vertex positions
        vti: triangle list mapping into texel coordinates
        texsize: Size of the generated maps
    """
    # vt = ((vt + 1. ) / 2.)[:, :2]
    vt = vt[:, :2]

    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    ntris = vi.shape[0]

    texu, texv = np.meshgrid(
        (np.arange(texsize) + 0.5) / texsize,
        (np.arange(texsize) + 0.5) / texsize)
    texuv = np.stack((texu, texv), axis=-1)

    vt = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)

    for i in list(range(ntris))[::-1]:
        bbox = (
            max(0, int(min(vt[i, 0, 0], min(vt[i, 1, 0], vt[i, 2, 0])) * texsize) - 1),
            min(texsize, int(max(vt[i, 0, 0], max(vt[i, 1, 0], vt[i, 2, 0])) * texsize) + 2),
            max(0, int(min(vt[i, 0, 1], min(vt[i, 1, 1], vt[i, 2, 1])) * texsize) - 1),
            min(texsize, int(max(vt[i, 0, 1], max(vt[i, 1, 1], vt[i, 2, 1])) * texsize) + 2))
        v0 = vt[None, None, i, 1, :] - vt[None, None, i, 0, :]
        v1 = vt[None, None, i, 2, :] - vt[None, None, i, 0, :]
        v2 = texuv[bbox[2]:bbox[3], bbox[0]:bbox[1], :] - vt[None, None, i, 0, :]
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01

        if denom != 0.:
            baryv = (d11 * d20 - d01 * d21) / denom
            baryw = (d00 * d21 - d01 * d20) / denom
            baryu = 1. - baryv - baryw

            baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                np.stack((baryu, baryv, baryw), axis=-1),
                baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                np.stack((vi[i, 0], vi[i, 1], vi[i, 2]), axis=-1),
                viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                np.stack((vti[i, 0], vti[i, 1], vti[i, 2]), axis=-1),
                vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

    return torch.LongTensor(viim), torch.Tensor(vtiim), torch.Tensor(baryim)


# modified from https://github.com/facebookresearch/pytorch3d
class Pytorch3dRasterizer(nn.Module):
    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
            'cull_backfaces': True
        }
        # raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings['image_size']
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings['blur_radius'],
            faces_per_pixel=raster_settings['faces_per_pixel'],
            bin_size=raster_settings['bin_size'],
            max_faces_per_bin=raster_settings['max_faces_per_bin'],
            perspective_correct=raster_settings['perspective_correct'],
            cull_backfaces=raster_settings['cull_backfaces']
        )

        return pix_to_face, bary_coords
    
#-------------------------------------------------------------------------------#

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    Indexing the coordinates of the three vertices on each face.

    Args:
        vertices:   [bs, V, 3]
        faces:      [bs, F, 3]

    Return: 
        face_to_vertices: [bs, F, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    # assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def uniform_sampling_barycoords(
        num_points:  int,
        tex_coord:   torch.Tensor,
        uv_faces:    torch.Tensor,
        d_size:      float=1.0,
        strict:      bool=False,
        use_mask:    bool=True,
    ):
    """
    Uniformly sampling barycentric coordinates using the rasterizer.

    Args:
        num_points:     int                 sampling points number
        tex_coord:      [5150, 2]           UV coords for each vert
        uv_faces:       [F,3]               UV faces to UV coords index
        d_size:         const               to control sampling points number
        use_mask:       use mask to mask valid points
    Returns:
        face_index      [num_points]        save which face each bary_coords belongs to
        bary_coords     [num_points, 3]
    """
    
    uv_size = int(math.sqrt(num_points) * d_size)
    uv_rasterizer = Pytorch3dRasterizer(uv_size)

    tex_coord   = tex_coord[None, ...]
    uv_faces    = uv_faces[None, ...]

    tex_coord_ = torch.cat([tex_coord, tex_coord[:,:,0:1]*0.+1.], -1)
    tex_coord_ = tex_coord_ * 2 - 1 
    tex_coord_[...,1] = - tex_coord_[...,1]

    pix_to_face, bary_coords = uv_rasterizer(tex_coord_.expand(1, -1, -1), uv_faces.expand(1, -1, -1))
    mask = (pix_to_face == -1)

    if use_mask:
        face_index = pix_to_face[~mask]
        bary_coords = bary_coords[~mask]
    else:
        return pix_to_face, bary_coords

    cur_n = face_index.shape[0]

    # fix sampling number to num_points
    if strict:
        if cur_n < num_points:
            pad_size        = num_points - cur_n
            new_face_index  = face_index[torch.randint(0, cur_n, (pad_size,))]
            new_bary_coords = torch.rand((pad_size, 3), device=bary_coords.device)
            new_bary_coords = new_bary_coords / new_bary_coords.sum(dim=-1, keepdim=True)
            face_index      = torch.cat([face_index, new_face_index], dim=0)
            bary_coords     = torch.cat([bary_coords, new_bary_coords], dim=0)
        elif cur_n > num_points:
            face_index  = face_index[:num_points]
            bary_coords = bary_coords[:num_points]

    return face_index, bary_coords

def random_sampling_barycoords(
        num_points:   int,
        vertices:     torch.Tensor,
        faces:        torch.Tensor
    ):
    """
    Randomly sampling barycentric coordinates using the rasterizer.

    Args:
        num_points:     int                 sampling points number
        vertices:       [V, 3]           
        faces:          [F,3]
    Returns:
        face_index      [num_points]        save which face each bary_coords belongs to
        bary_coords     [num_points, 3]
    """

    areas, _ = mesh_face_areas_normals(vertices.squeeze(0), faces)

    g1 = torch.Generator(device=vertices.device)
    g1.manual_seed(0)

    face_index = areas.multinomial(
            num_points, replacement=True, generator=g1
        )  # (N, num_samples)

    uvw = torch.rand((face_index.shape[0], 3), device=vertices.device)
    bary_coords = uvw / uvw.sum(dim=-1, keepdim=True)

    return face_index, bary_coords

def reweight_verts_by_barycoords(
        verts:       torch.Tensor,
        faces:       torch.Tensor,
        face_index:  torch.Tensor,
        bary_coords: torch.Tensor,
    ):
    """
    Reweights the vertices based on the barycentric coordinates for each face.

    Args:
        verts:          [bs, V, 3].
        faces:          [F, 3]
        face_index:     [N].
        bary_coords:    [N, 3].

    Returns:
        Reweighted vertex positions of shape [bs, N, 3].
    """
    
    # index attributes by face
    B               = verts.shape[0]

    face_verts      = face_vertices(verts,  faces.expand(B, -1, -1))   # [1, F, 3, 3]
    # gather idnex for every splat
    N               = face_index.shape[0]
    face_index_3    = face_index.view(1, N, 1, 1).expand(B, N, 3, 3)
    position_vals   = face_verts.gather(1, face_index_3)
    # reweight
    position_vals   = (bary_coords[..., None] * position_vals).sum(dim = -2)

    return position_vals

def reweight_uvcoords_by_barycoords(
        uvcoords:    torch.Tensor,
        uvfaces:     torch.Tensor,
        face_index:  torch.Tensor,
        bary_coords: torch.Tensor,
    ):
    """
    Reweights the UV coordinates based on the barycentric coordinates for each face.

    Args:
        uvcoords:       [bs, V', 2].
        uvfaces:        [F, 3].
        face_index:     [N].
        bary_coords:    [N, 3].

    Returns:
        Reweighted UV coordinates, shape [bs, N, 2].
    """

    # homogeneous coordinates
    num_v           = uvcoords.shape[0]
    uvcoords        = torch.cat([uvcoords, torch.ones((num_v, 1)).to(uvcoords.device)], dim=1)
    # index attributes by face
    uvcoords        = uvcoords[None, ...]
    face_verts      = face_vertices(uvcoords,  uvfaces.expand(1, -1, -1))   # [1, F, 3, 3]
    # gather idnex for every splat
    N               = face_index.shape[0]
    face_index_3    = face_index.view(1, N, 1, 1).expand(1, N, 3, 3)
    position_vals   = face_verts.gather(1, face_index_3)
    # reweight
    position_vals   = (bary_coords[..., None] * position_vals).sum(dim = -2)

    return position_vals

# modified from https://github.com/computational-imaging/GSM/blob/main/main/gsm/deformer/util.py
def get_shell_verts_from_base(
        template_verts: torch.Tensor,
        template_faces: torch.Tensor,
        offset_len: float,
        num_shells: int,
        deflat = False,
    ):
    """
    Generates shell vertices by offsetting the original mesh's vertices along their normals.

    Args:
        template_verts: [bs, V, 3].
        template_faces: [F, 3].
        offset_len:     Positive number specifying the offset length for generating shells.
        num_shells:     The number of shells to generate.
        deflat:         If True, performs a deflation process. Defaults to False.

    Returns:
        shell verts:    [bs, num_shells, n, 3]
    """
    out_offset_len = offset_len

    if deflat:
        in_offset_len = offset_len

    batch_size = template_verts.shape[0]
    mesh = Meshes(
        verts=template_verts, faces=template_faces[None].repeat(batch_size, 1, 1)
    )
    # bs, n, 3
    vertex_normal = mesh.verts_normals_padded()
    # only for inflating

    if deflat:
        n_inflated_shells = num_shells//2 + 1
    else:
        n_inflated_shells = num_shells
    
    linscale = torch.linspace(
        out_offset_len,
        0,
        n_inflated_shells,
        device=template_verts.device,
        dtype=template_verts.dtype,
    )
    offset = linscale.reshape(1,n_inflated_shells, 1, 1) * vertex_normal[:, None]
    
    if deflat:
        linscale = torch.linspace(0, -in_offset_len, num_shells - n_inflated_shells + 1, device=template_verts.device, dtype=template_verts.dtype)[1:]
        offset_in = linscale.reshape(1, -1, 1, 1) * vertex_normal[:, None]
        offset = torch.cat([offset, offset_in], dim=1)

    verts = template_verts[:, None] + offset
    assert verts.isfinite().all()
    return verts