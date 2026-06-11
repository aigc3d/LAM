# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import tyro
import matplotlib.pyplot as plt
import numpy as np
import torch
import nvdiffrast.torch as dr

from vhap.model.flame import FlameHead


FLAME_TEX_PATH = "asset/flame/FLAME_texture.npz"


def transform_vt(vt):
    """Transform uv vertices to clip space"""
    xy = vt * 2 - 1
    w = torch.ones([1, vt.shape[-2], 1]).to(vt)
    z = -w  # In the clip spcae of OpenGL, the camera looks at -z
    xyzw = torch.cat([xy[None, :, :], z, w], axis=-1)    
    return xyzw

def render_uvmap_vtex(glctx, pos, pos_idx, v_color, col_idx, resolution):
    """Render uv map with vertex color"""
    pos_clip = transform_vt(pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution)
    
    color, _ = dr.interpolate(v_color, rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def render_uvmap_texmap(glctx, pos, pos_idx, verts_uv, faces_uv, tex, resolution, enable_mip=True, max_mip_level=None):
    """Render uv map with texture map"""
    pos_clip = transform_vt(pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution)

    if enable_mip:
        texc, texd = dr.interpolate(verts_uv[None, ...], rast_out, faces_uv, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(verts_uv[None, ...], rast_out, faces_uv)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color


def main(
    use_texmap: bool = False,
    use_opengl: bool = False,
):
    n_shape = 300
    n_expr = 100
    print("Initialization FLAME model")
    flame_model = FlameHead(n_shape, n_expr)

    verts_uv = flame_model.verts_uvs.cuda()
    verts_uv[:, 1] = 1 - verts_uv[:, 1]
    faces_uv = flame_model.textures_idx.int().cuda()

    # Rasterizer context
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    h, w = 512, 512
    resolution = (h, w)
    
    if use_texmap:
        tex = torch.from_numpy(np.load(FLAME_TEX_PATH)['mean']).cuda().float().flip(dims=[-1]) / 255
        rgb = render_uvmap_texmap(glctx, verts_uv, faces_uv, verts_uv, faces_uv, tex, resolution, enable_mip=True)
    else:
        v_color = torch.ones(verts_uv.shape[0], 3).to(verts_uv)
        col_idx = faces_uv
        rgb = render_uvmap_vtex(glctx, verts_uv, faces_uv, v_color, col_idx, resolution)
    
    plt.imshow(rgb[0, :, :].cpu())
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
