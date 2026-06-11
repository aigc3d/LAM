# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from typing import Literal
import tyro
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import nvdiffrast.torch as dr
from vhap.util.render_uvmap import render_uvmap_vtex
from vhap.model.flame import FlameHead


FLAME_UV_MASK_FOLDER = "asset/flame/uv_masks"
FLAME_UV_MASK_NPZ = "asset/flame/uv_masks.npz"


def main(
    use_opengl: bool = False,
    device: Literal['cuda', 'cpu'] = 'cuda',
):
    n_shape = 300
    n_expr = 100
    print("Initializing FLAME model")
    flame_model = FlameHead(n_shape, n_expr, add_teeth=True)

    flame_model = FlameHead(
        n_shape, 
        n_expr, 
        add_teeth=True,
    ).cuda()

    faces = flame_model.faces.int().cuda()
    verts_uv = flame_model.verts_uvs.cuda()
    # verts_uv[:, 1] = 1 - verts_uv[:, 1]
    faces_uv = flame_model.textures_idx.int().cuda()
    col_idx = faces_uv

    # Rasterizer context
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    h, w = 2048, 2048
    resolution = (h, w)

    if not Path(FLAME_UV_MASK_FOLDER).exists():
        Path(FLAME_UV_MASK_FOLDER).mkdir(parents=True)
    
    # alpha_maps = {}
    masks = {}
    for region, vt_mask in flame_model.mask.vt:
        v_color = torch.zeros(verts_uv.shape[0], 1).to(device)  # alpha channel
        v_color[vt_mask] = 1

        alpha = render_uvmap_vtex(glctx, verts_uv, faces_uv, v_color, col_idx, resolution)[0]
        alpha = alpha.flip(0)
        # alpha_maps[region] = alpha.cpu().numpy()
        mask = (alpha > 0.5)  # to avoid overlap between hair and face
        mask = mask.squeeze(-1).cpu().numpy()
        masks[region] = mask  # (h, w)

        print(f"Saving uv mask for {region}...")
        # rgba = mask.expand(-1, -1, 4)  # (h, w, 4)
        # rgb = torch.ones_like(mask).expand(-1, -1, 3)  # (h, w, 3)
        # rgba = torch.cat([rgb, mask], dim=-1).cpu().numpy()  # (h, w, 4)
        img = mask
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(Path(FLAME_UV_MASK_FOLDER) / f"{region}.png")
    
    print(f"Saving uv mask into: {FLAME_UV_MASK_NPZ}")
    np.savez_compressed(FLAME_UV_MASK_NPZ, **masks)


if __name__ == "__main__":
    tyro.cli(main)