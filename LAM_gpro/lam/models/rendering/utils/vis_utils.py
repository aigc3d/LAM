import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PointLights,
    DirectionalLights,
    PerspectiveCameras,
    Materials,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
import torch
import torch.nn as nn

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def render_mesh(mesh, face, cam_param, bkg, blend_ratio=1.0, return_bg_mask=False, R=None, T=None, return_fragments=False):
    mesh = mesh.cuda()[None,:,:]
    face = torch.LongTensor(face.astype(np.int64)).cuda()[None,:,:]
    cam_param = {k: v.cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    if R is None:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                    principal_point=cam_param['princpt'], 
                                    device='cuda',
                                    in_ndc=False,
                                    image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    else:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                            principal_point=cam_param['princpt'], 
                            device='cuda',
                            in_ndc=False,
                            image_size=torch.LongTensor(render_shape).cuda().view(1,2),
                            R=R,
                            T=T)
    
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    materials = Materials(
	    device='cuda',
	    specular_color=[[0.0, 0.0, 0.0]],
	    shininess=0.0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
    
    # background masking
    is_bkg = (fragments.zbuf <= 0).float().cpu().numpy()[0]
    render = images[0,:,:,:3].cpu().numpy()
    fg = render * blend_ratio + bkg/255 * (1 - blend_ratio)
    render = fg * (1 - is_bkg) * 255 + bkg * is_bkg
    ret = [render]
    if return_bg_mask:
        ret.append(is_bkg)
    if return_fragments:
        ret.append(fragments)
    return tuple(ret)


def rasterize_mesh(mesh, face, cam_param, height, width, return_bg_mask=False, R=None, T=None):
    mesh = mesh.cuda()[None,:,:]
    face = face.long().cuda()[None,:,:]
    cam_param = {k: v.cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (height, width)

    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    if R is None:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                    principal_point=cam_param['princpt'], 
                                    device='cuda',
                                    in_ndc=False,
                                    image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    else:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                            principal_point=cam_param['princpt'], 
                            device='cuda',
                            in_ndc=False,
                            image_size=torch.LongTensor(render_shape).cuda().view(1,2),
                            R=R,
                            T=T)
    
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()

    # render
    fragments = rasterizer(mesh)

    ret = [fragments]

    if return_bg_mask:
        # background masking
        is_bkg = (fragments.zbuf <= 0).float().cpu().numpy()[0]
        ret.append(is_bkg)

    return tuple(ret)


def rasterize_points(points, cam_param, height, width, return_bg_mask=False, R=None, T=None, to_cpu=False, points_per_pixel=5, radius=0.01):
    points = torch.stack((-points[:, 0], -points[:, 1], points[:, 2]), 1) # reverse x- and y-axis following PyTorch3D axis direction
    device = points.device
    if len(points.shape) == 2:
        points = [points]
    pointclouds = Pointclouds(points=points)
    cam_param = {k: v.to(device)[None,:] for k,v in cam_param.items()}
    render_shape = (height, width) # height, width

    if R is None:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                    principal_point=cam_param['princpt'], 
                                    device=device,
                                    in_ndc=False,
                                    image_size=torch.LongTensor(render_shape).to(device).view(1,2))
    else:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                            principal_point=cam_param['princpt'], 
                            device=device,
                            in_ndc=False,
                            image_size=torch.LongTensor(render_shape).to(device).view(1,2),
                            R=R,
                            T=T)
    
    raster_settings = PointsRasterizationSettings(image_size=render_shape, radius=radius, points_per_pixel=points_per_pixel, max_points_per_bin=82000)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)

    # render
    fragments = rasterizer(pointclouds)

    # background masking
    ret = [fragments]
    if return_bg_mask:
        if to_cpu:
            is_bkg = (fragments.zbuf <= 0).all(dim=-1, keepdim=True).float().cpu().numpy()[0]
        else:
            is_bkg = (fragments.zbuf <= 0).all(dim=-1, keepdim=True).float()[0]
        ret.append(is_bkg)
    
    return tuple(ret)


def render_points(points, cam_param, bkg, blend_ratio=1.0, return_bg_mask=False, R=None, T=None, return_fragments=False, rgbs=None):
    points = torch.stack((-points[:, 0], -points[:, 1], points[:, 2]), 1) # reverse x- and y-axis following PyTorch3D axis direction
    if rgbs is None:
        rgbs = torch.ones_like(points)
    if len(points.shape) == 2:
        points = [points]
        rgbs = [rgbs]
    pointclouds = Pointclouds(points=points, features=rgbs).cuda()
    cam_param = {k: v.cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    if R is None:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                    principal_point=cam_param['princpt'], 
                                    device='cuda',
                                    in_ndc=False,
                                    image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    else:
        cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                            principal_point=cam_param['princpt'], 
                            device='cuda',
                            in_ndc=False,
                            image_size=torch.LongTensor(render_shape).cuda().view(1,2),
                            R=R,
                            T=T)
    
    raster_settings = PointsRasterizationSettings(image_size=render_shape, radius=0.01, points_per_pixel=5)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()

    # render
    with torch.no_grad():
        fragments = rasterizer(pointclouds)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor(background_color=(0, 0, 0)))
        images = renderer(pointclouds)
    
    # background masking
    is_bkg = (fragments.zbuf <= 0).all(dim=-1, keepdim=True).float().cpu().numpy()[0]
    render = images[0,:,:,:3].cpu().numpy()
    fg = render * blend_ratio + bkg/255 * (1 - blend_ratio)
    render = fg * (1 - is_bkg) * 255 + bkg * is_bkg

    ret = [render]
    if return_bg_mask:
        ret.append(is_bkg)
    if return_fragments:
        ret.append(fragments)
    return tuple(ret)


class RenderMesh(nn.Module):
    def __init__(self, image_size, obj_filename=None, faces=None, device='cpu'):
        super(RenderMesh, self).__init__()
        self.device = device
        self.image_size = image_size
        if obj_filename is not None:
            verts, faces, aux = load_obj(obj_filename, load_textures=False)
            self.faces = faces.verts_idx
        elif faces is not None:
            import numpy as np
            self.faces = torch.tensor(faces.astype(np.int32))
        else:
            raise NotImplementedError('Must have faces.')
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    def _build_cameras(self, transform_matrix, focal_length, principal_point=None, intr=None):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self.device
        ).float()[None].repeat(batch_size, 1)
        if principal_point is None:
            principal_point = torch.zeros(batch_size, 2, device=self.device).float()
        # print("==="*16, "principle_points:", principal_point)
        # print("==="*16, "focal_length:", focal_length)
        if intr is None:
            cameras_kwargs = {
                'principal_point': principal_point, 'focal_length': focal_length, 
                'image_size': screen_size, 'device': self.device,
            }
        else:
            cameras_kwargs = {
                'principal_point': principal_point, 'focal_length': torch.tensor([intr[0, 0], intr[1, 1]]).unsqueeze(0),
                'image_size': screen_size, 'device': self.device,
            }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    def forward(
            self, vertices, cameras=None, transform_matrix=None, focal_length=None, principal_point=None, only_rasterize=False, intr=None,
        ):
        if cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length, principal_point=principal_point, intr=intr)
        faces = self.faces[None].repeat(vertices.shape[0], 1, 1)
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        mesh = Meshes(
            verts=vertices.to(self.device),
            faces=faces.to(self.device),
            textures=textures
        )
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(cameras=cameras, lights=self.lights, device=self.device)
        )
        render_results, fragments = renderer(mesh)
        render_results = render_results.permute(0, 3, 1, 2)
        if only_rasterize:
            return fragments
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        images[alpha_images.expand(-1, 3, -1, -1)<0.5] = 0.0
        return images*255, alpha_images


class RenderPoints(nn.Module):
    def __init__(self, image_size, obj_filename=None, device='cpu'):
        super(RenderPoints, self).__init__()
        self.device = device
        self.image_size = image_size
        if obj_filename is not None:
            verts = load_obj(obj_filename, load_textures=False)
        self.raster_settings = PointsRasterizationSettings(image_size=image_size, radius=0.01, points_per_pixel=1)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    def _build_cameras(self, transform_matrix, focal_length, principal_point=None):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self.device
        ).float()[None].repeat(batch_size, 1)
        if principal_point is None:
            principal_point = torch.zeros(batch_size, 2, device=self.device).float()
        # print("==="*16, "principle_points:", principal_point)
        # print("==="*16, "focal_length:", focal_length)
        cameras_kwargs = {
            'principal_point': principal_point, 'focal_length': focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    def forward(
            self, vertices, cameras=None, transform_matrix=None, focal_length=None, principal_point=None, only_rasterize=False
        ):
        if cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length, principal_point=principal_point)
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
        pointclouds = Pointclouds(points=vertices, features=verts_rgb).cuda()

        # render
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings).cuda()
        if only_rasterize:
            fragments = rasterizer(pointclouds)
            return fragments
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor(background_color=(0, 0, 0)))
        render_results = renderer(pointclouds).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]

        return images*255, alpha_images