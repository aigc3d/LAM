import tyro
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch

from vhap.util.camera import OrbitCamera
from vhap.model.flame import FlameHead
from vhap.config.base import ModelConfig
from vhap.util.render_nvdiffrast import NVDiffRenderer


@dataclass
class Config:
    model: ModelConfig
    """FLAME model configuration"""
    param_path: Optional[Path] = None
    """Path to the npz file for FLAME parameters"""
    W: int = 1024
    """GUI width"""
    H: int = 1024
    """GUI height"""
    radius: float = 1
    """default GUI camera radius from center"""
    fovy: float = 30
    """default GUI camera fovy"""
    background_color: tuple[float] = (1., 1., 1.)
    """default GUI background color"""
    use_opengl: bool = False
    """use OpenGL or CUDA rasterizer"""


class FlameViewer:
    def __init__(self, cfg: Config):
        self.cfg = cfg  # shared with the trainer's cfg to support in-place modification of rendering parameters.

        # flame model
        self.flame_model = FlameHead(cfg.model.n_shape, cfg.model.n_expr, add_teeth=True)
        
        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention="opengl")
        self.last_time_fresh = None
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        # rendering settings
        self.mesh_renderer = NVDiffRenderer(use_opengl=cfg.use_opengl, lighting_space='camera')
        self.num_timesteps = 1
        self.timestep = 0

        self.define_gui()

    def __del__(self):
        dpg.destroy_context()
    
    def refresh(self):
        dpg.set_value("_texture", self.render_buffer)

        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{fps:.1f}')
        self.last_time_fresh = time.time()

    def define_gui(self):
        dpg.create_context()
        
        # register texture =================================================================================================
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        # register window ==================================================================================================
        # the window to display the rendered image
        with dpg.window(label="viewer", tag="_render_window", width=self.W, height=self.H, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image")

        # control window ==================================================================================================
        with dpg.window(label="Control", tag="_control_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ")
                dpg.add_text("", tag="_log_fps")

            # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):

                # timestep slider and buttons
                if self.num_timesteps != None:
                    def callback_set_current_frame(sender, app_data):
                        if sender == "_slider_timestep":
                            self.timestep = app_data
                        elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                            self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                        elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                            self.timestep = max(self.timestep - 1, 0)
                        elif sender == "_mvKey_Home":
                            self.timestep = 0
                        elif sender == "_mvKey_End":
                            self.timestep = self.num_timesteps - 1

                        dpg.set_value("_slider_timestep", self.timestep)

                        self.need_update = True
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                        dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                        dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=162, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True
                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy")

                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera)
            

        ### register mouse handlers ========================================================================================

        def callback_mouse_move(sender, app_data):
            self.cursor_x, self.cursor_y = app_data
            if not dpg.is_item_focused("_render_window"):
                return

            if self.drag_begin_x is None or self.drag_begin_y is None:
                self.drag_begin_x = self.cursor_x
                self.drag_begin_y = self.cursor_y
            else:
                dx = self.cursor_x - self.drag_begin_x
                dy = self.cursor_y - self.drag_begin_y

                # button=dpg.mvMouseButton_Left
                if self.drag_button is dpg.mvMouseButton_Left:
                    self.cam.orbit(dx, dy)
                    self.need_update = True
                elif self.drag_button is dpg.mvMouseButton_Middle:
                    self.cam.pan(dx, dy)
                    self.need_update = True

        def callback_mouse_button_down(sender, app_data):
            if not dpg.is_item_focused("_render_window"):
                return
            self.drag_begin_x = self.cursor_x
            self.drag_begin_y = self.cursor_y
            self.drag_button = app_data[0]
        
        def callback_mouse_release(sender, app_data):
            self.drag_begin_x = None
            self.drag_begin_y = None
            self.drag_button = None

            self.dx_prev = None
            self.dy_prev = None
        
        def callback_mouse_drag(sender, app_data):
            if not dpg.is_item_focused("_render_window"):
                return

            button, dx, dy = app_data
            if self.dx_prev is None or self.dy_prev is None:
                ddx = dx
                ddy = dy
            else:
                ddx = dx - self.dx_prev
                ddy = dy - self.dy_prev

            self.dx_prev = dx
            self.dy_prev = dy

            if ddx != 0 and ddy != 0:
                if button is dpg.mvMouseButton_Left:
                    self.cam.orbit(ddx, ddy)
                    self.need_update = True
                elif button is dpg.mvMouseButton_Middle:
                    self.cam.pan(ddx, ddy)
                    self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_render_window"):
                return
            delta = app_data
            self.cam.scale(delta)
            self.need_update = True

        with dpg.handler_registry():
            # this registry order helps avoid false fire
            dpg.add_mouse_release_handler(callback=callback_mouse_release)
            # dpg.add_mouse_drag_handler(callback=callback_mouse_drag)  # not using the drag callback, since it does not return the starting point
            dpg.add_mouse_move_handler(callback=callback_mouse_move)
            dpg.add_mouse_down_handler(callback=callback_mouse_button_down)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)

            # key press handlers
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

        def callback_viewport_resize(sender, app_data):
            while self.rendering:
                time.sleep(0.01)
            self.need_update = False
            self.W = app_data[0]
            self.H = app_data[1]
            self.cam.image_width = self.W
            self.cam.image_height = self.H
            self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)

            # delete and re-add the texture and image
            dpg.delete_item("_texture")
            dpg.delete_item("_image")

            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image", parent="_render_window")
            dpg.configure_item("_render_window", width=self.W, height=self.H)
            self.need_update = True
        dpg.set_viewport_resize_callback(callback_viewport_resize)

        ### global theme ==================================================================================================
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_render_window", theme_no_padding)

        ### finish setup ==================================================================================================
        dpg.create_viewport(title='FLAME Sequence Viewer', width=self.W, height=self.H, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def forward_flame(self,  flame_param):
        N = flame_param['expr'].shape[0]

        self.verts, self.verts_cano = self.flame_model(
            flame_param['shape'][None, ...].expand(N, -1),
            flame_param['expr'],
            flame_param['rotation'],
            flame_param['neck_pose'],
            flame_param['jaw_pose'],
            flame_param['eyes_pose'],
            flame_param['translation'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            # dynamic_offset=flame_param['dynamic_offset'],
        )

        self.num_timesteps = N
        dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)
    
    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    def run(self):
        if self.cfg.param_path is not None:
            if self.cfg.param_path.exists():
                self.flame_param = dict(np.load(self.cfg.param_path))
                for k, v in self.flame_param.items():
                    if v.dtype in [np.float64, np.float32]:
                        self.flame_param[k] = torch.from_numpy(v).float()
                self.forward_flame(self.flame_param)
            else:
                raise FileNotFoundError(f'{self.cfg.param_path} does not exist.')
        
        while dpg.is_dearpygui_running():

            if self.need_update:
                self.rendering = True

                with torch.no_grad():
                    RT = torch.from_numpy(self.cam.world_view_transform).cuda()[None]
                    K = torch.from_numpy(self.cam.intrinsics).cuda()[None]
                    image_size = self.cam.image_height, self.cam.image_width
                    verts = self.verts[[self.timestep]].cuda()
                    faces = self.flame_model.faces.cuda()
                    out_dict = self.mesh_renderer.render_without_texture(verts, faces, RT, K, image_size, self.cfg.background_color)

                    rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                    rgb_mesh = rgba_mesh[:3, :, :]

                self.render_buffer = rgb_mesh.permute(1, 2, 0).cpu().numpy()
                self.refresh()

                self.rendering = False
                self.need_update = False
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = FlameViewer(cfg)
    gui.run()
