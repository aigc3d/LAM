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
        self.flame_model = FlameHead(
            cfg.model.n_shape, 
            cfg.model.n_expr, 
            add_teeth=True, 
            include_lbs_color=True, 
        )
        self.reset_flame_param()
        
        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention="opengl")
        self.last_time_fresh = None
        self.render_mode = '-'
        self.selected_regions = '-'
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

                def callback_set_render_mode(sender, app_data):
                    self.render_mode = app_data
                    self.need_update = True
                dpg.add_combo(('-', 'lbs weights'), label='render mode', default_value=self.render_mode, tag="_combo_render_mode", callback=callback_set_render_mode)

                def callback_select_regions(sender, app_data):
                    self.selected_regions = app_data
                    self.need_update = True
                dpg.add_combo(['-']+sorted(self.flame_model.mask.v.keys()), label='regions', default_value='-', tag="_combo_regions", callback=callback_select_regions)
                
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
            

            # FLAME paraemter options
            with dpg.collapsing_header(label="Parameters", default_open=True):

                def callback_set_pose(sender, app_data):
                    joint, axis = sender.split('-')[1:3]
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    self.flame_param[joint][0, axis_idx] = app_data
                    self.need_update = True
                self.pose_sliders = []
                slider_width = 87
                for joint in ['neck', 'jaw']:
                    dpg.add_text(f'{joint:9s}')
                    if joint in self.flame_param:
                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(label="x", min_value=-1, max_value=1, format="%.2f", default_value=self.flame_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=slider_width)
                            dpg.add_slider_float(label="y", min_value=-1, max_value=1, format="%.2f", default_value=self.flame_param[joint][0, 1], callback=callback_set_pose, tag=f"_slider-{joint}-y", width=slider_width)
                            dpg.add_slider_float(label="z", min_value=-1, max_value=1, format="%.2f", default_value=self.flame_param[joint][0, 2], callback=callback_set_pose, tag=f"_slider-{joint}-z", width=slider_width)
                            self.pose_sliders.append(f"_slider-{joint}-x")
                            self.pose_sliders.append(f"_slider-{joint}-y")
                            self.pose_sliders.append(f"_slider-{joint}-z")
                
                def callback_set_expr(sender, app_data):
                    expr_i = int(sender.split('-')[2])
                    self.flame_param['expr'][0, expr_i] = app_data
                    self.need_update = True
                self.expr_sliders = []
                dpg.add_text(f'expr')
                for i in range(5):
                    dpg.add_slider_float(label=f"{i}", min_value=-5, max_value=5, format="%.2f", default_value=0, callback=callback_set_expr, tag=f"_slider-expr-{i}", width=300)
                    self.expr_sliders.append(f"_slider-expr-{i}")

                def callback_reset_flame(sender, app_data):
                    self.reset_flame_param()
                    self.need_update = True
                    for slider in self.pose_sliders + self.expr_sliders:
                        dpg.set_value(slider, 0)
                dpg.add_button(label="reset FLAME", tag="_button_reset_flame", callback=callback_reset_flame)

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
            # dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            # dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            # dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            # dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

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
        dpg.create_viewport(title='FLAME Editor', width=self.W, height=self.H, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def reset_flame_param(self):
        self.flame_param = {
            'shape': torch.zeros(1, self.cfg.model.n_shape),
            'expr': torch.zeros(1, self.cfg.model.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
            'static_offset': torch.zeros(1, 3),
            'dynamic_offset': torch.zeros(1, 3),
        }

    def forward_flame(self,  flame_param):
        N = flame_param['expr'].shape[0]

        self.verts, self.verts_cano = self.flame_model(
            **flame_param,
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
        )

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
          
        while dpg.is_dearpygui_running():

            if self.need_update:
                self.rendering = True

                with torch.no_grad():
                    # mesh
                    self.forward_flame(self.flame_param)
                    verts = self.verts.cuda()
                    faces = self.flame_model.faces.cuda()

                    # camera
                    RT = torch.from_numpy(self.cam.world_view_transform).cuda()[None]
                    K = torch.from_numpy(self.cam.intrinsics).cuda()[None]
                    image_size = self.cam.image_height, self.cam.image_width

                    if self.render_mode == 'lbs weights':
                        v_color = self.flame_model.lbs_color.cuda()
                    else:
                        v_color = torch.ones_like(verts)

                    if self.selected_regions != '-':
                        vid = self.flame_model.mask.get_vid_except_region(self.selected_regions)
                        v_color[..., vid, :] *= 0.3
                    
                    out_dict = self.mesh_renderer.render_v_color(verts, v_color, faces, RT, K, image_size, self.cfg.background_color)
                    
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
