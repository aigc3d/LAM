# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple
import tyro
import importlib
from vhap.util.log import get_logger
logger = get_logger(__name__)


def import_module(module_name: str):
    module_name, class_name = module_name.rsplit(".", 1)
    module = getattr(importlib.import_module(module_name), class_name)
    return module


class Config:
    def __getitem__(self, __name: str):
        if hasattr(self, __name):
            return getattr(self, __name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{__name}'")


@dataclass()
class DataConfig(Config):
    root_folder: Path = ''
    """The root folder for the dataset."""
    sequence: str = ''
    """The sequence name"""
    _target: str = "vhap.data.video_dataset.VideoDataset"
    """The target dataset class"""
    division: Optional[str] = None
    subset: Optional[str] = None
    calibrated: bool = False
    """Whether the cameras parameters are available"""
    align_cameras_to_axes: bool = True
    """Adjust how cameras distribute in the space with a global rotation"""
    camera_convention_conversion: str = 'opencv->opengl'
    target_extrinsic_type: Literal['w2c', 'c2w'] = 'w2c'
    n_downsample_rgb: Optional[int] = None
    """Load from downsampled RGB images to save data IO time"""
    scale_factor: float = 1.0
    """Further apply a scaling transformation after the downsampling of RGB"""
    background_color: Optional[Literal['white', 'black']] = 'white'
    use_alpha_map: bool = False
    use_landmark: bool = True
    landmark_source: Optional[Literal['face-alignment', 'star']] = "star"


@dataclass()
class ModelConfig(Config):
    n_shape: int = 300
    n_expr: int = 100
    n_tex: int = 100

    use_static_offset: bool = False
    """Optimize static offsets on top of FLAME vertices in the canonical space"""
    use_dynamic_offset: bool = False
    """Optimize dynamic offsets on top of the FLAME vertices in the canonical space"""
    add_teeth: bool = True
    """Add teeth to the FLAME model"""
    remove_lip_inside: bool = False
    """Remove the inner part of the lips from the FLAME model"""

    tex_resolution: int = 2048
    """The resolution of the extra texture map"""
    tex_painted: bool = True
    """Use a painted texture map instead the pca texture space as the base texture map"""
    tex_extra: bool = True
    """Optimize an extra texture map as the base texture map or the residual texture map"""
    # tex_clusters: tuple[str, ...] = ("skin", "hair", "sclerae", "lips_tight", "boundary")
    tex_clusters: tuple[str, ...] = ("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises")
    """Regions that are supposed to share a similar color inside"""
    residual_tex: bool = True
    """Use the extra texture map as a residual component on top of the base texture"""
    occluded: tuple[str, ...] = ()  # to be used for updating stage configs in __post_init__
    """The regions that are occluded by the hair or garments"""
    
    flame_params_path: Optional[Path] = None


@dataclass()
class RenderConfig(Config):
    backend: Literal['nvdiffrast', 'pytorch3d'] = 'nvdiffrast'
    """The rendering backend"""
    use_opengl: bool = False
    """Use OpenGL for NVDiffRast"""
    background_train: Literal['white', 'black', 'target'] = 'target'
    """Background color/image for training"""
    disturb_rate_fg: Optional[float] = 0.5
    """The rate of disturbance for the foreground"""
    disturb_rate_bg: Optional[float] = 0.5
    """The rate of disturbance for the background. 0.6 best for multi-view, 0.3 best for single-view"""
    background_eval: Literal['white', 'black', 'target'] = 'target'
    """Background color/image for evaluation"""
    lighting_type: Literal['constant', 'front', 'front-range', 'SH'] = 'SH'
    """The type of lighting"""
    lighting_space: Literal['world', 'camera'] = 'world'
    """The space of lighting"""


@dataclass()
class LearningRateConfig(Config):
    base: float = 5e-3
    """shape, texture, rotation, eyes, neck, jaw"""
    translation: float = 1e-3
    expr: float = 5e-2
    static_offset: float = 5e-4
    dynamic_offset: float = 5e-4
    camera: float = 5e-3
    light: float = 5e-3


@dataclass()
class LossWeightConfig(Config):
    landmark: Optional[float] = 10.
    always_enable_jawline_landmarks: bool = True
    """Always enable the landmark loss for the jawline landmarks. Ignore disable_jawline_landmarks in stages."""

    photo: Optional[float] = 30.

    reg_shape: float = 3e-1
    reg_expr: float = 3e-2
    reg_tex_pca: float = 1e-4  # will make it hard to model hair color when too high
    
    reg_tex_res: Optional[float] = None  # 1e2 (when w/o reg_var)
    """Regularize the residual texture map"""
    reg_tex_res_clusters: Optional[float] = 1e1
    """Regularize the residual texture map inside each texture cluster"""
    reg_tex_res_for: tuple[str, ...] = ("sclerae", "teeth")
    """Regularize the residual texture map for the clusters specified"""
    reg_tex_tv: Optional[float] = 1e4  # important to split regions apart
    """Regularize the total variation of the texture map"""

    reg_light: Optional[float] = None
    """Regularize lighting parameters"""
    reg_diffuse: Optional[float] = 1e2
    """Regularize lighting parameters by the diffuse term"""

    reg_offset: Optional[float] = 3e2
    """Regularize the norm of offsets"""
    reg_offset_relax_coef: float = 1.
    """The coefficient for relaxing reg_offset for the regions specified"""
    reg_offset_relax_for: tuple[str, ...] = ("hair", "ears")
    """Relax the offset loss for the regions specified"""

    reg_offset_lap: Optional[float] = 1e6
    """Regularize the difference of laplacian coordinate caused by offsets"""
    reg_offset_lap_relax_coef: float = 0.1
    """The coefficient for relaxing reg_offset_lap for the regions specified"""
    reg_offset_lap_relax_for: tuple[str, ...] = ("hair", "ears")
    """Relax the offset loss for the regions specified"""

    reg_offset_rigid: Optional[float] = 3e2
    """Regularize the the offsets to be as-rigid-as-possible"""
    reg_offset_rigid_for: tuple[str, ...] = ("left_ear", "right_ear", "neck", "left_eye", "right_eye", "lips_tight")
    """Regularize the the offsets to be as-rigid-as-possible for the regions specified"""

    reg_offset_dynamic: Optional[float] = 3e5
    """Regularize the dynamic offsets to be temporally smooth"""

    blur_iter: int = 0
    """The number of iterations for blurring vertex weights"""
    
    smooth_trans: float = 3e2
    """global translation"""
    smooth_rot: float = 3e1
    """global rotation"""

    smooth_neck: float = 3e1
    """neck joint"""
    smooth_jaw: float = 1e-1
    """jaw joint"""
    smooth_eyes: float = 0
    """eyes joints"""

    prior_neck: float = 3e-1
    """Regularize the neck joint towards neutral"""
    prior_jaw: float = 3e-1
    """Regularize the jaw joint towards neutral"""
    prior_eyes: float = 3e-2
    """Regularize the eyes joints towards neutral"""
    

@dataclass()
class LogConfig(Config):
    interval_scalar: Optional[int] = 100
    """The step interval of scalar logging. Using an interval of stage_tracking.num_steps // 5 unless specified."""
    interval_media: Optional[int] = 500
    """The step interval of media logging. Using an interval of stage_tracking.num_steps unless specified."""
    image_format: Literal['jpg', 'png'] = 'jpg'
    """Output image format"""
    view_indices: Tuple[int, ...] = ()
    """Manually specify the view indices for log"""
    max_num_views: int = 3
    """The maximum number of views for log"""
    stack_views_in_rows: bool = True


@dataclass()
class ExperimentConfig(Config):
    output_folder: Path = Path('output/track')
    reuse_landmarks: bool = True
    keyframes: Tuple[int, ...] = tuple()
    photometric: bool = False
    """enable photometric optimization, otherwise only landmark optimization"""

@dataclass()
class StageConfig(Config):
    disable_jawline_landmarks: bool = False
    """Disable the landmark loss for the jawline landmarks since they are not accurate"""

@dataclass()
class StageLmkInitRigidConfig(StageConfig):
    """The stage for initializing the rigid parameters"""
    num_steps: int = 300
    optimizable_params: tuple[str, ...] = ("cam", "pose")

@dataclass()
class StageLmkInitAllConfig(StageConfig):
    """The stage for initializing all the parameters optimizable with landmark loss"""
    num_steps: int = 300
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr")

@dataclass()
class StageLmkSequentialTrackingConfig(StageConfig):
    """The stage for sequential tracking with landmark loss"""
    num_steps: int = 50
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr")

@dataclass()
class StageLmkGlobalTrackingConfig(StageConfig):
    """The stage for global tracking with landmark loss"""
    num_epochs: int = 0
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr")

@dataclass()
class PhotometricStageConfig(StageConfig):
    align_texture_except: tuple[str, ...] = ()
    """Align the inner region of rendered FLAME to the image, except for the regions specified"""
    align_boundary_except: tuple[str, ...] = ("bottomline",)  # necessary to avoid the bottomline of FLAME from being stretched to the bottom of the image
    """Align the boundary of FLAME to the image, except for the regions specified"""

@dataclass()
class StageRgbInitTextureConfig(PhotometricStageConfig):
    """The stage for initializing the texture map with photometric loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "shape", "texture", "lights")
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    align_boundary_except: tuple[str, ...] = ("hair", "boundary")

@dataclass()
class StageRgbInitAllConfig(PhotometricStageConfig):
    """The stage for initializing all the parameters except the offsets with photometric loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr", "texture", "lights")
    disable_jawline_landmarks: bool = True
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    align_boundary_except: tuple[str, ...] = ("hair", "bottomline")

@dataclass()
class StageRgbInitOffsetConfig(PhotometricStageConfig):
    """The stage for initializing the offsets with photometric loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr", "texture", "lights", "static_offset")
    disable_jawline_landmarks: bool = True
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")

@dataclass()
class StageRgbSequentialTrackingConfig(PhotometricStageConfig):
    """The stage for sequential tracking with photometric loss"""
    num_steps: int = 50
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr", "texture", "dynamic_offset")
    disable_jawline_landmarks: bool = True

@dataclass()
class StageRgbGlobalTrackingConfig(PhotometricStageConfig):
    """The stage for global tracking with photometric loss"""
    num_epochs: int = 30
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr", "texture", "lights", "static_offset", "dynamic_offset")
    disable_jawline_landmarks: bool = True

@dataclass()
class PipelineConfig(Config):
    lmk_init_rigid: StageLmkInitRigidConfig
    lmk_init_all: StageLmkInitAllConfig
    lmk_sequential_tracking: StageLmkSequentialTrackingConfig
    lmk_global_tracking: StageLmkGlobalTrackingConfig
    rgb_init_texture: StageRgbInitTextureConfig
    rgb_init_all: StageRgbInitAllConfig
    rgb_init_offset: StageRgbInitOffsetConfig
    rgb_sequential_tracking: StageRgbSequentialTrackingConfig
    rgb_global_tracking: StageRgbGlobalTrackingConfig

    
@dataclass()
class BaseTrackingConfig(Config):
    data: DataConfig
    model: ModelConfig
    render: RenderConfig
    log: LogConfig
    exp: ExperimentConfig
    lr: LearningRateConfig
    w: LossWeightConfig
    pipeline: PipelineConfig

    begin_stage: Optional[str] = None
    """Begin from the specified stage for debugging"""
    begin_frame_idx: int = 0
    """Begin from the specified frame index for debugging"""
    async_func: bool = True
    """Allow asynchronous function calls for speed up"""
    device: Literal['cuda', 'cpu'] = 'cuda'

    def get_occluded(self):
        occluded_table = {
        }
        if self.data.sequence in occluded_table:
            logger.info(f"Automatically setting cfg.model.occluded to {occluded_table[self.data.sequence]}")
            self.model.occluded = occluded_table[self.data.sequence]

    def __post_init__(self):
        self.get_occluded()

        if not self.model.use_static_offset and not self.model.use_dynamic_offset:
            self.model.occluded = tuple(list(self.model.occluded) + ['hair'])  # disable boundary alignment for the hair region if no offset is used

        for cfg_stage in self.pipeline.__dict__.values():
            if isinstance(cfg_stage, PhotometricStageConfig):
                cfg_stage.align_texture_except = tuple(list(cfg_stage.align_texture_except) + list(self.model.occluded))
                cfg_stage.align_boundary_except = tuple(list(cfg_stage.align_boundary_except) + list(self.model.occluded))

        if self.begin_stage is not None:
            skip = True
            for cfg_stage in self.pipeline.__dict__.values():
                if cfg_stage.__class__.__name__.lower() == self.begin_stage:
                    skip = False
                if skip:
                    cfg_stage.num_steps = 0


if __name__ == "__main__":
    config = tyro.cli(BaseTrackingConfig)
    print(tyro.to_yaml(config))