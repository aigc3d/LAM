# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from typing import Optional, Literal
from dataclasses import dataclass
import tyro

from vhap.config.base import (
    StageRgbSequentialTrackingConfig, StageRgbGlobalTrackingConfig, PipelineConfig, 
    DataConfig, LossWeightConfig, BaseTrackingConfig,
)
from vhap.util.log import get_logger
logger = get_logger(__name__)


@dataclass()
class NersembleDataConfig(DataConfig):
    _target: str = "vhap.data.nersemble_dataset.NeRSembleDataset"
    calibrated: bool = True
    image_size_during_calibration: Optional[tuple[int, int]] = (3208, 2200)
    """(height, width). Will be use to convert principle points when the image size is not included in the camera parameters."""
    background_color: Optional[Literal['white', 'black']] = None
    landmark_source: Optional[Literal["face-alignment", 'star']] = "star"

    subject: str = ""
    """Subject ID. Such as 018, 218, 251, 253"""
    use_color_correction: bool = True
    """Whether to use color correction to harmonize the color of the input images."""

@dataclass()
class NersembleLossWeightConfig(LossWeightConfig):
    landmark: Optional[float] = 3.  # should not be lower to avoid collapse
    always_enable_jawline_landmarks: bool = False  # allow disable_jawline_landmarks in StageConfig to work
    reg_expr: float = 1e-2  # for best expressivness
    reg_tex_tv: Optional[float] = 1e5  # 10x of the base value

@dataclass()
class NersembleStageRgbSequentialTrackingConfig(StageRgbSequentialTrackingConfig):
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr", "dynamic_offset")

    align_texture_except: tuple[str, ...] = ("boundary",)
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Due to the limited flexibility in the lower neck region of FLAME, we relax the 
    alignment constraints for better alignment in the face region.
    """

@dataclass()
class NersembleStageRgbGlobalTrackingConfig(StageRgbGlobalTrackingConfig):
    align_texture_except: tuple[str, ...] = ("boundary",)
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Due to the limited flexibility in the lower neck region of FLAME, we relax the 
    alignment constraints for better alignment in the face region.
    """

@dataclass()
class NersemblePipelineConfig(PipelineConfig):
    rgb_sequential_tracking: NersembleStageRgbSequentialTrackingConfig
    rgb_global_tracking: NersembleStageRgbGlobalTrackingConfig

@dataclass()
class NersembleTrackingConfig(BaseTrackingConfig):
    data: NersembleDataConfig
    w: NersembleLossWeightConfig
    pipeline: NersemblePipelineConfig

    def get_occluded(self):
        occluded_table = {
            '018': ('neck_lower',),
            '218': ('neck_lower',),
            '251': ('neck_lower', 'boundary'),
            '253': ('neck_lower',),
        }
        if self.data.subject in occluded_table:
            logger.info(f"Automatically setting cfg.model.occluded to {occluded_table[self.data.subject]}")
            self.model.occluded = occluded_table[self.data.subject]


if __name__ == "__main__":
    config = tyro.cli(NersembleTrackingConfig)
    print(tyro.to_yaml(config))