# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import json
import numpy as np
import torch
from vhap.data.video_dataset import VideoDataset
from vhap.config.nersemble import NersembleDataConfig
from vhap.util import camera
from vhap.util.log import get_logger


logger = get_logger(__name__)


class NeRSembleDataset(VideoDataset):
    def __init__(
        self,
        cfg: NersembleDataConfig,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |---camera_params/
                |   |---<subject>/
                |       |---camera_params.json
                |
                |---color_correction/
                |   |---<subject>/
                |       |---<camera_id>.npy
                |
                |---<subject>/
                    |---<sequence>/
                        |---images/
                        |   |---cam_<camera_id>_<timestep_id>.jpg
                        |
                        |---alpha_maps/
                        |   |---cam_<camera_id>_<timestep_id>.png
                        |
                        |---landmark2d/
                                |---face-alignment/
                                |    |---<camera_id>.npz
                                |
                                |---STAR/
                                     |---<camera_id>.npz
        """
        self.cfg = cfg
        assert cfg.subject != "", "Please specify the subject name"

        super().__init__(
            cfg=cfg,
            img_to_tensor=img_to_tensor,
            batchify_all_views=batchify_all_views,
        )
    
    def match_sequences(self):
        logger.info(f"Subject: {self.cfg.subject}, sequence: {self.cfg.sequence}")
        return list(filter(lambda x: x.is_dir(), (self.cfg.root_folder / self.cfg.subject).glob(f"{self.cfg.sequence}*")))
    
    def define_properties(self):
        super().define_properties()
        self.properties['rgb']['cam_id_prefix'] = "cam_"
        self.properties['alpha_map']['cam_id_prefix'] = "cam_"
    
    def load_camera_params(self):
        load_path = self.cfg.root_folder / "camera_params" / self.cfg.subject / "camera_params.json"
        assert load_path.exists()
        param = json.load(open(load_path))

        K = torch.Tensor(param["intrinsics"])

        if "height" not in param or "width" not in param:
            assert self.cfg.image_size_during_calibration is not None
            H, W = self.cfg.image_size_during_calibration
        else:
            H, W = param["height"], param["width"]

        self.camera_ids =  list(param["world_2_cam"].keys())
        w2c = torch.tensor([param["world_2_cam"][k] for k in self.camera_ids])  # (N, 4, 4)
        R = w2c[..., :3, :3]
        T = w2c[..., :3, 3]

        orientation = R.transpose(-1, -2)  # (N, 3, 3)
        location = R.transpose(-1, -2) @ -T[..., None]  # (N, 3, 1)

        # adjust how cameras distribute in the space with a global rotation
        if self.cfg.align_cameras_to_axes:
            orientation, location = camera.align_cameras_to_axes(
                orientation, location, target_convention="opengl"
            )

        # modify the local orientation of cameras to fit in different camera conventions
        if self.cfg.camera_convention_conversion is not None:
            orientation, K = camera.convert_camera_convention(
                self.cfg.camera_convention_conversion, orientation, K, H, W
            )

        c2w = torch.cat([orientation, location], dim=-1)  # camera-to-world transformation

        if self.cfg.target_extrinsic_type == "w2c":
            R = orientation.transpose(-1, -2)
            T = orientation.transpose(-1, -2) @ -location
            w2c = torch.cat([R, T], dim=-1)  # world-to-camera transformation
            extrinsic = w2c
        elif self.cfg.target_extrinsic_type == "c2w":
            extrinsic = c2w
        else:
            raise NotImplementedError(f"Unknown extrinsic type: {self.cfg.target_extrinsic_type}")

        self.camera_params = {}
        for i, camera_id in enumerate(self.camera_ids):
            self.camera_params[camera_id] = {"intrinsic": K, "extrinsic": extrinsic[i]}

    def filter_division(self, division):
        if division is not None:
            cam_for_train = [8, 7, 9, 4, 10, 5, 13, 2, 12, 1, 14, 0]
            if division == "train":
                self.camera_ids = [
                    self.camera_ids[i]
                    for i in range(len(self.camera_ids))
                    if i in cam_for_train
                ]
            elif division == "val":
                self.camera_ids = [
                    self.camera_ids[i]
                    for i in range(len(self.camera_ids))
                    if i not in cam_for_train
                ]
            elif division == "front-view":
                self.camera_ids = self.camera_ids[8:9]
            elif division == "side-view":
                self.camera_ids = self.camera_ids[0:1]
            elif division == "six-view":
                self.camera_ids = [self.camera_ids[i] for i in [0, 1, 7, 8, 14, 15]]
            else:
                raise NotImplementedError(f"Unknown division type: {division}")
            logger.info(f"division: {division}")
    
    def apply_transforms(self, item):
        if self.cfg.use_color_correction:
            color_correction_path = self.cfg.root_folder / 'color_correction' / self.cfg.subject / f'{item["camera_id"]}.npy'
            affine_color_transform = np.load(color_correction_path)
            rgb = item["rgb"] / 255
            rgb = rgb @ affine_color_transform[:3, :3] + affine_color_transform[np.newaxis, :3, 3]
            item["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        super().apply_transforms(item)
        return item


if __name__ == "__main__":
    import tyro
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from vhap.config.nersemble import NersembleDataConfig
    from vhap.config.base import import_module

    cfg = tyro.cli(NersembleDataConfig)
    cfg.use_landmark = False
    dataset = import_module(cfg._target)(
        cfg=cfg,
        img_to_tensor=False,
        batchify_all_views=True,
    )

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())
    print(sample["rgb"].shape)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    for item in tqdm(dataloader):
        pass
