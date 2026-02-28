import os
from pathlib import Path
from copy import deepcopy
from typing import Optional
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, default_collate
import json
from vhap.util.log import get_logger
from vhap.config.base import DataConfig


logger = get_logger(__name__)


class VideoDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |---images/
                |   |---<timestep_id>.jpg
                |
                |---alpha_maps/
                |   |---<timestep_id>.png
                |
                |---landmark2d/
                        |---face-alignment/
                        |    |---<camera_id>.npz
                        |
                        |---STAR/
                                |---<camera_id>.npz
        """
        super().__init__()
        self.cfg = cfg
        self.img_to_tensor = img_to_tensor
        self.batchify_all_views = batchify_all_views
        
        sequence_paths = self.match_sequences()
        if len(sequence_paths) > 1:
            logger.info(f"Found multiple sequences: {sequence_paths}")
            raise ValueError(f"Found multiple sequences by '{cfg.sequence}': \n" + "\n\t".join([str(x) for x in sequence_paths]))
        elif len(sequence_paths) == 0:
            raise ValueError(f"Cannot find sequence: {cfg.sequence}")
        self.sequence_path = sequence_paths[0]
        logger.info(f"Initializing dataset from {self.sequence_path}")

        self.define_properties()
        self.load_camera_params()

        # timesteps
        self.timestep_ids = set(
            f.split('.')[0].split('_')[-1]
            for f in os.listdir(self.sequence_path / self.properties['rgb']['folder']) if f.endswith(self.properties['rgb']['suffix'])
        )
        self.timestep_ids = sorted(self.timestep_ids)
        self.timestep_indices = list(range(len(self.timestep_ids)))

        self.filter_division(cfg.division)
        self.filter_subset(cfg.subset)
            
        logger.info(f"number of timesteps: {self.num_timesteps}, number of cameras: {self.num_cameras}")

        # collect
        self.items = []
        for fi, timestep_index in enumerate(self.timestep_indices):
            for ci, camera_id in enumerate(self.camera_ids):
                self.items.append(
                    {
                        "timestep_index": fi,  # new index after filtering
                        "timestep_index_original": timestep_index,  # original index
                        "timestep_id": self.timestep_ids[timestep_index],
                        "camera_index": ci,
                        "camera_id": camera_id,
                    }
                )
    
    def match_sequences(self):
        logger.info(f"Looking for sequence '{self.cfg.sequence}' at {self.cfg.root_folder}")
        return list(filter(lambda x: x.is_dir(), self.cfg.root_folder.glob(f"{self.cfg.sequence}*")))
    
    def define_properties(self):
        self.properties = {
            "rgb": {
                "folder": f"images_{self.cfg.n_downsample_rgb}"
                if self.cfg.n_downsample_rgb
                else "images",
                "per_timestep": True,
                # "suffix": "jpg",
                "suffix": "png",
            },
            "alpha_map": {
                "folder": "alpha_maps",
                "per_timestep": True,
                "suffix": "jpg",
            },
            "landmark2d/face-alignment": {
                "folder": "landmark2d/face-alignment",
                "per_timestep": False,
                "suffix": "npz",
            },
            "landmark2d/STAR": {
                "folder": "landmark2d/STAR",
                "per_timestep": False,
                "suffix": "npz",
            },
            "landmark2d/lms": {
                "folder": "landmark2d/landmarks",
                "per_timestep": False,
                "suffix": "npz",
            },
        }

    @staticmethod
    def get_number_after_prefix(string, prefix):
        i = string.find(prefix)
        if i != -1:
            number_begin = i + len(prefix)
            assert number_begin < len(string), f"No number found behind prefix '{prefix}'"
            assert string[number_begin].isdigit(), f"No number found behind prefix '{prefix}'"

            non_digit_indices = [i for i, c in enumerate(string[number_begin:]) if not c.isdigit()]
            if len(non_digit_indices) > 0:
                number_end = number_begin + min(non_digit_indices)
                return int(string[number_begin:number_end])
            else:
                return int(string[number_begin:])
        else:
            return None
    
    def filter_division(self, division):
        pass

    def filter_subset(self, subset):
        if subset is not None:
            if 'ti' in subset:
                ti = self.get_number_after_prefix(subset, 'ti')
                if 'tj' in subset:
                    tj = self.get_number_after_prefix(subset, 'tj')
                    self.timestep_indices = self.timestep_indices[ti:tj+1]
                else:
                    self.timestep_indices = self.timestep_indices[ti:ti+1]
            elif 'tn' in subset:
                tn = self.get_number_after_prefix(subset, 'tn')
                tn_all = len(self.timestep_indices)
                tn = min(tn, tn_all)
                self.timestep_indices = self.timestep_indices[::tn_all // tn][:tn]
            elif 'ts' in subset:
                ts = self.get_number_after_prefix(subset, 'ts')
                self.timestep_indices = self.timestep_indices[::ts]
            if 'ci' in subset:
                ci = self.get_number_after_prefix(subset, 'ci')
                self.camera_ids = self.camera_ids[ci:ci+1]
            elif 'cn' in subset:
                cn = self.get_number_after_prefix(subset, 'cn')
                cn_all = len(self.camera_ids)
                cn = min(cn, cn_all)
                self.camera_ids = self.camera_ids[::cn_all // cn][:cn]
            elif 'cs' in subset:
                cs = self.get_number_after_prefix(subset, 'cs')
                self.camera_ids = self.camera_ids[::cs]
    
    def load_camera_params(self):
        self.camera_ids =  ['0']

        # Guessed focal length, height, width. Should be optimized or replaced by real values
        f, h, w = 512, 512, 512
        K = torch.Tensor([
            [f, 0, w],
            [0, f, h],
            [0, 0, 1]
        ])

        orientation = torch.eye(3)[None, ...]  # (1, 3, 3)
        location = torch.Tensor([0, 0, 1])[None, ..., None]  # (1, 3, 1)

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

        return self.camera_params

    def __len__(self):
        if self.batchify_all_views:
            return self.num_timesteps
        else:
            return len(self.items)

    def __getitem__(self, i):
        if self.batchify_all_views:
            return self.getitem_by_timestep(i)
        else:
            return self.getitem_single_image(i)

    def getitem_single_image(self, i):
        item = deepcopy(self.items[i])

        rgb_path = self.get_property_path("rgb", i)
        item["rgb"] = np.array(Image.open(rgb_path))[:, :, :3]

        camera_param = self.camera_params[item["camera_id"]]
        item["intrinsic"] = camera_param["intrinsic"].clone()
        item["extrinsic"] = camera_param["extrinsic"].clone()

        if self.cfg.use_alpha_map or self.cfg.background_color is not None:
            alpha_path = self.get_property_path("alpha_map", i)
            item["alpha_map"] = np.array(Image.open(alpha_path))

        if self.cfg.use_landmark:
            timestep_index = self.items[i]["timestep_index"]

            landmark_path = self.get_property_path("landmark2d/lms", i)
            landmark_npz = np.load(landmark_path)

            lms_eyes_path = os.path.join(os.path.dirname(landmark_path),'iris.json')

            item["lmk2d"] = landmark_npz["face_landmark_2d"][timestep_index]  # (num_points, 3)
            if (item["lmk2d"][:, :2] == -1).sum() > 0:
                item["lmk2d"][:, 2:] = 0.0
            else:
                item["lmk2d"][:, 2:] = 1.0

            if(os.path.exists(lms_eyes_path)):
                with open(lms_eyes_path,'r') as f:
                    lms_eye = json.load(f)
                lms_eye = np.array([lms_eye[key] for key in lms_eye][timestep_index]).reshape((2,2)) / 1024.
                lms_eye = np.concatenate([lms_eye,np.ones((2,1))],axis=1)[(1,0),:]
                item["lmk2d"] = np.concatenate([item["lmk2d"], lms_eye], 0)
            else:
                item["lmk2d"] = np.concatenate([item["lmk2d"]], 0)

        item = self.apply_transforms(item)
        return item

    def getitem_by_timestep(self, timestep_index):
        begin = timestep_index * self.num_cameras
        indices = range(begin, begin + self.num_cameras)
        item = default_collate([self.getitem_single_image(i) for i in indices])

        item["num_cameras"] = self.num_cameras
        return item

    def apply_transforms(self, item):
        item = self.apply_scale_factor(item)
        item = self.apply_background_color(item)
        item = self.apply_to_tensor(item)
        return item

    def apply_to_tensor(self, item):
        if self.img_to_tensor:
            if "rgb" in item:
                item["rgb"] = F.to_tensor(item["rgb"])

            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])
        return item

    def apply_scale_factor(self, item):
        assert self.cfg.scale_factor <= 1.0

        if "rgb" in item:
            H, W, _ = item["rgb"].shape
            h, w = int(H * self.cfg.scale_factor), int(W * self.cfg.scale_factor)
            rgb = Image.fromarray(item["rgb"]).resize(
                (w, h), resample=Image.BILINEAR
            )
            item["rgb"] = np.array(rgb)
    
        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h
        
        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

        # properties need to be scaled down when rgb is downsampled
        n_downsample_rgb = self.cfg.n_downsample_rgb if self.cfg.n_downsample_rgb else 1
        scale_factor = self.cfg.scale_factor / n_downsample_rgb
        item["scale_factor"] = scale_factor  # NOTE: not self.cfg.scale_factor
        if scale_factor < 1.0:
            if "intrinsic" in item:
                item["intrinsic"][:2] *= scale_factor
            if "alpha_map" in item:
                h, w = item["rgb"].shape[:2]
                alpha_map = Image.fromarray(item["alpha_map"]).resize(
                    (w, h), Image.Resampling.BILINEAR
                )
                item["alpha_map"] = np.array(alpha_map)
        return item

    def apply_background_color(self, item):
        if self.cfg.background_color is not None:
            assert (
                "alpha_map" in item
            ), "'alpha_map' is required to apply background color."
            fg = item["rgb"]
            if self.cfg.background_color == "white":
                bg = np.ones_like(fg) * 255
            elif self.cfg.background_color == "black":
                bg = np.zeros_like(fg)
            else:
                raise NotImplementedError(
                    f"Unknown background color: {self.cfg.background_color}."
                )

            # w = item["alpha_map"][..., None] / 255
            w = item["alpha_map"] / 255
            img = (w * fg + (1 - w) * bg).astype(np.uint8)
            item["rgb"] = img
        return item

    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        folder = p["folder"] if "folder" in p else None
        per_timestep = p["per_timestep"]
        suffix = p["suffix"]

        path = self.sequence_path
        if folder is not None:
            path = path / folder

        if self.num_cameras > 1:
            if camera_id is None:
                assert (
                    index is not None), "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_prefix" in p:
                camera_id = p["cam_id_prefix"] + camera_id
        else:
            camera_id = ""

        if per_timestep:
            if timestep_id is None:
                assert index is not None, "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            if len(camera_id) > 0:
                path /= f"{camera_id}_{timestep_id}.{suffix}"
            else:
                path /= f"{timestep_id}.{suffix}"
        else:
            if len(camera_id) > 0:
                path /= f"{camera_id}.{suffix}"
            else:
                path = Path(str(path) + f".{suffix}")

        return path
        
    def get_property_path_list(self, name):
        paths = []
        for i in range(len(self.items)):
            img_path = self.get_property_path(name, i)
            paths.append(img_path)
        return paths

    @property
    def num_timesteps(self):
        return len(self.timestep_indices)

    @property
    def num_cameras(self):
        return len(self.camera_ids)


if __name__ == "__main__":
    import tyro
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from vhap.config.base import DataConfig, import_module

    cfg = tyro.cli(DataConfig)
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
