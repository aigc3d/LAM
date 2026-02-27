from pathlib import Path
import json
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from vhap.util.log import get_logger


logger = get_logger(__name__)


class NeRFDataset(Dataset):
    def __init__(
        self,
        root_folder,
        division=None,
        camera_convention_conversion=None,
        target_extrinsic_type='w2c',
        use_fg_mask=False,
        use_flame_param=False,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |
                |---<images>/
                |   |---00000.jpg
                |   |...
                |
                |---<fg_masks>/
                |   |---00000.png
                |   |...
                |
                |---<flame_param>/
                |   |---00000.npz
                |   |...
                |
                |---transforms_backup.json          # backup of the original transforms.json
                |---transforms_backup_flame.json    # backup of the original transforms.json with flame_param
                |---transforms.json                 # the final transforms.json
                |---transforms_train.json   # the final transforms.json for training
                |---transforms_val.json     # the final transforms.json for validation
                |---transforms_test.json    # the final transforms.json for testing

                                   
        """

        super().__init__()
        self.root_folder = Path(root_folder)
        self.division = division
        self.camera_convention_conversion = camera_convention_conversion
        self.target_extrinsic_type = target_extrinsic_type
        self.use_fg_mask = use_fg_mask
        self.use_flame_param = use_flame_param

        logger.info(f"Loading NeRF scene from: {root_folder}")

        # data division
        if division is None:
            tranform_path = self.root_folder / "transforms.json"
        elif division == "train":
            tranform_path = self.root_folder / "transforms_train.json"
        elif division == "val":
            tranform_path = self.root_folder / "transforms_val.json"
        elif division == "test":
            tranform_path = self.root_folder / "transforms_test.json"
        else:
            raise NotImplementedError(f"Unknown division type: {division}")
        logger.info(f"division: {division}")

        self.transforms = json.load(open(tranform_path, "r"))
        logger.info(f"number of timesteps: {len(self.transforms['timestep_indices'])}, number of cameras: {len(self.transforms['camera_indices'])}")

        assert len(self.transforms['timestep_indices']) == max(self.transforms['timestep_indices']) + 1

    def __len__(self):
        return len(self.transforms['frames'])

    def __getitem__(self, i):
        frame = self.transforms['frames'][i]

        # 'timestep_index', 'timestep_index_original', 'timestep_id', 'camera_index', 'camera_id', 'cx', 'cy', 'fl_x', 'fl_y', 'h', 'w', 'camera_angle_x', 'camera_angle_y', 'transform_matrix', 'file_path', 'fg_mask_path', 'flame_param_path']

        K = torch.eye(3)
        K[[0, 1, 0, 1], [0, 1, 2, 2]] = torch.tensor(
            [frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]]
        )

        c2w = torch.tensor(frame['transform_matrix'])
        if self.target_extrinsic_type == "w2c":
            extrinsic = c2w.inverse()
        elif self.target_extrinsic_type == "c2w":
            extrinsic = c2w
        else:
            raise NotImplementedError(f"Unknown extrinsic type: {self.target_extrinsic_type}")
        
        img_path = self.root_folder / frame['file_path']

        item = {
            'timestep_index': frame['timestep_index'],
            'camera_index': frame['camera_index'],
            'intrinsics': K,
            'extrinsics': extrinsic,
            'image_height': frame['h'],
            'image_width': frame['w'],
            'image': np.array(Image.open(img_path)),
            'image_path': img_path,
        }

        if self.use_fg_mask and 'fg_mask_path' in frame:
            fg_mask_path = self.root_folder / frame['fg_mask_path']
            item["fg_mask"] = np.array(Image.open(fg_mask_path))
            item["fg_mask_path"] = fg_mask_path

        if self.use_flame_param and 'flame_param_path' in frame:
            npz = np.load(self.root_folder / frame['flame_param_path'], allow_pickle=True)
            item["flame_param"] = dict(npz)

        return item

    def apply_to_tensor(self, item):
        if self.img_to_tensor:
            if "rgb" in item:
                item["rgb"] = F.to_tensor(item["rgb"])
                # if self.rgb_range_shift:
                #     item["rgb"] = (item["rgb"] - 0.5) / 0.5

            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])
        return item


if __name__ == "__main__":
    from tqdm import tqdm
    from dataclasses import dataclass
    import tyro
    from torch.utils.data import DataLoader

    @dataclass
    class Args:
        root_folder: str
        subject: str
        sequence: str
        use_landmark: bool = False
        batchify_all_views: bool = False

    args = tyro.cli(Args)

    dataset = NeRFDataset(root_folder=args.root_folder)

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    for item in tqdm(dataloader):
        pass
