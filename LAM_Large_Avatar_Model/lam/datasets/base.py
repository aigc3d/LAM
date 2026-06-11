# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
import traceback
import json
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union
from megfile import smart_open, smart_path_join, smart_exists


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dirs: str, meta_path: Optional[Union[list, str]]):
        super().__init__()
        self.root_dirs = root_dirs
        self.uids = self._load_uids(meta_path)

    def __len__(self):
        return len(self.uids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            # raise e
            return self.__getitem__((idx + 1) % self.__len__())

    @staticmethod
    def _load_uids(meta_path: Optional[Union[list, str]]):
        # meta_path is a json file
        if isinstance(meta_path, str):
            with open(meta_path, 'r') as f:
                uids = json.load(f)
        else:
            uids_lst = []
            max_total = 0
            for pth, weight in meta_path:
                with open(pth, 'r') as f:
                    uids = json.load(f)
                    max_total = max(len(uids) / weight, max_total)
                uids_lst.append([uids, weight, pth])
            merged_uids = []
            for uids, weight, pth in uids_lst:
                repeat = 1
                if len(uids) < int(weight * max_total):
                    repeat = int(weight * max_total) // len(uids)
                cur_uids = uids * repeat
                merged_uids += cur_uids
                print("Data Path:", pth, "Repeat:", repeat, "Final Length:", len(cur_uids))
            uids = merged_uids
            print("Total UIDs:", len(uids))
        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0):
        ''' Load and blend RGBA image to RGB with certain background, 0-1 scaled '''
        rgba = np.array(Image.open(smart_open(file_path, 'rb')))
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])
        rgba[:, :3, ...] * rgba[:, 3:, ...] + (1 - rgba[:, 3:, ...])
        return rgb

    @staticmethod
    def _locate_datadir(root_dirs, uid, locator: str):
        for root_dir in root_dirs:
            datadir = smart_path_join(root_dir, uid, locator)
            if smart_exists(datadir):
                return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for uid {uid}")
