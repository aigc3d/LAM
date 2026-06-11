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


import math
from functools import partial
import torch

__all__ = ['MixerDataset']


class MixerDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 split: str,
                 subsets: dict,
                 **dataset_kwargs,
                 ):
        subsets = [e for e in subsets if e["meta_path"][split] is not None]
        self.subsets = [
            self._dataset_fn(subset, split)(**dataset_kwargs)
            for subset in subsets
        ]
        self.virtual_lens = [
            math.ceil(subset_config['sample_rate'] * len(subset_obj))
            for subset_config, subset_obj in zip(subsets, self.subsets)
        ]

    @staticmethod
    def _dataset_fn(subset_config: dict, split: str):
        name = subset_config['name']

        dataset_cls = None
        if name == "exavatar":
            from .exavatar import ExAvatarDataset
            dataset_cls = ExAvatarDataset  
        elif name == "humman":
            from .humman import HuMManDataset
            dataset_cls = HuMManDataset
        elif name == "humman_ori":
            from .humman_ori import HuMManOriDataset
            dataset_cls = HuMManOriDataset
        elif name == "static_human":
            from .static_human import StaticHumanDataset
            dataset_cls = StaticHumanDataset
        elif name == "singleview_human":
            from .singleview_human import SingleViewHumanDataset
            dataset_cls = SingleViewHumanDataset
        elif name == "singleview_square_human":
            from .singleview_square_human import SingleViewSquareHumanDataset
            dataset_cls = SingleViewSquareHumanDataset
        elif name == "bedlam":
            from .bedlam import BedlamDataset
            dataset_cls = BedlamDataset
        elif name == "dna_human":
            from .dna import DNAHumanDataset
            dataset_cls = DNAHumanDataset
        elif name == "video_human":
            from .video_human import VideoHumanDataset
            dataset_cls = VideoHumanDataset
        elif name == "video_head":
            from .video_head import VideoHeadDataset
            dataset_cls = VideoHeadDataset
        elif name == "video_head_gagtrack":
            from .video_head_gagtrack import VideoHeadGagDataset
            dataset_cls = VideoHeadGagDataset
        elif name == "objaverse":
            from .objaverse import ObjaverseDataset
            dataset_cls = ObjaverseDataset
        # elif name == 'mvimgnet':
        #     from .mvimgnet import MVImgNetDataset
        #     dataset_cls = MVImgNetDataset
        else:
            raise NotImplementedError(f"Dataset {name} not implemented")
        print("==="*16*3, "\nUse dataset loader:", name, "\n"+"==="*3*16)

        return partial(
            dataset_cls,
            root_dirs=subset_config['root_dirs'],
            meta_path=subset_config['meta_path'][split],
        )

    def __len__(self):
        return sum(self.virtual_lens)

    def __getitem__(self, idx):
        subset_idx = 0
        virtual_idx = idx
        while virtual_idx >= self.virtual_lens[subset_idx]:
            virtual_idx -= self.virtual_lens[subset_idx]
            subset_idx += 1
        real_idx = virtual_idx % len(self.subsets[subset_idx])
        return self.subsets[subset_idx][real_idx]
