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


import os
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def _remap_human_model_path(config: dict) -> dict:
    """Remap human_model_path if the config references a non-existent directory.

    HuggingFace model config.json may hardcode
    human_model_path="./pretrained_models/human_model_files" but Modal volumes
    store the files under model_zoo/human_parametric_models.
    """
    hmp = config.get("human_model_path")
    if not hmp or os.path.isdir(hmp):
        return config
    alt = hmp.replace(
        "pretrained_models/human_model_files",
        "model_zoo/human_parametric_models",
    )
    if alt != hmp and os.path.isdir(alt):
        print(f"[hf_hub] Remapping human_model_path: {hmp} -> {alt}")
        config = dict(config)
        config["human_model_path"] = alt
    return config


def wrap_model_hub(model_cls: nn.Module):
    class HfModel(model_cls, PyTorchModelHubMixin):
        def __init__(self, config: dict):
            config = _remap_human_model_path(config)
            super().__init__(**config)
            self.config = config
    return HfModel
