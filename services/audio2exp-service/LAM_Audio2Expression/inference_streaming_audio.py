"""
# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np

from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.infer import INFER
import librosa
from tqdm import tqdm
import time


def export_json(bs_array, json_path):
    from models.utils import export_blendshape_animation, ARKitBlendShape
    export_blendshape_animation(bs_array, json_path, ARKitBlendShape, fps=30.0)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/lam_audio2exp_config_streaming.py'
    cfg = default_config_parser(args.config_file, args.options)


    cfg = default_setup(cfg)
    infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
    infer.model.eval()

    audio, sample_rate = librosa.load(cfg.audio_input, sr=16000)
    context = None
    input_num = audio.shape[0]//16000+1
    gap = 16000
    all_exp = []
    for i in tqdm(range(input_num)):

        start = time.time()
        output, context = infer.infer_streaming_audio(audio[i*gap:(i+1)*gap], sample_rate, context)
        end = time.time()
        print('Inference time {}'.format(end - start))
        all_exp.append(output['expression'])

    all_exp = np.concatenate(all_exp,axis=0)

    export_json(all_exp, cfg.save_json_path)