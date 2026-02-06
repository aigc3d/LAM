"""
Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

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

import os
import math
import time
import librosa
import numpy as np
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn.functional as F

from .defaults import create_ddp_model
import utils.comm as comm
from models import build_model
from utils.logger import get_root_logger
from utils.registry import Registry
from utils.misc import (
    AverageMeter,
)

from models.utils import smooth_mouth_movements, apply_frame_blending, apply_savitzky_golay_smoothing, apply_random_brow_movement, \
    symmetrize_blendshapes, apply_random_eye_blinks, apply_random_eye_blinks_context, export_blendshape_animation, \
    RETURN_CODE, DEFAULT_CONTEXT, ARKitBlendShape

INFER = Registry("infer")

# Device detection for CPU/GPU support
def get_device():
    """Get the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class InferBase:
    def __init__(self, cfg, model=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.device = get_device()
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "infer.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.logger.info(f"=> Using device: {self.device}")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.to(self.device),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location=self.device)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}'".format(
                    self.cfg.weight
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model


    def infer(self):
        raise NotImplementedError



@INFER.register_module()
class Audio2ExpressionInfer(InferBase):
    def infer(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        self.model.eval()

        # process audio-input
        assert os.path.exists(self.cfg.audio_input)
        if(self.cfg.ex_vol):
            logger.info("Extract vocals ...")
            vocal_path = self.extract_vocal_track(self.cfg.audio_input)
            logger.info("=> Extract vocals at: {}".format(vocal_path if os.path.exists(vocal_path) else '... Failed'))
            if(os.path.exists(vocal_path)):
                self.cfg.audio_input = vocal_path

        with torch.no_grad():
            input_dict = {}
            input_dict['id_idx'] = F.one_hot(torch.tensor(self.cfg.id_idx),
                                             self.cfg.model.backbone.num_identity_classes).to(self.device)[None,...]
            speech_array, ssr = librosa.load(self.cfg.audio_input, sr=16000)
            input_dict['input_audio_array'] = torch.FloatTensor(speech_array).to(self.device)[None,...]

            end = time.time()
            output_dict = self.model(input_dict)
            batch_time.update(time.time() - end)

            logger.info(
                "Infer: [{}] "
                "Running Time: {batch_time.avg:.3f} ".format(
                    self.cfg.audio_input,
                    batch_time=batch_time,
                )
            )

        out_exp = output_dict['pred_exp'].squeeze().cpu().numpy()

        frame_length = math.ceil(speech_array.shape[0] / ssr * 30)
        volume = librosa.feature.rms(y=speech_array, frame_length=int(1 / 30 * ssr), hop_length=int(1 / 30 * ssr))[0]
        if (volume.shape[0] > frame_length):
            volume = volume[:frame_length]

        if(self.cfg.movement_smooth):
            out_exp = smooth_mouth_movements(out_exp, 0, volume)

        if (self.cfg.brow_movement):
            out_exp = apply_random_brow_movement(out_exp, volume)

        pred_exp = self.blendshape_postprocess(out_exp)

        if(self.cfg.save_json_path is not None):
            export_blendshape_animation(pred_exp,
                                        self.cfg.save_json_path,
                                        ARKitBlendShape,
                                        fps=self.cfg.fps)

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def infer_streaming_audio(self,
                           audio: np.ndarray,
                           ssr: float,
                           context: dict):

        if (context is None):
            context = DEFAULT_CONTEXT.copy()
        max_frame_length = 64

        frame_length = math.ceil(audio.shape[0] / ssr * 30)
        output_context = DEFAULT_CONTEXT.copy()

        volume = librosa.feature.rms(y=audio, frame_length=min(int(1 / 30 * ssr), len(audio)), hop_length=int(1 / 30 * ssr))[0]
        if (volume.shape[0] > frame_length):
            volume = volume[:frame_length]

        # resample audio
        if (ssr != self.cfg.audio_sr):
            in_audio = librosa.resample(audio.astype(np.float32), orig_sr=ssr, target_sr=self.cfg.audio_sr)
        else:
            in_audio = audio.copy()

        start_frame = int(max_frame_length - in_audio.shape[0] / self.cfg.audio_sr * 30)

        if (context['is_initial_input'] or (context['previous_audio'] is None)):
            blank_audio_length = self.cfg.audio_sr * max_frame_length // 30 - in_audio.shape[0]
            blank_audio = np.zeros(blank_audio_length, dtype=np.float32)

            # pre-append
            input_audio = np.concatenate([blank_audio, in_audio])
            output_context['previous_audio'] = input_audio

        else:
            clip_pre_audio_length = self.cfg.audio_sr * max_frame_length // 30 - in_audio.shape[0]
            clip_pre_audio = context['previous_audio'][-clip_pre_audio_length:]
            input_audio = np.concatenate([clip_pre_audio, in_audio])
            output_context['previous_audio'] = input_audio

        with torch.no_grad():
            try:
                input_dict = {}
                input_dict['id_idx'] = F.one_hot(torch.tensor(self.cfg.id_idx),
                                                 self.cfg.model.backbone.num_identity_classes).to(self.device)[
                    None, ...]
                input_dict['input_audio_array'] = torch.FloatTensor(input_audio).to(self.device)[None, ...]
                output_dict = self.model(input_dict)
                out_exp = output_dict['pred_exp'].squeeze().cpu().numpy()[start_frame:, :]
            except:
                self.logger.error('Error: faided to predict expression.')
                output_dict['pred_exp'] = torch.zeros((max_frame_length, 52)).float()
                return


        # post-process
        if (context['previous_expression'] is None):
            out_exp = self.apply_expression_postprocessing(out_exp, audio_volume=volume)
        else:
            previous_length = context['previous_expression'].shape[0]
            out_exp = self.apply_expression_postprocessing(expression_params = np.concatenate([context['previous_expression'], out_exp], axis=0),
                                                           audio_volume=np.concatenate([context['previous_volume'], volume], axis=0),
                                                           processed_frames=previous_length)[previous_length:, :]

        if (context['previous_expression'] is not None):
            output_context['previous_expression'] = np.concatenate([context['previous_expression'], out_exp], axis=0)[
                                                -max_frame_length:, :]
            output_context['previous_volume'] = np.concatenate([context['previous_volume'], volume], axis=0)[-max_frame_length:]
        else:
            output_context['previous_expression'] = out_exp.copy()
            output_context['previous_volume'] = volume.copy()

        output_context['first_input_flag'] = False

        return {"code": RETURN_CODE['SUCCESS'],
                "expression": out_exp,
                "headpose": None}, output_context
    def apply_expression_postprocessing(
            self,
            expression_params: np.ndarray,
            processed_frames: int = 0,
            audio_volume: np.ndarray = None
    ) -> np.ndarray:
        """Applies full post-processing pipeline to facial expression parameters.

        Args:
            expression_params: Raw output from animation model [num_frames, num_parameters]
            processed_frames: Number of frames already processed in previous batches
            audio_volume: Optional volume array for audio-visual synchronization

        Returns:
            Processed expression parameters ready for animation synthesis
        """
        # Pipeline execution order matters - maintain sequence
        expression_params = smooth_mouth_movements(expression_params, processed_frames, audio_volume)
        expression_params = apply_frame_blending(expression_params, processed_frames)
        expression_params, _ = apply_savitzky_golay_smoothing(expression_params, window_length=5)
        expression_params = symmetrize_blendshapes(expression_params)
        expression_params = apply_random_eye_blinks_context(expression_params, processed_frames=processed_frames)

        return expression_params

    def extract_vocal_track(
            self,
            input_audio_path: str
    ) -> str:
        """Isolates vocal track from audio file using source separation.

        Args:
            input_audio_path: Path to input audio file containing vocals+accompaniment

        Returns:
            Path to isolated vocal track in WAV format
        """
        separation_command = f'spleeter separate -p spleeter:2stems -o {self.cfg.save_path} {input_audio_path}'
        os.system(separation_command)

        base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        return os.path.join(self.cfg.save_path, base_name, 'vocals.wav')

    def blendshape_postprocess(self,
                               bs_array: np.ndarray
                               )->np.array:

        bs_array, _ = apply_savitzky_golay_smoothing(bs_array, window_length=5)
        bs_array = symmetrize_blendshapes(bs_array)
        bs_array = apply_random_eye_blinks(bs_array)

        return bs_array
