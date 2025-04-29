import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import tyro
import yaml
from loguru import logger
from PIL import Image

from external.human_matting import StyleMatteEngine as HumanMattingEngine
from external.landmark_detection.FaceBoxesV2.faceboxes_detector import \
    FaceBoxesDetector
from external.landmark_detection.infer_image import Alignment
from external.vgghead_detector import VGGHeadDetector
from vhap.config.base import BaseTrackingConfig
from vhap.export_as_nerf_dataset import (NeRFDatasetWriter,
                                         TrackedFLAMEDatasetWriter, split_json)
from vhap.model.tracker import GlobalTracker

# Define error codes for various processing failures.
ERROR_CODE = {'FailedToDetect': 1, 'FailedToOptimize': 2, 'FailedToExport': 3}


def expand_bbox(bbox, scale=1.1):
    """Expands the bounding box by a given scale."""
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    extension_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    x_min_expanded = center_x - extension_size / 2
    x_max_expanded = center_x + extension_size / 2
    y_min_expanded = center_y - extension_size / 2
    y_max_expanded = center_y + extension_size / 2
    return torch.stack(
        [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded],
        dim=-1)


def load_config(src_folder: Path):
    """Load configuration from the given source folder."""
    config_file_path = src_folder / 'config.yml'
    if not config_file_path.exists():
        src_folder = sorted(
            src_folder.iterdir())[-1]  # Get the last modified folder
        config_file_path = src_folder / 'config.yml'
    assert config_file_path.exists(), f'File not found: {config_file_path}'

    config_data = yaml.load(config_file_path.read_text(), Loader=yaml.Loader)
    return src_folder, config_data


class FlameTrackingSingleImage:
    """Class for tracking and processing a single image."""
    def __init__(
            self,
            output_dir,
            alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
            vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
            human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
            facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
            detect_iris_landmarks=False,
            args=None):

        logger.info(f'Output Directory: {output_dir}')

        start_time = time.time()
        logger.info('Loading Pre-trained Models...')

        self.output_dir = output_dir
        self.output_preprocess = os.path.join(output_dir, 'preprocess')
        self.output_tracking = os.path.join(output_dir, 'tracking')
        self.output_export = os.path.join(output_dir, 'export')
        self.device = 'cuda:0'

        # Load alignment model
        assert os.path.exists(
            alignment_model_path), f'{alignment_model_path} does not exist!'
        if args is None:
            args = self._parse_args()
        args.config_name = "alignment"
        args.model_path = alignment_model_path
        self.alignment = Alignment(args,
                                   alignment_model_path,
                                   dl_framework='pytorch',
                                   device_ids=[0])

        # Load VGG head model
        assert os.path.exists(
            vgghead_model_path), f'{vgghead_model_path} does not exist!'
        self.vgghead_encoder = VGGHeadDetector(
            device=self.device, vggheadmodel_path=vgghead_model_path)

        # Load human matting model
        assert os.path.exists(
            human_matting_path), f'{human_matting_path} does not exist!'
        self.matting_engine = HumanMattingEngine(
            device=self.device, human_matting_path=human_matting_path)

        # Load face box detector model
        assert os.path.exists(
            facebox_model_path), f'{facebox_model_path} does not exist!'
        self.detector = FaceBoxesDetector('FaceBoxes', facebox_model_path,
                                          True, self.device)

        self.detect_iris_landmarks_flag = detect_iris_landmarks
        if self.detect_iris_landmarks_flag:
            from fdlite import FaceDetection, FaceLandmark, IrisLandmark
            self.iris_detect_faces = FaceDetection()
            self.iris_detect_face_landmarks = FaceLandmark()
            self.iris_detect_iris_landmarks = IrisLandmark()

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(f'Finished Loading Pre-trained Models. Time: '
                    f'{end_time - start_time:.2f}s')

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='Evaluation script')
        parser.add_argument('--output_dir',
                            type=str,
                            help='Output directory',
                            default='output')
        parser.add_argument('--config_name',
                            type=str,
                            help='Configuration name',
                            default='alignment')
        parser.add_argument('--blender_path',
                            type=str,
                            help='Blender path')
        return parser.parse_args()

    def preprocess(self, input_image_path):
        """Preprocess the input image for tracking."""
        if not os.path.exists(input_image_path):
            logger.warning(f'{input_image_path} does not exist!')
            return ERROR_CODE['FailedToDetect']

        start_time = time.time()
        logger.info('Starting Preprocessing...')
        name_list = []
        frame_index = 0

        # Bounding box detection
        frame = torchvision.io.read_image(input_image_path)[:3, ...]
        try:
            _, frame_bbox, _ = self.vgghead_encoder(frame, frame_index)
        except Exception:
            logger.error('Failed to detect face')
            return ERROR_CODE['FailedToDetect']

        if frame_bbox is None:
            logger.error('Failed to detect face')
            return ERROR_CODE['FailedToDetect']

        # Expand bounding box
        name_list.append('00000.png')
        frame_bbox = expand_bbox(frame_bbox, scale=1.65).long()

        # Crop and resize
        cropped_frame = torchvision.transforms.functional.crop(
            frame,
            top=frame_bbox[1],
            left=frame_bbox[0],
            height=frame_bbox[3] - frame_bbox[1],
            width=frame_bbox[2] - frame_bbox[0])
        cropped_frame = torchvision.transforms.functional.resize(
            cropped_frame, (1024, 1024), antialias=True)

        # Apply matting
        cropped_frame, mask = self.matting_engine(cropped_frame / 255.0,
                                                  return_type='matting',
                                                  background_rgb=1.0)
        cropped_frame = cropped_frame.cpu() * 255.0
        saved_image = np.round(cropped_frame.cpu().permute(
            1, 2, 0).numpy()).astype(np.uint8)[:, :, (2, 1, 0)]

        # Create output directories if not exist
        self.sub_output_dir = os.path.join(
            self.output_preprocess,
            os.path.splitext(os.path.basename(input_image_path))[0])
        output_image_dir = os.path.join(self.sub_output_dir, 'images')
        output_mask_dir = os.path.join(self.sub_output_dir, 'mask')
        output_alpha_map_dir = os.path.join(self.sub_output_dir, 'alpha_maps')

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_alpha_map_dir, exist_ok=True)

        # Save processed image, mask and alpha map
        cv2.imwrite(os.path.join(output_image_dir, name_list[frame_index]),
                    saved_image)
        cv2.imwrite(os.path.join(output_mask_dir, name_list[frame_index]),
                    np.array((mask.cpu() * 255.0)).astype(np.uint8))
        cv2.imwrite(
            os.path.join(output_alpha_map_dir,
                         name_list[frame_index]).replace('.png', '.jpg'),
            (np.ones_like(saved_image) * 255).astype(np.uint8))

        # Landmark detection
        detections, _ = self.detector.detect(saved_image, 0.8, 1)
        for idx, detection in enumerate(detections):
            x1_ori, y1_ori = detection[2], detection[3]
            x2_ori, y2_ori = x1_ori + detection[4], y1_ori + detection[5]

            scale = max(x2_ori - x1_ori, y2_ori - y1_ori) / 180
            center_w, center_h = (x1_ori + x2_ori) / 2, (y1_ori + y2_ori) / 2
            scale, center_w, center_h = float(scale), float(center_w), float(
                center_h)

            face_landmarks = self.alignment.analyze(saved_image, scale,
                                                    center_w, center_h)

        # Normalize and save landmarks
        normalized_landmarks = np.zeros((face_landmarks.shape[0], 3))
        normalized_landmarks[:, :2] = face_landmarks / 1024

        landmark_output_dir = os.path.join(self.sub_output_dir, 'landmark2d')
        os.makedirs(landmark_output_dir, exist_ok=True)

        landmark_data = {
            'bounding_box': [],
            'face_landmark_2d': normalized_landmarks[None, ...],
        }

        landmark_path = os.path.join(landmark_output_dir, 'landmarks.npz')
        np.savez(landmark_path, **landmark_data)

        if self.detect_iris_landmarks_flag:
            self._detect_iris_landmarks(
                os.path.join(output_image_dir, name_list[frame_index]))

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(
            f'Finished Processing Image. Time: {end_time - start_time:.2f}s')

        return 0

    def optimize(self):
        """Optimize the tracking model using configuration data."""
        start_time = time.time()
        logger.info('Starting Optimization...')

        tyro.extras.set_accent_color('bright_yellow')
        from yaml import safe_load, safe_dump
        with open("configs/vhap_tracking/base_tracking_config.yaml", 'r') as yml_f:
            config_data = safe_load(yml_f)
        config_data = tyro.from_yaml(BaseTrackingConfig, config_data)

        config_data.data.sequence = self.sub_output_dir.split('/')[-1]
        config_data.data.root_folder = Path(
            os.path.dirname(self.sub_output_dir))

        if not os.path.exists(self.sub_output_dir):
            logger.error(f'Failed to load {self.sub_output_dir}')
            return ERROR_CODE['FailedToOptimize']

        config_data.exp.output_folder = Path(self.output_tracking)
        tracker = GlobalTracker(config_data)
        tracker.optimize()

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(
            f'Finished Optimization. Time: {end_time - start_time:.2f}s')

        return 0

    def _detect_iris_landmarks(self, image_path):
        """Detect iris landmarks in the given image."""
        from fdlite import face_detection_to_roi, iris_roi_from_face_landmarks

        img = Image.open(image_path)
        img_size = (1024, 1024)

        face_detections = self.iris_detect_faces(img)
        if len(face_detections) != 1:
            logger.warning('Empty iris landmarks')
        else:
            face_detection = face_detections[0]
            try:
                face_roi = face_detection_to_roi(face_detection, img_size)
            except ValueError:
                logger.warning('Empty iris landmarks')
                return

            face_landmarks = self.iris_detect_face_landmarks(img, face_roi)
            if len(face_landmarks) == 0:
                logger.warning('Empty iris landmarks')
                return

            iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)

            if len(iris_rois) != 2:
                logger.warning('Empty iris landmarks')
                return

            landmarks = []
            for iris_roi in iris_rois[::-1]:
                try:
                    iris_landmarks = self.iris_detect_iris_landmarks(
                        img, iris_roi).iris[0:1]
                except np.linalg.LinAlgError:
                    logger.warning('Failed to get iris landmarks')
                    break

                # For each landmark, append x and y coordinates scaled to 1024.
                for landmark in iris_landmarks:
                    landmarks.append(landmark.x * 1024)
                    landmarks.append(landmark.y * 1024)

            landmark_data = {'00000.png': landmarks}
            json.dump(
                landmark_data,
                open(
                    os.path.join(self.sub_output_dir, 'landmark2d',
                                 'iris.json'), 'w'))

    def export(self):
        """Export the tracking results to configured folder."""
        logger.info(f'Beginning export from {self.output_tracking}')
        start_time = time.time()
        if not os.path.exists(self.output_tracking):
            logger.error(f'Failed to load {self.output_tracking}')
            return ERROR_CODE['FailedToExport'], 'Failed'

        src_folder = Path(self.output_tracking)
        tgt_folder = Path(self.output_export,
                          self.sub_output_dir.split('/')[-1])
        src_folder, config_data = load_config(src_folder)

        nerf_writer = NeRFDatasetWriter(config_data.data, tgt_folder, None,
                                        None, 'white')
        nerf_writer.write()

        flame_writer = TrackedFLAMEDatasetWriter(config_data.model,
                                                 src_folder,
                                                 tgt_folder,
                                                 mode='param',
                                                 epoch=-1)
        flame_writer.write()

        split_json(tgt_folder)

        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(f'Finished Export. Time: {end_time - start_time:.2f}s')

        return 0, str(tgt_folder)
