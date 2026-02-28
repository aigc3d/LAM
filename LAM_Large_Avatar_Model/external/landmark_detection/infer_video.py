import cv2
import math
import copy
import numpy as np
import argparse
import torch
import json

# private package
from lib import utility
from FaceBoxesV2.faceboxes_detector import *

class GetCropMatrix():
    """
    from_shape -> transform_matrix
    """

    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix


class TransformPerspective():
    """
    image, matrix3x3 -> transformed_image
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class TransformPoints2D():
    """
    points (nx2), matrix (3x3) -> points (nx2)
    """

    def process(self, srcPoints, matrix):
        # nx3
        desPoints = np.concatenate([srcPoints, np.ones_like(srcPoints[:, [0]])], axis=1)
        desPoints = desPoints @ np.transpose(matrix)  # nx3
        desPoints = desPoints[:, :2] / desPoints[:, [2, 2]]
        return desPoints.astype(srcPoints.dtype)

class Alignment:
    def __init__(self, args, model_path, dl_framework, device_ids):
        self.input_size = 256
        self.target_face_scale = 1.0
        self.dl_framework = dl_framework

        # model
        if self.dl_framework == "pytorch":
            # conf
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            # set environment
            utility.set_environment(self.config)
            # self.config.init_instance()
            # if self.config.logger is not None:
            #     self.config.logger.info("Loaded configure file %s: %s" % (args.config_name, self.config.id))
            #     self.config.logger.info("\n" + "\n".join(["%s: %s" % item for item in self.config.__dict__.items()]))

            net = utility.get_net(self.config)
            if device_ids == [-1]:
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint["net"])

            if self.config.device_id == -1:
                net = net.cpu()
            else:
                net = net.to(self.config.device_id)
                
            net.eval()
            self.alignment = net
        else:
            assert False

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size, target_face_scale=self.target_face_scale,
                                           align_corners=True)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)
        self.transformPoints2D = TransformPoints2D()

    def norm_points(self, points, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            return points / torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            return (points * 2 + 1) / torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1

    def denorm_points(self, points, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            return (points + 1) / 2 * torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            return ((points + 1) * torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2

    def preprocess(self, image, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]

        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0

        if self.config.device_id == -1:
            input_tensor = input_tensor.cpu()
        else:
            input_tensor = input_tensor.to(self.config.device_id)
        
        return input_tensor, matrix

    def postprocess(self, srcPoints, coeff):
        # dstPoints = self.transformPoints2D.process(srcPoints, coeff)
        # matrix^(-1) * src = dst
        # src = matrix * dst
        dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
        for i in range(srcPoints.shape[0]):
            dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
            dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
        return dstPoints

    def analyze(self, image, scale, center_w, center_h):
        input_tensor, matrix = self.preprocess(image, scale, center_w, center_h)

        if self.dl_framework == "pytorch":
            with torch.no_grad():
                output = self.alignment(input_tensor)
            landmarks = output[-1][0]
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix))

        return landmarks
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument('--video_path', type=str, help='Path to videos',default='/media/yuanzhen/HH/DATASET/VFTH/TESTVIDEO/Clip+7CzHzeeVRlE+P0+C0+F101007-101139.mp4')
    args = parser.parse_args()

    # args.video_path = '/media/gyalex/Data/flame/ph_test/test.mp4'

    current_path = os.getcwd()

    use_gpu = True
    ########### face detection ############
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    current_path = os.getcwd()
    det_model_path = '/home/yuanzhen/code/landmark_detection/FaceBoxesV2/weights/FaceBoxesV2.pth'
    detector = FaceBoxesDetector('FaceBoxes', det_model_path, use_gpu, device)

    ########### facial alignment ############
    model_path = '/home/yuanzhen/code/landmark_detection/weights/68_keypoints_model.pkl'

    if use_gpu:
        device_ids = [0]
    else: 
        device_ids = [-1]

    args.config_name = 'alignment'
    alignment = Alignment(args, model_path, dl_framework="pytorch", device_ids=device_ids)

    video_file = args.video_path
    cap = cv2.VideoCapture(video_file)
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))

    # out_video_file = './output_video.mp4'
    # fps = 30
    # size = (frame_width, frame_height)
    # out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    count = 0
    kpts_code = dict()

    keypoint_data_path = args.video_path.replace('.mp4','.json')
    with open(keypoint_data_path,'r') as f:
        keypoint_data = json.load(f)

    ########### inference ############
    path = video_file[:-4]
    while(cap.isOpened()):
        ret, image = cap.read()

        if ret:
            detections, _ = detector.detect(image, 0.8, 1)
            image_draw = copy.deepcopy(image)

            cv2.imwrite(os.path.join(path, 'image', str(count+1)+'.png'), image_draw)    

            for idx in range(len(detections)):
                x1_ori = detections[idx][2]
                y1_ori = detections[idx][3]
                x2_ori = x1_ori + detections[idx][4] 
                y2_ori = y1_ori + detections[idx][5]
                
                scale    = max(x2_ori - x1_ori, y2_ori - y1_ori) / 180
                center_w = (x1_ori + x2_ori) / 2
                center_h = (y1_ori + y2_ori) / 2
                scale, center_w, center_h = float(scale), float(center_w), float(center_h)

                # landmarks_pv = alignment.analyze(image, scale, center_w, center_h)
                landmarks_pv = np.array(keypoint_data[str(count+1)+'.png'])

                landmarks_pv_list = landmarks_pv.tolist()

                for num in range(landmarks_pv.shape[0]):
                    cv2.circle(image_draw, (round(landmarks_pv[num][0]), round(landmarks_pv[num][1])), 
                            2, (0, 255, 0), -1)
                    cv2.putText(image_draw, str(num),
                                (round(landmarks_pv[num][0]) + 5, round(landmarks_pv[num][1]) + 5),  # 文本位置
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                kpts_code[str(count+1)+'.png'] = landmarks_pv_list
                cv2.imwrite(os.path.join(path, 'landmark', str(count+1)+'.png'), image_draw)        
        else:
            break

        count += 1
            
    cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    path = video_file[:-4]
    json.dump(kpts_code, open(os.path.join(path, 'keypoint.json'), 'w'))

    print(path)



