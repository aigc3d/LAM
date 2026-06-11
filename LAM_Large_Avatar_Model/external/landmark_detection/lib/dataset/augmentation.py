import os
import cv2
import math
import random
import numpy as np
from skimage import transform


class Augmentation:
    def __init__(self,
                 is_train=True,
                 aug_prob=1.0,
                 image_size=256,
                 crop_op=True,
                 std_lmk_5pts=None,
                 target_face_scale=1.0,
                 flip_rate=0.5,
                 flip_mapping=None,
                 random_shift_sigma=0.05,
                 random_rot_sigma=math.pi/180*18,
                 random_scale_sigma=0.1,
                 random_gray_rate=0.2,
                 random_occ_rate=0.4,
                 random_blur_rate=0.3,
                 random_gamma_rate=0.2,
                 random_nose_fusion_rate=0.2):
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.crop_op = crop_op
        self._flip = Flip(flip_mapping, flip_rate)
        if self.crop_op:
            self._cropMatrix = GetCropMatrix(
                                    image_size=image_size, 
                                    target_face_scale=target_face_scale, 
                                    align_corners=True)
        else:
            self._alignMatrix = GetAlignMatrix(
                                    image_size=image_size,
                                    target_face_scale=target_face_scale,
                                    std_lmk_5pts=std_lmk_5pts)
        self._randomGeometryMatrix = GetRandomGeometryMatrix(
                                        target_shape=(image_size, image_size),
                                        from_shape=(image_size, image_size),
                                        shift_sigma=random_shift_sigma,
                                        rot_sigma=random_rot_sigma,
                                        scale_sigma=random_scale_sigma,
                                        align_corners=True)
        self._transform = Transform(image_size=image_size)
        self._randomTexture = RandomTexture(
                                random_gray_rate=random_gray_rate, 
                                random_occ_rate=random_occ_rate, 
                                random_blur_rate=random_blur_rate, 
                                random_gamma_rate=random_gamma_rate, 
                                random_nose_fusion_rate=random_nose_fusion_rate)

    def process(self, img, lmk, lmk_5pts=None, scale=1.0, center_w=0, center_h=0, is_train=True):
        if self.is_train and random.random() < self.aug_prob:
            img, lmk, lmk_5pts, center_w, center_h = self._flip.process(img, lmk, lmk_5pts, center_w, center_h)
            matrix_geoaug = self._randomGeometryMatrix.process()
            if self.crop_op:
                matrix_pre = self._cropMatrix.process(scale, center_w, center_h)
            else:
                matrix_pre = self._alignMatrix.process(lmk_5pts)
            matrix = matrix_geoaug @ matrix_pre
            aug_img, aug_lmk = self._transform.process(img, lmk, matrix)
            aug_img = self._randomTexture.process(aug_img)
        else:
            if self.crop_op:
                matrix = self._cropMatrix.process(scale, center_w, center_h)
            else:
                matrix = self._alignMatrix.process(lmk_5pts)
            aug_img, aug_lmk = self._transform.process(img, lmk, matrix)
        return aug_img, aug_lmk, matrix


class GetCropMatrix:
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
            to_w, to_h = self.image_size-1, self.image_size-1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w/2.0, to_h/2.0])
        return matrix


class GetAlignMatrix:
    def __init__(self, image_size, target_face_scale, std_lmk_5pts):
        """
        points in std_lmk_5pts range from -1 to 1.
        """
        self.std_lmk_5pts = (std_lmk_5pts * target_face_scale + 1) * \
            np.array([image_size, image_size], np.float32) / 2.0

    def process(self, lmk_5pts):
        assert lmk_5pts.shape[-2:] == (5, 2)
        tform = transform.SimilarityTransform()
        tform.estimate(lmk_5pts, self.std_lmk_5pts)
        return tform.params


class GetRandomGeometryMatrix:
    def __init__(self, target_shape, from_shape,
                 shift_sigma=0.1, rot_sigma=18*math.pi/180, scale_sigma=0.1,
                 shift_mu=0.0, rot_mu=0.0, scale_mu=1.0,
                 shift_normal=True, rot_normal=True, scale_normal=True,
                 align_corners=False):
        self.target_shape = target_shape
        self.from_shape = from_shape
        self.shift_config = (shift_mu, shift_sigma, shift_normal)
        self.rot_config = (rot_mu, rot_sigma, rot_normal)
        self.scale_config = (scale_mu, scale_sigma, scale_normal)
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

    def _random(self, mu_sigma_normal, size=None):
        mu, sigma, is_normal = mu_sigma_normal
        if is_normal:
            return np.random.normal(mu, sigma, size=size)
        else:
            return np.random.uniform(low=mu-sigma, high=mu+sigma, size=size)

    def process(self):
        if self.align_corners:
            from_w, from_h = self.from_shape[1]-1, self.from_shape[0]-1
            to_w, to_h = self.target_shape[1]-1, self.target_shape[0]-1
        else:
            from_w, from_h = self.from_shape[1], self.from_shape[0]
            to_w, to_h = self.target_shape[1], self.target_shape[0]

        if self.shift_config[:2] != (0.0, 0.0) or \
           self.rot_config[:2] != (0.0, 0.0) or \
           self.scale_config[:2] != (1.0, 0.0):
            shift_xy = self._random(self.shift_config, size=[2]) * \
                min(to_h, to_w)
            rot_angle = self._random(self.rot_config)
            scale = self._random(self.scale_config)
            matrix_geoaug = self._compose_rotate_and_scale(
                rot_angle, scale, shift_xy,
                from_center=[from_w/2.0, from_h/2.0],
                to_center=[to_w/2.0, to_h/2.0])

        return matrix_geoaug


class Transform:
    def __init__(self, image_size):
        self.image_size = image_size

    def _transformPoints2D(self, points, matrix):
        """
        points (nx2), matrix (3x3) -> points (nx2)
        """
        dtype = points.dtype

        # nx3
        points = np.concatenate([points, np.ones_like(points[:, [0]])], axis=1)
        points = points @ np.transpose(matrix)
        points = points[:, :2] / points[:, [2, 2]]
        return points.astype(dtype)

    def _transformPerspective(self, image, matrix):
        """
        image, matrix3x3 -> transformed_image
        """
        return cv2.warpPerspective(
            image, matrix, 
            dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)

    def process(self, image, landmarks, matrix):
        t_landmarks = self._transformPoints2D(landmarks, matrix)
        t_image = self._transformPerspective(image, matrix)
        return t_image, t_landmarks


class RandomTexture:
    def __init__(self, random_gray_rate=0, random_occ_rate=0, random_blur_rate=0, random_gamma_rate=0, random_nose_fusion_rate=0):
        self.random_gray_rate = random_gray_rate
        self.random_occ_rate = random_occ_rate
        self.random_blur_rate = random_blur_rate
        self.random_gamma_rate = random_gamma_rate
        self.random_nose_fusion_rate = random_nose_fusion_rate
        self.texture_augs = (
            (self.add_occ, self.random_occ_rate),
            (self.add_blur, self.random_blur_rate), 
            (self.add_gamma, self.random_gamma_rate), 
            (self.add_nose_fusion, self.random_nose_fusion_rate)
        )

    def add_gray(self, image):
        assert image.ndim == 3 and image.shape[-1] == 3
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.tile(np.expand_dims(image, -1), [1, 1, 3])
        return image

    def add_occ(self, image):
        h, w, c = image.shape
        rh = 0.2 + 0.6 * random.random() # [0.2, 0.8]
        rw = rh - 0.2 + 0.4 * random.random()
        cx = int((h - 1) * random.random())
        cy = int((w - 1) * random.random())
        dh = int(h / 2 * rh)
        dw = int(w / 2 * rw)
        x0 = max(0, cx - dw // 2)
        y0 = max(0, cy - dh // 2)
        x1 = min(w - 1, cx + dw // 2)
        y1 = min(h - 1, cy + dh // 2)
        image[y0:y1+1, x0:x1+1] = 0
        return image

    def add_blur(self, image):
        blur_kratio = 0.05 * random.random()
        blur_ksize = int((image.shape[0] + image.shape[1]) / 2 * blur_kratio)
        if blur_ksize > 1:
            image = cv2.blur(image, (blur_ksize, blur_ksize))
        return image

    def add_gamma(self, image):
        if random.random() < 0.5:
            gamma = 0.25 + 0.75 * random.random()
        else:
            gamma = 1.0 + 3.0 * random.random()
        image = (((image / 255.0) ** gamma) * 255).astype("uint8")
        return image

    def add_nose_fusion(self, image):
        h, w, c = image.shape
        nose = np.array(bytearray(os.urandom(h * w * c)), dtype=image.dtype).reshape(h, w, c)
        alpha = 0.5 * random.random()
        image = (1 - alpha) * image + alpha * nose
        return image.astype(np.uint8)

    def process(self, image):
        image = image.copy()
        if random.random() < self.random_occ_rate:
            image = self.add_occ(image)
        if random.random() < self.random_blur_rate:
            image = self.add_blur(image)
        if random.random() < self.random_gamma_rate:
            image = self.add_gamma(image)
        if random.random() < self.random_nose_fusion_rate:
            image = self.add_nose_fusion(image)
        """
        orders = list(range(len(self.texture_augs)))
        random.shuffle(orders)
        for order in orders:
            if random.random() < self.texture_augs[order][1]:
                image = self.texture_augs[order][0](image)
        """

        if random.random() < self.random_gray_rate:
            image = self.add_gray(image)

        return image


class Flip:
    def __init__(self, flip_mapping, random_rate):
        self.flip_mapping = flip_mapping
        self.random_rate = random_rate

    def process(self, image, landmarks, landmarks_5pts, center_w, center_h):
        if random.random() >= self.random_rate or self.flip_mapping is None:
            return image, landmarks, landmarks_5pts, center_w, center_h

        # COFW
        if landmarks.shape[0] == 29:
            flip_offset = 0
        # 300W, WFLW
        elif landmarks.shape[0] in (68, 98):
            flip_offset = -1
        else:
            flip_offset = -1

        h, w, _ = image.shape
        #image_flip = cv2.flip(image, 1)
        image_flip = np.fliplr(image).copy()
        landmarks_flip = landmarks.copy()
        for i, j in self.flip_mapping:
            landmarks_flip[i] = landmarks[j]
            landmarks_flip[j] = landmarks[i]
        landmarks_flip[:, 0] = w + flip_offset - landmarks_flip[:, 0]
        if landmarks_5pts is not None:
            flip_mapping = ([0, 1], [3, 4])
            landmarks_5pts_flip = landmarks_5pts.copy()
            for i, j in flip_mapping:
                landmarks_5pts_flip[i] = landmarks_5pts[j]
                landmarks_5pts_flip[j] = landmarks_5pts[i]
            landmarks_5pts_flip[:, 0] = w + flip_offset - landmarks_5pts_flip[:, 0]
        else:
            landmarks_5pts_flip = None
        
        center_w = w + flip_offset - center_w
        return image_flip, landmarks_flip, landmarks_5pts_flip, center_w, center_h
