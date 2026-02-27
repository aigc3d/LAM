import cv2

class Detector(object):
    def __init__(self, model_arch, model_weights):
        self.model_arch = model_arch
        self.model_weights = model_weights

    def detect(self, image, thresh):
        raise NotImplementedError

    def crop(self, image, detections):
        crops = []
        for det in detections:
            xmin = max(det[2], 0)
            ymin = max(det[3], 0)
            width = det[4]
            height = det[5]
            xmax = min(xmin+width, image.shape[1])
            ymax = min(ymin+height, image.shape[0])
            cut = image[ymin:ymax, xmin:xmax,:]
            crops.append(cut)

        return crops

    def draw(self, image, detections, im_scale=None):
        if im_scale is not None:
            image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            detections = [[det[0],det[1],int(det[2]*im_scale),int(det[3]*im_scale),int(det[4]*im_scale),int(det[5]*im_scale)] for det in detections]

        for det in detections:
            xmin = det[2]
            ymin = det[3]
            width = det[4]
            height = det[5]
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        return image
