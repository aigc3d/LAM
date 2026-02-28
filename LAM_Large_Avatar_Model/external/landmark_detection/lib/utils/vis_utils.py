import cv2
import numpy as np
import numbers


def plot_points(vis, points, radius=1, color=(255, 255, 0), shift=4, indexes=0, is_index=False):
    if isinstance(points, list):
        num_point = len(points)
    elif isinstance(points, np.numarray):
        num_point = points.shape[0]
    else:
        raise NotImplementedError
    if isinstance(radius, numbers.Number):
        radius = np.zeros((num_point)) + radius

    if isinstance(indexes, numbers.Number):
        indexes = [indexes + i for i in range(num_point)]
    elif isinstance(indexes, list):
        pass
    else:
        raise NotImplementedError

    factor = (1 << shift)
    for (index, p, s) in zip(indexes, points, radius):
        cv2.circle(vis, (int(p[0] * factor + 0.5), int(p[1] * factor + 0.5)),
                   int(s * factor), color, 1, cv2.LINE_AA, shift=shift)
        if is_index:
            vis = cv2.putText(vis, str(index), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                              (255, 255, 255), 1)

    return vis
