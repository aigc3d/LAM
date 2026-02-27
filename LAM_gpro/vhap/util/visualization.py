# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_keypoints


connectivity_face = (
    [(i, i + 1) for i in list(range(0, 16))]
    + [(i, i + 1) for i in list(range(17, 21))]
    + [(i, i + 1) for i in list(range(22, 26))]
    + [(i, i + 1) for i in list(range(27, 30))]
    + [(i, i + 1) for i in list(range(31, 35))]
    + [(i, i + 1) for i in list(range(36, 41))]
    + [(36, 41)]
    + [(i, i + 1) for i in list(range(42, 47))]
    + [(42, 47)]
    + [(i, i + 1) for i in list(range(48, 59))]
    + [(48, 59)]
    + [(i, i + 1) for i in list(range(60, 67))]
    + [(60, 67)]
)


def plot_landmarks_2d(
    img: torch.tensor,
    lmks: torch.tensor,
    connectivity=None,
    colors="white",
    unit=1,
    input_float=False,
):
    if input_float:
        img = (img * 255).byte()

    img = draw_keypoints(
        img,
        lmks,
        connectivity=connectivity,
        colors=colors,
        radius=2 * unit,
        width=2 * unit,
    )

    if input_float:
        img = img.float() / 255
    return img


def blend(a, b, w):
    return (a * w + b * (1 - w)).byte()


if __name__ == "__main__":
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    from vhap.data.nersemble_dataset import NeRSembleDataset

    parser = ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--division", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--blend_weight", type=float, default=0.6)
    args = parser.parse_args()

    dataset = NeRSembleDataset(
        root_folder=args.root_folder,
        subject=args.subject,
        sequence=args.sequence,
        division=args.division,
        subset=args.subset,
        n_downsample_rgb=2,
        scale_factor=args.scale_factor,
        use_landmark=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for item in dataloader:
        unit = int(item["scale_factor"][0] * 3) + 1

        rgb = item["rgb"][0].permute(2, 0, 1)
        vis = rgb

        if "bbox_2d" in item:
            bbox = item["bbox_2d"][0][:4]
            tmp = draw_bounding_boxes(vis, bbox[None, ...], width=5 * unit)
            vis = blend(tmp, vis, args.blend_weight)

        if "lmk2d" in item:
            face_landmark = item["lmk2d"][0][:, :2]
            tmp = plot_landmarks_2d(
                vis,
                face_landmark[None, ...],
                connectivity=connectivity_face,
                colors="white",
                unit=unit,
            )
            vis = blend(tmp, vis, args.blend_weight)

        if "lmk2d_iris" in item:
            iris_landmark = item["lmk2d_iris"][0][:, :2]
            tmp = plot_landmarks_2d(
                vis,
                iris_landmark[None, ...],
                colors="blue",
                unit=unit,
            )
            vis = blend(tmp, vis, args.blend_weight)

        vis = vis.permute(1, 2, 0).numpy()
        plt.imshow(vis)
        plt.draw()
        while not plt.waitforbuttonpress(timeout=-1):
            pass
