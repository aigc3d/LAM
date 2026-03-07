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
import numpy as np
import torch

def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False):
    import imageio
    # images: torch.tensor (T, C, H, W), 0-1  or numpy: (T, H, W, 3) 0-255
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        if isinstance(images, torch.Tensor):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        else:
            frame = images[i]
        frames.append(frame)
    frames = np.stack(frames)
    if gradio_codec:
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)
    else:
        # imageio.mimwrite(output_path, frames, fps=fps, codec='mpeg4', quality=10)
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)

    if verbose:
        print(f"Using gradio codec option {gradio_codec}")
        print(f"Saved video to {output_path}")


def save_images2video(img_lst, v_pth, fps):
    import moviepy.editor as mpy
    # Convert the list of NumPy arrays to a list of ImageClip objects
    clips = [mpy.ImageClip(img).set_duration(0.1) for img in img_lst]  # 0.1 seconds per frame

    # Concatenate the ImageClips into a single VideoClip
    video = mpy.concatenate_videoclips(clips, method="compose")

    # Write the VideoClip to a file
    video.write_videofile(v_pth, fps=fps)  # setting fps to 10 as example
    print("save video to:", v_pth)


if __name__ == "__main__":
    from glob import glob
    clip_name = "clip1"
    ptn = f"./assets/sample_motion/export/{clip_name}/images/*.png"
    images_pths = glob(ptn)
    import cv2
    import numpy as np
    images = [cv2.imread(pth) for pth in images_pths]
    save_images2video(images, "./assets/sample_mption/export/{clip_name}/video.mp4", 25, True)