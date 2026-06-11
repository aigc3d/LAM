import os
import pdb
import torch
import numpy as np
import imageio
import cv2
import imageio.v3 as iio

VIDEO_TYPE_LIST = {'.avi','.mp4','.gif','.AVI','.MP4','.GIF'}

def encodeffmpeg(inputs, frame_rate, output, format="png"):
    """output: need video_name"""
    assert (
        os.path.splitext(output)[-1] in VIDEO_TYPE_LIST
    ), "output is the format of video, e.g., mp4"
    assert os.path.isdir(inputs), "input dir is NOT file format"

    inputs = inputs[:-1] if inputs[-1] == "/" else inputs

    output = os.path.abspath(output)

    cmd = (
        f"ffmpeg -r {frame_rate} -pattern_type glob -i '{inputs}/*.{format}' "
        + f'-vcodec libx264 -crf 10 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        + f"-pix_fmt yuv420p {output} > /dev/null 2>&1"
    )

    print(cmd)

    output_dir = os.path.dirname(output)
    if os.path.exists(output):
        os.remove(output)
    os.makedirs(output_dir, exist_ok=True)

    print("encoding imgs to video.....")
    os.system(cmd)
    print("video done!")

def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False, bitrate="2M"):
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
        width, height = frame.shape[1], frame.shape[0]
        # reshape to limit the export time
        # if width > 1200 or height > 1200 or images.shape[0] > 200:
        #     frames.append(cv2.resize(frame, (width // 2, height // 2)))
        # else:
        frames.append(frame)
    # limit the frames directly @NOTE huggingface only！
    frames = frames[:200]
    
    frames = np.stack(frames)

    print("start saving {} using imageio.v3 .".format(output_path))
    iio.imwrite(output_path,frames,fps=fps,codec="libx264",pixelformat="yuv420p",bitrate=bitrate, macro_block_size=32)
    print("saved {} using imageio.v3 .".format(output_path))