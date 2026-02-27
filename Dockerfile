# ============================================================
# Dockerfile for HF Spaces Docker SDK (GPU)
# ============================================================
# Reproduces the exact environment from concierge_modal.py's
# Modal Image definition, but as a standard Dockerfile.
#
# Build: docker build -t lam-concierge .
# Run:   docker run --gpus all -p 7860:7860 lam-concierge
# HF:    Push to a HF Space with SDK=Docker, Hardware=GPU
# ============================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl ffmpeg tree \
    libgl1-mesa-glx libglib2.0-0 libusb-1.0-0 \
    build-essential ninja-build clang llvm libclang-dev \
    xz-utils libxi6 libxxf86vm1 libxfixes3 \
    libxrender1 libxkbcommon0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# numpy first (pinned for compatibility — must stay <2.0 for PyTorch 2.4 + mediapipe)
RUN pip install 'numpy==1.26.4'

# ============================================================
# PyTorch 2.4.0 + CUDA 12.1
# ============================================================
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# xformers — CRITICAL for DINOv2 MemEffAttention
# Without it, model produces garbage output ("bird monster").
# ============================================================
RUN pip install xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA build environment
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CC=clang
ENV CXX=clang++

# CUDA extensions (require no-build-isolation)
RUN pip install chumpy==0.70 --no-build-isolation

# pytorch3d — build from source (C++17 required for CUDA 12.1)
ENV CXXFLAGS="-std=c++17"
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation

# diff-gaussian-rasterization — patch CUDA 12.1 header issues then build
RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git /tmp/dgr && \
    find /tmp/dgr -name '*.cu' -exec sed -i '1i #include <cfloat>' {} + && \
    find /tmp/dgr -name '*.h' -path '*/cuda_rasterizer/*' -exec sed -i '1i #include <cstdint>' {} + && \
    pip install /tmp/dgr --no-build-isolation && \
    rm -rf /tmp/dgr

# simple-knn — patch cfloat for CUDA 12.1 then build
RUN git clone https://github.com/camenduru/simple-knn.git /tmp/simple-knn && \
    sed -i '1i #include <cfloat>' /tmp/simple-knn/simple_knn.cu && \
    pip install /tmp/simple-knn --no-build-isolation && \
    rm -rf /tmp/simple-knn

# nvdiffrast — JIT compilation at runtime (requires -devel image)
RUN pip install git+https://github.com/ShenhanQian/nvdiffrast.git@backface-culling --no-build-isolation

# ============================================================
# Python dependencies
# ============================================================
RUN pip install \
    "gradio==4.44.0" \
    "gradio_client==1.3.0" \
    "fastapi" \
    "uvicorn" \
    "omegaconf==2.3.0" \
    "pandas" \
    "scipy<1.14.0" \
    "opencv-python-headless==4.9.0.80" \
    "imageio[ffmpeg]" \
    "moviepy==1.0.3" \
    "rembg" \
    "scikit-image" \
    "pillow" \
    "huggingface_hub>=0.24.0" \
    "filelock" \
    "typeguard" \
    "transformers==4.44.2" \
    "diffusers==0.30.3" \
    "accelerate==0.34.2" \
    "tyro==0.8.0" \
    "mediapipe==0.10.21" \
    "tensorboard" \
    "rich" \
    "loguru" \
    "Cython" \
    "PyMCubes" \
    "trimesh" \
    "einops" \
    "plyfile" \
    "jaxtyping" \
    "ninja" \
    "patool" \
    "safetensors" \
    "decord" \
    "numpy==1.26.4"

# onnxruntime-gpu for CUDA 12 — MUST be installed AFTER rembg to prevent
# rembg from pulling in the PyPI default (CUDA 11) build
RUN pip install onnxruntime-gpu==1.18.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# FBX SDK Python bindings (for OBJ -> FBX -> GLB avatar export)
RUN pip install https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl

# ============================================================
# Blender 4.2 LTS (for GLB generation)
# ============================================================
RUN wget -q https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    mkdir -p /opt/blender && \
    tar xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && \
    ln -sf /opt/blender/blender /usr/local/bin/blender && \
    rm /tmp/blender.tar.xz

# ============================================================
# Clone LAM repo and build cpu_nms
# ============================================================
RUN git clone https://github.com/aigc3d/LAM.git /app/LAM

# Build cpu_nms for FaceBoxesV2
RUN cd /app/LAM/external/landmark_detection/FaceBoxesV2/utils/nms && \
    python -c "\
from setuptools import setup, Extension; \
from Cython.Build import cythonize; \
import numpy; \
setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), \
include_dirs=[numpy.get_include()])" \
    build_ext --inplace

# ============================================================
# Download model weights (cached in Docker layer)
# ============================================================
COPY download_models.py /app/download_models.py
RUN python /app/download_models.py

# ============================================================
# Copy application code (after model download for cache)
# ============================================================
WORKDIR /app/LAM

# Copy our app into the container
COPY app_concierge.py /app/LAM/app_concierge.py

# HF Spaces expects port 7860
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app_concierge.py"]