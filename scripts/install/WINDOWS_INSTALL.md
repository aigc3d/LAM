


## Windows Installation Guide

### Base software

- Python 3.10
- Nvidia Cuda Toolkit 11.8 (You can also change to others)
- Visual Studio 2019: 2022 will cause some compilation error on cuda operators. Download it from [techspot](https://www.techspot.com/downloads/7241-visual-studio-2019.html)



### Install Dependencies

Note we use "x64 Native Tools" from Visual Studio as the compilation and do not use powershell or cmd. It offer MSVC environment for python package compilation.

Open "x64 Native Tools" terminal and install dependencies:

- Prepare environment:
    We recommend to use venv (or conda) to create a python environment:
    ```bash
    python -m venv lam_env
    lam_env\Scripts\activate
    git clone https://github.com/aigc3d/LAM.git
    git checkout feat/windows
    ```

- Install torch 2.3.0 and xformers
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118
    ```

- Install python packages which do not need compilation:
    ```bash
    # pip install -r requirements.txt without the last 4 lines:
    head -n $((total_lines - 4)) requirements.txt | pip install -r /dev/stdin
    ```

- Install packages which need compilation
    ```bash
    # Install Pytorch3d, which follows:
    # https://blog.csdn.net/m0_70229101/article/details/127196699
    # https://blog.csdn.net/qq_61247019/article/details/139927752
    set DISTUTILS_USE_SDK=1
    set PYTORCH3D_NO_NINJA=1
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    # modify setup.py
    # add "-DWIN32_LEAN_AND_MEAN" in nvcc_args
    python setup.py install

    # Install other packages
    pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
    pip install nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling
    pip install git+https://github.com/camenduru/simple-knn/

    cd external/landmark_detection/FaceBoxesV2/utils/
    python3 build.py build_ext --inplace
    cd ../../../../
    ```


### Run

```bash
python app_lam.py
```