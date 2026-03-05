import os
import sys
import subprocess
import time
import shutil
import modal
import base64

# アプリ名を変更
app = modal.App("lam-final-v33-ui-fix-v2")

# --- 事前チェック ---
local_assets_path = "./assets/human_parametric_models/flame_assets/flame/flame2023.pkl"
if __name__ == "__main__":
    if not os.path.exists(local_assets_path):
        print(f"❌ CRITICAL ERROR: Local asset not found at: {local_assets_path}")
        sys.exit(1)

# --- UI修復パッチ (Base64) ---
# 1. GradioのExamplesを無効化
# 2. サーバーポートを8080に固定
PATCH_SCRIPT = """
import re
import os

path = '/root/LAM/app_lam.py'
if os.path.exists(path):
    print("🛠️ Applying UI patch...")
    with open(path, 'r') as f:
        code = f.read()

    # 1. Examples機能を無効化するコードを注入
    patch_code = '''
import gradio as gr
# --- PATCH START ---
try:
    class DummyExamples:
        def __init__(self, *args, **kwargs): pass
        def attach_load_event(self, *args, **kwargs): pass
        def render(self): pass
    gr.Examples = DummyExamples
    print("✅ Gradio Examples disabled to prevent UI crash.")
except Exception as e:
    print(f"⚠️ Failed to disable examples: {e}")
# --- PATCH END ---
'''
    code = code.replace('import gradio as gr', patch_code)

    # 2. 起動設定の強制書き換え
    if '.launch(' in code:
        code = re.sub(r'\.launch\s*\(', ".launch(server_name='0.0.0.0', server_port=8080, ", code)
        print("✅ Server port forced to 8080.")

    with open(path, 'w') as f:
        f.write(code)
    print("🚀 Patch applied successfully.")
"""

# スクリプトをBase64化
patch_b64 = base64.b64encode(PATCH_SCRIPT.encode('utf-8')).decode('utf-8')
patch_cmd = f"python -c \"import base64; exec(base64.b64decode('{patch_b64}'))\""


# --- 1. 環境構築 ---
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "tree",
        "libusb-1.0-0", "build-essential", "ninja-build",
        "clang", "llvm", "libclang-dev"
    )

    # 1. Base setup
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy==1.23.5'"
    )
    # 2. PyTorch 2.2.0
    .run_commands(
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118"
    )

    # 3. Build Environment
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "CC": "clang",
        "CXX": "clang++"
    })
    
    # 4. Critical Build (no-build-isolation)
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7 --no-build-isolation"
    )
    
    # 5. Dependencies
    .pip_install(
        "gradio==3.50.2", 
        "omegaconf==2.3.0", 
        "pandas", 
        "scipy<1.14.0",
        "opencv-python-headless", 
        "imageio[ffmpeg]", 
        "moviepy==1.0.3", 
        "rembg[gpu]",
        "scikit-image", 
        "pillow", 
        "onnxruntime-gpu", 
        "huggingface_hub>=0.24.0", 
        "filelock",
        "typeguard", 
        
        "transformers==4.44.2",
        "diffusers==0.30.3",
        "accelerate==0.34.2",
        "tyro==0.8.0",
        "mediapipe==0.10.21", 
        
        "tensorboard", 
        "rich", 
        "loguru", 
        "Cython", 
        "PyMCubes", 
        "trimesh", 
        "einops", 
        "plyfile", 
        "jaxtyping", 
        "ninja",
        "numpy==1.23.5"
    )
    
    # 6. LAM 3D Libs
    .run_commands(
        "pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git --no-build-isolation",
        "pip install git+https://github.com/ShenhanQian/nvdiffrast.git@backface-culling --no-build-isolation"
    )
    
    # 7. LAM Setup with UI Patch
    .run_commands(
        "mkdir -p /root/LAM",
        "rm -rf /root/LAM",
        "git clone https://github.com/aigc3d/LAM.git /root/LAM",
        
        # cpu_nms ビルド
        "cd /root/LAM/external/landmark_detection/FaceBoxesV2/utils/nms && "
        "echo \"from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), include_dirs=[numpy.get_include()])\" > setup.py && "
        "python setup.py build_ext --inplace",
        
        # ★パッチ適用（UIのサンプル機能を無効化）
        patch_cmd
    )
)

# --- 2. サーバー準備 ---
def setup_server():
    from huggingface_hub import snapshot_download
    print("📥 Downloading checkpoints...")
    try:
        snapshot_download(
            repo_id="3DAIGC/LAM-20K",
            local_dir="/root/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500",
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Checkpoints download warning: {e}")

image = (
    image
    .run_function(setup_server)
    .add_local_dir("./assets", remote_path="/root/LAM/model_zoo", copy=True)
)

# --- 3. アプリ起動 ---
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600
)
@modal.web_server(8080)
def ui():
    os.chdir("/root/LAM")
    import sys
    print(f"🚀 Launching LAM App (Python {sys.version})")
    
    cmd = "python -u app_lam.py"
    subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()