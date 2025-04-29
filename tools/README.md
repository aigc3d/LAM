## Export Chatting Avatar File
### 🛠️ Environment Setup
#### Prerequisites
```
Python FBX SDK 2020.2+
Blender (version > 4.0.0)
```
#### Step1: download and install python fbx-sdk and other requirements
```bash
# FBX SDK: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-2
# Download FBX SDK installation package, example for Linux
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/fbxsdk_linux.tar
# Extract and install
tar -xf fbxsdk_linux.tar
sh tools/install_fbx_sdk.sh

# install other requirements
pip install pathlib
pip install patool
```
#### Step2: download blender
```bash
# Download latest Blender (>=4.0.0)
# Choose appropriate version from: https://www.blender.org/download/
# Example for Linux
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xvf blender-4.0.2-linux-x64.tar.xz -C ~/software/
```
#### Step3: download chatting avatar template file
```bash
# Download and extract sample files
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/sample_oac.tar
tar -xf sample_oac.tar -C assets/
```

### Gradio Run
```bash
# Example path for Blender executable
python app_lam.py --blender_path ~/software/blender-4.0.2-linux-x64/blender
```