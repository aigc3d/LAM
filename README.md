# LAM: Official Pytorch Implementation

[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://aigc3d.github.io/projects/LAM/) 
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2503-10625)](https://arxiv.org/pdf/2502.17796)
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

### <p align="center"> LAM: Large Avatar Model for One-shot Animatable Gaussian Head </p>

#####  <p align="center"> Yisheng He*, Xiaodong Gu*, Xiaodan Ye, Chao Xu, Zhengyi Zhao, Yuan Dong†, Weihao Yuan†, Zilong Dong, Liefeng Bo </p>

#####  <p align="center"> Tongyi Lab, Alibaba Group</p>

####  <p align="center"> **"Build 3D Interactive Chatting Avatar with One Image!"** </p>

<p align="center">
  <img src="./assets/images/teaser.jpg" width="100%">
</p>

## Core Highlights 🔥🔥🔥
- **Ultra-realistic 3D Avatar Creation from One Image**
- **Super-fast Cross-platform Animating and Rendering on Any Devices**
- **Low-latency SDK for Realtime Interactive Chatting Avatar**

## 📢 News

### To do list
- [x] Release LAM-small trained on VFHQ and Nersemble.
- [ ] Release Huggingface space.
- [ ] Release Modelscope space.
- [ ] Release LAM-large trained on a self-constructed large dataset.
- [ ] Release WebGL Render for cross-platform animation and rendering.
- [ ] Release audio driven model: Audio2Expression.
- [ ] Release Interactive Chatting Avatar SDK, including LLM, ASR, TTS, Avatar.



## 🚀 Get Started
### Environment Setup
```bash
git clone git@github.com:aigc3d/LAM.git
cd LAM

# install with cuda 11.8
sh ./scripts/install/install_cu118.sh

# or install with cuda 12.1
sh  ./scripts/install/install_cu121.sh
```

### Model Weights
| Model | Training Data | Link | Inference Time|
| :--- | :--- | :--- | :--- |
| LAM-20K | VFHQ | OSS | 1.4 s |
| LAM-20K | VFHQ+Nersemble | [OSS](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_20K.tar) | 1.4 s |
| LAM-20K | self-constructed large dataset | TBD  | 1.4 s |

```
# Download from OSS. 
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_20K.tar
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_assets.tar
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_human_model.tar

tar -xvf LAM-20K.tar 
tar -xvf LAM_assets.tar
tar -xvf LAM_human_model.tar
```


### Gradio Run
```
python app_lam.py
```

### Acknowledgement
This work is built on many amazing research works and open-source projects:
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [VHAP](https://github.com/ShenhanQian/VHAP)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)

Thanks for their excellent works and great contribution.


### More Works
Welcome to follow our other interesting works:
- [LHM](https://github.com/aigc3d/LHM)


### Citation
```
@inproceedings{he2025LAM,
  title={LAM: Large Avatar Model for One-shot Animatable Gaussian Head},
  author={
    Yisheng He and Xiaodong Gu and Xiaodan Ye and Chao Xu and Zhengyi Zhao and Yuan Dong and Weihao Yuan and Zilong Dong and Liefeng Bo
  },
  booktitle={arXiv preprint arXiv:2502.17796},
  year={2025}
}
```
