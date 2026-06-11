# LAM: Official Pytorch Implementation

[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://aigc3d.github.io/projects/LAM/) 
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2503-10625)](https://arxiv.org/pdf/2502.17796)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/3DAIGC/LAM)
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./assets/images/logo.jpeg" width="20%">
</p>

### <p align="center"> LAM: Large Avatar Model for One-shot Animatable Gaussian Head </p>

#####  <p align="center"> Yisheng He*, Xiaodong Gu*, Xiaodan Ye, Chao Xu, Zhengyi Zhao, Yuan Dong†, Weihao Yuan†, Zilong Dong, Liefeng Bo </p>

#####  <p align="center"> Tongyi Lab, Alibaba Group</p>

####  <p align="center"> **"Build 3D Interactive Chatting Avatar with One Image in Seconds!"** </p>

<p align="center">
  <img src="./assets/images/teaser.jpg" width="100%">
</p>

## Core Highlights 🔥🔥🔥
- **Ultra-realistic 3D Avatar Creation from One Image in Seconds**
- **Super-fast Cross-platform Animating and Rendering on Any Devices**
- **Low-latency SDK for Realtime Interactive Chatting Avatar**

## 📢 News

### To do list
- [x] Release LAM-small trained on VFHQ and Nersemble.
- [x] Release Huggingface space.
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
# Install with Cuda 12.1
sh  ./scripts/install/install_cu121.sh
# Or Install with Cuda 11.8
sh ./scripts/install/install_cu118.sh
```

### Model Weights

| Model   | Training Data                  | HuggingFace | OSS | Reconstruction Time | A100 (A & R) |   XiaoMi 14 Phone (A & R)          |
|---------|--------------------------------|----------|----------|---------------------|-----------------------------|-----------|
| LAM-20K | VFHQ                          | TBD       | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | VFHQ + NeRSemble                | [Link](https://huggingface.co/3DAIGC/LAM-20K) | [Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_yisheng/LAM/LAM_20K.tar)   | 1.4 s               | 562.9FPS                    | 110+FPS   |
| LAM-20K | Our large dataset | TBD      | TBD      | 1.4 s               | 562.9FPS                    | 110+FPS   |

(**A & R:** Animating & Rendering )

```
# HuggingFace download
# Download Assets
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_human_model.tar && rm ./tmp/LAM_human_model.tar
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
huggingface-cli download yuandong513/flametracking_model --local-dir ./tmp/
tar -xf ./tmp/pretrain_model.tar && rm -r ./tmp/
# Download Model Weights
huggingface-cli download 3DAIGC/LAM-20K --local-dir ./exps/releases/lam/lam-20k/step_045500/


# Or OSS Download (In case of HuggingFace download failing)
# Download assets
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_assets.tar
tar -xf LAM_assets.tar && rm LAM_assets.tar
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_human_model.tar
tar -xf LAM_human_model.tar && rm LAM_human_model.tar
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/tracking_pretrain_model.tar
tar -xf tracking_pretrain_model.tar && rm tracking_pretrain_model.tar
# Download Model Weights
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_20K.tar
tar -xf LAM-20K.tar && rm LAM-20K.tar
```


### Gradio Run
```
python app_lam.py
```

### Inference
```bash
sh ./scripts/inference.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_PATH_OR_FOLDER} ${MOTION_SEQ}
```

### Acknowledgement
This work is built on many amazing research works and open-source projects:
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
- [VHAP](https://github.com/ShenhanQian/VHAP)

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
