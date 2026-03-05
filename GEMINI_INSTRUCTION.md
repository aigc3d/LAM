# Gemini作業指示書: ModelScope app.py → Modal移植

## 目的

ModelScope上で正常動作している`app.py`を、Modal（https://modal.com）上で動くように移植する。
出力ファイル名: `app_modal.py`

---

## 最重要ルール

1. **推論禁止**: LAMのGitHubリポジトリの知識を使わないこと。この指示書とソースコードだけを見て作業する
2. **写経方式**: `app.py`の関数をそのままコピーし、パス変更など必要最小限の修正のみ行う
3. **変更箇所の明示**: 原本から1行でも変えた箇所には `# [MODAL変更] 理由` コメントを書く
4. **追加禁止**: 原本に無いロジック（最適化、修正、改善）は一切追加しない

---

## アーキテクチャ概要

```
[ユーザーブラウザ]
    ↓ Gradio UI
[Modal: web() 関数] ← 軽量コンテナ（CPU only）
    ↓ bytes送信
[Modal: Generator クラス] ← GPU (L4) コンテナ
    ↓ LAM推論 + GLB生成
[Modal Volume: concierge-output] → ZIP + Preview動画
```

---

## Modal インフラ（そのまま使う、変更不要）

### Volume

| Volume名 | マウントパス | 内容 |
|-----------|-------------|------|
| `lam-storage` | `/vol/lam-storage` | 重みファイル、モデル、アセット全て |
| `concierge-output` | `/vol/output` | 生成物の受け渡し |

### Volume内のディレクトリ構造

```
/vol/lam-storage/LAM/
├── model_zoo/
│   ├── lam_models/releases/lam/lam-20k/step_045500/
│   │   ├── config.json          ← from_pretrained()が読む
│   │   └── model.safetensors    ← 重みファイル
│   ├── flame_tracking_models/
│   │   ├── 68_keypoints_model.pkl
│   │   ├── vgghead/vgg_heads_l.trcd
│   │   ├── matting/stylematte_synth.pt
│   │   └── FaceBoxesV2.pth
│   ├── human_parametric_models/
│   │   └── flame_assets/flame/flame2023.pkl
│   ├── sample_motion/export/*/    ← サンプルモーション
│   └── sample_oac/
│       ├── template_file.fbx
│       └── animation.glb
├── assets/  (sample_motion等のコピー)
├── pretrained_models/  (存在する場合あり)
└── configs/
```

### ソースコード配置

Modal Image Build時に `git clone https://github.com/aigc3d/LAM.git /root/LAM` している。
ランタイムの作業ディレクトリは `/root/LAM`。

---

## パス変換表（これだけ変える）

| ModelScope `app.py`のパス | Modal上のパス | 備考 |
|---------------------------|---------------|------|
| `./exps/releases/lam/lam-20k/step_045500/` | `./model_zoo/lam_models/releases/lam/lam-20k/step_045500/` | APP_MODEL_NAME |
| `./configs/inference/lam-20k-8gpu.yaml` | `./configs/inference/lam-20k-8gpu.yaml` | 変更なし（git cloneに含まれる） |
| `./pretrained_models/68_keypoints_model.pkl` | `./model_zoo/flame_tracking_models/68_keypoints_model.pkl` | |
| `./pretrained_models/vgghead/vgg_heads_l.trcd` | `./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd` | |
| `./pretrained_models/matting/stylematte_synth.pt` | `./model_zoo/flame_tracking_models/matting/stylematte_synth.pt` | |
| `./pretrained_models/FaceBoxesV2.pth` | `./model_zoo/flame_tracking_models/FaceBoxesV2.pth` | |
| `./pretrained_models/human_model_files` | シンボリックリンク先を参照（後述） | |
| `./assets/sample_motion/export/` | `./model_zoo/sample_motion/export/` | |
| `./assets/sample_oac/template_file.fbx` | `./model_zoo/sample_oac/template_file.fbx` | |
| `./assets/sample_oac/animation.glb` | `./model_zoo/sample_oac/animation.glb` | |
| `./blender-4.0.2-linux-x64/blender` | `/usr/local/bin/blender` | Blender 4.2がインストール済 |

### human_model_files のブリッジ

推論configの`lam-20k-8gpu.yaml`内に以下の記述がある:
```yaml
human_model_path: "./pretrained_models/human_model_files"
```

Volume上の実体は `/vol/lam-storage/LAM/model_zoo/human_parametric_models/`。
そのため、初期化時にシンボリックリンクを作成する:
```python
os.symlink(
    "/vol/lam-storage/LAM/model_zoo/human_parametric_models",  # 実体
    "/root/LAM/pretrained_models/human_model_files"            # configが参照するパス
)
```

---

## 写経対象の関数と変更指示

### 1. `_build_model(cfg)` — **一切変更しない**

```python
# app.py 641-648行をそのままコピー
def _build_model(cfg):
    from lam.models import model_dict
    from lam.utils.hf_hub import wrap_model_hub

    hf_model_cls = wrap_model_hub(model_dict["lam"])
    model = hf_model_cls.from_pretrained(cfg.model_name)

    return model
```

**注意**: 元コードは`from_pretrained()`で重み読み込みまで一括で行う。
手動で`ModelLAM()`を構築したり`load_file()`で重みを読んではいけない。

### 2. `parse_configs()` — パス変更のみ

`app.py` 248-306行の`parse_configs()`をそのままコピー。
変更点:
- `blender_path`のデフォルトを `/usr/local/bin/blender` に変更

### 3. `launch_gradio_app()` のモデル初期化部分 — パス変更のみ

```python
# app.py 651-672行を参考に
os.environ.update({
    'APP_ENABLED': '1',
    'APP_MODEL_NAME': './model_zoo/lam_models/releases/lam/lam-20k/step_045500/',  # [MODAL変更] パス
    'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
    'APP_TYPE': 'infer.lam',
    'NUMBA_THREADING_LAYER': 'forseq',
})

cfg, _ = parse_configs()
lam = _build_model(cfg)
lam.to('cuda')

flametracking = FlameTrackingSingleImage(
    output_dir='tracking_output',
    alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',     # [MODAL変更] パス
    vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',     # [MODAL変更] パス
    human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',   # [MODAL変更] パス
    facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',              # [MODAL変更] パス
    detect_iris_landmarks=False,
)
```

### 4. `core_fn()` 推論ロジック — **最も重要。一切変更しない**

`app.py` 311-471行の`core_fn()`をそのままコピーする。
Gradio固有の戻り値だけModal向けに調整。

特に以下のパラメータは原本のまま:
```python
# preprocess_image の呼び出し（app.py 365-370行）
image, _, _, shape_param = preprocess_image(
    image_path, mask_path=mask_path, intr=None, pad_ratio=0,
    bg_color=1., max_tgt_size=None, aspect_standard=aspect_standard,
    enlarge_ratio=[1.0, 1.0], render_tgt_size=source_size,
    multiply=14, need_mask=True, get_shape_param=True
)

# prepare_motion_seqs の呼び出し（app.py 381-386行）
motion_seq = prepare_motion_seqs(
    motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
    bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1, 0],
    render_image_res=render_size, multiply=16,
    need_mask=motion_img_need_mask, vis_motion=vis_motion,
    shape_param=shape_param, test_sample=False, cross_id=False,
    src_driven=src_driven, max_squen_length=300
)

# infer_single_view の呼び出し（app.py 392-398行）
motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
device, dtype = "cuda", torch.float32
with torch.no_grad():
    res = lam.infer_single_view(
        image.unsqueeze(0).to(device, dtype), None, None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()}
    )
```

### 5. OAC (GLB + ZIP) 生成 — パス変更のみ

`app.py` 402-446行のOAC生成部分をコピー。変更点:
- `Path("./assets/sample_oac/template_file.fbx")` → `Path("./model_zoo/sample_oac/template_file.fbx")`
- `Path(cfg.blender_path)` → `Path("/usr/local/bin/blender")`
- `'./assets/sample_oac/animation.glb'` → `'./model_zoo/sample_oac/animation.glb'`

### 6. 結果の後処理（RGB→動画） — 一切変更しない

`app.py` 448-468行をそのままコピー。

### 7. ヘルパー関数 — そのままコピー

以下はそのまま:
- `save_imgs_2_video()` (app.py 207-222行)
- `add_audio_to_video()` (app.py 225-245行)
- `compile_module()` (app.py 76-98行) ← Modal版では不要（ビルド済）
- `preprocess_fn()` (app.py 187-198行) ← Modal版では不要

---

## Modal固有の構造

### Image Build

既存の`app_modal.py`の`image = (...)` 定義（32-199行目）をそのまま流用する。
これはModelScopeの`app.py`冒頭の`os.system("pip install ...")`群に相当する。

ただし以下の点に注意:
- `@torch.compile`デコレータの除去は**Image Build時のsed**で行う（既存コードにあり）
- 原本の`lam-20k-8gpu.yaml`に`compile.disable: true`があるが、`@torch.compile`デコレータはクラス定義時に評価されるため、sedでの除去が必要

### _setup_model_paths()

Volume上のファイルを`/root/LAM`配下にシンボリックリンクする関数。
既存のものを流用するが、簡略化して以下だけ行う:

```python
def _setup_model_paths():
    """Volume上のディレクトリを/root/LAM配下にリンク"""
    import shutil
    lam_root = "/root/LAM"
    vol_lam = "/vol/lam-storage/LAM"

    for subdir in ["model_zoo", "assets", "pretrained_models"]:
        src = os.path.join(vol_lam, subdir)
        dst = os.path.join(lam_root, subdir)
        if not os.path.isdir(src):
            continue
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        os.symlink(src, dst)

    # human_model_filesブリッジ
    pretrained_hm = os.path.join(lam_root, "pretrained_models", "human_model_files")
    model_zoo_hpm = os.path.join(lam_root, "model_zoo", "human_parametric_models")
    if not os.path.exists(pretrained_hm) and os.path.isdir(model_zoo_hpm):
        os.makedirs(os.path.dirname(pretrained_hm), exist_ok=True)
        os.symlink(model_zoo_hpm, pretrained_hm)
```

### _init_lam_pipeline() — 新規作成（写経ベース）

```python
def _init_lam_pipeline():
    """app.pyのlaunch_gradio_app()を写経。Gradio部分を除去"""
    os.chdir("/root/LAM")
    sys.path.insert(0, "/root/LAM")
    _setup_model_paths()

    # ここからapp.py 652-672行の写経
    os.environ.update({...})  # パス変更のみ
    cfg, _ = parse_configs()  # app.pyの関数をそのまま使う
    lam = _build_model(cfg)   # app.pyの関数をそのまま使う
    lam.to('cuda')

    from flame_tracking_single_image import FlameTrackingSingleImage
    flametracking = FlameTrackingSingleImage(...)  # パス変更のみ

    return cfg, lam, flametracking
```

### Generator クラス

```python
@app.cls(gpu="L4", image=image,
         volumes={"/vol/output": output_vol, "/vol/lam-storage": storage_vol},
         timeout=600)
class Generator:
    @modal.enter()
    def setup(self):
        self.cfg, self.lam, self.flametracking = _init_lam_pipeline()

    @modal.method()
    def generate(self, image_bytes, video_bytes, motion_name, job_id):
        # image_bytes → tmpファイルに保存
        # app.pyのcore_fn()を写経して実行
        # 結果をVolume(/vol/output)に保存
```

### web() 関数（Gradio UI）

軽量コンテナ（GPUなし）で動作。
ユーザーからの入力を受け取り、`Generator.generate.remote()`にbytesで渡す。
結果はVolume経由で取得。

---

## 絶対にやってはいけないこと

1. `from lam.models import ModelLAM` → `ModelLAM(**config)`で手動構築すること
   - 必ず`from_pretrained()`を使う
2. `safetensors.torch.load_file()`で手動重み読み込み
   - `from_pretrained()`に任せる
3. `torch.compile`のmonkey-patch（`torch.compile = lambda fn: fn`的なもの）
   - Image Build時のsedで除去済。ランタイムでの介入は不要
4. `app_lam.py`を参照すること
   - このファイルは別バージョンのコード。正解は`app.py`のみ
5. 独自の「修正」「最適化」「改善」の追加
   - shape_guardの追加、dtype変換の追加、etc. 全て禁止

---

## flame_tracking_single_image.py のimportパスについて

ModelScope版は:
```python
from flame_tracking_single_image import FlameTrackingSingleImage
```

Modal版では、このファイルは`/root/LAM/tools/`にある場合がある:
```python
# まず試す
try:
    from flame_tracking_single_image import FlameTrackingSingleImage
except ImportError:
    from tools.flame_tracking_single_image import FlameTrackingSingleImage
```

または、`sys.path`に追加:
```python
sys.path.insert(0, "/root/LAM/tools")
from flame_tracking_single_image import FlameTrackingSingleImage
```

---

## generateARKITGLBWithBlender.py のimportパスについて

ModelScope版は:
```python
from generateARKITGLBWithBlender import generate_glb
```

Modal版では:
```python
try:
    from generateARKITGLBWithBlender import generate_glb
except ImportError:
    from tools.generateARKITGLBWithBlender import generate_glb
```

---

## 作業完了時のチェックリスト

- [ ] `_build_model()`が`from_pretrained()`を使っている
- [ ] `parse_configs()`が原本と同一（パス以外）
- [ ] `preprocess_image()`の引数が原本と完全一致
- [ ] `prepare_motion_seqs()`の引数が原本と完全一致（`enlarge_ratio=[1.0, 1, 0]`含む ← 原本にあるtypo的な値もそのまま）
- [ ] `infer_single_view()`の引数が原本と完全一致
- [ ] `motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)` が推論前にある
- [ ] OAC生成が原本と同一順序（save_shaped_mesh → save_ply → generate_glb → copy animation.glb → remove mesh）
- [ ] RGB後処理が原本と同一（comp_rgb → mask処理 → save_imgs_2_video → add_audio）
- [ ] `torch.compile`のランタイムmonkey-patchが**無い**
- [ ] `ModelLAM(**config)`による手動構築が**無い**
- [ ] `load_file()`による手動重み読み込みが**無い**
- [ ] 全ての `# [MODAL変更]` コメントの数が10箇所以内

---

## 参考: 既存Modal Image Build定義

以下は動作確認済のImage Build定義。そのまま使う:

```python
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "tree",
        "libusb-1.0-0", "build-essential",
        "gcc", "g++", "ninja-build",
        "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libxkbcommon0", "libsm6",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy==1.26.4'",
    )
    .run_commands(
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        "pip install xformers==0.0.27.post2 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CC": "gcc",
        "CXX": "g++",
        "CXXFLAGS": "-std=c++17",
        "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
        "TORCHDYNAMO_DISABLE": "1",
    })
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        # chumpy numpy互換パッチ
        "CHUMPY_INIT=$(python -c \"import importlib.util; print(importlib.util.find_spec('chumpy').origin)\") && "
        "sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/"
        "from numpy import nan, inf; import numpy; bool = numpy.bool_; int = numpy.int_; "
        "float = numpy.float64; complex = numpy.complex128; object = numpy.object_; "
        "unicode = numpy.str_; str = numpy.str_/' "
        "\"$CHUMPY_INIT\" && "
        "find $(dirname \"$CHUMPY_INIT\") -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true",
        "pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation",
    )
    .pip_install(
        "gradio==4.44.0", "gradio_client==1.3.0", "fastapi",
        "omegaconf==2.3.0", "pandas", "scipy<1.14.0",
        "opencv-python-headless==4.9.0.80", "imageio[ffmpeg]",
        "moviepy==1.0.3", "rembg[gpu]", "scikit-image", "pillow",
        "huggingface_hub>=0.24.0", "filelock", "typeguard",
        "transformers==4.44.2", "diffusers==0.30.3", "accelerate==0.34.2",
        "tyro==0.8.0", "mediapipe==0.10.21", "tensorboard", "rich",
        "loguru", "Cython", "PyMCubes", "trimesh", "einops", "plyfile",
        "jaxtyping", "ninja", "patool", "safetensors", "decord",
        "numpy==1.26.4",
    )
    .run_commands(
        "pip install onnxruntime-gpu==1.18.1 "
        "--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
    )
    # diff-gaussian-rasterization
    .run_commands(
        "git clone https://github.com/ashawkey/diff-gaussian-rasterization.git /tmp/dgr && "
        "git clone https://github.com/g-truc/glm.git /tmp/dgr/third_party/glm && "
        "find /tmp/dgr -name '*.cu' -exec sed -i '1i #include <cfloat>' {} + && "
        "find /tmp/dgr -name '*.h' -path '*/cuda_rasterizer/*' -exec sed -i '1i #include <cstdint>' {} + && "
        "pip install /tmp/dgr --no-build-isolation && rm -rf /tmp/dgr",
    )
    # simple-knn
    .run_commands(
        "git clone https://github.com/camenduru/simple-knn.git /tmp/simple-knn && "
        "sed -i '1i #include <cfloat>' /tmp/simple-knn/simple_knn.cu && "
        "pip install /tmp/simple-knn --no-build-isolation && rm -rf /tmp/simple-knn",
    )
    # nvdiffrast
    .run_commands(
        "pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation",
    )
    # FBX SDK
    .run_commands(
        "pip install https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl",
    )
    # Blender 4.2 LTS
    .run_commands(
        "wget -q https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender",
        "tar xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1",
        "ln -sf /opt/blender/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
    # LAMソースコード + cpu_nmsビルド
    .run_commands(
        "git clone https://github.com/aigc3d/LAM.git /root/LAM",
        "sed -i 's/dtype=np\\.int)/dtype=np.intp)/' "
        "/root/LAM/external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.pyx",
        "cd /root/LAM/external/landmark_detection/FaceBoxesV2/utils/nms && "
        "python -c \""
        "from setuptools import setup, Extension; "
        "from Cython.Build import cythonize; "
        "import numpy; "
        "setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), "
        "include_dirs=[numpy.get_include()])\" "
        "build_ext --inplace",
    )
    # @torch.compile デコレータ除去（Modal L4で数値破損を起こすため）
    .run_commands(
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/models/modeling_lam.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/models/encoders/dinov2_fusion_wrapper.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/losses/tvloss.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/root/LAM/lam/losses/pixelwise.py",
    )
    # DINOv2事前ダウンロード
    .run_commands(
        "python -c \""
        "import torch; "
        "url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'; "
        "torch.hub.load_state_dict_from_url(url, map_location='cpu'); "
        "print('DINOv2 cached OK')\"",
    )
)

# nvdiffrast事前コンパイル
def _precompile_nvdiffrast():
    import torch
    import nvdiffrast.torch as dr
    print("nvdiffrast pre-compiled OK")

image = image.run_function(_precompile_nvdiffrast)
```

---

## 参考: ModelScope app.py 全文

このファイルを直接渡す: `LAM_Large_Avatar_Model/app.py`
（lam-large-uploadブランチ上にある）

---

## 成果物

`app_modal.py` 1ファイル。以下の構造:

1. Modal Image Build定義（上記をそのまま）
2. `_setup_model_paths()` — Volume→/root/LAMリンク
3. `parse_configs()` — app.pyからの写経（パス変更のみ）
4. `_build_model(cfg)` — app.pyからの写経（変更なし）
5. `_init_lam_pipeline()` — launch_gradio_app()の写経（Gradio除去、パス変更のみ）
6. `_generate_concierge_zip()` — core_fn()の写経（パス変更のみ）+ OAC生成 + 動画保存
7. `Generator` クラス — Modal GPU実行
8. `web()` 関数 — Gradio UI + ファイル配信
