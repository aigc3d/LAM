"""
Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import base64

import gradio as gr
import argparse
from omegaconf import OmegaConf
from gradio_gaussian_render import gaussian_render

from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.infer import INFER
from pathlib import Path

try:
    import spaces
except:
    pass

import patoolib

h5_rendering = True


def assert_input_image(input_image,input_zip_textbox):
    if(os.path.exists(input_zip_textbox)):
        return
    if input_image is None:
        raise gr.Error('No image selected or uploaded!')


def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def get_image_base64(path):
    with open(path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'


def do_render():
    print('WebGL rendering ....')
    return

def audio_loading():
    print("Audio loading ....")
    return "None"

def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def create_zip_archive(output_zip='assets/arkitWithBSData.zip', base_dir=""):
    if os.path.exists(output_zip):
        os.remove(output_zip)
        print(f"Remove previous file: {output_zip}")
    
    try:
        # 创建压缩包
        patoolib.create_archive(
            archive=output_zip,
            filenames=[base_dir],  # 要压缩的目录
            verbosity=-1,         # 静默模式
            program='zip'         # 指定使用zip格式
        )
        print(f"Archive created successfully: {output_zip}")
    except Exception as e:
        raise ValueError(f"Archive creation failed: {str(e)}")


def demo_lam_audio2exp(infer, cfg):
    def core_fn(image_path: str, audio_params, working_dir, input_zip_textbox):

        if(os.path.exists(input_zip_textbox)):
            base_id = os.path.basename(input_zip_textbox).split(".")[0]
            output_dir = os.path.join('assets', 'sample_lam', base_id)
            # unzip_dir
            if (not os.path.exists(os.path.join(output_dir, 'arkitWithBSData'))):
                run_command = 'unzip -d '+output_dir+' '+input_zip_textbox
                os.system(run_command)
                rename_command = 'mv '+os.path.join(output_dir,base_id)+' '+os.path.join(output_dir,'arkitWithBSData')
                os.system(rename_command)
        else:
            base_id = os.path.basename(image_path).split(".")[0]

        # set input audio
        cfg.audio_input = audio_params
        cfg.save_json_path = os.path.join("./assets/sample_lam", base_id, 'arkitWithBSData', 'bsData.json')
        infer.infer()

        output_file_name = base_id+'_'+os.path.basename(audio_params).split(".")[0]+'.zip'
        assetPrefix = 'gradio_api/file=assets/'
        output_file_path = os.path.join('./assets',output_file_name)

        create_zip_archive(output_zip=output_file_path, base_dir=os.path.join("./assets/sample_lam", base_id))

        return 'gradio_api/file='+audio_params, assetPrefix+output_file_name

    with gr.Blocks(analytics_enabled=False) as demo:
        logo_url = './assets/images/logo.jpeg'
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(f"""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1> LAM-A2E: Audio to Expression</h1>
            </div>
            </div>
            """)

        gr.HTML(
            """<p><h4 style="color: blue;"> Notes: This project leverages audio input to generate ARKit blendshapes-driven facial expressions in ⚡real-time⚡, powering ultra-realistic 3D avatars generated by <a href="https://github.com/aigc3d/LAM">LAM</a>.</h4></p>"""
        )

        # DISPLAY
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id='lam_input_image'):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label='Input Image',
                                                   image_mode='RGB',
                                                   height=480,
                                                   width=270,
                                                   sources='upload',
                                                   type='filepath',  # 'numpy',
                                                   elem_id='content_image',
                                                   interactive=False)
                # EXAMPLES
                with gr.Row():
                    examples = [
                        ['assets/sample_input/barbara.jpg'],
                        ['assets/sample_input/status.png'],
                        ['assets/sample_input/james.png'],
                        ['assets/sample_input/vfhq_case1.png'],
                    ]
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image],
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Tabs(elem_id='lam_input_audio'):
                    with gr.TabItem('Input Audio'):
                        with gr.Row():
                            audio_input = gr.Audio(label='Input Audio',
                                                   type='filepath',
                                                   waveform_options={
                                                       'sample_rate': 16000,
                                                       'waveform_progress_color': '#4682b4'
                                                   },
                                                   elem_id='content_audio')

                examples = [
                    ['assets/sample_audio/Nangyanwen_chinese.wav'],
                    ['assets/sample_audio/LiBai_TTS_chinese.wav'],
                    ['assets/sample_audio/LinJing_TTS_chinese.wav'],
                    ['assets/sample_audio/BarackObama_english.wav'],
                    ['assets/sample_audio/HillaryClinton_english.wav'],
                    ['assets/sample_audio/XitongShi_japanese.wav'],
                    ['assets/sample_audio/FangXiao_japanese.wav'],
                ]
                gr.Examples(
                    examples=examples,
                    inputs=[audio_input],
                    examples_per_page=10,
                )

        # SETTING
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                input_zip_textbox = gr.Textbox(
                    label="Input Local Path to LAM-Generated ZIP File",
                    interactive=True,
                    placeholder="Input Local Path to LAM-Generated ZIP File",
                    visible=True
                )
                submit = gr.Button('Generate',
                                   elem_id='lam_generate',
                                   variant='primary')

        if h5_rendering:
            gr.set_static_paths(Path.cwd().absolute() / "assets/")
            with gr.Row():
                gs = gaussian_render(width=380, height=680)

        working_dir = gr.State()
        selected_audio = gr.Textbox(visible=False)
        selected_render_file = gr.Textbox(visible=False)

        submit.click(
            fn=assert_input_image,
            inputs=[input_image,input_zip_textbox],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=core_fn,
            inputs=[input_image, audio_input,
                    working_dir, input_zip_textbox],
            outputs=[selected_audio, selected_render_file],
            queue=False,
        ).success(
            fn=audio_loading,
            outputs=[selected_audio],
            js='''(output_component) => window.loadAudio(output_component)'''
        ).success(
            fn=do_render(),
            outputs=[selected_render_file],
            js='''(selected_render_file) => window.start(selected_render_file)'''
        )

        demo.queue()
        demo.launch(inbrowser=True)



def launch_gradio_app():
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME':'',
        'APP_INFER': 'configs/lam_audio2exp_streaming_config.py',
        'APP_TYPE': 'infer.audio2exp',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    args = default_argument_parser().parse_args()
    args.config_file = 'configs/lam_audio2exp_config_streaming.py'
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)

    cfg.ex_vol = True
    infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))

    demo_lam_audio2exp(infer, cfg)


if __name__ == '__main__':
    launch_gradio_app()
