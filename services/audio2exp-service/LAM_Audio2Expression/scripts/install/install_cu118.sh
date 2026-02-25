# install torch 2.1.2
# or conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# install dependencies
pip install -r requirements.txt

# install H5-render
pip install wheels/gradio_gaussian_render-0.0.3-py3-none-any.whl