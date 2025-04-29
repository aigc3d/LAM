cd fbxsdk_linux
chmod +x fbx202034_fbxsdk_linux fbx202034_fbxpythonbindings_linux
mkdir -p ./python_binding ./python_binding/fbx_sdk
yes yes | ./fbx202034_fbxpythonbindings_linux ./python_binding
yes yes | ./fbx202034_fbxsdk_linux ./python_binding/fbx_sdk
cd ./python_binding
export FBXSDK_ROOT=./fbx_sdk
pip install .
cd -