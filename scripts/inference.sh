# step1. set TRAIN_CONFIG path to config file

TRAIN_CONFIG="configs/inference/lam-20k-8gpu.yaml"
MODEL_NAME="exps/releases/lam/lam-20k/step_045500/"
IMAGE_INPUT="assets/sample_input/status.png"
MOTION_SEQS_DIR="assets/sample_motion/export/Look_In_My_Eyes/"


TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"


MOTION_IMG_DIR=null
SAVE_PLY=false
SAVE_IMG=false
VIS_MOTION=false
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=true
CROSS_ID=false
TEST_SAMPLE=false
GAGA_TRACK_TYPE=""

device=0
nodes=0

export PYTHONPATH=$PYTHONPATH:$pwd


CUDA_VISIBLE_DEVICES=$device python -m lam.launch infer.lam --config $TRAIN_CONFIG \
        model_name=$MODEL_NAME image_input=$IMAGE_INPUT \
        export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
        save_ply=$SAVE_PLY save_img=$SAVE_IMG \
        gaga_track_type=$GAGA_TRACK_TYPE cross_id=$CROSS_ID \
        test_sample=$TEST_SAMPLE rank=$device nodes=$nodes
