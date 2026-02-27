  ACC_CONFIG="./configs/accelerate-train-4gpu.yaml"
  TRAIN_CONFIG="./configs/train-sample-human.yaml"

  if [ -n "$1" ]; then
    TRAIN_CONFIG=$1
  else
    TRAIN_CONFIG="./configs/train-sample-human.yaml"
  fi

  if [ -n "$2" ]; then
    MAIN_PORT=$2
  else
    MAIN_PORT=12345
  fi

  accelerate launch --config_file $ACC_CONFIG  --main_process_port=$MAIN_PORT   -m openlrm.launch train.human_lrm --config $TRAIN_CONFIG