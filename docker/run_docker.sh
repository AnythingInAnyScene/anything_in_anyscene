export CURRENT_DIR=${PWD}
# export XDG_RUNTIME_DIR=/tmp
# xhost +local:docker 
sudo docker run --rm -it -v ${CURRENT_DIR}/3d_models:/anything_in_anyscene/3d_models \
  -v ${CURRENT_DIR}/models:/anything_in_anyscene/models \
  -v ${CURRENT_DIR}/data:/anything_in_anyscene/data \
  -v ${CURRENT_DIR}:/all \
  -w /all \
  --runtime=nvidia \
  --gpus=all --net=host --ipc=host -e DISPLAY=$DISPLAY anything_in_anyscene:all_in_base