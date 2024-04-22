#!/bin/bash
USER=$(whoami)

docker run -it --net=host \
    --gpus all \
    -v $(pwd):/home/voxblox_ws/src/voxblox \
    -v /home/$USER/.ssh:/root/.ssh \
    voxblox