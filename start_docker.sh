#!/bin/bash
USER=$(whoami)

docker run -it --net=host \
    --gpus all \
    -v $(pwd):/home/voxblox_ws/src/voxblox \
    -v $CODA_ROOT:/home/data/coda \
    -v /home/$USER/.ssh:/root/.ssh \
    voxblox