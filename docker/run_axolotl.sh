#!/bin/bash

# Directory to mount inside container
MOUNT_DIR="/fsx/home/janvijay.singh"

# Run Docker container with GPU support and resource configurations
docker run \
    --privileged \
    --gpus '"all"' \
    --shm-size 10g \
    --rm -d -it \
    --name axolotl \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $MOUNT_DIR:$MOUNT_DIR \
    docker.io/iamjanvijay/axolotl:mod-sigmoid-sft-dpo-loss-v2