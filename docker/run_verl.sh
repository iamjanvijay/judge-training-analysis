#!/bin/bash

# Directory to mount inside container
MOUNT_DIR="/shared/storage-01/users/jvsingh2"

# Run Docker container with GPU support and resource configurations
docker run \
    --privileged \
    --gpus '"all"' \
    --shm-size 10g \
    --rm -d -it \
    --name verl \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $MOUNT_DIR:$MOUNT_DIR \
    docker.io/iamjanvijay/verl:mod-vllm-fsdp-with-verl-installed-v2