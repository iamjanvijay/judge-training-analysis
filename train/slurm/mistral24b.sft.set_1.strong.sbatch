#!/bin/bash
#SBATCH --job-name=mistral24b-strong-sft
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low          # use -low + --account=low-pri if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/fsx/home/janvijay.singh
#SBATCH --output=/fsx/home/janvijay.singh/slurm_outputs/%j/log.out
#SBATCH --error=/fsx/home/janvijay.singh/slurm_outputs/%j/err.out

set -euo pipefail
mkdir -p /fsx/home/janvijay.singh/slurm_outputs/${SLURM_JOB_ID}

# your requested exports (consider rotating later)
export HF_TOKEN='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'
export WANDB_API_KEY='f39df9c1230a9046b1ce65b5f9407de8e9cd590b'

IMAGE="docker.io/iamjanvijay/axolotl:mod-sigmoid-sft-dpo-loss-v2"
HOST_MNT="/fsx/home/janvijay.singh"
CONT_MNT="/fsx/home/janvijay.singh"
CONFIG_YAML="/fsx/home/janvijay.singh/latest/axolotl_configs_latest/set_1/sft/set_1_strong.mistral24b.deepscaler_cosine_2.5e-6.yaml"
CONTAINER="axolotl-${SLURM_JOB_ID}"
MAIN_PORT=29500

cleanup(){ docker stop "$CONTAINER" >/dev/null 2>&1 || true; docker rm "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"

docker pull "$IMAGE" || true

docker run \
  --name "$CONTAINER" \
  --privileged \
  --gpus "all" \
  --shm-size 10g \
  --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  -e NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NCCL_DEBUG=WARN \
  -v "${HOST_MNT}:${CONT_MNT}" \
  -w "${CONT_MNT}" \
  "$IMAGE" bash -lc "
    set -euo pipefail
    accelerate launch \
      --multi-gpu \
      --num-machines 1 \
      --num-processes 8 \
      --main_process_port ${MAIN_PORT} \
      --same-network \
      --rdzv_backend static \
      --mixed_precision bf16 \
      -m axolotl.cli.train ${CONFIG_YAML}
  "
