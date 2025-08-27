# setting up the wandb
export WANDB_API_KEY=f39df9c1230a9046b1ce65b5f9407de8e9cd590b # if you log online
export WANDB_PROJECT=judge-generalisation
export WANDB_ENTITY=ksartik-georgia-institute-of-technology
export HF_TOKEN=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

# setting up the gpu devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_DIR=/shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis

# running the training script
python3 -m verl.trainer.main_ppo \
        --config-path $BASE_DIR/configs/trainings/set_1/grpo \
        --config-name set_1_unflipped_weak.ministral8b.deepscaler_const_1e-6.yaml \
        base_dir=$BASE_DIR  

