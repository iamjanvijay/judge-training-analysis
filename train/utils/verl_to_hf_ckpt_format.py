# python convert_fsdp_to_hf.py /jvsingh2/sf-intern/grpo-setup/train/checkpoints/judge-grpo/qwen2.5-1.5b.math500.grpo/global_step_440/actor /jvsingh2/sf-intern/grpo-setup/train/checkpoints/judge-grpo/qwen2.5-1.5b.math500.grpo/global_step_440/actor/huggingface /jvsingh2/sf-intern/grpo-setup/train/checkpoints/judge-grpo/qwen2.5-1.5b.math500.grpo/converted_ckpts/global_step_440

# Sample command to run this script:
# CUDA_VISIBLE_DEVICES=-1 python train/utils/verl_to_hf_ckpt_format.py \
#     --verl_ckpt_dir /shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis/train-output-dir/checkpoints/judge-generalisation/set_1_unflipped_weak.ministral8b.deepscaler_const_1e-6 \
#     --hf_out_dir /shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis/ckpts/grpo.set_1_weak.ministral8b.deepscaler \
#     --world_size 8

"""
Convert VERL FSDP checkpoints to HuggingFace format.

This script converts distributed FSDP checkpoints from VERL training runs
into standard HuggingFace model checkpoints that can be easily loaded
and used for inference or further fine-tuning.
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from collections import defaultdict
import argparse
import os
import json


def convert_verl_to_hf(fsdp_checkpoint_path, huggingface_model_path, output_path, world_size):
    """
    Convert a single VERL FSDP checkpoint to HuggingFace format.
    
    Args:
        fsdp_checkpoint_path (str): Path to the FSDP checkpoint directory containing model shards
        huggingface_model_path (str): Path to the base HuggingFace model for config and tokenizer
        output_path (str): Directory where the converted checkpoint will be saved
    """
    # Dictionary to collect state dict tensors from all ranks
    state_dict = defaultdict(list)

    # Process all 8 distributed model shards
    for rank in range(world_size):
        # Construct path to this rank's checkpoint file
        checkpoint_filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print(f'Loading checkpoint from rank {rank}: {checkpoint_filepath}')
        
        # Load the checkpoint for this rank (weights_only=False for DTensors)
        this_rank_state_dict = torch.load(checkpoint_filepath, weights_only=False)
        
        # Extract tensors and convert DTensors to local tensors
        for key, tensor_value in this_rank_state_dict.items():
            state_dict[key].append(tensor_value.to_local())

    # Concatenate all rank tensors along dimension 0 to reconstruct full model
    print("Reconstructing full model from distributed shards...")
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    # Load the base model configuration and create model instance
    print(f"Loading base model configuration from: {huggingface_model_path}")
    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    
    # Load the reconstructed state dict into the model
    print("Loading reconstructed weights into model...")
    model.load_state_dict(state_dict)
    
    # Save the converted model in HuggingFace format
    print(f"Saving converted model to: {output_path}")
    model.save_pretrained(output_path, max_shard_size="10GB")

    # Copy the tokenizer from the base model
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Successfully converted checkpoint to: {output_path}")


def fetch_args():
    """
    Parse command line arguments for the checkpoint conversion script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Convert VERL FSDP checkpoint to HuggingFace format.")
    parser.add_argument(
        "--verl_ckpt_dir",
        type=str,
        required=True,
        help="Path to the VERL FSDP checkpoint directory (e.g., .../actor). "
             "This is the path that was specified in default_local_dir in the yaml file for the VERL training."
    )
    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="World size of the VERL training. This is used to determine the number of model shards to load."
    )
    parser.add_argument(
        "--hf_out_dir",
        type=str,
        required=True,
        help="Path to save the converted HuggingFace checkpoint"
    )
    args = parser.parse_args()
    return args


def get_params_for_convert_verl_to_hf(args):
    """
    Process all checkpoint directories and convert them to HuggingFace format.
    
    This function:
    1. Finds all global_step_* directories in the checkpoint parent directory
    2. Generates conversion parameters for each checkpoint
    3. Executes the conversion for each checkpoint
    
    Args:
        args: Command line arguments containing verl_ckpt_dir and hf_out_dir
    """
    verl_ckpt_parent_dir, hf_out_dir, world_size = args.verl_ckpt_dir, args.hf_out_dir, args.world_size
    
    # Find all checkpoint directories that follow the global_step_* pattern
    verl_ckpt_dirs = [
        d for d in os.listdir(verl_ckpt_parent_dir) 
        if os.path.isdir(os.path.join(verl_ckpt_parent_dir, d)) and "global_step_" in d
    ]
    
    print(f"Found {len(verl_ckpt_dirs)} VERL checkpoint directories in {verl_ckpt_parent_dir}")
    print(f"Checkpoint directories: {verl_ckpt_dirs}")

    # Generate conversion parameters for each checkpoint
    convert_verl_to_hf_params = []
    for verl_ckpt_dir in verl_ckpt_dirs:
        # Extract checkpoint step number from directory name
        ckpt_steps = int(verl_ckpt_dir[len("global_step_"):])
        
        # Construct paths for this checkpoint
        fsdp_ckpt_path = os.path.join(verl_ckpt_parent_dir, verl_ckpt_dir, "actor")
        hf_model_path = os.path.join(fsdp_ckpt_path, "huggingface")
        output_path = os.path.join(hf_out_dir, f"checkpoint-{ckpt_steps}")
        
        # Add conversion parameters for this checkpoint
        convert_verl_to_hf_params.append({
            "fsdp_checkpoint_path": fsdp_ckpt_path, 
            "huggingface_model_path": hf_model_path, 
            "output_path": output_path,
            "world_size": world_size
        })
    
    # Display all conversion operations that will be performed
    print("Converting these checkpoints:")
    print(json.dumps(convert_verl_to_hf_params, indent=4))

    # Execute conversion for each checkpoint
    for param in convert_verl_to_hf_params:
        print(f"\n{'='*60}")
        print(f"Converting checkpoint: {param['output_path']}")
        print(f"{'='*60}")
        convert_verl_to_hf(**param)


if __name__ == "__main__":
    # Parse command line arguments and execute checkpoint conversion
    args = fetch_args()
    get_params_for_convert_verl_to_hf(args)



