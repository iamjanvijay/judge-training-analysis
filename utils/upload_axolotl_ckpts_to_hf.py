#!/usr/bin/env python3
"""
Upload Axolotl checkpoints to HuggingFace Hub.

This script uploads model checkpoints from local directories to HuggingFace Hub repositories.
It reads source-destination pairs from a JSONL file and handles the upload process with
appropriate file filtering.

Requirements:
    pip install -U huggingface_hub hf_transfer
"""

import os
import sys
from typing import List, Dict, Any
import json
import jsonlines
from huggingface_hub import create_repo, upload_folder

# Configuration Constants
HF_TOKEN = json.load(open("./utils/tokens.json"))["HF_TOKEN"]  # private token
CHECKPOINT_PREFIX = "checkpoint-"

# Files to ignore during upload (training artifacts and temporary files)
IGNORE_PATTERNS = [
    # Trainer state and runtime artifacts
    "trainer_state.json",
    "training_args.bin",
    "scheduler*.pt",
    "rng_state_*.pth",
    "global_step*",         # e.g. global_step1400
    "latest*",              # symlink/marker created by trainers
    "zero_to_fp32.py",
    "events.out.tfevents*", 
    "optimizer*",
    "state_*",
    "wandb/*",
    "*.log",
    "__pycache__/*",
    "*.tmp",
]

def setup_environment() -> None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_TOKEN"] = HF_TOKEN

def load_source_dest_pairs(jsonl_path: str) -> List[Dict[str, str]]:
    try:
        with jsonlines.open(jsonl_path) as reader:
            return [item for item in reader]
    except (FileNotFoundError, jsonlines.Error) as e:
        print(f"Error reading JSONL file: {e}")
        sys.exit(1)

def upload_checkpoint(ckpt_path: str, repo_path: str, ckpt_dir: str) -> None:
    print(f">>> Uploading checkpoint: {ckpt_path}")
    upload_folder(
        folder_path=ckpt_path,
        repo_id=repo_path,
        repo_type="model",
        path_in_repo=ckpt_dir,
        ignore_patterns=IGNORE_PATTERNS
    )

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python upload_axolotl_ckpts_to_hf.py <source_dest_pairs.jsonl>")
        sys.exit(1)

    # Setup environment
    setup_environment()
    
    # Load configurations
    source_dest_pairs = load_source_dest_pairs(sys.argv[1])
    
    # Process each source-destination pair
    for config in source_dest_pairs:
        ckpt_basedir = config["ckpt_basedir"]
        repo_path = config["repo_path"]
        
        try:
            # Get all checkpoint directories
            ckpt_dirs = [
                subpath for subpath in os.listdir(ckpt_basedir)
                if subpath.startswith(CHECKPOINT_PREFIX)
            ]
            
            if not ckpt_dirs:
                print(f"No checkpoint folders found in: {ckpt_basedir}")
                continue
                
            print(f"Found checkpoint folders: {ckpt_dirs}")
            
            # Create or verify repository exists
            create_repo(repo_path, private=True, exist_ok=True)
            
            # Upload each checkpoint
            for ckpt_dir in ckpt_dirs:
                ckpt_path = os.path.join(ckpt_basedir, ckpt_dir)
                upload_checkpoint(ckpt_path, repo_path, ckpt_dir)
                
        except Exception as e:
            print(f"Error processing {ckpt_basedir}: {e}")
            continue

if __name__ == "__main__":
    main()