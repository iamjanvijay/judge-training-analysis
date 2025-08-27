#!/usr/bin/env python3
"""
Download Axolotl checkpoints from HuggingFace Hub.

This script downloads model checkpoints from HuggingFace Hub repositories to local directories.
It reads source-destination pairs from a JSONL file and handles the download process.

Requirements:
    pip install -U huggingface_hub hf_transfer
"""

import os
import sys
from typing import List, Dict
import json
import jsonlines
from huggingface_hub import snapshot_download

# Add utils to path for importing common functions
sys.path.append("./utils")
from common import check_working_directory

# Configuration Constants
HF_TOKEN = json.load(open("./utils/tokens.json"))["HF_TOKEN"]
CHECKPOINT_PREFIX = "checkpoint-"

def setup_environment() -> None:
    """Set up environment variables for HF transfer."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_TOKEN"] = HF_TOKEN

def load_source_dest_pairs(jsonl_path: str) -> List[Dict[str, str]]:
    """Load repository and local directory pairs from JSONL file."""
    try:
        with jsonlines.open(jsonl_path) as reader:
            return [item for item in reader]
    except (FileNotFoundError, jsonlines.Error) as e:
        print(f"Error reading JSONL file: {e}")
        sys.exit(1)

def download_checkpoint(repo_path: str, local_dir: str) -> None:
    print(f">>> Downloading from {repo_path} to {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_path,
            local_dir=local_dir,
            token=HF_TOKEN,
            repo_type="model"
        )
    except Exception as e:
        print(f"Error downloading from {repo_path}: {e}")

def main() -> None:
    # Verify working directory
    if not check_working_directory():
        exit(1)

    if len(sys.argv) != 3:
        print("Usage: python download_axolotl_ckpts_from_hf.py <source_dest_pairs.jsonl> <local_dir>")
        sys.exit(1)

    # Setup environment
    setup_environment()
    
    # Load configurations
    source_dest_pairs = load_source_dest_pairs(sys.argv[1])
    local_dir = sys.argv[2]
    
    # Process each source-destination pair
    for config in source_dest_pairs:
        repo_path = config["repo_path"]
        cur_local_dir = os.path.join(local_dir, repo_path.split("/")[-1])
        
        try:
            # Create local directory if it doesn't exist
            os.makedirs(cur_local_dir, exist_ok=True)
            
            # Download checkpoint
            download_checkpoint(repo_path, cur_local_dir)
                
        except Exception as e:
            print(f"Error processing {repo_path}: {e}")
            continue

if __name__ == "__main__":
    main()
