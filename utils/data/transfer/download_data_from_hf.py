from huggingface_hub import HfApi, hf_hub_download, list_repo_files
import os
import json
import argparse
from pathlib import Path
import time
import sys

sys.path.append("./utils") # to import the common.py file
from common import check_working_directory

def get_repo_files(repo_id: str, token: str) -> list:
    """Get all files from the Hugging Face dataset repository."""
    api = HfApi(token=token)
    
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        # Filter for data files and documentation
        target_files = [f for f in files if f.endswith(('.jsonl', '.parquet', '.md'))]
        return sorted(target_files)
    except Exception as e:
        print(f"❌ Error listing repository files: {str(e)}")
        raise


def filter_files_by_subfolders(repo_files: list, subfolders: list) -> list:
    """Filter repository files to only include specified subfolders."""
    if not subfolders:
        return repo_files
    
    filtered_files = []
    for file_path in repo_files:
        # Check if file is in any of the specified subfolders
        if any(file_path.startswith(subfolder) for subfolder in subfolders):
            filtered_files.append(file_path)
    
    return filtered_files


def download_single_file(file_path: str, repo_id: str, token: str, local_dir: Path) -> None:
    """Download a single file from Hugging Face repository."""
    # Create the local directory structure if it doesn't exist
    local_file_path = local_dir / file_path
    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading: {file_path} -> {local_file_path}")
    
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file_path,
            local_dir=str(local_dir),
            token=token
        )
        print(f"✅ Downloaded: {file_path}")
    except Exception as e:
        print(f"❌ Failed to download {file_path}: {str(e)}")
        raise


def download_files_sequentially(repo_id: str, token: str, local_dir: str, subfolders: list = None) -> None:
    """Download files one by one while maintaining folder structure."""
    local_dir_path = Path(local_dir)
    
    # Create local directory if it doesn't exist
    local_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of files from repository
    print(f"Fetching file list from hf://{repo_id}")
    repo_files = get_repo_files(repo_id, token)
    
    # Filter files by subfolders if specified
    if subfolders:
        repo_files = filter_files_by_subfolders(repo_files, subfolders)
        print(f"Filtering to subfolders: {', '.join(subfolders)}")
    
    if not repo_files:
        print("No files found matching the specified criteria.")
        return
    
    print(f"Found {len(repo_files)} files to download")
    print(f"Local directory: {local_dir_path}")
    
    # Download files sequentially
    for i, file_path in enumerate(repo_files, 1):
        print(f"\n[{i}/{len(repo_files)}] ", end="")
        try:
            download_single_file(file_path, repo_id, token, local_dir_path)
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Failed to download {file_path}: {str(e)}")
            raise
    
    print(f"\n🎉 Successfully downloaded {len(repo_files)} files!")


def verify_download(local_dir: str, repo_id: str, token: str, subfolders: list = None) -> None:
    """Verify that all files were downloaded correctly."""
    local_dir_path = Path(local_dir)
    repo_files = get_repo_files(repo_id, token)
    
    # Filter files by subfolders if specified
    if subfolders:
        repo_files = filter_files_by_subfolders(repo_files, subfolders)
    
    print("\n🔍 Verifying download...")
    
    missing_files = []
    for file_path in repo_files:
        local_file = local_dir_path / file_path
        if not local_file.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    else:
        print("✅ All files downloaded successfully!")
    
    # Show local file count
    local_files = list(local_dir_path.rglob("*.jsonl")) + list(local_dir_path.rglob("*.parquet")) + list(local_dir_path.rglob("*.md"))
    print(f"📁 Local files: {len(local_files)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download sf-judge-data from Hugging Face with optional subfolder filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download only formatted_data_splits (default)
  python download_data_from_hf.py
  
  # Download specific subfolders
  python download_data_from_hf.py --subfolders data data_splits
  
  # Download all subfolders
  python download_data_from_hf.py --subfolders data data_splits formatted_data_splits
  
  # Download only data subfolder
  python download_data_from_hf.py --subfolders data
        """
    )
    
    parser.add_argument(
        '--subfolders',
        nargs='+',
        choices=['data', 'data_splits', 'formatted_data_splits'],
        default=['formatted_data_splits'],
        help='Subfolders to download (default: formatted_data_splits)'
    )
    
    return parser.parse_args()


def check_working_directory():
    """Check if the script is being run from inside the judge-training-analysis directory."""
    current_dir = Path.cwd()
    
    # Check if we're inside the judge-training-analysis directory
    if current_dir.name != "judge-training-analysis":
        print("❌ Error: Script must be run from inside the 'judge-training-analysis' directory")
        print(f"Current directory: {current_dir}")
        print(f"Current directory name: {current_dir.name}")
        print("\n💡 To fix this:")
        print("1. Navigate to the judge-training-analysis directory")
        print("2. Run: python utils/data/data_transfer/download_data_from_hf.py")
        print("\nExample:")
        print("   cd /shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis")
        print("   python utils/data/data_transfer/download_data_from_hf.py")
        return False
    
    # Check if the required files exist
    tokens_file = current_dir / "utils" / "tokens.json"
    if not tokens_file.exists():
        print("❌ Error: tokens.json file not found")
        print(f"Expected location: {tokens_file}")
        return False
    
    print(f"✅ Working directory check passed")
    print(f"📁 Running from: {current_dir}")
    return True


def main():
    """Main function to download sf-judge-data folder from Hugging Face."""
    # Check working directory first
    if not check_working_directory():
        exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configuration
    HF_TOKEN = json.load(open("./utils/tokens.json"))["HF_TOKEN"]
    REPO_ID = "iamjanvijay/sf-judge-data"
    LOCAL_DIR = "./sf-judge-data"
    
    try:
        print(f"🚀 Starting download from hf://{REPO_ID}")
        print(f"📁 Target directory: {LOCAL_DIR}")
        print(f"📂 Selected subfolders: {', '.join(args.subfolders)}")
        
        # Download files sequentially
        download_files_sequentially(REPO_ID, HF_TOKEN, LOCAL_DIR, args.subfolders)
        
        # Verify download
        verify_download(LOCAL_DIR, REPO_ID, HF_TOKEN, args.subfolders)
        
        print("✅ Download completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during download: {str(e)}")
        raise


if __name__ == "__main__":
    main()
