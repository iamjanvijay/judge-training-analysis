from huggingface_hub import HfApi, create_repo, upload_file
import os
import json
from pathlib import Path
import sys

sys.path.append("./utils") # to import the common.py file
from common import check_working_directory

def create_hf_repo(repo_id: str, token: str) -> None:
    """Create a Hugging Face dataset repository."""
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        private=True,
        exist_ok=True
    )


def get_target_files(folder_path: str) -> list:
    """Get all .jsonl and .parquet files from the folder and subfolders."""
    target_files = []
    folder = Path(folder_path)
    
    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.jsonl', '.parquet', '.md']:
            target_files.append(file_path)
    
    return sorted(target_files)


def get_relative_path(file_path: Path, base_folder: Path) -> str:
    """Get the relative path for the file in the repository."""
    return str(file_path.relative_to(base_folder))


def upload_single_file(file_path: Path, repo_id: str, token: str, base_folder: Path) -> None:
    """Upload a single file to Hugging Face repository."""
    relative_path = get_relative_path(file_path, base_folder)
    
    print(f"Uploading: {file_path.name} -> {relative_path}")
    
    upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=relative_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    print(f"✅ Uploaded: {file_path.name}")


def upload_files_sequentially(folder_path: str, repo_id: str, token: str) -> None:
    """Upload files one by one while maintaining folder structure."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    base_folder = Path(folder_path)
    target_files = get_target_files(folder_path)
    
    if not target_files:
        print("No .jsonl or .parquet files found in the folder.")
        return
    
    print(f"Found {len(target_files)} files to upload")
    print(f"Base folder: {base_folder}")
    
    for i, file_path in enumerate(target_files, 1):
        print(f"\n[{i}/{len(target_files)}] ", end="")
        try:
            upload_single_file(file_path, repo_id, token, base_folder)
        except Exception as e:
            print(f"❌ Failed to upload {file_path.name}: {str(e)}")
            raise
    
    print(f"\n🎉 Successfully uploaded {len(target_files)} files!")


def main():
    """Main function to upload sf-judge-data folder to Hugging Face."""
    # Check working directory first
    if not check_working_directory():
        exit(1)
    
    # Configuration
    HF_TOKEN = json.load(open("./utils/tokens.json"))["HF_TOKEN"]
    REPO_ID = "iamjanvijay/sf-judge-data"
    LOCAL_FOLDER = "./sf-judge-data"
    
    try:
        # Initialize Hugging Face API
        api = HfApi(token=HF_TOKEN)
        
        # Create repository
        print(f"Creating/ensuring repository: {REPO_ID}")
        create_hf_repo(REPO_ID, HF_TOKEN)
        
        # Upload files sequentially
        upload_files_sequentially(LOCAL_FOLDER, REPO_ID, HF_TOKEN)
        
        print("✅ All uploads completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        raise


if __name__ == "__main__":
    main()