from pathlib import Path

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
        print("2. Example Run: python utils/data/data_transfer/download_data_from_hf.py")
        print("\nExample:")
        print("   cd /shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis")
        print("   python utils/data/data_transfer/download_data_from_hf.py --subfolders data")
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