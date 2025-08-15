# Judge-Training Analysis

## Usage

Upload Axolotl checkpoints to HuggingFace:
```bash
python utils/upload_axolotl_ckpts_to_hf.py configs/hf_uploads/upload_src_dest_pairs.jsonl
```

Download checkpoints from HuggingFace:
```bash
mkdir -p ckpts
python utils/download_axolotl_ckpts_from_hf.py configs/hf_uploads/upload_src_dest_pairs.jsonl ckpts
```