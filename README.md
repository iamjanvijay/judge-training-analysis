# Judge-Training Analysis

## Usage

Upload data to HuggingFace:
```bash
python utils/data/transfer/upload_data_to_hf.py
```

Download data from HuggingFace:
```bash
python utils/data/transfer/download_data_from_hf.py [--subfolders data data_splits formatted_data_splits]
```

Generate formatted_data_splits from data_splits:
```bash
python utils/data/format/train_data.py
python utils/data/format/eval_data.py
```

Download Axolotl checkpoints from HuggingFace:
```bash
python utils/ckpt_transfer/download_axolotl_ckpts_from_hf.py <source_dest_pairs.jsonl> <local_dir>
```

Upload Axolotl checkpoints to HuggingFace:
```bash
python utils/ckpt_transfer/upload_axolotl_ckpts_to_hf.py <source_dest_pairs.jsonl>
```

Compute overlap statistics for formatted_data_splits and data_splits subfolders:
```bash
python utils/data/stats/compute_overlap_stats.py
```