# Example usage:
# python eval/run_batched_eval.py \
#   --configs configs/evaluations/set_2/dpo/llama8b.jsonl \
#   --gpu_ids 0 1 2 3 \
#   --max_tokens 4096 \
#   --temperature 0.0 \
#   --top_p 1.0 \
#   --gpu_mem_util 0.9 \
#   --dry_run

import argparse, json, os, pathlib, subprocess, shlex
import multiprocessing
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_eval(model, input_files, output_files, gpu_id, extra_args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "python", "./eval/run_eval.py",
        "--model", model,
        "--inputs"
    ] + input_files + [
        "--outputs"
    ] + output_files + [
        "--max_tokens", str(extra_args.max_tokens),
        "--temperature", str(extra_args.temperature),
        "--top_p", str(extra_args.top_p),
        "--gpu_mem_util", str(extra_args.gpu_mem_util),
    ]
    if extra_args.trust_remote_code:
        cmd.append("--trust_remote_code")
    print(f"[GPU{gpu_id}] {shlex.join(cmd)}")
    if not extra_args.dry_run:
        subprocess.run(cmd, check=True, env=env)
    else:
        print(f"[GPU{gpu_id}] DRY RUN - Command not executed")

def get_model_path_and_input_output_pairs(config):
    all_tasks = []

    model_path = config["model_path"]
    is_local_model = os.path.exists(model_path)

    model_paths = []
    if not is_local_model: # if huggingface model -> this is the only model path
        model_paths.append(model_path)
    else: # if local model -> this is the base folder with all checkpoint-xxxx folders
        if os.path.isdir(model_path):
            for entry in os.listdir(model_path):
                full_path = os.path.join(model_path, entry)
                if (
                    os.path.isdir(full_path)
                    and entry.startswith("checkpoint-")
                    and entry[len("checkpoint-"):].isdigit()
                    and int(entry[len("checkpoint-"):]) > 0
                ):
                    model_paths.append(full_path)
    
    input_output_pairs = config["input_output_pairs"]
    if not is_local_model:
        for model_path in model_paths:
            input_output_pairs_copy = deepcopy(input_output_pairs)
            all_tasks.append((model_path, input_output_pairs_copy))
    else:
        for model_path in model_paths:
            checkpoint_name = os.path.basename(model_path)
            input_output_pairs_copy = deepcopy(input_output_pairs)
            for idx in range(len(input_output_pairs_copy)):
                output_path = input_output_pairs_copy[idx][1]
                output_folder = os.path.dirname(output_path)
                output_filename = os.path.basename(output_path)
                output_filename_splitted = output_filename.split(".")
                assert output_filename_splitted[1].startswith("train_")
                output_filename_splitted[1] = output_filename_splitted[1] + f"_{checkpoint_name}"
                output_path = os.path.join(output_folder, ".".join(output_filename_splitted))
                input_output_pairs_copy[idx][1] = output_path
            all_tasks.append((model_path, input_output_pairs_copy))
    return all_tasks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True, nargs="+", help="Paths to config JSONL files")
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU IDs to use")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gpu_mem_util", type=float, default=0.9)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    args = p.parse_args()

    # Read all config files
    configs = []
    for config_file in args.configs:
        configs.extend(list(read_jsonl(config_file)))
    
    # Create output directories
    for config in configs:
        for input_file, output_file in config["input_output_pairs"]:
            pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    all_tasks = []
    for config in configs:
        model_path_and_input_output_pairs = get_model_path_and_input_output_pairs(config)
        all_tasks.extend(model_path_and_input_output_pairs)

    total_tasks = len(all_tasks)
    # Run evaluations sequentially by GPU
    # Each GPU gets one task at a time, and only gets a new task after completing the previous one
    print(f"Starting evaluation of {total_tasks} tasks across {len(args.gpu_ids)} GPUs")
    print(f"GPU assignment order: {args.gpu_ids}")
    
    # Distribute tasks evenly across GPUs in round-robin fashion
    gpu_tasks = {gpu_id: [] for gpu_id in args.gpu_ids}
    for i, task in enumerate(all_tasks):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        gpu_tasks[gpu_id].append(task)
    
    # Print task distribution
    for gpu_id, tasks in gpu_tasks.items():
        print(f"GPU {gpu_id} will process {len(tasks)} tasks")
    
    # Execute tasks sequentially per GPU using ThreadPoolExecutor
    # This ensures each GPU processes one task at a time
    with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
        # Submit initial tasks for each GPU
        futures = {}
        for gpu_id, tasks in gpu_tasks.items():
            if tasks:
                future = executor.submit(run_eval, tasks[0][0], [pair[0] for pair in tasks[0][1]], [pair[1] for pair in tasks[0][1]], gpu_id, args)
                futures[gpu_id] = (future, tasks[1:])  # Keep remaining tasks
        
        # Process remaining tasks as GPUs complete their current task
        while futures:
            # Check which futures are done
            for gpu_id, (future, remaining_tasks) in list(futures.items()):
                if future.done():
                    # Remove completed future
                    del futures[gpu_id]
                    
                    # Submit next task for this GPU if available
                    if remaining_tasks:
                        next_task = remaining_tasks[0]
                        next_future = executor.submit(run_eval, next_task[0], [pair[0] for pair in next_task[1]], [pair[1] for pair in next_task[1]], gpu_id, args)
                        futures[gpu_id] = (next_future, remaining_tasks[1:])
            
            # Small delay to prevent busy waiting
            time.sleep(0.1)
    
    print("All evaluations completed!")

if __name__ == "__main__":
    main()
