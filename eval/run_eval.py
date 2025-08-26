import argparse, json, os, pathlib, sys
from vllm import LLM, SamplingParams
from json_repair import repair_json

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_responses_from_prompts(args, llm, prompts):
    sampling_params = SamplingParams(temperature=args.temperature, n=args.n, max_tokens=args.max_tokens, top_p=args.top_p)
    chat_prompts = [llm.get_tokenizer().apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
    outputs = llm.generate(chat_prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

# get_label: this is the older version of the get_label function.
# def get_label(response):
#     # Take the last 50 characters of the response
#     text = response[-50:]
#     # If a code block with json is present, extract from there
#     if '```json' in text:
#         text = '```json' + text.split('```json', 1)[1]
#     # Attempt to repair and parse the JSON
#     label_dict = repair_json(text, return_objects=True)

#     # If the result is an empty list, return 'C'
#     if isinstance(label_dict, list) and not label_dict:
#         return 'C'

#     # If the result is a list, take the last element
#     if isinstance(label_dict, list):
#         label_dict = label_dict[-1]

#     # If the result is not a dict with a valid verdict, return 'C'
#     if not (
#         isinstance(label_dict, dict)
#         and "verdict" in label_dict
#         and label_dict["verdict"] in ['A', 'B']
#     ):
#         return 'C'

#     # Otherwise, return the predicted label
#     return label_dict["verdict"]

def check_text_for_verdict_a(text):
    """Check if text contains indicators for verdict A."""
    return (
        '{"verdict": "A"}' in text
        or '{"verdict": "Response A"}' in text
        or "Response A is better" in " ".join(text.replace("*", "").split())
        or "Response A is the better" in " ".join(text.replace("*", "").split())
        or "Response A is than better" in " ".join(text.replace("*", "").split())
    )

def check_text_for_verdict_b(text):
    """Check if text contains indicators for verdict B."""
    return (
        '{"verdict": "B"}' in text
        or '{"verdict": "Response B"}' in text
        or "Response B is better" in " ".join(text.replace("*", "").split())
        or "Response B is the better" in " ".join(text.replace("*", "").split())
        or "Response B is than better" in " ".join(text.replace("*", "").split())
    )

def extract_json_from_codeblock(text):
    """Extract JSON content from code block if present."""
    if '```json' in text:
        text = '```json' + text.split('```json', 1)[1]
    return text

def process_label_dict(label_dict):
    """Process label dictionary to extract valid verdict."""
    # If the result is an empty list, return 'C'
    if isinstance(label_dict, list) and not label_dict:
        return 'C'

    # If the result is a list, take the last element
    if isinstance(label_dict, list):
        for item in label_dict:
            if isinstance(item, dict) and "verdict" in item and item["verdict"] in ['A', 'Response A', 'B', 'Response B']:
                label_dict = item
                break
        else:
            label_dict = label_dict[-1]

    return label_dict

def validate_verdict_dict(label_dict):
    """Validate if label_dict contains a valid verdict."""
    return (
        isinstance(label_dict, dict)
        and "verdict" in label_dict
        and label_dict["verdict"] in ['A', 'Response A', 'B', 'Response B']
    )

def extract_verdict_from_dict(label_dict):
    """Extract the final verdict from validated label_dict."""
    if label_dict["verdict"].strip() == "Response A":
        return "A"
    elif label_dict["verdict"].strip() == "Response B":
        return "B"
    else:
        return label_dict["verdict"]

def get_label(response):
    text = response
    # If a code block with json is present, extract from there
    text = extract_json_from_codeblock(text)

    # Attempt to repair and parse the JSON
    label_dict = repair_json(text, return_objects=True)

    # Process the label dictionary
    label_dict = process_label_dict(label_dict)

    # If the result is not a dict with a valid verdict, check for strings in text that indicate the verdict, otherwise return 'C'
    if not validate_verdict_dict(label_dict):
        if check_text_for_verdict_a(text):
            return 'A'
        elif check_text_for_verdict_b(text):
            return 'B'
        else:
            return 'C'

    # Otherwise, return the predicted label
    return extract_verdict_from_dict(label_dict)

def process_single_eval(args, llm, input_file, output_file):
    """Process a single evaluation file"""
    
    # Check if output file already exists and skip if overwrite is False
    if not args.overwrite and os.path.exists(output_file):
        print(f"Skipping {input_file} -> {output_file} (output already exists)")
        return None
    
    # read prompts and labels from the input jsonl file.
    prompts, labels = [], []
    for ex in read_jsonl(input_file):
        prompt = ex.get("prompt")
        if prompt is None:
            raise ValueError(f"Input JSONL {input_file} must contain 'prompt'.")
        prompts.append(prompt)

        label = ex.get("label")
        if label is None:
            raise ValueError(f"Input JSONL {input_file} must contain 'label'.")
        labels.append(label)

    responses = get_responses_from_prompts(args, llm, prompts)

    jsonl_output = []
    for prompt, label, response in zip(prompts, labels, responses):
        response_label = get_label(response)
        jsonl_output.append({
            "prompt": prompt,
            "label": label,
            "response": response,
            "response_label": response_label,
        })

    correct, incorrect, incorrect_format = 0, 0, 0
    for jsonl_row in jsonl_output:
        if jsonl_row["response_label"] == jsonl_row["label"]:
            correct += 1
        else:
            incorrect += 1
            if jsonl_row["response_label"] == 'C':
                incorrect_format += 1

    metrics_output = {
        "correct": correct,
        "incorrect": incorrect,
        "incorrect_format": incorrect_format,
        "accuracy": correct / (correct + incorrect),
        "incorrect_format_rate": incorrect_format / (correct + incorrect),
        "input_file": input_file,
        "output_file": output_file,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": args.n,
        "tp_size": args.tp_size,
        "gpu_mem_util": args.gpu_mem_util,
        "max_tokens": args.max_tokens
    }

    if input_file.endswith("pairs_flip.jsonl"):
        assert len(jsonl_output) % 2 == 0
        consistent_correct, consistent_incorrect = 0, 0
        for i in range(0, len(jsonl_output), 2):
            if (jsonl_output[i]["response_label"] == jsonl_output[i]["label"]) and \
                (jsonl_output[i+1]["response_label"] == jsonl_output[i+1]["label"]):
                consistent_correct += 1
            else:
                consistent_incorrect += 1

        metrics_output["consistent_correct"] = consistent_correct
        metrics_output["consistent_incorrect"] = consistent_incorrect
        metrics_output["consistent_accuracy"] = consistent_correct / (consistent_correct + consistent_incorrect)

    write_jsonl(pathlib.Path(output_file), jsonl_output)
    with open(pathlib.Path(output_file).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=4)
    
    return metrics_output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path or HF id of the checkpoint")
    p.add_argument("--inputs", required=True, nargs="+", help="List of input JSONL files. Expected keys: {'prompt': ...}")
    p.add_argument("--outputs", required=True, nargs="+", help="List of output JSONL files")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--gpu_mem_util", type=float, default=0.9)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files. If False, skip files that already exist.")
    args = p.parse_args()

    # Validate that inputs and outputs lists have the same length
    if len(args.inputs) != len(args.outputs):
        raise ValueError("Number of input files must match number of output files")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        max_model_len=10000,
        enable_prefix_caching=True
    )

    # Process each input/output pair
    for input_file, output_file in zip(args.inputs, args.outputs):
        print(f"Processing {input_file} -> {output_file}")
        metrics = process_single_eval(args, llm, input_file, output_file)
        
        if metrics is None: # File was skipped (already exists)
            continue

        if "consistent_accuracy" in metrics:
            print(f"Completed {input_file}: accuracy={metrics['accuracy']:.4f}, consistent_accuracy={metrics['consistent_accuracy']:.4f}, unparsed={metrics['incorrect_format_rate']:.4f}")
        else:
            print(f"Completed {input_file}: accuracy={metrics['accuracy']:.4f}, unparsed={metrics['incorrect_format_rate']:.4f}")

if __name__ == "__main__":
    main()