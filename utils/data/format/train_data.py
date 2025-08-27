import os
from collections import defaultdict
import jsonlines
import random
from datasets import Dataset
import argparse

def get_filenames(args, dataset_path):
    answers_type = 'answers_one' if args.include_two_response_pairs == 'one_res' else 'answers_two'
    dataset_files = [fname for fname in os.listdir(dataset_path) if fname.endswith('.jsonl') and answers_type in fname and args.set_name in fname]
    return dataset_files

def create_dataset_dict(dataset_files, dataset_path):
    dataset_dict = {}
    all_splits, all_model_strengths, all_set_names = set(), set(), set()
    for filename in dataset_files:
            # reading the filename into parts.
            filepath = os.path.join(dataset_path, filename)
            filename_split = filename[:-6].split('_')
            set_name = '_'.join(filename_split[:2])
            model_strength = filename_split[2]
            split_name = '_'.join(filename_split[3:])

            all_splits.add(split_name)
            all_model_strengths.add(model_strength)
            all_set_names.add(set_name)

            # setting up the dataset_dict
            if set_name not in dataset_dict:
                dataset_dict[set_name] = {}
            if model_strength not in dataset_dict[set_name]:
                dataset_dict[set_name][model_strength] = {}
            if split_name not in dataset_dict[set_name][model_strength]:
                dataset_dict[set_name][model_strength][split_name] = []

            # reading the data from the file to dataset_dict
            with jsonlines.open(filepath) as reader:
                for data in reader:
                    dataset_dict[set_name][model_strength][split_name].append(data)
    return dataset_dict, all_splits, all_model_strengths, all_set_names

def get_question_wise_intersection(dataset_dict, all_set_names, all_splits, all_model_strengths):  
    assert set(all_model_strengths) == set(["weak", "strong"])
    
    random.seed(42)
    for set_name in all_set_names:
        for split_name in all_splits:
            weak_data = dataset_dict[set_name]['weak'][split_name]
            strong_data = dataset_dict[set_name]['strong'][split_name]

            random.shuffle(weak_data)
            weak_data_temp, weak_data_temp_keys = [], set()
            for item in weak_data:
                question, correct_response, incorrect_response = item['question'], item['examtaker_response_correct'], item['examtaker_response_incorrect']
                if (question, correct_response, incorrect_response) not in weak_data_temp_keys:
                    weak_data_temp.append(item)
                    weak_data_temp_keys.add((question, correct_response, incorrect_response))
            weak_data = weak_data_temp

            random.shuffle(strong_data)
            strong_data_temp, strong_data_temp_keys = [], set()
            for item in strong_data:
                question, correct_response, incorrect_response = item['question'], item['examtaker_response_correct'], item['examtaker_response_incorrect']
                if (question, correct_response, incorrect_response) not in strong_data_temp_keys:
                    strong_data_temp.append(item)
                    strong_data_temp_keys.add((question, correct_response, incorrect_response))
            strong_data = strong_data_temp

            question_weak_set = {}
            for item in weak_data:
                question = item['question']
                if question not in question_weak_set:
                    question_weak_set[question] = []
                question_weak_set[question].append(item)

            question_strong_set = {}
            for item in strong_data:
                question = item['question']
                if question not in question_strong_set:
                    question_strong_set[question] = []
                question_strong_set[question].append(item)

            question_common_weak_set = {}
            question_common_strong_set = {}
            for question in set(question_weak_set.keys()).intersection(set(question_strong_set.keys())):
                weak_items = question_weak_set[question]
                strong_items = question_strong_set[question]

                random.shuffle(weak_items)
                random.shuffle(strong_items)

                min_len = min(len(weak_items), len(strong_items))
                weak_items = weak_items[:min_len]
                strong_items = strong_items[:min_len]

                question_common_weak_set[question] = weak_items
                question_common_strong_set[question] = strong_items

            weak_common_data = []
            strong_common_data = []
            for question in question_common_weak_set:
                weak_items = question_common_weak_set[question]
                strong_items = question_common_strong_set[question]

                for weak_item, strong_item in zip(weak_items, strong_items):
                    weak_common_data.append(weak_item)
                    strong_common_data.append(strong_item)

            dataset_dict[set_name]['weak'][split_name] = weak_common_data
            dataset_dict[set_name]['strong'][split_name] = strong_common_data

    return dataset_dict

def get_judge_prompt(question, response_a, response_b):
    system_prompt = '''
You are a rigorous evaluator tasked with comparing two assistant responses to the same mathematical question. Your goal is to **evaluate the responses step by step** and finally output which one is **better**, **based solely on logical soundness and correctness**.

You will be given:
* A **Question**
* Two assistant responses: **Response A** and **Response B**

### Evaluation Guidelines:
To identify the **better** response, consider the following points:

1. **Prioritize correctness.** Focus on factual accuracy and logical reasoning. Select the response that is free from mistakes or flawed logic both in its reasoning and final answer.
2. **If both responses contain inaccuracies**, choose the one that ultimately reaches the correct final answer, even if its reasoning has minor flaws.
3. **Remain objective.** Do **not** consider irrelevant factors such as response order, length, style, or formatting. Focus purely on correctness and reasoning quality.

### First, **evaluate the responses step by step** and then output your final verdict. Your **final verdict** must be **exactly one of the following JSON objects**:
1. **If the response A is better than response B, either in terms of overall reasoning and final answer or just in final answer, respond with:
```json
{"verdict": "A"}
```
2. **If the response B is better than response A, either in terms of overall reasoning and final answer or just in final answer, respond with:
```json
{"verdict": "B"}
```
'''
    user_prompt = f'''
**Question**
{question}

**Response A**
{response_a}

**Response B**
{response_b}
'''
    return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"


def format_sft_data(judge_prompt, correct_response):
    return {"conversations": [{"from": "user", "value": judge_prompt}, {"from": "assistant", "value": correct_response}]}

def format_dpo_data(judge_prompt, correct_response, incorrect_response):
    return {"question": judge_prompt, "chosen": correct_response, "rejected": incorrect_response}

def format_grpo_flipped_data(question, correct_response, incorrect_response):
    converted_example_label_A = get_judge_prompt(question, correct_response, incorrect_response)
    converted_example_label_B = get_judge_prompt(question, incorrect_response, correct_response)
    return [converted_example_label_A, converted_example_label_B]

def format_grpo_unflipped_data(judge_prompt, label, idx):
    converted_example = {
        "data_source": "deepscaler",
        "prompt": [
            {
                "role": "user",
                "content": judge_prompt
            }
        ],  # chat-style prompt:contentReference[oaicite:4]{index=4}
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": label},  # correct answer for reward:contentReference[oaicite:5]{index=5}
        "extra_info": {"split": "train", "index": idx}
    }
    return converted_example

def format_data(data, train_type):
    all_message_dicts, all_labels = [], []
    for idx, item in enumerate(data):
        judge_prompt, question, correct_response, incorrect_response, correct_examtaker_response, incorrect_examtaker_response = item['judge_prompt'], item['question'], item['correct_response'], item['incorrect_response'], item['examtaker_response_correct'], item['examtaker_response_incorrect']
        label = item['label']
        all_labels.append(label)
        if train_type == 'sft':
            message_dict = [format_sft_data(judge_prompt, correct_response),]
        elif train_type == 'dpo':
            message_dict = [format_dpo_data(judge_prompt, correct_response, incorrect_response),]
        elif train_type == 'grpo_flipped':
            message_dict = format_grpo_flipped_data(question, correct_examtaker_response, incorrect_examtaker_response)
        elif train_type == 'grpo_unflipped':
            message_dict = [(judge_prompt, label),]
        else:
            raise ValueError(f"Invalid train type: {train_type}")
        all_message_dicts.extend(message_dict)

    if train_type == 'grpo_flipped':
        print("=> Intial number of message dicts: ", len(all_message_dicts))

        list_judge_pair_keys = [(all_message_dicts[i], all_message_dicts[i+1]) for i in range(0, len(all_message_dicts), 2)]
        print("=> [GRPO FLIPPED] Number of judge-prompt pairs before removing duplicates: ", len(list_judge_pair_keys))
        set_judge_pair_keys = list(set(list_judge_pair_keys))
        print("=> [GRPO FLIPPED] Number of judge-prompt pairs after removing duplicates: ", len(set_judge_pair_keys))

        import random
        random.seed(42)
        random.shuffle(set_judge_pair_keys)
        final_message_dicts = []
        for idx, judge_prompt_label_pair in enumerate(set_judge_pair_keys):
            if random.random() < 0.5: # this is just to ensure that A, B are random uniformly distributed in the final dataset.
                final_message_dicts.append(format_grpo_unflipped_data(judge_prompt_label_pair[0], "A", 2*idx))
                final_message_dicts.append(format_grpo_unflipped_data(judge_prompt_label_pair[1], "B", 2*idx+1))
            else:
                final_message_dicts.append(format_grpo_unflipped_data(judge_prompt_label_pair[1], "B", 2*idx))
                final_message_dicts.append(format_grpo_unflipped_data(judge_prompt_label_pair[0], "A", 2*idx+1))

        all_message_dicts = final_message_dicts

    if train_type == 'grpo_unflipped':
        all_message_dicts_label_A, all_message_dicts_label_B = [], []
        print(f"=> [GRPO UNFLIPPED] Intial number of judge-prompt pairs: {len(all_message_dicts)} | Intial number of unique judge prompts: {len(set(judge_prompt for judge_prompt, _ in all_message_dicts))}")

        for judge_prompt, label in all_message_dicts:
            if label == "A":
                all_message_dicts_label_A.append((judge_prompt, label))
            elif label == "B":
                all_message_dicts_label_B.append((judge_prompt, label))

        import random
        random.seed(42)
        random.shuffle(all_message_dicts_label_A)
        random.shuffle(all_message_dicts_label_B)
        if len(all_message_dicts_label_A) < len(all_message_dicts_label_B): # A is smaller, upsample it
            num_samples_needed = len(all_message_dicts_label_B) - len(all_message_dicts_label_A)
            additional_samples = random.choices(all_message_dicts_label_A, k=num_samples_needed)
            all_message_dicts_label_A.extend(additional_samples)
        elif len(all_message_dicts_label_B) < len(all_message_dicts_label_A): # B is smaller, upsample it
            num_samples_needed = len(all_message_dicts_label_A) - len(all_message_dicts_label_B)
            additional_samples = random.choices(all_message_dicts_label_B, k=num_samples_needed)
            all_message_dicts_label_B.extend(additional_samples)
        print(f"=> [GRPO UNFLIPPED] Number of message dicts after upsampling: {2*len(all_message_dicts_label_A)}")

        final_message_dicts = []
        for a, b in zip(all_message_dicts_label_A, all_message_dicts_label_B):
            a_judge_prompt, a_label = a
            b_judge_prompt, b_label = b
            assert a_label == "A" and b_label == "B"
            if random.random() < 0.5: # this is just to ensure that A, B are random uniformly distributed in the final dataset.
                final_message_dicts.append(format_grpo_unflipped_data(a_judge_prompt, a_label, 2*idx))
                final_message_dicts.append(format_grpo_unflipped_data(b_judge_prompt, b_label, 2*idx+1))
            else:
                final_message_dicts.append(format_grpo_unflipped_data(b_judge_prompt, b_label, 2*idx))
                final_message_dicts.append(format_grpo_unflipped_data(a_judge_prompt, a_label, 2*idx+1))

        all_message_dicts = final_message_dicts

    if 'grpo' in train_type:
        # print(json.dumps(all_message_dicts[0], indent=4))
        # print(json.dumps(all_message_dicts[1], indent=4))
        # print("-" * 60)
        all_message_dicts = Dataset.from_list(all_message_dicts)

    return all_message_dicts, all_labels

def format_sft_dpo_data(args, dataset_dict, all_set_names, all_model_strengths, all_splits, output_dir):
    intersect = 'ques_wise' if args.question_intersection else 'no_ques_wise'
    answers_type = 'answers_one' if args.include_two_response_pairs == 'one_res' else 'answers_two'

    train_types = set(['sft', 'dpo']) & set(args.format_algos)
    for train_type in train_types:
        for set_name in all_set_names:
            for model_strength in all_model_strengths:
                for split_name in all_splits:
                    if "train" not in split_name:
                        continue

                    data = dataset_dict[set_name][model_strength][split_name]
                    all_message_dicts, all_labels = format_data(data, train_type)

                    label_A_count, label_B_count = 0, 0
                    for label in all_labels:
                        if label == "A":
                            label_A_count += 1
                        elif label == "B":
                            label_B_count += 1
                    total = label_A_count + label_B_count
                    label_A_pct = (label_A_count / total) * 100
                    label_B_pct = (label_B_count / total) * 100
                    print(f"Set Name: {set_name} | Model Strength: {model_strength} | Split Name: {split_name} | Label A: {label_A_pct:.1f}% | Label B: {label_B_pct:.1f}%")

                    output_path = os.path.join(output_dir, f"{train_type}_{set_name}_{model_strength}_{intersect}_{answers_type}_train.jsonl")
                    with jsonlines.open(output_path, 'w') as writer:
                        for message_dict in all_message_dicts:
                            writer.write(message_dict)

def format_grpo_data(args, dataset_dict, all_set_names, all_model_strengths, all_splits, output_dir):
    intersect = 'ques_wise' if args.question_intersection else 'no_ques_wise'
    answers_type = 'answers_one' if args.include_two_response_pairs == 'one_res' else 'answers_two'

    for train_type in ['grpo_unflipped', 'grpo_flipped']:
        for set_name in all_set_names:
            for model_strength in all_model_strengths:
                for split_name in all_splits:
                    if ("train" in split_name) or ("eval" in split_name and "_seen_questions_unseen_answers" in split_name and train_type == 'grpo_flipped'):

                        if "train" in split_name:
                            split_name_str = "train"
                        elif "eval" in split_name:
                            split_name_str = "eval"
                        else:
                            raise ValueError(f"Invalid split name: {split_name}")
                        
                        data = dataset_dict[set_name][model_strength][split_name]
                        all_message_dicts, all_labels = format_data(data, train_type)

                        print("=> Set Name: ", set_name)
                        print("=> Model Strength: ", model_strength)
                        print("=> Split Name: ", split_name)
                        print("=> Number of examples: ", len(all_message_dicts))
                        print("-" * 50)

                        label_A_count, label_B_count = 0, 0
                        for label in all_labels:
                            if label == "A":
                                label_A_count += 1
                            elif label == "B":
                                label_B_count += 1
                        total = label_A_count + label_B_count
                        label_A_pct = (label_A_count / total) * 100
                        label_B_pct = (label_B_count / total) * 100
                        print(f"Set Name: {set_name} | Model Strength: {model_strength} | Split Name: {split_name} | Label A: {label_A_pct:.1f}% | Label B: {label_B_pct:.1f}%")

                        output_path = os.path.join(output_dir, f"{train_type}_{set_name}_{model_strength}_{intersect}_{answers_type}_{split_name_str}.parquet")
                        all_message_dicts.to_parquet(output_path)
                        print(f"+++++ Saved to {output_path} | Number of examples: {len(all_message_dicts)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Format training data for SFT/DPO/GRPO.")
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./sf-judge-data/data_splits',
        help='Path to the dataset directory.'
    )
    parser.add_argument(
        '--include-two-response-pairs', 
        choices=['two_res', 'one_res'],
        default='one_res',
        help='What type of data split to use, one_res or two_res.'
    )
    parser.add_argument(
        "--set_name",
        type=str,
        choices=["set_1", "set_2"],
        required=True,
        help="Dataset set name: 'set_1' or 'set_2'."
    )
    parser.add_argument(
        '--question-intersection',
        action='store_true',
        default=False,
        help='Whether to use question-wise intersection between weak and strong models in the train set, so that every question has same number of responses in both strong and weak splits.'
    )
    parser.add_argument(
        '--format_algos',
        nargs='+',
        choices=['sft', 'dpo', 'grpo'],
        required=True,
        help='Which algorithms to format data for. Can be one or both of: sft_dpo, grpo. Example: --format_algos sft_dpo grpo'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./sf-judge-data/formatted_data_splits/train',
        help='Output directory for formatted data.'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # read all the relevant filenames and create the dataset_dict.
    dataset_files = get_filenames(args, args.dataset_path)
    dataset_dict, all_splits, all_model_strengths, all_set_names = create_dataset_dict(dataset_files, args.dataset_path)
    if args.question_intersection:
        dataset_dict = get_question_wise_intersection(dataset_dict, all_set_names, all_splits, all_model_strengths)

    if 'sft' in args.format_algos or 'dpo' in args.format_algos:
        print("=> Formatting SFT/DPO data...")
        format_sft_dpo_data(args, dataset_dict, all_set_names, all_model_strengths, all_splits, args.output_dir)

    if 'grpo' in args.format_algos:
        print("=> Formatting GRPO data...")
        format_grpo_data(args, dataset_dict, all_set_names, all_model_strengths, all_splits, args.output_dir) # for grpo, we also save a eval split.

if __name__ == "__main__":
    main()