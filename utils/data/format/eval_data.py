import os
import jsonlines
import random
import argparse
import json
import sys

# Add utils to path for importing common functions
sys.path.append("./utils")
from common import check_working_directory

"""
python utils/data/format/eval_data.py --set-name set_1
python utils/data/format/eval_data.py --set-name set_2
"""

def parse_arguments():
    """Parse command line arguments for dataset processing configuration."""
    parser = argparse.ArgumentParser(description='Process dataset splits with configurable parameters')

    parser.add_argument(
        '--include-two-response-pairs', 
        choices=['two_res', 'one_res'],
        default='one_res',
        help='What type of data split to use, one_res or two_res.'
    )

    parser.add_argument(
        '--set-name',
        choices=['set_1', 'set_2'], 
        default='set_1',
        help='Which dataset set to use, whether take examtaker responses from set_1 or set_2.'
    )

    parser.add_argument(
        '--question-intersection',
        action='store_true',
        default=False,
        help='Whether to use question-wise intersection between weak and strong models in the train set, so that every question has same number of responses in both strong and weak splits.'
    )

    return parser.parse_args()

def setup_answers_type(args):
    """Convert argument choice to internal answers type format."""
    answers_type = args.include_two_response_pairs
    if answers_type == 'one_res':
        answers_type = 'answers_one'
    else:
        answers_type = 'answers_two'
    return answers_type

def load_dataset_files(dataset_path, answers_type, set_name):
    """Load and parse dataset files into structured dictionary."""
    # Filter files based on criteria
    dataset_files = [
        fname for fname in os.listdir(dataset_path) 
        if fname.endswith('.jsonl') and answers_type in fname and set_name in fname
    ]

    dataset_dict = {}
    all_splits, all_model_strengths, all_set_names = set(), set(), set()
    
    for filename in dataset_files:
        # Parse filename components
        filepath = os.path.join(dataset_path, filename)
        filename_split = filename[:-6].split('_')
        
        set_name = '_'.join(filename_split[:2])
        model_strength = filename_split[2]
        split_name = '_'.join(filename_split[3:])

        # Collect unique values
        all_splits.add(split_name)
        all_model_strengths.add(model_strength)
        all_set_names.add(set_name)

        # Initialize nested dictionary structure
        if set_name not in dataset_dict:
            dataset_dict[set_name] = {}
        if model_strength not in dataset_dict[set_name]:
            dataset_dict[set_name][model_strength] = {}
        if split_name not in dataset_dict[set_name][model_strength]:
            dataset_dict[set_name][model_strength][split_name] = []

        # Load data from file
        with jsonlines.open(filepath) as reader:
            for data in reader:
                dataset_dict[set_name][model_strength][split_name].append(data)
    
    return dataset_dict, all_splits, all_model_strengths, all_set_names

def apply_question_wise_intersection(dataset_dict, all_set_names, all_splits):
    """Apply question-wise intersection to balance weak and strong model data."""
    random.seed(42)
    
    for set_name in all_set_names:
        for split_name in all_splits:
            weak_data = dataset_dict[set_name]['weak'][split_name]
            strong_data = dataset_dict[set_name]['strong'][split_name]

            # Process weak data: remove duplicates and shuffle
            random.shuffle(weak_data)
            weak_data_temp, weak_data_temp_keys = [], set()
            for item in weak_data:
                question = item['question']
                correct_response = item['examtaker_response_correct']
                incorrect_response = item['examtaker_response_incorrect']
                
                if (question, correct_response, incorrect_response) not in weak_data_temp_keys:
                    weak_data_temp.append(item)
                    weak_data_temp_keys.add((question, correct_response, incorrect_response))
            weak_data = weak_data_temp

            # Process strong data: remove duplicates and shuffle
            random.shuffle(strong_data)
            strong_data_temp, strong_data_temp_keys = [], set()
            for item in strong_data:
                question = item['question']
                correct_response = item['examtaker_response_correct']
                incorrect_response = item['examtaker_response_incorrect']
                
                if (question, correct_response, incorrect_response) not in strong_data_temp_keys:
                    strong_data_temp.append(item)
                    strong_data_temp_keys.add((question, correct_response, incorrect_response))
            strong_data = strong_data_temp

            # Group data by question
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

            # Find common questions and balance data
            question_common_weak_set = {}
            question_common_strong_set = {}
            
            common_questions = set(question_weak_set.keys()).intersection(set(question_strong_set.keys()))
            
            for question in common_questions:
                weak_items = question_weak_set[question]
                strong_items = question_strong_set[question]

                # Shuffle and balance to minimum length
                random.shuffle(weak_items)
                random.shuffle(strong_items)

                min_len = min(len(weak_items), len(strong_items))
                weak_items = weak_items[:min_len]
                strong_items = strong_items[:min_len]

                question_common_weak_set[question] = weak_items
                question_common_strong_set[question] = strong_items

            # Reconstruct balanced datasets
            weak_common_data = []
            strong_common_data = []
            
            for question in question_common_weak_set:
                weak_items = question_common_weak_set[question]
                strong_items = question_common_strong_set[question]

                for weak_item, strong_item in zip(weak_items, strong_items):
                    weak_common_data.append(weak_item)
                    strong_common_data.append(strong_item)

            # Update dataset with balanced data
            dataset_dict[set_name]['weak'][split_name] = weak_common_data
            dataset_dict[set_name]['strong'][split_name] = strong_common_data

def get_judge_prompt(question, response_a, response_b):
    """Generate judge prompt for comparing two responses."""
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


def get_prompt_label_pairs(data, flip=False):
    """Generate prompt-label pairs for training/evaluation."""
    all_judge_prompt_label_pairs = []
    
    if flip:
        # Create flipped pairs for data augmentation
        unique_qr_triples = set([
            (item['question'], item['examtaker_response_correct'], item['examtaker_response_incorrect']) 
            for item in data
        ])

        for qr_triple in unique_qr_triples:
            q, r_correct, r_incorrect = qr_triple

            # Original order: correct=A, incorrect=B
            judge_prompt = get_judge_prompt(question=q, response_a=r_correct, response_b=r_incorrect)
            judge_label = "A"
            all_judge_prompt_label_pairs.append({"prompt": judge_prompt, "label": judge_label})

            # Flipped order: incorrect=A, correct=B
            judge_prompt = get_judge_prompt(question=q, response_a=r_incorrect, response_b=r_correct)
            judge_label = "B"
            all_judge_prompt_label_pairs.append({"prompt": judge_prompt, "label": judge_label})
    
    else:
        # Use existing judge prompts and labels
        for item in data:
            judge_prompt = item['judge_prompt']
            judge_label = item['label']
            all_judge_prompt_label_pairs.append({"prompt": judge_prompt, "label": judge_label})

    return all_judge_prompt_label_pairs

def process_and_save_data(dataset_dict, ques_wise_intersection, answers_type):
    """Process dataset and save formatted data to output files."""
    # Setup output directory
    output_data_path = "./sf-judge-data/formatted_data_splits/eval"
    os.makedirs(output_data_path, exist_ok=True)

    stats_dict = {}
    
    # Process each dataset combination
    for set_name in dataset_dict:
        for model_strength in dataset_dict[set_name]:
            for split_name in dataset_dict[set_name][model_strength]:
                # Skip training splits
                if "train" in split_name:
                    continue
                
                # Generate both flipped and unflipped versions
                for flip in [True, False]:
                    # Create output filename
                    intersection_type = 'ques_wise' if ques_wise_intersection else 'no_ques_wise'
                    flip_suffix = 'flip' if flip else 'no_flip'
                    
                    output_file_name = f"{set_name}_{model_strength}_{intersection_type}_{answers_type}_{split_name}_{flip_suffix}.jsonl"
                    
                    # Generate prompt-label pairs
                    all_prompt_label_pairs = get_prompt_label_pairs(
                        dataset_dict[set_name][model_strength][split_name], 
                        flip=flip
                    )
                    
                    # Save to file
                    output_file_path = os.path.join(output_data_path, output_file_name)
                    stats_dict[output_file_path] = len(all_prompt_label_pairs)
                    
                    with jsonlines.open(output_file_path, mode='w') as writer:
                        for item in all_prompt_label_pairs:
                            writer.write(item)

    # Print statistics
    print(json.dumps(stats_dict, indent=4))

def main():
    """Main execution function."""
    # Verify working directory
    if not check_working_directory():
        exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup configuration
    answers_type = setup_answers_type(args)
    set_name = args.set_name
    ques_wise_intersection = args.question_intersection
    
    # Load and parse dataset files
    dataset_path = "./sf-judge-data/data_splits"
    dataset_dict, all_splits, all_model_strengths, all_set_names = load_dataset_files(
        dataset_path, answers_type, set_name
    )
    
    # Apply question-wise intersection if requested
    if ques_wise_intersection:
        apply_question_wise_intersection(dataset_dict, all_set_names, all_splits)
    
    # Process and save formatted data
    process_and_save_data(dataset_dict, ques_wise_intersection, answers_type)


if __name__ == "__main__":
    main()
