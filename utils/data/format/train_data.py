### This scrip is yet to be verified if it is generating the correct data.

import os
import json
from collections import defaultdict
import jsonlines
import asyncio
from openai import AsyncOpenAI
from json_repair import repair_json
from tqdm import tqdm
import random
import sys
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch


class DataFormatter:
    """Formats training data for different training types."""
    
    def __init__(self, answers_type='answers_one', set_name='set_1'):
        self.answers_type = answers_type
        self.set_name = set_name
        self.dataset_path = "/shared/storage-01/users/jvsingh2/sf-intern/judge-analysis/curate_axolotl_data/weak_strong_deepscaler_new_final"
        self.dataset_dict = {}
        self.all_splits, self.all_model_strengths, self.all_set_names = set(), set(), set()
        
        # Model configuration
        self.model_port_dict = {
            "set_1": {
                "model_name": "mistralai/Ministral-8B-Instruct-2410",
            },
            "set_2": {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            }
        }

    def load_dataset_files(self):
        """Load and parse dataset files - preserving original logic exactly."""
        dataset_files = [fname for fname in os.listdir(self.dataset_path) if fname.endswith('.jsonl') and self.answers_type in fname and self.set_name in fname]

        for filename in dataset_files:
            # reading the filename into parts.
            filepath = os.path.join(self.dataset_path, filename)
            filename_split = filename[:-6].split('_')
            set_name = '_'.join(filename_split[:2])
            model_strength = filename_split[2]
            split_name = '_'.join(filename_split[3:])

            self.all_splits.add(split_name)
            self.all_model_strengths.add(model_strength)
            self.all_set_names.add(set_name)

            # setting up the dataset_dict
            if set_name not in self.dataset_dict:
                self.dataset_dict[set_name] = {}
            if model_strength not in self.dataset_dict[set_name]:
                self.dataset_dict[set_name][model_strength] = {}
            if split_name not in self.dataset_dict[set_name][model_strength]:
                self.dataset_dict[set_name][model_strength][split_name] = []

            # reading the data from the file to dataset_dict
            with jsonlines.open(filepath) as reader:
                for data in reader:
                    self.dataset_dict[set_name][model_strength][split_name].append(data)

    def apply_question_wise_intersection(self):
        """Apply question-wise intersection between weak and strong data - preserving original logic exactly."""
        random.seed(42)
        for set_name in self.all_set_names:
            for split_name in self.all_splits:
                weak_data = self.dataset_dict[set_name]['weak'][split_name]
                strong_data = self.dataset_dict[set_name]['strong'][split_name]

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

                self.dataset_dict[set_name]['weak'][split_name] = weak_common_data
                self.dataset_dict[set_name]['strong'][split_name] = strong_common_data

    def print_dataset_statistics(self):
        """Print dataset statistics - preserving original logic exactly."""
        print("-" * 60)
        print("=" * 60)

        print("\nDataset Statistics:")
        print("Set Name | Model Strength | Split Name | Number of Examples")
        print("-" * 60)

        # print the stats in each of the splits
        for set_name in self.dataset_dict:
            for model_strength in self.dataset_dict[set_name]:
                for split_name in self.dataset_dict[set_name][model_strength]:
                    num_examples = len(self.dataset_dict[set_name][model_strength][split_name])
                    print(f"{set_name:8} | {model_strength:13} | {split_name:30} | {num_examples}")
        print("=" * 60)

    def get_question_iou(self, data_1, data_2):
        """Get question intersection over union - preserving original logic exactly."""
        questions_1 = set([item['question'] for item in data_1])
        questions_2 = set([item['question'] for item in data_2])
        return len(questions_1.intersection(questions_2)) / len(questions_1.union(questions_2))

    def print_intersection_statistics(self):
        """Print intersection statistics - preserving original logic exactly."""
        # compute the iou for the splits.
        for set_name in self.all_set_names:
            for split_name in self.all_splits:
                iou = self.get_question_iou(self.dataset_dict[set_name]['weak'][split_name], self.dataset_dict[set_name]['strong'][split_name])
                print(f"Across model strengths: {set_name:8} | {split_name:30} | {iou:.2f}")
            print("-" * 60)
        print("=" * 60)

        for set_name in self.all_set_names:
            for model_strength in self.all_model_strengths:
                iou = self.get_question_iou(self.dataset_dict[set_name][model_strength]['train_seen_questions_seen_answers_one_res_pairs'], self.dataset_dict[set_name][model_strength]['train_seen_questions_seen_answers_one_res_pairs'])
                print(f"Across splits: {set_name:8} | {model_strength:13} | {iou:.2f}")
            print("-" * 60)

    def get_responses(self, llm, tokenizer, prompts, n=1):
        """Get responses from LLM - preserving original logic exactly."""
        chat_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
        sampling_params = SamplingParams(temperature=0.0, n=n, max_tokens=2000)
        outputs = llm.generate(chat_prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses

    def get_label(self, assistant_response):
        """Get label from assistant response - preserving original logic exactly."""
        text = assistant_response[-50:]
        if '```json' in text:
            text = '```json' + text.split('```json')[1]
        label_dict = repair_json(text, return_objects=True)
        
        if isinstance(label_dict, list) and len(label_dict) == 0:
            return None

        if isinstance(label_dict, list):
            label_dict = label_dict[-1]

        if not (isinstance(label_dict, dict) and \
                "verdict" in label_dict and \
                label_dict["verdict"] in ['A', 'B']):
            return None

        predicted_label = label_dict["verdict"]
        return predicted_label

    def eval_judge_responses(self, responses, labels):
        """Evaluate judge responses - preserving original logic exactly."""
        predicted_labels = []
        for response, label in zip(responses, labels):
            predicted_label = self.get_label(response)
            predicted_labels.append(predicted_label)

        assert len(predicted_labels) == len(labels)

        correct_items, total_items = 0, 0
        for idx in range(len(predicted_labels)):
            if predicted_labels[idx] == labels[idx]:
                correct_items += 1
            total_items += 1
        accuracy_pointwise = correct_items / total_items
        print(f"Correct items: {correct_items} | Total items: {total_items} | Accuracy (pointwise) : {accuracy_pointwise:.2f}")

        correct_items, total_items = 0, 0
        for idx in range(0, len(predicted_labels), 2):
            if predicted_labels[idx] == labels[idx] and predicted_labels[idx+1] == labels[idx+1]:
                correct_items += 1
            total_items += 1
        accuracy_pairwise = correct_items / total_items
        print(f"Correct items: {correct_items} | Total items: {total_items} | Accuracy (pairwise) : {accuracy_pairwise:.2f}")

        return accuracy_pointwise, accuracy_pairwise

    def get_judge_prompt(self, question, response_a, response_b):
        """Get judge prompt - preserving original logic exactly."""
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

    def get_correctly_flipped_pairs(self, data):
        """Get correctly flipped pairs - preserving original logic exactly."""
        unique_qr_triples = set([(item['question'], item['examtaker_response_correct'], item['examtaker_response_incorrect']) for item in data])

        all_judge_prompt_label_pairs = []
        for qr_triple in unique_qr_triples:
            q, r_correct, r_incorrect = qr_triple

            judge_prompt = self.get_judge_prompt(question=q, response_a=r_correct, response_b=r_incorrect)
            judge_label = "A"
            all_judge_prompt_label_pairs.append([judge_prompt, judge_label])

            judge_prompt = self.get_judge_prompt(question=q, response_a=r_incorrect, response_b=r_correct)
            judge_label = "B"
            all_judge_prompt_label_pairs.append([judge_prompt, judge_label])

        return all_judge_prompt_label_pairs

    def process_all_evaluations(self):
        """Process all evaluations - preserving original logic exactly."""
        for set_name in self.all_set_names:
            model_name = self.model_port_dict[set_name]["model_name"]
            num_gpus = torch.cuda.device_count()
            llm = LLM(
                model=model_name,
                dtype="bfloat16",
                tensor_parallel_size=num_gpus,
                enable_prefix_caching=True,
                max_model_len=7000,
                gpu_memory_utilization=0.9
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            all_prompt_label_pairs = []
            all_split_names = []
            for model_strength in self.all_model_strengths:
                for split_name in self.all_splits:
                    if "train" in split_name:
                        continue
                    data = self.dataset_dict[set_name][model_strength][split_name]
                    judge_prompt_label_pairs = self.get_correctly_flipped_pairs(data)
                    all_prompt_label_pairs.append(judge_prompt_label_pairs)
                    all_split_names.append((model_strength, split_name))

            # Split prompts across GPUs
            all_prompts = [judge_prompt for judge_prompt_label_pairs in all_prompt_label_pairs for judge_prompt, label in judge_prompt_label_pairs]
            responses = self.get_responses(llm, tokenizer, all_prompts)

            i = -1
            for j in range(len(all_prompt_label_pairs)):
                for k in range(len(all_prompt_label_pairs[j])):
                    i += 1
                    assert all_prompt_label_pairs[j][k][0] == all_prompts[i]
                    all_prompt_label_pairs[j][k].append(responses[i])

            for idx, (model_strength, split_name) in enumerate(all_split_names):
                prompt_label_response_pairs = all_prompt_label_pairs[idx]

                accuracy_pointwise, accuracy_pairwise = self.eval_judge_responses([response for _, _, response in prompt_label_response_pairs], [label for _, label, _ in prompt_label_response_pairs])

                print("Split Name: ", split_name, "| Set Name: ", set_name, "| Model Strength: ", model_strength)
                print(f">>>>> Number of responses: {len(prompt_label_response_pairs)} | Accuracy (pointwise) : {accuracy_pointwise:.4f} | Accuracy (pairwise) : {accuracy_pairwise:.4f}")

    def format_sft_data(self, judge_prompt, correct_response):
        """Format SFT data - preserving original logic exactly."""
        return {"conversations": [{"from": "user", "value": judge_prompt}, {"from": "assistant", "value": correct_response}]}

    def format_dpo_data(self, judge_prompt, correct_response, incorrect_response):
        """Format DPO data - preserving original logic exactly."""
        return {"question": judge_prompt, "chosen": correct_response, "rejected": incorrect_response}

    def format_grpo_flipped_data(self, question, correct_response, incorrect_response):
        """Format GRPO flipped data - preserving original logic exactly."""
        converted_example_label_A = self.get_judge_prompt(question, correct_response, incorrect_response)
        converted_example_label_B = self.get_judge_prompt(question, incorrect_response, correct_response)
        return [converted_example_label_A, converted_example_label_B]

    def format_grpo_unflipped_data(self, judge_prompt, label, idx):
        """Format GRPO unflipped data - preserving original logic exactly."""
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

    def format_data(self, data, train_type):
        """Format data based on training type - preserving original logic exactly."""
        all_message_dicts, all_labels = [], []
        for idx, item in enumerate(data):
            judge_prompt, question, correct_response, incorrect_response, correct_examtaker_response, incorrect_examtaker_response = item['judge_prompt'], item['question'], item['correct_response'], item['incorrect_response'], item['examtaker_response_correct'], item['examtaker_response_incorrect']
            label = item['label']
            all_labels.append(label)
            if train_type == 'sft':
                message_dict = [self.format_sft_data(judge_prompt, correct_response),]
            elif train_type == 'dpo':
                message_dict = [self.format_dpo_data(judge_prompt, correct_response, incorrect_response),]
            elif train_type == 'grpo_flipped':
                message_dict = self.format_grpo_flipped_data(question, correct_examtaker_response, incorrect_examtaker_response)
            elif train_type == 'grpo_unflipped':
                message_dict = [(judge_prompt, label),]
            else:
                raise ValueError(f"Invalid train type: {train_type}")
            all_message_dicts.extend(message_dict)

        if train_type == 'grpo_flipped':
            print("=> Intial number of message dicts: ", len(all_message_dicts))

            list_judge_pair_keys = [(all_message_dicts[i], all_message_dicts[i+1]) for i in range(0, len(all_message_dicts), 2)]
            print("=> Number of judge-prompt pairs before removing duplicates: ", len(list_judge_pair_keys))
            set_judge_pair_keys = list(set(list_judge_pair_keys))
            print("=> Number of judge-prompt pairs after removing duplicates: ", len(set_judge_pair_keys))

            import random
            random.seed(42)
            random.shuffle(set_judge_pair_keys)
            final_message_dicts = []
            for idx, judge_prompt_label_pair in enumerate(set_judge_pair_keys):
                if random.random() < 0.5:
                    final_message_dicts.append(self.format_grpo_unflipped_data(judge_prompt_label_pair[0], "A", 2*idx))
                    final_message_dicts.append(self.format_grpo_unflipped_data(judge_prompt_label_pair[1], "B", 2*idx+1))
                else:
                    final_message_dicts.append(self.format_grpo_unflipped_data(judge_prompt_label_pair[1], "B", 2*idx))
                    final_message_dicts.append(self.format_grpo_unflipped_data(judge_prompt_label_pair[0], "A", 2*idx+1))

            all_message_dicts = final_message_dicts

        if train_type == 'grpo_unflipped':
            all_message_dicts_label_A, all_message_dicts_label_B = [], []

            print("=> Intial number of message dicts: ", len(all_message_dicts))
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
            print("=> Number of message dicts after upsampling: ", 2*len(all_message_dicts_label_A))

            final_message_dicts = []
            for a, b in zip(all_message_dicts_label_A, all_message_dicts_label_B):
                a_judge_prompt, a_label = a
                b_judge_prompt, b_label = b
                if random.random() < 0.5:
                    final_message_dicts.append(self.format_grpo_unflipped_data(a_judge_prompt, a_label, 2*idx))
                    final_message_dicts.append(self.format_grpo_unflipped_data(b_judge_prompt, b_label, 2*idx+1))
                else:
                    final_message_dicts.append(self.format_grpo_unflipped_data(b_judge_prompt, b_label, 2*idx))
                    final_message_dicts.append(self.format_grpo_unflipped_data(a_judge_prompt, a_label, 2*idx+1))

            all_message_dicts = final_message_dicts

        if 'grpo' in train_type:
            # print(json.dumps(all_message_dicts[0], indent=4))
            # print(json.dumps(all_message_dicts[1], indent=4))
            # print("-" * 60)
            all_message_dicts = Dataset.from_list(all_message_dicts)

        return all_message_dicts, all_labels

    def process_grpo_formats(self, intersect):
        """Process GRPO formats - preserving original logic exactly."""
        # for train_type in ['grpo_flipped', 'grpo_unflipped']:
        for train_type in ['grpo_unflipped', 'grpo_flipped']:
            for set_name in self.all_set_names:
                for model_strength in self.all_model_strengths:
                    for split_name in self.all_splits:
                        if ("train" in split_name) or ("eval" in split_name and "_seen_questions_unseen_answers" in split_name and train_type == 'grpo_flipped'):

                            if "train" in split_name:
                                split_name_str = "train"
                            elif "eval" in split_name:
                                split_name_str = "eval"
                            else:
                                raise ValueError(f"Invalid split name: {split_name}")
                            
                            data = self.dataset_dict[set_name][model_strength][split_name]
                            all_message_dicts, all_labels = self.format_data(data, train_type)

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

                            output_path = f"/shared/storage-01/users/jvsingh2/sf-intern/judge-analysis/curate_axolotl_data/weak_strong_deepscaler_new_final_formatted/{train_type}_{set_name}_{model_strength}_{intersect}_{self.answers_type}_{split_name_str}.parquet"
                            all_message_dicts.to_parquet(output_path)
                            print(f"+++++ Saved to {output_path} | Number of examples: {len(all_message_dicts)}")

    def process_sft_dpo_formats(self, intersect):
        """Process SFT and DPO formats - preserving original logic exactly."""
        for train_type in ['sft', 'dpo']:
            for set_name in self.all_set_names:
                for model_strength in self.all_model_strengths:
                    for split_name in self.all_splits:
                        if "train" not in split_name:
                            continue

                        data = self.dataset_dict[set_name][model_strength][split_name]
                        all_message_dicts, all_labels = self.format_data(data, train_type)

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

                        output_path = f"/shared/storage-01/users/jvsingh2/sf-intern/judge-analysis/curate_axolotl_data/weak_strong_deepscaler_new_final_formatted/{train_type}_{set_name}_{model_strength}_{intersect}_{self.answers_type}_train.jsonl"
                        with jsonlines.open(output_path, 'w') as writer:
                            for message_dict in all_message_dicts:
                                writer.write(message_dict)

    def run_full_pipeline(self, intersect):
        """Run the complete data formatting pipeline - preserving original logic exactly."""
        # Load dataset files
        self.load_dataset_files()
        
        # Apply question-wise intersection if needed
        if intersect == 'ques_wise':
            self.apply_question_wise_intersection()
        
        # Print statistics
        self.print_dataset_statistics()
        self.print_intersection_statistics()
        
        # Process all formats
        self.process_grpo_formats(intersect)
        self.process_sft_dpo_formats(intersect)


def main():
    """Main function - preserving original logic exactly."""
    answers_type = 'answers_one'
    # answers_type = 'answers_two'

    set_name = 'set_1'
    # set_name = 'set_2'

    intersect = sys.argv[1]
    if intersect == 'ques_wise':
        ques_wise_intersection = True
    else: # no_ques_wise
        assert intersect == 'no_ques_wise'
        ques_wise_intersection = False

    # Create formatter and run pipeline
    formatter = DataFormatter(answers_type=answers_type, set_name=set_name)
    formatter.run_full_pipeline(intersect)

    # Uncomment to run evaluations
    # formatter.process_all_evaluations()


if __name__ == "__main__":
    main()







