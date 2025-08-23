import os
import jsonlines
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import sys

sys.path.append("./utils") # to import the common.py file
from common import check_working_directory

class OverlapAnalyzer:
    """Analyzes overlap statistics between formatted and unformatted data splits."""
    
    def __init__(self, formatted_folder: str = "./sf-judge-data/formatted_data_splits", 
                 unformatted_folder: str = "./sf-judge-data/data_splits"):
        self.formatted_folder = Path(formatted_folder)
        self.unformatted_folder = Path(unformatted_folder)
        
        # Define file pairs for comparison - now including both set_1 and set_2
        self.pair_files = [
            #### Set 1 pairs ####

             # eval files
            ("eval/set_1_strong_no_ques_wise_answers_one_eval_seen_questions_seen_answers_one_res_pairs_no_flip.jsonl", 
             "set_1_strong_eval_seen_questions_seen_answers_one_res_pairs.jsonl", "eval", "set_1"),
            ("eval/set_1_strong_no_ques_wise_answers_one_eval_seen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_1_strong_eval_seen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_1"),
            ("eval/set_1_strong_no_ques_wise_answers_one_eval_unseen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_1_strong_eval_unseen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_1"),
            ("eval/set_1_weak_no_ques_wise_answers_one_eval_seen_questions_seen_answers_one_res_pairs_no_flip.jsonl", 
             "set_1_weak_eval_seen_questions_seen_answers_one_res_pairs.jsonl", "eval", "set_1"),
            ("eval/set_1_weak_no_ques_wise_answers_one_eval_seen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_1_weak_eval_seen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_1"),
            ("eval/set_1_weak_no_ques_wise_answers_one_eval_unseen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_1_weak_eval_unseen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_1"),

            # train files
            ("train/sft_set_1_weak_no_ques_wise_answers_one_train.jsonl", 
             "set_1_weak_train_seen_questions_seen_answers_one_res_pairs.jsonl", "sft", "set_1"),
            ("train/sft_set_1_strong_no_ques_wise_answers_one_train.jsonl", 
             "set_1_strong_train_seen_questions_seen_answers_one_res_pairs.jsonl", "sft", "set_1"),
            ("train/dpo_set_1_weak_no_ques_wise_answers_one_train.jsonl", 
             "set_1_weak_train_seen_questions_seen_answers_one_res_pairs.jsonl", "dpo", "set_1"),
            ("train/dpo_set_1_strong_no_ques_wise_answers_one_train.jsonl", 
             "set_1_strong_train_seen_questions_seen_answers_one_res_pairs.jsonl", "dpo", "set_1"),

            # #### Set 2 pairs ####

             # eval files
            ("eval/set_2_strong_no_ques_wise_answers_one_eval_seen_questions_seen_answers_one_res_pairs_no_flip.jsonl", 
             "set_2_strong_eval_seen_questions_seen_answers_one_res_pairs.jsonl", "eval", "set_2"),
            ("eval/set_2_strong_no_ques_wise_answers_one_eval_seen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_2_strong_eval_seen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_2"),
            ("eval/set_2_strong_no_ques_wise_answers_one_eval_unseen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_2_strong_eval_unseen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_2"),
            ("eval/set_2_weak_no_ques_wise_answers_one_eval_seen_questions_seen_answers_one_res_pairs_no_flip.jsonl", 
             "set_2_weak_eval_seen_questions_seen_answers_one_res_pairs.jsonl", "eval", "set_2"),
            ("eval/set_2_weak_no_ques_wise_answers_one_eval_seen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_2_weak_eval_seen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_2"),
            ("eval/set_2_weak_no_ques_wise_answers_one_eval_unseen_questions_unseen_answers_one_res_pairs_no_flip.jsonl", 
             "set_2_weak_eval_unseen_questions_unseen_answers_one_res_pairs.jsonl", "eval", "set_2"),

            # train files
            ("train/sft_set_2_weak_no_ques_wise_answers_one_train.jsonl", 
             "set_2_weak_train_seen_questions_seen_answers_one_res_pairs.jsonl", "sft", "set_2"),
            ("train/sft_set_2_strong_no_ques_wise_answers_one_train.jsonl", 
             "set_2_strong_train_seen_questions_seen_answers_one_res_pairs.jsonl", "sft", "set_2"),
            ("train/dpo_set_2_weak_no_ques_wise_answers_one_train.jsonl", 
             "set_2_weak_train_seen_questions_seen_answers_one_res_pairs.jsonl", "dpo", "set_2"),
            ("train/dpo_set_2_strong_no_ques_wise_answers_one_train.jsonl", 
             "set_2_strong_train_seen_questions_seen_answers_one_res_pairs.jsonl", "dpo", "set_2"),
        ]
        
        # Model definitions - updated to include set_2 models
        self.weak_exam_takers_set_1 = {
            "google.gemma-2-9b-it", 
            "Qwen.Qwen2-7B-Instruct", 
            "meta-llama.Llama-3.1-8B-Instruct"
        }
        self.strong_exam_takers_set_1 = {
            "Qwen.Qwen2.5-7B-Instruct", 
            "meta-llama.Llama-3.3-70B-Instruct", 
            "google.gemma-3-12b-it"
        }
        
        # Set 2 has different models
        self.weak_exam_takers_set_2 = {
            "google.gemma-2-9b-it", 
            "Qwen.Qwen2-7B-Instruct", 
            "mistralai.Ministral-8B-Instruct-2410"
        }
        self.strong_exam_takers_set_2 = {
            "Qwen.Qwen2.5-32B-Instruct", 
            "google.gemma-3-12b-it", 
            "mistralai.Mistral-Small-24B-Instruct-2501"
        }

    def read_jsonl_file(self, file_path: str) -> List[Dict]:
        """Read and return data from a JSONL file."""
        with jsonlines.open(file_path) as reader:
            return [item for item in reader]

    def extract_sft_data(self, formatted_data: List[Dict], unformatted_data: List[Dict]) -> Tuple[List, List, Dict]:
        """Extract and process SFT data for comparison - preserving original logic exactly."""
        # Extract formatted data - using lists as in original
        judge_formatted_data = []
        for item in formatted_data:
            judge_prompt = item["conversations"][0]["value"]
            judge_response = item["conversations"][1]["value"]
            judge_formatted_data.append((judge_prompt, judge_response))
        
        # Extract unformatted data - using lists as in original
        judge_unformatted_data = []
        label_count = {}
        for item in unformatted_data:
            judge_prompt = item["judge_prompt"]
            judge_response = item["correct_response"]
            judge_label = item["label"]
            if judge_label not in label_count:
                label_count[judge_label] = 0
            label_count[judge_label] += 1
            judge_unformatted_data.append((judge_prompt, judge_response))
        
        return judge_formatted_data, judge_unformatted_data, label_count

    def extract_dpo_data(self, formatted_data: List[Dict], unformatted_data: List[Dict]) -> Tuple[List, List]:
        """Extract and process DPO data for comparison - preserving original logic exactly."""
        # Extract formatted data - using lists as in original
        judge_formatted_data = []
        for item in formatted_data:
            judge_prompt = item["question"]
            judge_chosen_response = item["chosen"]
            judge_rejected_response = item["rejected"]
            judge_formatted_data.append((judge_prompt, judge_chosen_response, judge_rejected_response))
        
        # Extract unformatted data - using lists as in original
        judge_unformatted_data = []
        for item in unformatted_data:
            judge_prompt = item["judge_prompt"]
            judge_chosen_response = item["correct_response"]
            judge_rejected_response = item["incorrect_response"]
            judge_unformatted_data.append((judge_prompt, judge_chosen_response, judge_rejected_response))
        
        return judge_formatted_data, judge_unformatted_data
    
    def extract_eval_data(self, formatted_data: List[Dict], unformatted_data: List[Dict]) -> Tuple[List, List]:
        """Extract and process eval data for comparison - preserving original logic exactly."""
        # Extract formatted data - using lists as in original
        judge_formatted_data = []
        for item in formatted_data:
            judge_prompt = item["prompt"]
            judge_label = item["label"]
            judge_formatted_data.append((judge_prompt.strip(), judge_label))
        
        # Extract unformatted data - using lists as in original
        judge_unformatted_data = []
        for item in unformatted_data:
            judge_prompt = item["judge_prompt"]
            judge_label = item["label"]
            judge_unformatted_data.append((judge_prompt.strip(), judge_label))
        
        return judge_formatted_data, judge_unformatted_data

    def compute_intersection_stats(self, set1: List, set2: List, set1_name: str, set2_name: str) -> None:
        """Compute and print intersection statistics between two lists - preserving original logic exactly."""
        # Convert to sets for intersection operations as in original
        set1_set = set(set1)
        set2_set = set(set2)
        
        print(f"Total unique examples in {set1_name}: {len(set1_set)}")
        print(f"Total examples in {set1_name} but not in {set2_name}: {len(set1_set - set2_set)}")
        print(f"Total unique examples in {set2_name}: {len(set2_set)}")
        print(f"Total examples in {set2_name} but not in {set1_name}: {len(set2_set - set1_set)}")

    def analyze_sft_overlap(self, formatted_data: List[Dict], unformatted_data: List[Dict]) -> None:
        """Analyze overlap between SFT formatted and unformatted data - preserving original logic exactly."""
        judge_formatted_data, judge_unformatted_data, label_count = self.extract_sft_data(formatted_data, unformatted_data)
        
        print("=== SFT Data Overlap Analysis ===")
        self.compute_intersection_stats(judge_formatted_data, judge_unformatted_data, "formatted data", "unformatted data")
        
        # Calculate label percentages exactly as in original
        label_count["total"] = sum(label_count.values())
        label_count["A"] = round(100.0 * label_count["A"] / label_count["total"], 2)
        label_count["B"] = round(100.0 * label_count["B"] / label_count["total"], 2)

        print(json.dumps(label_count, indent=4))

    def analyze_dpo_overlap(self, formatted_data: List[Dict], unformatted_data: List[Dict]) -> None:
        """Analyze overlap between DPO formatted and unformatted data - preserving original logic exactly."""
        judge_formatted_data, judge_unformatted_data = self.extract_dpo_data(formatted_data, unformatted_data)
        
        print("=== DPO Data Overlap Analysis ===")
        self.compute_intersection_stats(judge_formatted_data, judge_unformatted_data, "formatted data", "unformatted data")

    def analyze_eval_overlap(self, formatted_data: List[Dict], unformatted_data: List[Dict]) -> None:
        """Analyze overlap between eval formatted and unformatted data - preserving original logic exactly."""
        judge_formatted_data, judge_unformatted_data = self.extract_eval_data(formatted_data, unformatted_data)
        
        print("=== Eval Data Overlap Analysis ===")
        self.compute_intersection_stats(judge_formatted_data, judge_unformatted_data, "formatted data", "unformatted data")

    def analyze_formatted_unformatted_overlap(self) -> None:
        """Analyze overlap between all formatted and unformatted file pairs."""
        print("🔍 Analyzing overlap between formatted and unformatted data...")
        
        for formatted_file, unformatted_file, algo_name, set_name in self.pair_files:
            formatted_file_path = self.formatted_folder / formatted_file
            unformatted_file_path = self.unformatted_folder / unformatted_file
            
            print(f"\n📁 Comparing: {formatted_file} vs {unformatted_file}")
            print(f"   Set: {set_name.upper()}")
            print(f"   Algorithm: {algo_name.upper()}")
            print(f"   Formatted: {formatted_file_path}")
            print(f"   Unformatted: {unformatted_file_path}")
            
            if not formatted_file_path.exists():
                print(f"❌ Formatted file not found: {formatted_file_path}")
                continue
            if not unformatted_file_path.exists():
                print(f"❌ Unformatted file not found: {unformatted_file_path}")
                continue
            
            formatted_data = self.read_jsonl_file(str(formatted_file_path))
            unformatted_data = self.read_jsonl_file(str(unformatted_file_path))
            
            if algo_name == "sft":
                self.analyze_sft_overlap(formatted_data, unformatted_data)
            elif algo_name == "dpo":
                self.analyze_dpo_overlap(formatted_data, unformatted_data)
            elif algo_name == "eval":
                self.analyze_eval_overlap(formatted_data, unformatted_data)
            else:
                raise ValueError(f"Invalid algorithm name: {algo_name}")

    def get_eval_split_files(self, set_name: str = "set_1", res_type: str = "one_res") -> List[str]:
        """Get evaluation split files for a specific set and resolution type."""
        folder_path = self.unformatted_folder
        fnames = [
            f for f in os.listdir(folder_path) 
            if f.endswith(".jsonl") and set_name in f and res_type in f and "eval" in f
        ]
        return sorted(fnames)

    def compute_intersection_matrix(self, fnames: List[str], field: str) -> List[List[float]]:
        """Compute intersection ratios matrix for a specific field."""
        n = len(fnames)
        intersection_matrix = [[1.0 for _ in range(n)] for _ in range(n)]
        
        for i, fname_1 in enumerate(fnames):
            for j, fname_2 in enumerate(fnames):
                if i == j:
                    continue
                
                fname_1_path = self.unformatted_folder / fname_1
                fname_2_path = self.unformatted_folder / fname_2
                
                data_1 = self.read_jsonl_file(str(fname_1_path))
                data_2 = self.read_jsonl_file(str(fname_2_path))
                
                set_1 = set(item[field] for item in data_1)
                set_2 = set(item[field] for item in data_2)
                
                intersection = set_1.intersection(set_2)
                min_size = min(len(set_1), len(set_2))
                intersection_ratio = len(intersection) / min_size if min_size > 0 else 0.0
                
                intersection_matrix[i][j] = intersection_ratio
        
        return intersection_matrix

    def print_matrix(self, matrix: List[List[float]], title: str, fnames: List[str]) -> None:
        """Print a formatted matrix with row/column labels."""
        print(f"\n{title}")
        print("    " + " ".join(f"{i:4d}" for i in range(len(matrix[0]))))
        
        for i, row in enumerate(matrix):
            print(f"{i:2d} " + " ".join(f"{val:4.2f}" for val in row))
        
        print("\nFile mapping:")
        for idx, fname in enumerate(fnames):
            print(f"  {idx:2d}: {fname}")

    def analyze_eval_splits_overlap(self, set_name: str = "set_1", res_type: str = "one_res") -> None:
        """Analyze overlap between evaluation splits."""
        print(f"\n🔍 Analyzing evaluation splits overlap for {set_name} ({res_type})...")
        
        fnames = self.get_eval_split_files(set_name, res_type)
        if not fnames:
            print(f"No evaluation files found for {set_name} ({res_type})")
            return
        
        print(f"Found {len(fnames)} evaluation files")
        
        # Compute intersection matrices
        judge_prompt_matrix = self.compute_intersection_matrix(fnames, "judge_prompt")
        question_prompt_matrix = self.compute_intersection_matrix(fnames, "question")
        
        # Print results
        self.print_matrix(judge_prompt_matrix, "Judge Prompt Intersection Ratios", fnames)
        self.print_matrix(question_prompt_matrix, "Question Prompt Intersection Ratios", fnames)

    def analyze_model_distribution(self, set_name: str = "set_1", res_type: str = "one_res") -> None:
        """Analyze model distribution across data splits."""
        print(f"\n🤖 Analyzing model distribution for {set_name} ({res_type})...")
        
        folder_path = self.unformatted_folder
        fnames = [
            f for f in os.listdir(folder_path) 
            if f.endswith(".jsonl") and set_name in f and res_type in f
        ]
        
        for fname in fnames:
            print(f"\n📊 File: {fname}")
            print("=" * 80)
            
            file_path = folder_path / fname
            data = self.read_jsonl_file(str(file_path))
            
            # Count model occurrences
            model_counts = {}
            for item in data:
                model_name = item["exam_taker_model_name"]
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
            
            # Determine set and strength from filename
            if set_name == "set_1":
                if '_strong_' in fname:
                    expected_models = self.strong_exam_takers_set_1
                elif '_weak_' in fname:
                    expected_models = self.weak_exam_takers_set_1
                else:
                    print(f"⚠️  Warning: Could not determine model strength for {fname}")
                    continue
            elif set_name == "set_2":
                if '_strong_' in fname:
                    expected_models = self.strong_exam_takers_set_2
                elif '_weak_' in fname:
                    expected_models = self.weak_exam_takers_set_2
                else:
                    print(f"⚠️  Warning: Could not determine model strength for {fname}")
                    continue
            else:
                print(f"⚠️  Warning: Unknown set name: {set_name}")
                continue
            
            # Verify all models are in expected set
            unexpected_models = set(model_counts.keys()) - expected_models
            if unexpected_models:
                print(f"❌ Unexpected models found: {unexpected_models}")
                continue
            
            # Print distribution
            total_count = sum(model_counts.values())
            for model_name, count in sorted(model_counts.items()):
                percentage = (count / total_count) * 100
                print(f"{model_name}: {count} ({percentage:.2f}%)")

    def run_full_analysis(self) -> None:
        """Run the complete overlap analysis."""
        print("🚀 Starting comprehensive overlap analysis...")
    
        print("🔍 Analyzing unformatted overlap across unformatted data splits...")

        # 1. Analyze unformatted evaluation splits distribution
        self.analyze_eval_splits_overlap("set_1", "one_res")
        self.analyze_eval_splits_overlap("set_2", "one_res")
        
        # 2. Analyze unformatted evaluation splits distribution
        self.analyze_model_distribution("set_1", "one_res")
        self.analyze_model_distribution("set_2", "one_res")

        print("🔍 Analyzing formatted vs unformatted overlap...")

        # 3. Analyze formatted vs unformatted overlap
        self.analyze_formatted_unformatted_overlap()
        
        print("\n✅ Analysis completed!")


def main():
    """Main function to run the overlap analysis."""
    # Check working directory first
    if not check_working_directory():
        exit(1)
    
    analyzer = OverlapAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()




        





