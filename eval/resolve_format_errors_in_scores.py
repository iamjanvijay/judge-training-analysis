import jsonlines
from json_repair import repair_json
import os
import json

def check_text_for_verdict_a(text):
    """Check if text contains indicators for verdict A."""
    return (
        '{"verdict": "A"}' in text
        or '{"verdict": "Response A"}' in text
        or "Response A is better" in " ".join(text.replace("*", "").split())
        or "Response A is the better" in " ".join(text.replace("*", "").split())
        or "Response A is than better" in " ".join(text.replace("*", "").split())
        or "better response is Response A" in " ".join(text.replace("*", "").split())
    )

def check_text_for_verdict_b(text):
    """Check if text contains indicators for verdict B."""
    return (
        '{"verdict": "B"}' in text
        or '{"verdict": "Response B"}' in text
        or "Response B is better" in " ".join(text.replace("*", "").split())
        or "Response B is the better" in " ".join(text.replace("*", "").split())
        or "Response B is than better" in " ".join(text.replace("*", "").split())
        or "better response is Response B" in " ".join(text.replace("*", "").split())
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

def get_label_updated(response):
    """Keep modifying this function to better parse the responses, and better handle the output formatting errors from the LLM."""
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

def get_label(response):
    """Extract label using the original method."""
    # Take the last 50 characters of the response
    text = response[-50:]
    # If a code block with json is present, extract from there
    text = extract_json_from_codeblock(text)
    
    # Attempt to repair and parse the JSON
    label_dict = repair_json(text, return_objects=True)

    # If the result is an empty list, return 'C'
    if isinstance(label_dict, list) and not label_dict:
        return 'C'

    # If the result is a list, take the last element
    if isinstance(label_dict, list):
        label_dict = label_dict[-1]

    # If the result is not a dict with a valid verdict, return 'C'
    if not (
        isinstance(label_dict, dict)
        and "verdict" in label_dict
        and label_dict["verdict"] in ['A', 'B']
    ):
        return 'C'

    # Otherwise, return the predicted label
    return label_dict["verdict"]

def process_single_response(response, label):
    """Process a single response and return accuracy metrics."""
    response_label_updated = get_label_updated(response)
    response_label = get_label(response)

    # with the original get_label function
    correct = (label == response_label)
    incorrect_format = (response_label == "C")

    # with the updated get_label function
    correct_updated = (label == response_label_updated)
    incorrect_format_updated = (response_label_updated == "C")

    return correct, incorrect_format, correct_updated, incorrect_format_updated

def calculate_consistent_accuracy(correct_list, correct_list_updated):
    """Calculate consistent accuracy for both original and updated methods."""
    consistent_correct, consistent_incorrect = 0, 0
    consistent_correct_updated, consistent_incorrect_updated = 0, 0
    
    for i in range(0, len(correct_list), 2):
        # with the original get_label function
        if correct_list[i] and correct_list[i+1]:
            consistent_correct += 1
        else:
            consistent_incorrect += 1

        # with the updated get_label function
        if correct_list_updated[i] and correct_list_updated[i+1]:
            consistent_correct_updated += 1
        else:
            consistent_incorrect_updated += 1

    return consistent_correct, consistent_incorrect, consistent_correct_updated, consistent_incorrect_updated

def compute_accuracy(path):
    """Compute accuracy metrics for all responses in a file."""
    correct, incorrect_format, incorrect, correct_list = 0.0, 0.0, 0.0, []
    correct_updated, incorrect_format_updated, incorrect_updated, correct_list_updated = 0.0, 0.0, 0.0, []
    
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            response, label = obj["response"], obj["label"]

            # Process single response
            single_correct, single_incorrect_format, single_correct_updated, single_incorrect_format_updated = process_single_response(response, label)

            # Accumulate results
            if single_correct:
                correct += 1
                correct_list.append(True)
            else:
                incorrect += 1
                correct_list.append(False)

            if single_correct_updated:
                correct_updated += 1
                correct_list_updated.append(True)
            else:
                incorrect_updated += 1
                correct_list_updated.append(False)

            if single_incorrect_format:
                incorrect_format += 1
            if single_incorrect_format_updated:
                incorrect_format_updated += 1

    # Calculate consistent accuracy
    consistent_correct, consistent_incorrect, consistent_correct_updated, consistent_incorrect_updated = calculate_consistent_accuracy(correct_list, correct_list_updated)

    return correct, incorrect_format, incorrect, consistent_correct, consistent_incorrect, \
        correct_updated, incorrect_format_updated, incorrect_updated, consistent_correct_updated, consistent_incorrect_updated

def calculate_metrics(correct, incorrect, correct_updated, incorrect_updated, 
                     incorrect_format, incorrect_format_updated,
                     consistent_correct, consistent_incorrect,
                     consistent_correct_updated, consistent_incorrect_updated):
    """Calculate all the accuracy and format rate metrics."""
    accuracy = (float(correct) / (correct + incorrect))
    updated_accuracy = (float(correct_updated) / (correct_updated + incorrect_updated))

    consistent_accuracy = (float(consistent_correct) / (consistent_correct + consistent_incorrect))
    updated_consistent_accuracy = (float(consistent_correct_updated) / (consistent_correct_updated + consistent_incorrect_updated))

    incorrect_format_rate = (float(incorrect_format) / (correct + incorrect))
    updated_incorrect_format_rate = (float(incorrect_format_updated) / (correct_updated + incorrect_updated))

    return (accuracy, updated_accuracy, consistent_accuracy, updated_consistent_accuracy,
            incorrect_format_rate, updated_incorrect_format_rate)

def print_metrics(file, accuracy, updated_accuracy, consistent_accuracy, updated_consistent_accuracy,
                 incorrect_format_rate, updated_incorrect_format_rate):
    """Print the computed metrics for a file."""
    print(f"File: {file}")
    print(">>> Accuracy Increment: ", updated_accuracy - accuracy)  
    print(">>> Consistent Accuracy Increment: ", updated_consistent_accuracy - consistent_accuracy)
    if incorrect_format_rate - updated_incorrect_format_rate > 0.05:
        print("\033[91m>>> Incorrect Format Rate Decrement: ", incorrect_format_rate - updated_incorrect_format_rate, "\033[0m")
    else:
        print(">>> Incorrect Format Rate Decrement: ", incorrect_format_rate - updated_incorrect_format_rate)
    print("--------------------------------")

def update_scores_file(scores_filepath, scores, correct_updated, incorrect_updated, 
                      incorrect_format_updated, updated_accuracy, updated_incorrect_format_rate,
                      consistent_correct_updated, consistent_incorrect_updated, updated_consistent_accuracy):
    """Update the scores file with new metrics."""
    scores["correct"] = correct_updated
    scores["incorrect"] = incorrect_updated
    scores["incorrect_format"] = incorrect_format_updated
    scores["accuracy"] = updated_accuracy
    scores["incorrect_format_rate"] = updated_incorrect_format_rate
    scores["consistent_correct"] = consistent_correct_updated
    scores["consistent_incorrect"] = consistent_incorrect_updated
    scores["consistent_accuracy"] = updated_consistent_accuracy

    with open(scores_filepath, "w") as f:
        json.dump(scores, f)

def process_single_file(file, folder_path, overwrite_scores):
    """Process a single file to compute and update accuracy metrics."""
    # read the responses from the file
    responses_filepath = os.path.join(folder_path, file)
    correct, incorrect_format, incorrect, consistent_correct, consistent_incorrect, \
    correct_updated, incorrect_format_updated, incorrect_updated, consistent_correct_updated, consistent_incorrect_updated \
            = compute_accuracy(responses_filepath)
    
    # read the corresponding scores file and update the scores
    scores_filepath = os.path.join(folder_path, file.replace(".jsonl", ".json"))
    with open(scores_filepath, "r") as f:
        scores = json.load(f)

    # Calculate metrics
    (accuracy, updated_accuracy, consistent_accuracy, updated_consistent_accuracy,
     incorrect_format_rate, updated_incorrect_format_rate) = calculate_metrics(
        correct, incorrect, correct_updated, incorrect_updated,
        incorrect_format, incorrect_format_updated,
        consistent_correct, consistent_incorrect,
        consistent_correct_updated, consistent_incorrect_updated
    )

    # Print metrics
    print_metrics(file, accuracy, updated_accuracy, consistent_accuracy, updated_consistent_accuracy,
                 incorrect_format_rate, updated_incorrect_format_rate)

    # Update scores file if requested
    if overwrite_scores:
        update_scores_file(scores_filepath, scores, correct_updated, incorrect_updated,
                          incorrect_format_updated, updated_accuracy, updated_incorrect_format_rate,
                          consistent_correct_updated, consistent_incorrect_updated, updated_consistent_accuracy)

def main():
    """Main function to process all files in the folder."""
    folder_path = "/shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis/eval-results"
    filenames = [filename for filename in os.listdir(folder_path) if filename.endswith(".jsonl")]

    overwrite_scores = True
    for file in sorted(filenames):
        # read the responses from the file
        responses_filepath = os.path.join(folder_path, file)
        correct, incorrect_format, incorrect, consistent_correct, consistent_incorrect, \
        correct_updated, incorrect_format_updated, incorrect_updated, consistent_correct_updated, consistent_incorrect_updated \
                = compute_accuracy(responses_filepath)
        
        # read the corresponding scores file and update the scores
        scores_filepath = os.path.join(folder_path, file.replace(".jsonl", ".json"))
        with open(scores_filepath, "r") as f:
            scores = json.load(f)

        accuracy = (float(correct) / (correct + incorrect))
        updated_accuracy = (float(correct_updated) / (correct_updated + incorrect_updated))

        consistent_accuracy = (float(consistent_correct) / (consistent_correct + consistent_incorrect))
        updated_consistent_accuracy = (float(consistent_correct_updated) / (consistent_correct_updated + consistent_incorrect_updated))

        incorrect_format_rate = (float(incorrect_format) / (correct + incorrect))
        updated_incorrect_format_rate = (float(incorrect_format_updated) / (correct_updated + incorrect_updated))

        print(f"File: {file}")
        print(">>> Accuracy Increment: ", updated_accuracy - accuracy)  
        print(">>> Consistent Accuracy Increment: ", updated_consistent_accuracy - consistent_accuracy)
        if incorrect_format_rate - updated_incorrect_format_rate > 0.05:
            print("\033[91m>>> Incorrect Format Rate Decrement: ", incorrect_format_rate - updated_incorrect_format_rate, "\033[0m")
        else:
            print(">>> Incorrect Format Rate Decrement: ", incorrect_format_rate - updated_incorrect_format_rate)
        print("-" * 50)

        if overwrite_scores:
            scores["correct"] = correct_updated
            scores["incorrect"] = incorrect_updated
            scores["incorrect_format"] = incorrect_format_updated
            scores["accuracy"] = updated_accuracy
            scores["incorrect_format_rate"] = updated_incorrect_format_rate
            scores["consistent_correct"] = consistent_correct_updated
            scores["consistent_incorrect"] = consistent_incorrect_updated
            scores["consistent_accuracy"] = updated_consistent_accuracy

            with open(scores_filepath, "w") as f:
                json.dump(scores, f)
            print(f"Updated scores file: {scores_filepath}")
            print("+" * 50)

if __name__ == "__main__":
    main()
