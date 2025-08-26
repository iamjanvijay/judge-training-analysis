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
    correct_updated = (label == response_label_updated)
    incorrect_format_updated = (response_label_updated == "C")

    return correct_updated, incorrect_format_updated

def print_format_errors(path):    
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            response, label = obj["response"], obj["label"]

            # Process single response
            single_correct_updated, single_incorrect_format_updated = process_single_response(response, label)

            # Accumulate results
            if single_incorrect_format_updated:
                print("Response: ", response[-1000:])
                print("+" * 100)
                print("Label: ", label)
                print("-" * 100)

def main():
    folder_path = "/shared/storage-01/users/jvsingh2/sf-intern/github/judge-training-analysis/eval-results"
    filename = "set_1.train_dpo_strong_checkpoint-2800.mistral24b.eval_weak_unseen_questions_unseen_answers.jsonl"
    responses_filepath = os.path.join(folder_path, filename)

    print_format_errors(responses_filepath)

if __name__ == "__main__":
    main()
