from json_repair import repair_json

def check_text_for_verdict_a(text):
    """Return True if text contains a correctly formatted verdict A."""
    return '{"verdict": "A"}' in text

def check_text_for_verdict_b(text):
    """Return True if text contains a correctly formatted verdict B."""
    return '{"verdict": "B"}' in text

def extract_json_from_codeblock(text):
    """Extract JSON content from a code block if present."""
    if '```json' in text:
        return '```json' + text.split('```json', 1)[1]
    return text

def process_label_dict(label_dict):
    """
    Process the label dictionary to extract a valid verdict.
    If the result is an empty list, return 'C'.
    If the result is a list, try to find a dict with a valid verdict.
    """
    if isinstance(label_dict, list):
        if not label_dict:
            return 'C'
        for item in reversed(label_dict):
            if isinstance(item, dict) and item.get("verdict") in ['A', 'B']:
                return item
        return label_dict[-1]
    return label_dict

def validate_verdict_dict(label_dict):
    """Return True if label_dict is a dict with a valid verdict ('A' or 'B')."""
    return (
        isinstance(label_dict, dict)
        and label_dict.get("verdict") in ['A', 'B']
    )

def extract_verdict_from_dict(label_dict):
    """Extract the verdict value from a validated label_dict."""
    return label_dict["verdict"]

def get_label(response):
    """
    Attempt to extract the verdict label from the response.
    Returns 'A', 'B', or 'C' (cannot parse).
    """
    response = response[-2000:] # only consider the last 2000 characters of the response.
    text = extract_json_from_codeblock(response)

    try:
        label_dict = repair_json(text, return_objects=True)
        label_dict = process_label_dict(label_dict)
    except Exception as e:
        # Catch RecursionError, ValueError, etc.
        print(f"[DEBUG] repair_json failed: {e}")
        return 'C'

    if not validate_verdict_dict(label_dict):
        if check_text_for_verdict_a(text):
            return 'A'
        if check_text_for_verdict_b(text):
            return 'B'
        return 'C'
    return extract_verdict_from_dict(label_dict)

def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None
):
    """
    Reward function for judge responses.
    Returns:
        1.0   if parsed and correct,
        0.0001 if parsed but wrong verdict,
        0.0   if unparsed.
    """
    pred = get_label(solution_str)
    if pred == ground_truth:
        return 1.0
    if pred in ['A', 'B']:
        return 0.0001
    return 0.0


if __name__ == "__main__":
    solution_str = """
Let's evaluate the responses step-by-step.\n\n**Response A:**\n\n1. **Understanding Relationships:** Correctly states the formulas for area and circumference and relates diameter to radius.\n2. **Setting up the Inequality:** Correctly sets up the inequality \\( \\pi r^2 < 2\\pi r \\).\n3. **Simplifying the Inequality:** Correctly simplifies to \\( r < 2 \\).\n4. **Consider Dice Rolls:** Correctly identifies the possible sums of two dice as ranging from 2 to 12.\n5. **Finding Favorable Outcomes:** Incorrectly concludes that the sum must be less than 4, considering sums 2 and 3 only.\n6. **Calculating Probabilities:** Incorrectly calculates the probability as 1/18 due to the earlier mistake in identifying favorable outcomes.\n\n**Response B:**\n\n1. **Determine Possible Sums:** Correctly identifies the possible sums from 2 to 12.\n2. **Formulating Relationships:** Correctly sets up the diameter as \\( s \\) and uses it to express area and circumference.\n3. **Setting up the Inequality:** Correctly sets up \\( \\pi (s/2)^2 < \\pi s \\).\n4. **Simplifying and Solving the Inequality:** Correctly simplifies to \\( s(s - 4) < 0 \\), concluding \\( 0 < s < 4 \\).\n5. **Identifying Favorable Outcomes:** Incorrectly identifies the favorable outcomes as 1, 2, and 3. The sum of dice cannot be 1.\n6. **Calculating Probability:** Incorrectly calculates the probability as 1/12 due to the mistake in identifying favorable outcomes.\n\n**Final Evaluation:**\n\nBoth responses contain errors. Response A incorrectly identifies the favorable outcomes and probability due to a misunderstanding of the inequality's implication on possible sums. Response B correctly sets up and solves the inequality but mistakenly includes an impossible sum (1) in the favorable outcomes. However, Response B's logical process leading to the inequality is correct, and it has one more favorable outcome than Response A, leading to a closer probability to the correct one.\n\nThus, based on the logical process and the closeness to the correct answer, **Response B** is better.\n\n```json\n{\"verdict\": \"B\"}\n```
"""
    print(compute_reward(data_source="", solution_str=solution_str.strip(), ground_truth="B"))
