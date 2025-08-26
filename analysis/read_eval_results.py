import json
import os


def file_name_to_eval_meta(fname):
    set_name, train_name, model_name, split_name, _ = fname.split('.')

    if train_name == "train_zero":
        ckpt_steps = [0, 0, 0, 0]
        train_algos = ["sft", "dpo", "sft_dpo", "grpo"]
        train_caps = ["weak", "strong"]
    else:
        ckpt_steps = [int(train_name.split('-')[-1])]
        if "_sft_dpo_" in train_name:
            train_algos = ["sft_dpo"]
        elif "_sft_" in train_name:
            train_algos = ["sft"]
        elif "_dpo_" in train_name:
            train_algos = ["dpo"]
        elif "_grpo_" in train_name:
            train_algos = ["grpo"]
        else:
            raise ValueError(f"Unknown train name: {train_name}")
        assert "weak" in train_name or "strong" in train_name
        train_caps = ["weak" if "weak" in train_name else "strong"]
        
    split_category, split_capability, split_type = \
        split_name.split('_')[0], split_name.split('_')[1], '_'.join(split_name.split('_')[2:])
    assert split_category == "eval"

    return set_name, train_algos, train_caps, ckpt_steps, model_name, split_capability, split_type

def assert_correctness(set_name, train_algo, train_cap, ckpt_step, model_name, split_capability, split_type, scores):
    if ckpt_step != 0:
        assert set_name in scores["model"]
        assert train_algo == scores["model"].split('/')[2].split('.')[0]
        assert ckpt_step == int(scores["model"].split('/')[3].split('-')[1])
        assert model_name == scores["model"].split('/')[2].split('.')[2]
        assert train_cap in scores["model"]

    assert set_name in scores["input_file"] and set_name in scores["output_file"]

    assert split_capability in scores["input_file"].split('/')[4]
    assert split_type in scores["input_file"].split('/')[4]

def read_eval_results(path):
    agg_scores_dict = {}

    score_fpaths = [f for f in os.listdir(path) if f.endswith(".json")]
    for score_fpath in score_fpaths:
        with open(os.path.join(path, score_fpath), "r") as f:
            set_name, train_algos, train_caps, ckpt_steps, model_name, eval_cap, eval_type = file_name_to_eval_meta(score_fpath)
            scores = json.load(f)

            for train_algo, train_cap, ckpt_step in zip(train_algos, train_caps, ckpt_steps):
                assert_correctness(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type, scores)

            for train_algo, train_cap, ckpt_step in zip(train_algos, train_caps, ckpt_steps):
                assert (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) not in agg_scores_dict, f"Duplicate score file: {score_fpath} \n\n {(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)}"
                # print((set_name, train_algo, train_cap, ckpt_step, model_name, split_capability, split_type))
                # ('set_2', 'sft', 'weak', 0, 'llama8b', 'strong', 'seen_questions_seen_answers')
                agg_scores_dict[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)] = scores

    return agg_scores_dict