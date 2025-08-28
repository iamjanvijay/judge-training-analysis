# Standard library imports
import json
import os
import sys

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Local imports
sys.path.append("./analysis")
from read_eval_results import read_eval_results


def plot_six_block_advantage(df, metric_type, degradation_type, save_path=None):
    """
    2x3 grid of barplots.
    Y-axis labels include the direction per row:
      - Row 0: Degradation (weak→strong)
      - Row 1: Degradation (strong→weak)
    Legend is above the grid. Bars annotated with rounded values.
    """

    row_gen = ["weak->strong", "strong->weak"]
    col_models = ["llama8b", "ministral8b", "mistral24b"]

    pairs = [
        ("sft",      "seen_questions_unseen_answers"),
        ("sft",      "unseen_questions_unseen_answers"),
        ("dpo",      "seen_questions_unseen_answers"),
        ("dpo",      "unseen_questions_unseen_answers"),
        ("sft_dpo",  "seen_questions_unseen_answers"),
        ("sft_dpo",  "unseen_questions_unseen_answers"),
    ]

    # base colors (no explicit colors; use default cycle)
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(prop_cycle) < 3:
        prop_cycle = prop_cycle + [f"C{i}" for i in range(10)]
    algo_colors = {"sft": prop_cycle[0], "dpo": prop_cycle[1], "sft_dpo": prop_cycle[2]}

    def _lookup_value(gen_type, model, algo, split):
        sel = df[
            (df["generalisation_type"] == gen_type) &
            (df["model_name"] == model) &
            (df["train_algo"] == algo) &
            (df["eval_split"] == split)
        ]["value"].values
        
        if len(sel) == 0:
            raise ValueError(
                f"Zero rows found for gen_type={gen_type}, "
                f"model={model}, algo={algo}, split={split}. "
                f"Expected exactly 1."
            )
        if len(sel) > 1:
            raise ValueError(
                f"Multiple rows found for gen_type={gen_type}, "
                f"model={model}, algo={algo}, split={split}. "
                f"Expected exactly 1."
            )
        return float(sel[0])

    # global ylim across all panels for comparability
    all_vals = []
    for r in row_gen:
        for c in col_models:
            for algo, split in pairs:
                all_vals.append(_lookup_value(r, c, algo, split))
    global_max = np.nanmax(all_vals) if len(all_vals) else 1.0
    ylim_top = 1.2 * (global_max if np.isfinite(global_max) else 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, constrained_layout=True)
    bar_positions = np.arange(6)
    bar_width = 0.75

    for i, gen in enumerate(row_gen):
        for j, model in enumerate(col_models):
            ax = axes[i, j]
            for k, (algo, split) in enumerate(pairs):
                val = _lookup_value(gen, model, algo, split)
                alpha = 0.5 if split == "seen_questions_unseen_answers" else 1.0
                ax.bar(
                    bar_positions[k], val,
                    width=bar_width,
                    color=algo_colors[algo],
                    alpha=alpha,
                    edgecolor='black', linewidth=0.6
                )
                # annotate value
                if not np.isnan(val) and abs(val) > 1e-3:
                    ax.text(
                        bar_positions[k], val + ylim_top*0.015,
                        str(round(val, 2)),
                        ha="center", va="bottom", fontsize=8
                    )

            ax.set_ylim(0, ylim_top)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(["sft-s","sft-u","dpo-s","dpo-u","sft_dpo-s","sft_dpo-u"],
                               fontsize=8, rotation=20)

            # per-row y-axis label with direction
            if j == 0:
                ylab = "Degradation (weak\u2192strong)" if gen == "weak->strong" else "Degradation (strong\u2192weak)"
                ax.set_ylabel(ylab, fontsize=9)

            if i == 0:
                ax.set_title(model, fontsize=11, weight="bold")

            ax.grid(axis="y", linestyle="--", alpha=0.25)

    # legend above
    legend_handles = [
        Patch(facecolor=algo_colors["sft"], edgecolor='black', alpha=0.5, label="sft (seen questions)"),
        Patch(facecolor=algo_colors["sft"], edgecolor='black', alpha=1.0, label="sft (unseen questions)"),
        Patch(facecolor=algo_colors["dpo"], edgecolor='black', alpha=0.5, label="dpo (seen questions)"),
        Patch(facecolor=algo_colors["dpo"], edgecolor='black', alpha=1.0, label="dpo (unseen questions)"),
        Patch(facecolor=algo_colors["sft_dpo"], edgecolor='black', alpha=0.5, label="sft_dpo (seen questions)"),
        Patch(facecolor=algo_colors["sft_dpo"], edgecolor='black', alpha=1.0, label="sft_dpo (unseen questions)"),
    ]
    fig.legend(handles=legend_handles, ncols=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)

    fig.suptitle(
        f"Degradation by Train Capability Shift Direction and Model\n"
        f"Metric: {metric_type.replace('_', ' ').title()} | Degradation: {degradation_type.replace('_', ' ').title()}",
        y=1.12, fontsize=13, weight="bold"
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def compute_advantage_metrics_util(scores, eval_cap, split_types, metrics, degradation_types, advantage_types):
    """
    example of a advantage metric:  (Acc(train=strong, eval=weak) - Acc(train=weak, eval=weak)) / (Acc(train=weak,eval=weak))
    """
    advantage_metrics = {}
    
    train_baseline_cap = eval_cap
    train_comparator_cap = "weak" if eval_cap == "strong" else "strong"
    advantage_type = f"{train_baseline_cap}->{train_comparator_cap}"
    assert advantage_type in advantage_types, f"Advantage type {advantage_type} not in {advantage_types}"

    for eval_type in split_types:
        if eval_type not in advantage_metrics:
            advantage_metrics[eval_type] = {}

        for accuracy_type in metrics:
            if accuracy_type not in advantage_metrics[eval_type]:
                advantage_metrics[eval_type][accuracy_type] = {}

            baseline = scores[train_baseline_cap][eval_type][accuracy_type] # eval-capability same as train-capability. Acc(train=weak, eval=weak)
            comparator = scores[train_comparator_cap][eval_type][accuracy_type] # eval-capability different from train-capability. Acc(train=weak, eval=strong)
            
            absolute_degradation = 100.0 * (baseline - comparator) # ideally this should be >= 0. since baseline should be >= comparator in our case.
            relative_degradation = absolute_degradation / baseline # ideally this should be >= 0 too.

            advantage_metrics[eval_type][accuracy_type] = {
                "absolute_degradation": absolute_degradation,
                "relative_degradation": relative_degradation
            }
            for degradation_type in advantage_metrics[eval_type][accuracy_type]:
                assert degradation_type in degradation_types, f"Degradation type {degradation_type} not in {degradation_types}"

    return advantage_type, advantage_metrics

def compute_advantage_metrics(agg_scores, split_types, metrics, degradation_types, advantage_types):

    # Reorganize the scores into a dictionary of train_meta_to_eval_scores[train_meta][train_cap][eval_cap][eval_type] -> score
    train_meta_to_eval_scores = {}
    for (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) in agg_scores:
        meta = (set_name, train_algo, ckpt_step, model_name)
        if meta not in train_meta_to_eval_scores:
            train_meta_to_eval_scores[meta] = {}

        if eval_cap not in train_meta_to_eval_scores[meta]:
            train_meta_to_eval_scores[meta][eval_cap] = {}
        if train_cap not in train_meta_to_eval_scores[meta][eval_cap]:
            train_meta_to_eval_scores[meta][eval_cap][train_cap] = {}
        if eval_type not in train_meta_to_eval_scores[meta][eval_cap][train_cap]:
            train_meta_to_eval_scores[meta][eval_cap][train_cap][eval_type] = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)]

    # All eval types: {'seen_questions_seen_answers', 'seen_questions_unseen_answers', 'unseen_questions_unseen_answers'}
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200} # for llama series of models we trained longer until 4200 steps for rest of the models we trained until 2800 steps.

    # Compute the generalization metrics, at the last ckpt step.
    adv_meta_to_adv_scores = {}
    for train_meta in train_meta_to_eval_scores:
        set_name, train_algo, ckpt_step, model_name = train_meta
        if ckpt_step != last_ckpt_steps[model_name]: # if not the last ckpt step, skip
            continue

        for eval_cap in ["weak", "strong"]:
            advantage_type, advantage_metrics = compute_advantage_metrics_util(train_meta_to_eval_scores[train_meta][eval_cap], eval_cap, split_types, metrics, degradation_types, advantage_types)
            advantage_meta = (train_algo, model_name, advantage_type)
            assert advantage_meta not in adv_meta_to_adv_scores, f"Advantage meta {advantage_meta} already exists"
            adv_meta_to_adv_scores[advantage_meta] = advantage_metrics

    return adv_meta_to_adv_scores

def create_dataframe_for_plots(gen_meta_to_gen_scores):

    #### STEP 1: Convert the gen_meta_to_gen_scores to a pandas dataframe
    rows = []
    for (train_algo, model_name, gen_type) in gen_meta_to_gen_scores:
        for split in ["seen_questions_unseen_answers", "unseen_questions_unseen_answers"]:
            for metric in ["accuracy", "consistent_accuracy"]:
                for deg_type in ["absolute_degradation", "relative_degradation"]:
                    value = gen_meta_to_gen_scores[(train_algo, model_name, gen_type)][split][metric][deg_type]
                    rows.append({
                        "train_algo": train_algo,
                        "model_name": model_name,
                        "generalisation_type": gen_type,
                        "eval_split": split,
                        "metric": metric,
                        "degradation_type": deg_type,
                        "value": value
                    })

    df = pd.DataFrame(rows)
    return df

def create_generalization_plots(df, plot_dir, model_names, split_types, metrics, degradation_types, generalization_types):
    for current_metric in metrics:
        for current_degradation_type in degradation_types:
            # Filter the dataframe for the current metric and degradation type.
            filtered_df = df[(df["metric"] == current_metric) & (df["degradation_type"] == current_degradation_type)].drop(columns=["metric", "degradation_type"])
            plot_six_block_advantage(filtered_df, current_metric, current_degradation_type, save_path=os.path.join(plot_dir, f"advantage_{current_metric}_{current_degradation_type}.png"))
            
def main():
    agg_scores = read_eval_results("./eval-results")

    model_names = ["ministral8b", "mistral24b", "llama8b"]
    split_types = ["seen_questions_unseen_answers", "unseen_questions_unseen_answers"]
    advantage_types = ["weak->strong", "strong->weak"]
    metrics = ["accuracy", "consistent_accuracy"]
    degradation_types = ["absolute_degradation", "relative_degradation"]

    plot_dir = "./eval-plots/plot_type_4"
    os.makedirs(plot_dir, exist_ok=True)

    gen_meta_to_gen_scores = compute_advantage_metrics(agg_scores, split_types, metrics, degradation_types, advantage_types)
    df = create_dataframe_for_plots(gen_meta_to_gen_scores)
    create_generalization_plots(df, plot_dir, model_names, split_types, metrics, degradation_types, advantage_types)

    # Save the dataframe as a CSV file in the plot_dir
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)


if __name__ == "__main__":
    main()