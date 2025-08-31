import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import defaultdict

# Local imports
sys.path.append("./analysis")
from read_eval_results import read_eval_results


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_six_block_generalisation(df, metric_type, degradation_type, save_path=None):
    """
    2x3 grid of barplots with support for negative values.
    Rows:
      - Row 0: Generalisation (weak→strong)
      - Row 1: Generalisation (strong→weak)
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

    # colors per algo (use default cycle)
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(prop_cycle) < 3:
        prop_cycle += [f"C{i}" for i in range(10)]
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
                f"Zero rows found for gen_type={gen_type}, model={model}, "
                f"algo={algo}, split={split}. Expected exactly 1."
            )
        if len(sel) > 1:
            raise ValueError(
                f"Multiple rows found for gen_type={gen_type}, model={model}, "
                f"algo={algo}, split={split}. Expected exactly 1."
            )
        return float(sel[0])

    # ----- global y-limits (allow negatives) -----
    all_vals = []
    for r in row_gen:
        for c in col_models:
            for algo, split in pairs:
                all_vals.append(_lookup_value(r, c, algo, split))

    if all_vals:
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)
        # Headroom on both sides
        ylim_bottom = 1.2 * vmin if np.isfinite(vmin) else -1.0
        ylim_top    = 1.2 * vmax if np.isfinite(vmax) else 1.0
        if vmin >= 0: ylim_bottom = 0.0
        if vmax <= 0: ylim_top = 0.0
    else:
        ylim_bottom, ylim_top = -1.0, 1.0

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
                # place label above/below depending on sign
                if not np.isnan(val) and abs(val) > 1e-3:
                    offset = (ylim_top - ylim_bottom) * 0.015
                    va = "bottom" if val >= 0 else "top"
                    ax.text(
                        bar_positions[k], val + (offset if val >= 0 else -offset),
                        f"{val:.2f}", ha="center", va=va, fontsize=8
                    )

            ax.set_ylim(ylim_bottom, ylim_top)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(
                ["sft-s","sft-u","dpo-s","dpo-u","sft_dpo-s","sft_dpo-u"],
                fontsize=8, rotation=20
            )

            # y-axis label per row
            if j == 0:
                ylab = "Generalisation (weak\u2192strong)" if gen == "weak->strong" else "Generalisation (strong\u2192weak)"
                ax.set_ylabel(ylab, fontsize=9)

            if i == 0:
                ax.set_title(model, fontsize=11, weight="bold")

            ax.axhline(0, color="black", linewidth=0.8)  # zero baseline
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
        f"Generalisation by Generalisation Direction and Model\n"
        f"Metric: {metric_type.replace('_', ' ').title()} | Generalisation: {degradation_type.replace('_', ' ').title()}",
        y=1.12, fontsize=13, weight="bold"
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def compute_generalization_metrics_util_type_3(scores, train_cap, split_types, metrics, degradation_types, generalization_types):
    """
    example of a generalization metric:  
        weak->strong:
            absolute degradation: Acc(train=weak, eval=strong) - Acc(train=weak, eval=weak)
            relative degradation: (Acc(train=weak, eval=strong) - Acc(train=weak, eval=weak)) / (Acc(train=weak,eval=weak))
        strong->weak:
            absolute degradation: Acc(train=strong, eval=weak) - Acc(train=strong, eval=strong)
            relative degradation: (Acc(train=strong, eval=weak) - Acc(train=strong, eval=strong)) / (Acc(train=strong,eval=strong))
    train_cap is fixed, eval_cap is varied.
    """
    generalization_metrics = {}
    
    eval_baseline_cap = train_cap
    eval_comparator_cap = "weak" if train_cap == "strong" else "strong"
    generalization_type = f"{eval_baseline_cap}->{eval_comparator_cap}"
    assert generalization_type in generalization_types, f"Generalization type {generalization_type} not in {generalization_types}"

    for eval_type in split_types:
        if eval_type not in generalization_metrics:
            generalization_metrics[eval_type] = {}

        for accuracy_type in metrics:
            if accuracy_type not in generalization_metrics[eval_type]:
                generalization_metrics[eval_type][accuracy_type] = {}

            baseline = scores[eval_baseline_cap][eval_type][accuracy_type] # eval-capability same as train-capability. Acc(train=weak, eval=weak)
            comparator = scores[eval_comparator_cap][eval_type][accuracy_type] # eval-capability different from train-capability. Acc(train=weak, eval=strong)
            
            absolute_degradation = 100.0 * (comparator - baseline) # ideally this should be >= 0. since baseline should be >= comparator in our case.
            relative_degradation = absolute_degradation / baseline # ideally this should be >= 0 too.

            generalization_metrics[eval_type][accuracy_type] = {
                "absolute_degradation": absolute_degradation,
                "relative_degradation": relative_degradation
            }
            for degradation_type in generalization_metrics[eval_type][accuracy_type]:
                assert degradation_type in degradation_types, f"Degradation type {degradation_type} not in {degradation_types}"

    return generalization_type, generalization_metrics

def compute_generalization_metrics_type_3(agg_scores, split_types, metrics, degradation_types, generalization_types):

    # Reorganize the scores into a dictionary of train_meta_to_eval_scores[train_meta][train_cap][eval_cap][eval_type] -> score
    train_meta_to_eval_scores = {}
    for (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) in agg_scores:
        meta = (set_name, train_algo, ckpt_step, model_name)
        if meta not in train_meta_to_eval_scores:
            train_meta_to_eval_scores[meta] = {}
        
        if train_cap not in train_meta_to_eval_scores[meta]:
            train_meta_to_eval_scores[meta][train_cap] = {}
        if eval_cap not in train_meta_to_eval_scores[meta][train_cap]:
            train_meta_to_eval_scores[meta][train_cap][eval_cap] = {}
        if eval_type not in train_meta_to_eval_scores[meta][train_cap][eval_cap]:
            # this is different from the advantage case, where we have: train_meta_to_eval_scores[meta][eval_cap][train_cap][eval_type] instead, 
            # since we want to vary train_cap there, but here we want to vary eval_cap so eval_cap appears at a deeper level in dictionary.
            train_meta_to_eval_scores[meta][train_cap][eval_cap][eval_type] = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)]

    # All eval types: {'seen_questions_seen_answers', 'seen_questions_unseen_answers', 'unseen_questions_unseen_answers'}
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200} # for llama series of models we trained longer until 4200 steps for rest of the models we trained until 2800 steps.

    # Compute the generalization metrics, at the last ckpt step.
    gen_meta_to_gen_scores = {}
    for train_meta in train_meta_to_eval_scores:
        set_name, train_algo, ckpt_step, model_name = train_meta
        if ckpt_step != last_ckpt_steps[model_name]: # if not the last ckpt step, skip
            continue

        for train_cap in ["weak", "strong"]:
            generalization_type, generalization_metrics = compute_generalization_metrics_util_type_3(train_meta_to_eval_scores[train_meta][train_cap], train_cap, split_types, metrics, degradation_types, generalization_types)
            generalization_meta = (train_algo, model_name, generalization_type)
            assert generalization_meta not in gen_meta_to_gen_scores, f"Generalization meta {generalization_meta} already exists"
            gen_meta_to_gen_scores[generalization_meta] = generalization_metrics

    return gen_meta_to_gen_scores

def create_dataframe_for_plots_type_3(gen_meta_to_gen_scores):

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

def create_generalization_plots_type_3(df, plot_dir, model_names, split_types, metrics, degradation_types, generalization_types):
    for current_metric in metrics:
        for current_degradation_type in degradation_types:
            # Filter the dataframe for the current metric and degradation type.
            filtered_df = df[(df["metric"] == current_metric) & (df["degradation_type"] == current_degradation_type)].drop(columns=["metric", "degradation_type"])
            plot_six_block_generalisation(filtered_df, current_metric, current_degradation_type, save_path=os.path.join(plot_dir, f"generalization_{current_metric}_{current_degradation_type}.png"))

def plot_in_out_scatter_type_6_util(d, title="Out-Dist vs In-Dist by Model/Algo/Cap", save_path=None):
    """
    d[(model_name, train_cap, train_algo)]["in_dist"]  -> float
    d[(model_name, train_cap, train_algo)]["out_dist"] -> float
    train_cap in {"weak", "strong"}

    Args:
        d : dict of metrics
        title : str, plot title
        save_path : str or None, if provided saves to file (png/pdf/svg etc.)
    """

    # Collect categories
    models, algos, caps, points = [], [], [], []
    for (model, cap, algo), metrics in d.items():
        if metrics is None or "in_dist" not in metrics or "out_dist" not in metrics:
            continue
        x = metrics["in_dist"]   # x-axis
        y = metrics["out_dist"]  # y-axis
        if x is None or y is None:
            continue
        models.append(model)
        algos.append(algo)
        caps.append(cap)
        points.append((x, y, model, cap, algo))

    if not points:
        raise ValueError("No valid data points found in dict.")

    uniq_models = list(dict.fromkeys(models))
    uniq_algos  = list(dict.fromkeys(algos))
    uniq_caps   = list(dict.fromkeys(caps))

    # Color map for algos
    color_cycle = plt.cm.tab10.colors
    color_map = {a: color_cycle[i % len(color_cycle)] for i, a in enumerate(uniq_algos)}

    # Marker map for models
    base_markers = ['o','s','^','D','P','X','*','v','<','>','h','H','p']
    marker_map = {m: base_markers[i % len(base_markers)] for i, m in enumerate(uniq_models)}

    # Shade map via alpha
    cap_alpha = defaultdict(lambda: 0.9)
    cap_alpha.update({"weak": 0.45, "strong": 0.95})

    plt.figure(figsize=(8.5, 7), dpi=140)
    ax = plt.gca()

    # Plot points
    for x, y, model, cap, algo in points:
        ax.scatter(
            x, y,
            s=70,
            marker=marker_map[model],
            c=[color_map[algo]],
            alpha=cap_alpha[cap],
            edgecolor="black",
            linewidths=0.6
        )
        
        # # Add text labels showing the values
        # ax.annotate(
        #     f"({x:.1f}, {y:.1f})",
        #     (x, y),
        #     xytext=(5, 5),
        #     textcoords='offset points',
        #     fontsize=8,
        #     alpha=0.8,
        #     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
        # )

    # Diagonal y=x baseline
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    min_val = min(min(all_x), min(all_y))
    max_val = max(max(all_x), max(all_y))
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle="--", color="grey", linewidth=1.2, alpha=0.7,
            label="y = x (no-generalisation)")

    # Labels
    ax.set_xlabel("In-Distribution Score", labelpad=8)
    ax.set_ylabel("Out-of-Distribution Score", labelpad=8)
    ax.set_title(title, pad=12)
    # Set ticks at 1.0 intervals
    ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    
    # Add gridlines at 1.0 intervals
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Legends
    # Color = algo
    color_handles = [Line2D([0], [0], marker='o', color='none',
                            markerfacecolor=color_map[a], markeredgecolor="black",
                            markersize=8, label=a) for a in uniq_algos]
    legend_algos = ax.legend(handles=color_handles, title="Train Algo (color)",
                              loc="upper left", bbox_to_anchor=(1.02, 1.00))
    ax.add_artist(legend_algos)

    # Shape = model
    marker_handles = [Line2D([0], [0], marker=marker_map[m], color="black",
                             linestyle="none", markersize=8, label=m) for m in uniq_models]
    legend_models = ax.legend(handles=marker_handles, title="Model (shape)",
                              loc="upper left", bbox_to_anchor=(1.02, 0.60))
    ax.add_artist(legend_models)

    # Shade = cap
    shade_color = (0.2, 0.2, 0.2)
    shade_handles = [Line2D([0], [0], marker='o', linestyle='none',
                            markerfacecolor=shade_color, markeredgecolor="black",
                            alpha=cap_alpha[cap], markersize=8, label=cap)
                     for cap in uniq_caps]
    legend_caps = ax.legend(handles=shade_handles, title="Train Cap (shade)",
                            loc="upper left", bbox_to_anchor=(1.02, 0.28))
    ax.add_artist(legend_caps)

    # Add baseline legend entry
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, -0.02))

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def compute_generalization_metrics_type_6(agg_scores, split_types, metrics, degradation_types, generalization_types):
    # All eval types: {'seen_questions_seen_answers', 'seen_questions_unseen_answers', 'unseen_questions_unseen_answers'}
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200} # for llama series of models we trained longer until 4200 steps for rest of the models we trained until 2800 steps.

    train_meta_to_eval_scores = {}
    for (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) in agg_scores:
        if ckpt_step != last_ckpt_steps[model_name]: # if not the last ckpt step, skip
            continue
        if eval_type == "seen_questions_seen_answers":
            continue

        accuracy_meta = (eval_type, "accuracy")
        if accuracy_meta not in train_meta_to_eval_scores:
            train_meta_to_eval_scores[accuracy_meta] = {}
        if (model_name, train_cap, train_algo) not in train_meta_to_eval_scores[accuracy_meta]:
            train_meta_to_eval_scores[accuracy_meta][(model_name, train_cap, train_algo)] = {}

        consistent_accuracy_meta = (eval_type, "consistent_accuracy")
        if consistent_accuracy_meta not in train_meta_to_eval_scores:
            train_meta_to_eval_scores[consistent_accuracy_meta] = {}
        if (model_name, train_cap, train_algo) not in train_meta_to_eval_scores[consistent_accuracy_meta]:
            train_meta_to_eval_scores[consistent_accuracy_meta][(model_name, train_cap, train_algo)] = {}

        eval_in_dist = True if (eval_cap == train_cap) else False

        for accuracy_meta in [accuracy_meta, consistent_accuracy_meta]:
            accuracy_type = accuracy_meta[1]
            accuracy = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)][accuracy_type]
            if eval_in_dist:
                if (model_name, train_cap, train_algo) not in train_meta_to_eval_scores[(eval_type, accuracy_type)]:
                    train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)] = {}
                train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)]["in_dist"] = 100.0 * accuracy
            else:
                if (model_name, train_cap, train_algo) not in train_meta_to_eval_scores[(eval_type, accuracy_type)]:
                    train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)] = {}
                train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)]["out_dist"] = 100.0 * accuracy

    return train_meta_to_eval_scores

def plot_in_out_scatter_type_6(train_meta_to_eval_scores, plot_dir):
    for eval_type, accuracy_type in train_meta_to_eval_scores:
        plot_data = train_meta_to_eval_scores[(eval_type, accuracy_type)]
        plot_in_out_scatter_type_6_util(plot_data, title=f"{eval_type} {accuracy_type}", save_path=os.path.join(plot_dir, f"in_out_scatter_{eval_type}_{accuracy_type}.png"))

def create_dataframe_for_plots_type_6(train_meta_to_eval_scores):
    rows = []
    for (eval_type, accuracy_type), model_dict in train_meta_to_eval_scores.items():
        for (model_name, train_cap, train_algo), inout_dict in model_dict.items():
            row = {
                "eval_type": eval_type,
                "accuracy_type": accuracy_type,
                "model_name": model_name,
                "train_cap": train_cap,
                "train_algo": train_algo,
                "in_dist": inout_dict.get("in_dist", None),
                "out_dist": inout_dict.get("out_dist", None)
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

def main():
    agg_scores = read_eval_results("./eval-results")

    model_names = ["ministral8b", "mistral24b", "llama8b"]
    split_types = ["seen_questions_unseen_answers", "unseen_questions_unseen_answers"]
    generalization_types = ["weak->strong", "strong->weak"]
    metrics = ["accuracy", "consistent_accuracy"]
    degradation_types = ["absolute_degradation", "relative_degradation"]


    # create the type 3 plots and save the dataframe corresponding to it.
    plot_dir = "./eval-plots/plot_type_3"
    os.makedirs(plot_dir, exist_ok=True)

    gen_meta_to_gen_scores = compute_generalization_metrics_type_3(agg_scores, split_types, metrics, degradation_types, generalization_types)
    df = create_dataframe_for_plots_type_3(gen_meta_to_gen_scores)
    create_generalization_plots_type_3(df, plot_dir, model_names, split_types, metrics, degradation_types, generalization_types)

    # Save the dataframe as a CSV file in the plot_dir
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)

    # create the type 6 plots (scatter plots) and save the dataframe corresponding to it.
    plot_dir = "./eval-plots/plot_type_6"
    os.makedirs(plot_dir, exist_ok=True)
    train_meta_to_eval_scores = compute_generalization_metrics_type_6(agg_scores, split_types, metrics, degradation_types, generalization_types)
    plot_in_out_scatter_type_6(train_meta_to_eval_scores, plot_dir)
    df = create_dataframe_for_plots_type_6(train_meta_to_eval_scores)
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)


if __name__ == "__main__":
    main()