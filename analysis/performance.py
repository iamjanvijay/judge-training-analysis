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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_six_block_performance(filtered_df, score_name, save_path=None):
    """
    2x3 grid of barplots.

    Rows (top->bottom):   train_cap = ["weak", "strong"]
    Cols (left->right):   model_name = ["llama8b", "mistral8b", "mistral24b"]

    Within each panel: 12 bars = 3 algos × 4 bars each.
      For train_cap=weak:
        [same-seen (light green), same-unseen (dark green),
         cross-seen (light red), cross-unseen (dark red)]
      For train_cap=strong:
        [cross-seen (light red), cross-unseen (dark red),
         same-seen (light green), same-unseen (dark green)]

    Expected columns in filtered_df:
      train_cap, model_name, train_algo, eval_cap, eval_type, and the metric column `score_name`.

    Raises:
      ValueError if more than one row exists for a (train_cap, model_name, train_algo, eval_cap, eval_type).
    """

    row_caps    = ["weak", "strong"]
    col_models  = ["llama8b", "ministral8b", "mistral24b"]
    algos       = ["sft", "dpo", "sft_dpo"]
    pretty_algo = {"sft": "SFT", "dpo": "DPO", "sft_dpo": "SFT+DPO"}

    # Different bar order depending on row
    types_in_algo_row = {
        "weak": [
            ("same",  "seen_questions_unseen_answers"),   # light green
            ("same",  "unseen_questions_unseen_answers"), # dark green
            ("cross", "seen_questions_unseen_answers"),   # light red
            ("cross", "unseen_questions_unseen_answers"), # dark red
        ],
        "strong": [
            ("cross", "seen_questions_unseen_answers"),   # light red
            ("cross", "unseen_questions_unseen_answers"), # dark red
            ("same",  "seen_questions_unseen_answers"),   # light green
            ("same",  "unseen_questions_unseen_answers"), # dark green
        ],
    }

    # Colors
    color_map = {
        ("same",  "seen_questions_unseen_answers"):   "#A6E3A1",  # light green
        ("same",  "unseen_questions_unseen_answers"): "#2E7D32",  # dark green
        ("cross", "seen_questions_unseen_answers"):   "#F2A7A7",  # light red
        ("cross", "unseen_questions_unseen_answers"): "#C62828",  # dark red
    }

    def _get_value(train_cap, model_name, train_algo, eval_cap, eval_type):
        m = (
            (filtered_df["train_cap"]  == train_cap) &
            (filtered_df["model_name"] == model_name) &
            (filtered_df["train_algo"] == train_algo) &
            (filtered_df["eval_cap"]   == eval_cap) &
            (filtered_df["eval_type"]  == eval_type)
        )
        vals = filtered_df.loc[m, score_name].values
        if vals.size == 0:
            return np.nan
        if vals.size > 1:
            raise ValueError(
                f"Duplicate rows for train_cap={train_cap}, model={model_name}, "
                f"train_algo={train_algo}, eval_cap={eval_cap}, eval_type={eval_type}"
            )
        return float(vals[0])

    # Collect all values for global y-limit
    all_vals = []
    for tc in row_caps:
        for model in col_models:
            for algo in algos:
                for rel, etype in types_in_algo_row[tc]:
                    ec = tc if rel == "same" else ("strong" if tc == "weak" else "weak")
                    all_vals.append(_get_value(tc, model, algo, ec, etype))
    vmax = np.nanmax(all_vals) if len(all_vals) else 1.0
    ymax = 1.10 * (vmax if np.isfinite(vmax) else 1.0)

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True, constrained_layout=True)

    group_gap = 0.8
    group_width = 4
    bar_w = 0.75
    group_starts = np.array([0,
                             group_width + group_gap,
                             2*group_width + 2*group_gap], dtype=float)
    panel_xlim = -0.6, group_starts[-1] + group_width - 1 + 0.6

    for i_row, tc in enumerate(row_caps):
        for j_col, model in enumerate(col_models):
            ax = axes[i_row, j_col]

            # Draw bars
            for g, algo in enumerate(algos):
                base = group_starts[g]
                for k, (rel, etype) in enumerate(types_in_algo_row[tc]):
                    xv = base + k
                    eval_cap = tc if rel == "same" else ("strong" if tc == "weak" else "weak")
                    val = _get_value(tc, model, algo, eval_cap, etype)
                    ax.bar(
                        xv, val, width=bar_w,
                        color=color_map[(rel, etype)],
                        edgecolor="black", linewidth=0.6
                    )
                    if not np.isnan(val):
                        ax.text(xv, val + 0.015*ymax, f"{val:.2f}",
                                ha="center", va="bottom", fontsize=8)

            # Algo labels centered
            group_centers = [start + 1.5 for start in group_starts]
            ax.set_xticks(group_centers)
            ax.set_xticklabels([pretty_algo[a] for a in algos],
                               fontsize=10, fontweight="bold")

            # cosmetics
            ax.set_ylim(0, ymax)
            ax.set_xlim(*panel_xlim)
            ax.grid(axis="y", linestyle="--", alpha=0.25)

            if j_col == 0:
                ax.set_ylabel(f"Performance ({score_name})\n(train_cap={tc})", fontsize=10)
            if i_row == 0:
                ax.set_title(model, fontsize=12, weight="bold")

    # Legend
    legend_handles = [
        Patch(facecolor=color_map[("same",  "seen_questions_unseen_answers")],  edgecolor='black', label="same-cap (seen)"),
        Patch(facecolor=color_map[("same",  "unseen_questions_unseen_answers")], edgecolor='black', label="same-cap (unseen)"),
        Patch(facecolor=color_map[("cross", "seen_questions_unseen_answers")],  edgecolor='black', label="cross-cap (seen)"),
        Patch(facecolor=color_map[("cross", "unseen_questions_unseen_answers")], edgecolor='black', label="cross-cap (unseen)"),
    ]
    fig.legend(handles=legend_handles, ncols=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=9)

    fig.suptitle("Performance by Train Capability (rows), Model (cols), Algo (groups), Eval Conditions (bar colors)",
                 y=1.08, fontsize=13, weight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def create_performance_plots(df, plot_dir, accuracy_types):

    # Drop all rows where model_name does not have the last_ckpt_step as specified in last_ckpt_steps
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200} # for llama series of models we trained longer until 4200 steps for rest of the models we trained until 2800 steps.
    df = df[df.apply(lambda row: row['ckpt_step'] == last_ckpt_steps.get(row['model_name'], row['ckpt_step']), axis=1)]

    # Drop the set_name column, since we don't need it for the plots
    df = df.drop(columns=["set_name"])

    for current_accuracy_type in accuracy_types:
        drop_accuracy_type = "consistent_accuracy" if current_accuracy_type == "accuracy" else "accuracy"

        filtered_df = df.copy()
        filtered_df = filtered_df.drop(columns=[drop_accuracy_type])

        plot_six_block_performance(filtered_df, current_accuracy_type, save_path=os.path.join(plot_dir, f"performance_{current_accuracy_type}.png"))

    return df

def create_dataframe_for_plots(agg_scores):
    rows = []
    for (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) in agg_scores:
        value = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)]
        accuracy = value.get("accuracy", None)
        consistent_accuracy = value.get("consistent_accuracy", None)
        
        rows.append({
            "set_name": set_name,
            "train_algo": train_algo,
            "train_cap": train_cap,
            "ckpt_step": ckpt_step,
            "model_name": model_name,
            "eval_cap": eval_cap,
            "eval_type": eval_type,
            "accuracy": accuracy,
            "consistent_accuracy": consistent_accuracy
        })

    df = pd.DataFrame(rows)
    return df
   
def main():
    agg_scores = read_eval_results("./eval-results")

    accuracy_types = ["accuracy", "consistent_accuracy"]

    plot_dir = "./eval-plots/plot_type_5"
    os.makedirs(plot_dir, exist_ok=True)

    df = create_dataframe_for_plots(agg_scores)
    df = create_performance_plots(df, plot_dir, accuracy_types)

    # Save the dataframe as a CSV file in the plot_dir
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)



if __name__ == "__main__":
    main()