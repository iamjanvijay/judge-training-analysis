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

def create_table_1(df, plot_dir):

    # Keep all rows where ckpt_step is 0, or where model_name has the last_ckpt_step as specified in last_ckpt_steps
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200}
    df = df[df.apply(lambda row: (row['ckpt_step'] == 0) or (row['ckpt_step'] == last_ckpt_steps.get(row['model_name'], row['ckpt_step'])), axis=1)]

    # Drop the set_name column, since we don't need it for the plots
    df = df.drop(columns=["set_name"])

    df.to_csv(os.path.join(plot_dir, "table_1.csv"), index=False)

    return df

def create_latex_table(df, plot_dir, metric_name="accuracy"):
    """
    Create a LaTeX table from the dataframe with the structure from the paper.
    Maps the data to the table format: Wk-Se, St-Se, Wk-Us, St-Us, Se, Us, Wk, St
    
    Args:
        df: DataFrame containing the evaluation results
        plot_dir: Directory to save the LaTeX table
        metric_name: Name of the metric to use ('accuracy' or 'consistent_accuracy')
    """
    
    # Model name mapping
    model_mapping = {
        "llama8b": "Llama3-8B",
        "ministral8b": "Ministral-8B", 
        "mistral24b": "Mistral-24B"
    }
    
    # Training algorithm mapping
    algo_mapping = {
        "sft": "SFT",
        "dpo": "DPO", 
        "sft_dpo": "SFT+DPO"
    }
    
    # Last checkpoint steps for each model
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200}
    
    # Create the LaTeX table
    latex_content = """% ---- Color palettes ----
% Group A (columns 2–5): 4-level ranks (1=darkest, 4=lightest)
\\definecolor{colA1}{RGB}{ 49,130,189}  % blue (dark)
\\definecolor{colA2}{RGB}{107,174,214}  % blue (midtone)
\\definecolor{colA3}{RGB}{158,202,225}  % blue (light)
\\definecolor{colA4}{RGB}{222,235,247}  % blue (very light)
\\newcommand{\\Aone}[1]{\\cellcolor{colA1}{#1}}
\\newcommand{\\Atwo}[1]{\\cellcolor{colA2}{#1}}
\\newcommand{\\Athr}[1]{\\cellcolor{colA3}{#1}}
\\newcommand{\\Afour}[1]{\\cellcolor{colA4}{#1}}

% Group B (columns 6–7): 2-level ranks (1=best dark, 2=lighter)
\\definecolor{colBbest}{RGB}{140, 86, 75} % brownish
\\definecolor{colBrest}{RGB}{233,203,194}
\\newcommand{\\Bbest}[1]{\\cellcolor{colBbest}{#1}}
\\newcommand{\\Bsec}[1]{\\cellcolor{colBrest}{#1}} % use for "rank 2"

% Group C (columns 8–9): 2-level ranks (1=best dark, 2=lighter)
\\definecolor{colCbest}{RGB}{ 44,160, 44} % green
\\definecolor{colCrest}{RGB}{199,233,192}
\\newcommand{\\Cbest}[1]{\\cellcolor{colCbest}{#1}}
\\newcommand{\\Csec}[1]{\\cellcolor{colCrest}{#1}}
% ===========================================================================

% ======================= TABLE ==========================
\\begin{table}[t]
\\centering
\\footnotesize
\\renewcommand{\\arraystretch}{1.12}
\\setlength{\\tabcolsep}{6pt}
\\begin{tabular}{l|*{4}{S[table-format=2.2]}|*{2}{S[table-format=2.2]}|*{2}{S[table-format=2.2]}}
\\toprule
\\textbf{Judge Model} &
{\\boldmath$D_k^{\\text{Wk-Se}}$} &
{\\boldmath$D_k^{\\text{St-Se}}$} &
{\\boldmath$D_k^{\\text{Wk-Us}}$} &
{\\boldmath$D_k^{\\text{St-Us}}$} &
{\\boldmath$D_k^{\\text{Se}}$} &
{\\boldmath$D_k^{\\text{Us}}$} &
{\\boldmath$D_k^{\\text{Wk}}$} &
{\\boldmath$D_k^{\\text{St}}$} \\\\
\\midrule
"""
    
    # Process each model
    for model_key, model_display in model_mapping.items():
        model_df = df[df['model_name'] == model_key]
        
        # Collect all metrics for this model to calculate rankings
        model_metrics = []
        model_labels = []
        
        # Base model (ckpt_step = 0)
        base_df = model_df[model_df['ckpt_step'] == 0]
        if not base_df.empty:
            base_metrics = calculate_aggregated_metrics(base_df, metric_name)
            model_metrics.append(base_metrics)
            model_labels.append(f"\\texttt{{{model_display}}}")
        
        # Training algorithms
        for algo_key, algo_display in algo_mapping.items():
            algo_df = model_df[model_df['train_algo'] == algo_key]
            if not algo_df.empty:
                # Get the appropriate dataset indicator
                if model_key == "llama8b":
                    dataset_indicator = "D_2"
                else:
                    dataset_indicator = "D_1"
                
                # Calculate metrics for each training capability
                for train_cap in ['weak', 'strong']:
                    # Get the last checkpoint step for this model
                    last_ckpt_step = last_ckpt_steps.get(model_key, 0)
                    cap_df = algo_df[(algo_df['train_cap'] == train_cap) & (algo_df['ckpt_step'] == last_ckpt_step)]
                    if not cap_df.empty:
                        cap_metrics = calculate_aggregated_metrics(cap_df, metric_name)
                        cap_display = "Wk" if train_cap == 'weak' else "St"
                        model_metrics.append(cap_metrics)
                        model_labels.append(f"\\hspace{{1em}}+ {algo_display} : ${dataset_indicator}^{{\\text{{{cap_display}}}}}$")
        
        # Get ranked values with colors for this model
        if model_metrics:
            ranked_values = get_ranked_values_with_colors(model_metrics)
            
            # Add rows to table
            for i, (label, values) in enumerate(zip(model_labels, ranked_values)):
                latex_content += f"{label} & {values} \\\\\n"
        
        # Add midrule between models (except for the last one)
        if model_key != list(model_mapping.keys())[-1]:
            latex_content += "\\midrule\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\caption{Rank-highlighted table with grouped columns: (2–5) four fine-grained splits ranked 1–4 within each row (dark$\\to$light); (6–7) \\emph{Se/Us} ranked 1–2 within each row (best vs second); (8–9) \\emph{Wk/St} ranked 1–2 within each row. Rankings are calculated within each row based on actual performance values.}
\\label{tab:training-setup-iclr}
\\end{table}
% ========================================================
"""
    
    # Save the LaTeX table
    filename = f"table_1_row_ranked_{metric_name}.tex"
    with open(os.path.join(plot_dir, filename), 'w') as f:
        f.write(latex_content)
    
    print(f"Row-ranked LaTeX table saved to {os.path.join(plot_dir, filename)}")
    return latex_content

def calculate_aggregated_metrics(df, metric_name="accuracy"):
    """
    Calculate the aggregated metrics for a given dataframe subset.
    Returns a dictionary with the 8 metrics needed for the table.
    
    The table structure should be:
    - Wk-Se: Performance when evaluated on weak capability, seen questions
    - St-Se: Performance when evaluated on strong capability, seen questions  
    - Wk-Us: Performance when evaluated on weak capability, unseen questions
    - St-Us: Performance when evaluated on strong capability, unseen questions
    - Se: Average of seen (Wk-Se + St-Se) / 2
    - Us: Average of unseen (Wk-Us + St-Us) / 2
    - Wk: Average of weak evaluation (Wk-Se + Wk-Us) / 2
    - St: Average of strong evaluation (St-Se + St-Us) / 2
    
    Args:
        df: DataFrame containing the evaluation results
        metric_name: Name of the metric to use ('accuracy' or 'consistent_accuracy')
    """
    metrics = {}
    
    # Helper function to get accuracy value
    def get_accuracy(eval_cap, eval_type):
        subset = df[(df['eval_cap'] == eval_cap) & (df['eval_type'] == eval_type)]
        if len(subset) == 0:
            return 0.0
        # For base models, there might be multiple rows with same values, just take the first one
        return subset[metric_name].iloc[0] * 100  # Convert to percentage
    
    # Wk-Se: Weak evaluation, Seen questions
    metrics['wk_se'] = get_accuracy('weak', 'seen_questions_unseen_answers')
    
    # St-Se: Strong evaluation, Seen questions  
    metrics['st_se'] = get_accuracy('strong', 'seen_questions_unseen_answers')
    
    # Wk-Us: Weak evaluation, Unseen questions
    metrics['wk_us'] = get_accuracy('weak', 'unseen_questions_unseen_answers')
    
    # St-Us: Strong evaluation, Unseen questions
    metrics['st_us'] = get_accuracy('strong', 'unseen_questions_unseen_answers')
    
    # Se: Average of seen (Wk-Se + St-Se) / 2
    metrics['se'] = (metrics['wk_se'] + metrics['st_se']) / 2
    
    # Us: Average of unseen (Wk-Us + St-Us) / 2  
    metrics['us'] = (metrics['wk_us'] + metrics['st_us']) / 2
    
    # Wk: Average of weak evaluation (Wk-Se + Wk-Us) / 2
    metrics['wk'] = (metrics['wk_se'] + metrics['wk_us']) / 2
    
    # St: Average of strong evaluation (St-Se + St-Us) / 2
    metrics['st'] = (metrics['st_se'] + metrics['st_us']) / 2
    
    return metrics

def get_ranked_values_with_colors(metrics_list):
    """
    Calculate rankings for each group of columns within each row and return formatted values with colors.
    
    Args:
        metrics_list: List of dictionaries containing metrics for each row
    
    Returns:
        List of formatted strings with color commands
    """
    if not metrics_list:
        return []
    
    # Group A: columns 2-5 (Wk-Se, St-Se, Wk-Us, St-Us) - 4-level ranking
    group_a_keys = ['wk_se', 'st_se', 'wk_us', 'st_us']
    
    # Group B: columns 6-7 (Se, Us) - 2-level ranking
    group_b_keys = ['se', 'us']
    
    # Group C: columns 8-9 (Wk, St) - 2-level ranking
    group_c_keys = ['wk', 'st']
    
    def get_row_ranks(values, num_ranks):
        """Get ranks for values within a single row (1=best, higher=worse)"""
        # Create list of (value, original_index) pairs
        indexed_values = [(val, i) for i, val in enumerate(values)]
        # Sort by value in descending order
        sorted_values = sorted(indexed_values, key=lambda x: x[0], reverse=True)
        # Create rank mapping
        ranks = [0] * len(values)
        for rank, (val, orig_idx) in enumerate(sorted_values, 1):
            ranks[orig_idx] = rank
        return ranks
    
    # Format values with colors
    formatted_rows = []
    for metrics in metrics_list:
        # Group A: Get values and ranks within this row
        group_a_values = [metrics[key] for key in group_a_keys]
        group_a_ranks = get_row_ranks(group_a_values, 4)
        
        # Group B: Get values and ranks within this row
        group_b_values = [metrics[key] for key in group_b_keys]
        group_b_ranks = get_row_ranks(group_b_values, 2)
        
        # Group C: Get values and ranks within this row
        group_c_values = [metrics[key] for key in group_c_keys]
        group_c_ranks = get_row_ranks(group_c_values, 2)
        
        # Group A colors (4-level)
        a_colors = []
        for rank in group_a_ranks:
            if rank == 1:
                a_colors.append("\\Aone")
            elif rank == 2:
                a_colors.append("\\Atwo")
            elif rank == 3:
                a_colors.append("\\Athr")
            else:
                a_colors.append("\\Afour")
        
        # Group B colors (2-level)
        b_colors = []
        for rank in group_b_ranks:
            if rank == 1:
                b_colors.append("\\Bbest")
            else:
                b_colors.append("\\Bsec")
        
        # Group C colors (2-level)
        c_colors = []
        for rank in group_c_ranks:
            if rank == 1:
                c_colors.append("\\Cbest")
            else:
                c_colors.append("\\Csec")
        
        # Format the row
        formatted_row = f"{a_colors[0]}{{{metrics['wk_se']:.2f}}} & {a_colors[1]}{{{metrics['st_se']:.2f}}} & {a_colors[2]}{{{metrics['wk_us']:.2f}}} & {a_colors[3]}{{{metrics['st_us']:.2f}}} & {b_colors[0]}{{{metrics['se']:.2f}}} & {b_colors[1]}{{{metrics['us']:.2f}}} & {c_colors[0]}{{{metrics['wk']:.2f}}} & {c_colors[1]}{{{metrics['st']:.2f}}}"
        formatted_rows.append(formatted_row)
    
    return formatted_rows

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
    
    # create the colored LaTeX tables for both metrics
    create_latex_table(df.copy(), plot_dir, "accuracy")
    create_latex_table(df.copy(), plot_dir, "consistent_accuracy")

    # create the performance plots
    df = create_performance_plots(df, plot_dir, accuracy_types)

    # Save the dataframe as a CSV file in the plot_dir
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)



if __name__ == "__main__":
    main()