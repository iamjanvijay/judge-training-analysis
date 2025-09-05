# Standard library imports
import os
import sys
import math
from collections import defaultdict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import beta

# Local imports
sys.path.append("./analysis")
from read_eval_results import read_eval_results

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


def compute_generalization_metrics_util_type_3(scores, initial_scores, train_cap, split_types, metrics, degradation_types, generalization_types):
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

            baseline_normalization = initial_scores[eval_baseline_cap][eval_type][accuracy_type] # eval-capability same as train-capability. Acc(train=weak, eval=weak)
            comparator_normalization = initial_scores[eval_comparator_cap][eval_type][accuracy_type] # eval-capability different from train-capability. Acc(train=weak, eval=strong)
            
            absolute_degradation = 100.0 * (comparator - baseline) # ideally this should be >= 0. since baseline should be >= comparator in our case.
            relative_degradation = absolute_degradation / baseline # ideally this should be >= 0 too.

            normalized_baseline = baseline - initial_scores[eval_baseline_cap][eval_type][accuracy_type]
            normalized_comparator = comparator - initial_scores[eval_comparator_cap][eval_type][accuracy_type]

            normalized_absolute_degradation = 100.0 * (normalized_comparator - normalized_baseline) # ideally this should be >= 0. since baseline should be >= comparator in our case.
            normalized_relative_degradation = normalized_absolute_degradation / normalized_baseline # ideally this should be >= 0 too.

            generalization_metrics[eval_type][accuracy_type] = {
                "absolute_degradation": absolute_degradation,
                "relative_degradation": relative_degradation,
                "normalized_absolute_degradation": normalized_absolute_degradation,
                "normalized_relative_degradation": normalized_relative_degradation,
                "baseline": baseline,
                "comparator": comparator,
                "baseline_normalization": baseline_normalization,
                "comparator_normalization": comparator_normalization
            }
            for degradation_type in generalization_metrics[eval_type][accuracy_type]:
                # assert degradation_type in degradation_types, f"Degradation type {degradation_type} not in {degradation_types}"
                pass
    return generalization_type, generalization_metrics

def compute_generalization_metrics_type_3(agg_scores, split_types, metrics, degradation_types, generalization_types):

    # All eval types: {'seen_questions_seen_answers', 'seen_questions_unseen_answers', 'unseen_questions_unseen_answers'}
    last_ckpt_steps = {"ministral8b": 2800, "mistral24b": 2800, "llama8b": 4200} # for llama series of models we trained longer until 4200 steps for rest of the models we trained until 2800 steps.

    # Reorganize the scores into a dictionary of train_meta_to_eval_scores[train_meta][train_cap][eval_cap][eval_type] -> score
    train_meta_to_eval_scores = {}
    for (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) in agg_scores:
        meta = (set_name, train_algo, model_name)

        if train_algo not in ["sft", "dpo", "sft_dpo"]: continue
        
        ckpt_type = "last" if ckpt_step == last_ckpt_steps[model_name] else "first" if ckpt_step == 0 else "intermediate";  # put this in one line
        if ckpt_type == "intermediate": continue
    
        if meta not in train_meta_to_eval_scores:
            train_meta_to_eval_scores[meta] = {}
        if ckpt_type not in train_meta_to_eval_scores[meta]:
            train_meta_to_eval_scores[meta][ckpt_type] = {}
        
        if train_cap not in train_meta_to_eval_scores[meta][ckpt_type]:
            train_meta_to_eval_scores[meta][ckpt_type][train_cap] = {}
        if eval_cap not in train_meta_to_eval_scores[meta][ckpt_type][train_cap]:
            train_meta_to_eval_scores[meta][ckpt_type][train_cap][eval_cap] = {}
        if eval_type not in train_meta_to_eval_scores[meta][ckpt_type][train_cap][eval_cap]:
            # this is different from the advantage case, where we have: train_meta_to_eval_scores[meta][eval_cap][train_cap][eval_type] instead, 
            # since we want to vary train_cap there, but here we want to vary eval_cap so eval_cap appears at a deeper level in dictionary.
            train_meta_to_eval_scores[meta][ckpt_type][train_cap][eval_cap][eval_type] = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)]

    # Compute the generalization metrics, at the last ckpt step.
    gen_meta_to_gen_scores = {}
    for train_meta in train_meta_to_eval_scores:
        set_name, train_algo, model_name = train_meta
        last_ckpt_scores_data = train_meta_to_eval_scores[train_meta]["last"]
        first_ckpt_scores_data = train_meta_to_eval_scores[train_meta]["first"]
        
        for train_cap in ["weak", "strong"]:
            generalization_type, generalization_metrics = compute_generalization_metrics_util_type_3(last_ckpt_scores_data[train_cap], first_ckpt_scores_data[train_cap], train_cap, split_types, metrics, degradation_types, generalization_types)
            generalization_meta = (train_algo, model_name, generalization_type)
            gen_meta_to_gen_scores[generalization_meta] = generalization_metrics

    return gen_meta_to_gen_scores

def create_dataframe_for_plots_type_3(gen_meta_to_gen_scores):

    rows = []
    for (train_algo, model_name, gen_type) in gen_meta_to_gen_scores:
        for split in ["seen_questions_unseen_answers", "unseen_questions_unseen_answers"]:
            for metric in ["accuracy", "consistent_accuracy"]:
                # Create a single row with all degradation types as columns
                row = {
                    "train_algo": train_algo,
                    "model_name": model_name,
                    "generalisation_type": gen_type,
                    "eval_split": split,
                    "metric": metric
                }
                
                # Add each degradation type as a column
                for deg_type in ["absolute_degradation", "relative_degradation", \
                                 "normalized_absolute_degradation", "normalized_relative_degradation", \
                                 "baseline", "comparator", "baseline_normalization", "comparator_normalization"]:
                    value = gen_meta_to_gen_scores[(train_algo, model_name, gen_type)][split][metric][deg_type]
                    row[deg_type] = value
                
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def create_generalization_plots_type_3(df, plot_dir, model_names, split_types, metrics, degradation_types, generalization_types):
    for current_metric in metrics:
        for current_degradation_type in degradation_types:
            # Filter the dataframe for the current metric and create a new dataframe with the degradation type as a value column
            filtered_df = df[df["metric"] == current_metric].copy()
            filtered_df = filtered_df.rename(columns={current_degradation_type: "value"})
            filtered_df = filtered_df[["train_algo", "model_name", "generalisation_type", "eval_split", "value"]]
            plot_six_block_generalisation(filtered_df, current_metric, current_degradation_type, save_path=os.path.join(plot_dir, f"generalization_{current_metric}_{current_degradation_type}.png"))


def _cp_interval_scipy(k: int, n: int, alpha: float = 0.05):
    """
    Clopper–Pearson (exact) CI via Beta quantiles using SciPy.
    Returns (lo, hi) in probability space [0,1].
    """
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2.0, k + 1, n - k)
    return float(lo), float(hi)

def _to_prob_and_scale(vals):
    """
    Detect if values are probabilities [0,1] or percentages [0,100].
    Returns (values_in_prob_[0,1], was_percent_bool).
    """
    mx = max(vals) if vals else 1.0
    is_percent = mx > 1.5  # heuristic: anything >~1 means percent
    return ([v / 100.0 for v in vals], True) if is_percent else (vals, False)

def plot_in_out_scatter_type_6_util(
    d,
    title="Out-Dist vs In-Dist by Model/Algo/Cap",
    save_path=None,
    show_ci=True,
    ci_alpha=0.05,
    equal_aspect=False
):
    """
    Parameters
    ----------
    d : dict
        Mapping:
          d[(model_name, train_cap, train_algo)] = {
              "in_dist": float,              # accuracy (prob in [0,1] or % in [0,100])
              "out_dist": float,             # accuracy (prob in [0,1] or % in [0,100])
              "in_dist_count": int (opt),    # trials for x-axis accuracy
              "out_dist_count": int (opt),   # trials for y-axis accuracy
              # Optional (if you have exact hits, recommended to avoid rounding):
              # "in_dist_hits": int,
              # "out_dist_hits": int,
          }

    Visual encodings
    ----------------
    Color  -> train_algo
    Shape  -> model_name
    Shade  -> train_cap ('weak' lighter alpha, 'strong' darker alpha)

    Other niceties
    --------------
    * Auto-detect % vs prob and scale labels/limits accordingly
    * Data-driven axis limits with small padding (clipped to [0,1] or [0,100])
    * y = x baseline (no-generalisation)
    * Optional equal_aspect so the baseline is visually 45°
    * Optional 95% Clopper–Pearson error bars via SciPy
    """

    # --- Collect rows from dict ---
    rows = []
    for (model, cap, algo), metrics in d.items():
        if not metrics:
            continue
        if "in_dist" not in metrics or "out_dist" not in metrics:
            continue
        x_raw = metrics["in_dist"]
        y_raw = metrics["out_dist"]
        if x_raw is None or y_raw is None:
            continue
        rows.append((
            model, cap, algo,
            x_raw, y_raw,
            metrics.get("in_dist_count"),
            metrics.get("out_dist_count"),
            metrics.get("in_dist_hits"),
            metrics.get("out_dist_hits"),
        ))

    if not rows:
        raise ValueError("No valid data points found in dict.")

    # --- Detect scale and convert to probabilities for CI math ---
    all_x_raw = [r[3] for r in rows]
    all_y_raw = [r[4] for r in rows]
    xs_prob, x_was_pct = _to_prob_and_scale(all_x_raw)
    ys_prob, y_was_pct = _to_prob_and_scale(all_y_raw)
    use_percent_scale = x_was_pct or y_was_pct

    # --- Rebuild points list in prob space ---
    points = []
    for (row, x_p, y_p) in zip(rows, xs_prob, ys_prob):
        model, cap, algo, _, _, N, M, hx, hy = row
        points.append((x_p, y_p, model, cap, algo, N, M, hx, hy))

    # --- Category lists for legends ---
    uniq_models = list(dict.fromkeys([p[2] for p in points]))
    uniq_algos  = list(dict.fromkeys([p[4] for p in points]))
    uniq_caps   = list(dict.fromkeys([p[3] for p in points]))

    # --- Style maps ---
    color_cycle = plt.cm.tab10.colors
    color_map = {a: color_cycle[i % len(color_cycle)] for i, a in enumerate(uniq_algos)}

    base_markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', 'p']
    marker_map = {m: base_markers[i % len(base_markers)] for i, m in enumerate(uniq_models)}

    cap_alpha = defaultdict(lambda: 0.9)
    cap_alpha.update({"weak": 0.45, "strong": 0.95})

    # --- Figure ---
    plt.figure(figsize=(8.8, 7.4), dpi=150)
    ax = plt.gca()

    # --- Plot points and optional CIs ---
    x_disp_vals, y_disp_vals = [], []
    for x, y, model, cap, algo, N, M, hits_x, hits_y in points:
        # Compute CIs axis-by-axis if requested and counts known
        x_lo = x_hi = y_lo = y_hi = None
        if show_ci:
            # X-axis (in-dist)
            if isinstance(N, int) and N > 0:
                kx = hits_x if isinstance(hits_x, int) else int(round(x * N))
                kx = min(max(kx, 0), N)
                x_lo, x_hi = _cp_interval_scipy(kx, N, alpha=ci_alpha)
            # Y-axis (out-dist)
            if isinstance(M, int) and M > 0:
                ky = hits_y if isinstance(hits_y, int) else int(round(y * M))
                ky = min(max(ky, 0), M)
                y_lo, y_hi = _cp_interval_scipy(ky, M, alpha=ci_alpha)

        # Convert point to display scale
        x_plot = 100 * x if use_percent_scale else x
        y_plot = 100 * y if use_percent_scale else y
        x_disp_vals.append(x_plot)
        y_disp_vals.append(y_plot)

        # Error bars (for whichever axes we have)
        if show_ci and (x_lo is not None or y_lo is not None):
            xd_lo = xd_hi = yd_lo = yd_hi = None
            if x_lo is not None:
                xd_lo = 100 * x_lo if use_percent_scale else x_lo
                xd_hi = 100 * x_hi if use_percent_scale else x_hi
            if y_lo is not None:
                yd_lo = 100 * y_lo if use_percent_scale else y_lo
                yd_hi = 100 * y_hi if use_percent_scale else y_hi

            xerr = None
            yerr = None
            if xd_lo is not None:
                left  = max(0.0, x_plot - xd_lo)
                right = max(0.0, xd_hi - x_plot)
                xerr = [[left], [right]]
            if yd_lo is not None:
                lower = max(0.0, y_plot - yd_lo)
                upper = max(0.0, yd_hi - y_plot)
                yerr = [[lower], [upper]]

            ax.errorbar(
                x_plot, y_plot,
                xerr=xerr, yerr=yerr,
                fmt='none',
                ecolor='gray',
                elinewidth=1.0,
                capsize=2,
                alpha=0.7,
                zorder=1
            )

        # Scatter point
        ax.scatter(
            x_plot, y_plot,
            s=72,
            marker=marker_map[model],
            c=[color_map[algo]],
            alpha=cap_alpha[cap],
            edgecolor="black",
            linewidths=0.6,
            zorder=2
        )

    # --- y = x baseline (draw in display scale) ---
    lo_val = min(min(x_disp_vals), min(y_disp_vals))
    hi_val = max(max(x_disp_vals), max(y_disp_vals))
    span = max(1e-12, hi_val - lo_val)
    pad = 0.05 * span
    v0, v1 = lo_val - pad, hi_val + pad
    ax.plot([v0, v1], [v0, v1],
            ls="--", color="grey", lw=1.2, alpha=0.8,
            label="y = x (no-generalisation)")

    # --- Labels & limits ---
    if use_percent_scale:
        ax.set_xlabel("In-Distribution Accuracy (%)", labelpad=8)
        ax.set_ylabel("Out-of-Distribution Accuracy (%)", labelpad=8)
        ax.set_xlim(max(0.0, v0), min(100.0, v1))
        ax.set_ylim(max(0.0, v0), min(100.0, v1))
    else:
        ax.set_xlabel("In-Distribution Accuracy", labelpad=8)
        ax.set_ylabel("Out-of-Distribution Accuracy", labelpad=8)
        ax.set_xlim(max(0.0, v0), min(1.0, v1))
        ax.set_ylim(max(0.0, v0), min(1.0, v1))

    ax.set_title(title, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')

    # --- Legends (separate blocks) ---
    color_handles = [Line2D([0], [0], marker='o', color='none',
                            markerfacecolor=color_map[a], markeredgecolor="black",
                            markersize=8, label=a) for a in uniq_algos]
    legend_algos = ax.legend(handles=color_handles, title="Train Algo (color)",
                             loc="upper left", bbox_to_anchor=(1.02, 1.00))
    ax.add_artist(legend_algos)

    marker_handles = [Line2D([0], [0], marker=marker_map[m], color="black",
                             linestyle="none", markersize=8, label=m) for m in uniq_models]
    legend_models = ax.legend(handles=marker_handles, title="Model (shape)",
                              loc="upper left", bbox_to_anchor=(1.02, 0.62))
    ax.add_artist(legend_models)

    shade_color = (0.25, 0.25, 0.25)
    shade_handles = [Line2D([0], [0], marker='o', linestyle='none',
                            markerfacecolor=shade_color, markeredgecolor="black",
                            alpha=cap_alpha[cap], markersize=8, label=cap)
                     for cap in uniq_caps]
    legend_caps = ax.legend(handles=shade_handles, title="Train Cap (shade)",
                            loc="upper left", bbox_to_anchor=(1.02, 0.30))
    ax.add_artist(legend_caps)

    # Baseline legend entry
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.02))
    plt.tight_layout()

    # --- Save or show ---
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
            scores_and_meta_dict = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)]
            accuracy = scores_and_meta_dict[accuracy_type]
            total_count = int(scores_and_meta_dict["correct"] + scores_and_meta_dict["incorrect"])
            if accuracy_type == "consistent_accuracy":
                total_count = total_count // 2
            dist_type = "in_dist" if eval_in_dist else "out_dist"
            if (model_name, train_cap, train_algo) not in train_meta_to_eval_scores[(eval_type, accuracy_type)]:
                train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)] = {}
            train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)][dist_type] = 100.0 * accuracy
            train_meta_to_eval_scores[(eval_type, accuracy_type)][(model_name, train_cap, train_algo)][f"{dist_type}_count"] = total_count

    return train_meta_to_eval_scores

def plot_in_out_scatter_type_6(train_meta_to_eval_scores, plot_dir):
    for eval_type, accuracy_type in train_meta_to_eval_scores:
        plot_data = train_meta_to_eval_scores[(eval_type, accuracy_type)]
        plot_in_out_scatter_type_6_util(plot_data, 
                                        title=f"{eval_type} {accuracy_type}", save_path=os.path.join(plot_dir, f"in_out_scatter_{eval_type}_{accuracy_type}_ci.png"), 
                                        equal_aspect=True, 
                                        show_ci=True, 
                                        ci_alpha=0.05)
        plot_in_out_scatter_type_6_util(plot_data, 
                                        title=f"{eval_type} {accuracy_type}", save_path=os.path.join(plot_dir, f"in_out_scatter_{eval_type}_{accuracy_type}.png"), 
                                        equal_aspect=True, 
                                        show_ci=False, 
                                        ci_alpha=0.05)

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

def create_generalization_latex_table(gen_meta_to_gen_scores, plot_dir):
    """Create a LaTeX table for generalization metrics."""
    
    # Create the table data
    table_data = []
    
    for model in ["llama8b", "ministral8b", "mistral24b"]:
        for algo in ['sft', 'dpo', 'sft_dpo']:
            # Get the generalization metrics for this model-algorithm combination
            ws_metrics = gen_meta_to_gen_scores.get((algo, model, "weak->strong"), {})
            sw_metrics = gen_meta_to_gen_scores.get((algo, model, "strong->weak"), {})
            
            if ws_metrics and sw_metrics:
                # Extract the values we need for the table
                # CM_St^Se: strong->weak seen (absolute degradation)
                cm_st_se = ws_metrics.get("seen_questions_unseen_answers", {}).get("accuracy", {}).get("absolute_degradation", 0.0)
                
                # CM_Wk^Se: weak->strong seen (absolute degradation) 
                cm_wk_se = sw_metrics.get("seen_questions_unseen_answers", {}).get("accuracy", {}).get("absolute_degradation", 0.0)
                
                # CM_St^Us: strong->weak unseen (absolute degradation)
                cm_st_us = ws_metrics.get("unseen_questions_unseen_answers", {}).get("accuracy", {}).get("absolute_degradation", 0.0)
                
                # CM_Wk^Us: weak->strong unseen (absolute degradation)
                cm_wk_us = sw_metrics.get("unseen_questions_unseen_answers", {}).get("accuracy", {}).get("absolute_degradation", 0.0)
                
                # Calculate deltas
                delta_se = cm_st_se - cm_wk_se  # Δ^Se = CM_St^Se - CM_Wk^Se
                delta_us = cm_st_us - cm_wk_us  # Δ^Us = CM_St^Us - CM_Wk^Us
                delta_st = cm_st_se - cm_st_us  # Δ_St = CM_St^Se - CM_St^Us
                delta_wk = cm_wk_se - cm_wk_us  # Δ_Wk = CM_Wk^Se - CM_Wk^Us
                
                table_data.append({
                    'model': model,
                    'algo': algo,
                    'cm_st_se': cm_st_se,
                    'cm_wk_se': cm_wk_se,
                    'cm_st_us': cm_st_us,
                    'cm_wk_us': cm_wk_us,
                    'delta_se': delta_se,
                    'delta_us': delta_us,
                    'delta_st': delta_st,
                    'delta_wk': delta_wk
                })
    
    # Generate LaTeX table
    latex_table = generate_generalization_latex_table(table_data)
    
    # Save to file
    output_path = os.path.join(plot_dir, "generalization_table.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"Generalization table saved to: {output_path}")
    
    return latex_table

def generate_generalization_latex_table(table_data):
    """Generate the LaTeX table with proper formatting."""
    
    # Define color commands for deltas
    color_definitions = """% Color definitions for deltas
\\definecolor{pos1}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{pos2}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{pos3}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{pos4}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{pos5}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{neg1}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{neg2}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{neg3}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{neg4}{RGB}{255, 255, 255}  % white (no color)
\\definecolor{neg5}{RGB}{255, 255, 255}  % white (no color)

\\newcommand{\\DposOne}[1]{\\textcolor{pos1}{#1}}
\\newcommand{\\DposTwo}[1]{\\textcolor{pos2}{#1}}
\\newcommand{\\DposThree}[1]{\\textcolor{pos3}{#1}}
\\newcommand{\\DposFour}[1]{\\textcolor{pos4}{#1}}
\\newcommand{\\DposFive}[1]{\\textcolor{pos5}{#1}}
\\newcommand{\\DnegOne}[1]{\\textcolor{neg1}{#1}}
\\newcommand{\\DnegTwo}[1]{\\textcolor{neg2}{#1}}
\\newcommand{\\DnegThree}[1]{\\textcolor{neg3}{#1}}
\\newcommand{\\DnegFour}[1]{\\textcolor{neg4}{#1}}
\\newcommand{\\DnegFive}[1]{\\textcolor{neg5}{#1}}
"""
    
    latex = color_definitions + """
% ======================= TABLE ==========================
\\begin{table}[t]
\\centering
\\footnotesize
\\renewcommand{\\arraystretch}{1.1}
\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{l|
  S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] |
  S[table-format=2.2] S[table-format=2.2] |
  S[table-format=2.2] S[table-format=2.2]}
\\toprule
\\textbf{Judge Model} &
\\multicolumn{1}{|c}{$CT_{\\mathrm{St}}^{\\mathrm{Se}}$} &
\\multicolumn{1}{c}{$CT_{\\mathrm{Wk}}^{\\mathrm{Se}}$} &
\\multicolumn{1}{c}{$CT_{\\mathrm{St}}^{\\mathrm{Us}}$} &
\\multicolumn{1}{c}{$CT_{\\mathrm{Wk}}^{\\mathrm{Us}}$} &
\\multicolumn{1}{|c}{$\\Delta^{\\mathrm{Se}}$} &
\\multicolumn{1}{c}{$\\Delta^{\\mathrm{Us}}$} &
\\multicolumn{1}{|c}{$\\Delta_{\\mathrm{St}}$} &
\\multicolumn{1}{c}{$\\Delta_{\\mathrm{Wk}}$} \\\\
\\midrule
"""
    
    # Group data by model
    models_data = {}
    for row in table_data:
        model = row['model']
        if model not in models_data:
            models_data[model] = []
        models_data[model].append(row)
    
    # Add rows for each model
    for model in ["llama8b", "ministral8b", "mistral24b"]:
        if model in models_data:
            # Model name
            model_display = model.replace("ministral", "Ministral").replace("mistral", "Mistral").replace("llama", "Llama3")
            latex += f"\\texttt{{{model_display}-8B}}" if "8B" not in model_display else f"\\texttt{{{model_display}}}"
            latex += " \\\\\n"
            
            # Algorithm rows
            for row in models_data[model]:
                algo = row['algo']
                algo_display = algo.replace('_', '+').upper()
                
                # Format values
                cm_st_se = f"{row['cm_st_se']:.3f}"
                cm_wk_se = f"{row['cm_wk_se']:.3f}"
                cm_st_us = f"{row['cm_st_us']:.3f}"
                cm_wk_us = f"{row['cm_wk_us']:.3f}"
                
                # Format deltas with color commands
                delta_se = format_generalization_delta(row['delta_se'])
                delta_us = format_generalization_delta(row['delta_us'])
                delta_st = format_generalization_delta(row['delta_st'])
                delta_wk = format_generalization_delta(row['delta_wk'])
                
                latex += f"\\hspace{{1em}}+ {algo_display:<8} & {cm_st_se} & {cm_wk_se} & {cm_st_us} & {cm_wk_us} & {delta_se} & {delta_us} & {delta_st} & {delta_wk} \\\\\n"
            
            # Add midrule between models (except for the last one)
            if model != "mistral24b":
                latex += "\\midrule\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\caption{
\\textit{Capability-Transfer Gain (CT).}
}
\\label{tab:ct-deltas-right}
\\end{table}
% ========================================================"""
    
    return latex

def format_generalization_delta(value):
    """Format delta values with appropriate color commands."""
    if value >= 0:
        if value >= 5:
            return f"\\DposFive{{{value:6.3f}}}"
        elif value >= 2:
            return f"\\DposFour{{{value:6.3f}}}"
        elif value >= 1:
            return f"\\DposThree{{{value:6.3f}}}"
        elif value >= 0.5:
            return f"\\DposTwo{{{value:6.3f}}}"
        else:
            return f"\\DposOne{{{value:6.3f}}}"
    else:
        if value <= -5:
            return f"\\DnegFive{{{value:6.3f}}}"
        elif value <= -2:
            return f"\\DnegFour{{{value:6.3f}}}"
        elif value <= -1:
            return f"\\DnegThree{{{value:6.3f}}}"
        elif value <= -0.5:
            return f"\\DnegTwo{{{value:6.3f}}}"
        else:
            return f"\\DnegOne{{{value:6.3f}}}"

def main():
    agg_scores = read_eval_results("./eval-results")

    model_names = ["ministral8b", "mistral24b", "llama8b"]
    split_types = ["seen_questions_unseen_answers", "unseen_questions_unseen_answers"]
    generalization_types = ["weak->strong", "strong->weak"]
    metrics = ["accuracy", "consistent_accuracy"]
    degradation_types = ["absolute_degradation", "relative_degradation",
                         "normalized_absolute_degradation", "normalized_relative_degradation"]

    # create the type 3 plots and save the dataframe corresponding to it.
    plot_dir = "./eval-plots/plot_type_3"
    os.makedirs(plot_dir, exist_ok=True)

    gen_meta_to_gen_scores = compute_generalization_metrics_type_3(agg_scores, split_types, metrics, degradation_types, generalization_types)
    df = create_dataframe_for_plots_type_3(gen_meta_to_gen_scores)
    create_generalization_plots_type_3(df, plot_dir, model_names, split_types, metrics, degradation_types, generalization_types)

    # Save the dataframe as a CSV file in the plot_dir
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)
    
    # Create and save the LaTeX table
    create_generalization_latex_table(gen_meta_to_gen_scores, plot_dir)

    # create the type 6 plots (scatter plots) and save the dataframe corresponding to it.
    plot_dir = "./eval-plots/plot_type_6"
    os.makedirs(plot_dir, exist_ok=True)
    train_meta_to_eval_scores = compute_generalization_metrics_type_6(agg_scores, split_types, metrics, degradation_types, generalization_types)
    plot_in_out_scatter_type_6(train_meta_to_eval_scores, plot_dir)
    df = create_dataframe_for_plots_type_6(train_meta_to_eval_scores)
    df.to_csv(os.path.join(plot_dir, "dataframe.csv"), index=False)


if __name__ == "__main__":
    main()