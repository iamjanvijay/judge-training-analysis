import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add analysis to path for importing common functions
sys.path.append("./analysis")
from read_eval_results import read_eval_results

def plot_accuracy_curve(scores, set_name, model_name, train_algo, train_cap, accuracy_type, plot_path):
    """
    Input: 
        - a (set_name, model_name, train_algo, train_cap)
        - scores: a dictionary of scores for a (set_name, model_name, train_algo, train_cap)
            - accuracy at for a eval_type, eval_cap at a ckpt_step is given by scores[eval_type][eval_cap][ckpt_step]["accuracy"]
    Output: a plot of accuracy (y-axis) vs. ckpt_step (x-axis).
    Each plot has five line curves. 
        - if eval_cap == train_cap and eval_type == "seen_questions_seen_answers", then live curve is gray in color. # train subset
        - if eval_cap == train_cap and eval_type == "seen_questions_unseen_answers", then live curve is light-green in color. (same answer dist)
        - if eval_cap == train_cap and eval_type == "unseen_questions_unseen_answers", then live curve is dark-green in color. # unseen questions (same answer dist)
        - if eval_cap != train_cap and eval_type == "seen_questions_unseen_answers", then live curve is light-red in color. (different answer dist)
        - if eval_cap != train_cap and eval_type == "unseen_questions_unseen_answers", then live curve is dark-red in color. # unseen questions (different answer dist)

    """
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for different evaluation types
    colors = {
        ('seen_questions_seen_answers', True): 'gray',      # eval_cap == train_cap
        ('seen_questions_unseen_answers', True): 'lightgreen',  # eval_cap == train_cap
        ('unseen_questions_unseen_answers', True): 'darkgreen', # eval_cap == train_cap
        ('seen_questions_unseen_answers', False): 'lightcoral', # eval_cap != train_cap
        ('unseen_questions_unseen_answers', False): 'darkred',  # eval_cap != train_cap
    }
    
    # Define line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Track all checkpoint steps to set x-axis properly
    all_ckpt_steps = set()
    
    # Plot each evaluation type
    for eval_type in scores:
        for eval_cap in scores[eval_type]:
            # Determine if eval_cap matches train_cap
            cap_match = (eval_cap == train_cap)
            
            # Get the color for this combination
            color_key = (eval_type, cap_match)
            if color_key in colors:
                color = colors[color_key]
            else:
                # Fallback color if not in our defined set
                color = 'blue'
            
            # Extract checkpoint steps and accuracies
            ckpt_steps = []
            accuracies = []
            
            for ckpt_step in scores[eval_type][eval_cap]:
                all_ckpt_steps.add(ckpt_step)
                ckpt_steps.append(ckpt_step)
                accuracy = scores[eval_type][eval_cap][ckpt_step][accuracy_type]
                accuracies.append(accuracy)
            
            # Sort by checkpoint step
            sorted_data = sorted(zip(ckpt_steps, accuracies))
            ckpt_steps, accuracies = zip(*sorted_data)
            
            # Create label for legend
            cap_status = "same" if cap_match else "different"
            label = f"{eval_type.replace('_', ' ')} ({cap_status} cap)"
            
            # Plot the line
            plt.plot(ckpt_steps, accuracies, 
                    color=color, 
                    linewidth=2.5, 
                    marker='o', 
                    markersize=6, 
                    label=label,
                    alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Checkpoint Step', fontsize=14, fontweight='bold')
    plt.ylabel(accuracy_type.upper(), fontsize=14, fontweight='bold')
    
    # Set title with model and training information
    title = f"{accuracy_type.upper()} Curves: {model_name.upper()} - {train_algo.upper()} - {train_cap.upper()}\nDataset: {set_name}"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Customize grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis to show all checkpoint steps
    if all_ckpt_steps:
        plt.xticks(sorted(all_ckpt_steps), fontsize=12)
    
    # Set y-axis limits with some padding
    plt.ylim(0, 1.05)
    
    # Add legend with better positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")


def create_plots(agg_scores, plot_base_path):

    ### PLOTS TYPE 1: 
    ### accuracy with x-axis as ckpt_step, and y-axis as accuracy, and multiple eval splits as line curves.
    plots = {}
    for (set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type) in agg_scores:
        if (set_name, model_name, train_algo, train_cap) not in plots:
            plots[(set_name, model_name, train_algo, train_cap)] = {}
        if eval_type not in plots[(set_name, model_name, train_algo, train_cap)]:
            plots[(set_name, model_name, train_algo, train_cap)][eval_type] = {}
        if eval_cap not in plots[(set_name, model_name, train_algo, train_cap)][eval_type]:
            plots[(set_name, model_name, train_algo, train_cap)][eval_type][eval_cap] = {}
        plots[(set_name, model_name, train_algo, train_cap)][eval_type][eval_cap][ckpt_step] = agg_scores[(set_name, train_algo, train_cap, ckpt_step, model_name, eval_cap, eval_type)]
    
    # Remove plots with just 1 ckpt_steps
    plots_to_remove = []
    for (set_name, model_name, train_algo, train_cap) in plots:
        for eval_type in plots[(set_name, model_name, train_algo, train_cap)]:
            for eval_cap in plots[(set_name, model_name, train_algo, train_cap)][eval_type]:
                ckpt_steps_count = len(plots[(set_name, model_name, train_algo, train_cap)][eval_type][eval_cap])
                if ckpt_steps_count <= 1:
                    plots_to_remove.append((set_name, model_name, train_algo, train_cap, eval_type, eval_cap))
    for set_name, model_name, train_algo, train_cap, eval_type, eval_cap in plots_to_remove:
        del plots[(set_name, model_name, train_algo, train_cap)][eval_type][eval_cap]
    for set_name, model_name, train_algo, train_cap, eval_type, _ in plots_to_remove:
        if eval_type in plots[(set_name, model_name, train_algo, train_cap)] and len(plots[(set_name, model_name, train_algo, train_cap)][eval_type]) == 0:
            del plots[(set_name, model_name, train_algo, train_cap)][eval_type]
    for set_name, model_name, train_algo, train_cap, _, _ in plots_to_remove:
        if (set_name, model_name, train_algo, train_cap) in plots and len(plots[(set_name, model_name, train_algo, train_cap)]) == 0:
            del plots[(set_name, model_name, train_algo, train_cap)]

    # Create the plots
    plot_path_prefix = os.path.join(plot_base_path, "plot_type_1")
    os.makedirs(plot_path_prefix, exist_ok=True)
    for (set_name, model_name, train_algo, train_cap) in plots:
        plot_path = os.path.join(plot_path_prefix, f"{set_name}.{model_name}.{train_algo}.{train_cap}.accuracy.png")
        plot_accuracy_curve(plots[(set_name, model_name, train_algo, train_cap)], set_name, model_name, train_algo, train_cap, "accuracy", plot_path)
        plot_path = os.path.join(plot_path_prefix, f"{set_name}.{model_name}.{train_algo}.{train_cap}.consistent_accuracy.png")
        plot_accuracy_curve(plots[(set_name, model_name, train_algo, train_cap)], set_name, model_name, train_algo, train_cap, "consistent_accuracy", plot_path)

    ### PLOTS TYPE 2: 
    ### accuracy with x-axis as ckpt_step, and y-axis as incorrect_format_rate, and multiple eval splits as line curves.

    # Create the plots
    plot_path_prefix = os.path.join(plot_base_path, "plot_type_2")
    os.makedirs(plot_path_prefix, exist_ok=True)
    for (set_name, model_name, train_algo, train_cap) in plots:
        plot_path = os.path.join(plot_path_prefix, f"{set_name}.{model_name}.{train_algo}.{train_cap}.incorrect_format_rate.png")
        plot_accuracy_curve(plots[(set_name, model_name, train_algo, train_cap)], set_name, model_name, train_algo, train_cap, "incorrect_format_rate", plot_path)

def main():
    agg_scores = read_eval_results("./eval-results")
    create_plots(agg_scores, "./eval-plots")

if __name__ == "__main__":
    main()