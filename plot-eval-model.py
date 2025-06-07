from huggingface_hub import HfApi
import os
from datetime import datetime, timezone
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate

def list_commit_dates(repo_id: str, branch: str = "main", repo_type: str = "model", *, token: str | None = None):
    api = HfApi(token=token)

    commits = api.list_repo_commits(
        repo_id,
        repo_type=repo_type,
        revision=branch,          # branch, tag, or SHA
    )

    first_commit_date = None
    for c in reversed(commits):
        dt = c.created_at
        if first_commit_date is None:
            first_commit_date = dt.date()
    
    return first_commit_date

def classify_model(model_name):
    model_date = list_commit_dates(
        repo_id=model_name,
        token=os.getenv("HF_TOKEN") 
    )

    model_family = "unknown"
    if "qwen" in model_name.lower():
        model_family = "qwen"
    elif "llama" in model_name.lower():
        model_family = "llama"
    elif "mistral" in model_name.lower() or "ministral" in model_name.lower():
        model_family = "mistral"
    elif "gemma" in model_name.lower():
        model_family = "gemma"
    else:
        raise ValueError(f"Unknown model family: {model_name}")
    
    model_size = "unknown"
    model_name_parts = model_name.split("-")
    for part in model_name_parts:
        if len(part) <= 3 and 'b' in part.lower():
            model_size = part
            break
    assert model_size != "unknown", f"Unknown model size: {model_name}"
    model_size = int(model_size[:-1])

    if model_size <= 5:
        model_size_type = "tiny"
    elif model_size <= 10:
        model_size_type = "small"
    elif model_size <= 20:
        model_size_type = "medium"
    else:
        model_size_type = "large"
    assert model_size_type != "unknown", f"Unknown model size type: {model_name}"

    return model_name.split("/")[-1], model_family, model_size, model_size_type, model_date

def read_eval_files(folder_path, dataset_hash):
    results = {}
    eval_dir = folder_path
    
    # Read all json files in the eval directory
    for filename in os.listdir(eval_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(eval_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                config = data["config"]
                accuracy = data["accuracy"]
                if dataset_hash == data["dataset_hash"]:
                    model_name, model_family, model_size, model_size_type, model_date = classify_model(config["model"])
                    results[os.path.basename(filename)] = {
                        "model": model_name,
                        "model_family": model_family,
                        "model_size": f"{model_size}B",
                        "model_size_type": model_size_type,
                        "model_date": model_date.strftime("%Y-%m-%d"),
                        "accuracy": round(100.0 * accuracy, 2)
                    }
                
    return results


if __name__ == "__main__":
    results = read_eval_files("/shared/storage-01/users/jvsingh2/judge-gen-eval/outputs/eval-model", "1f76947c3c9aa9707e05ee3e5f7b662d479693995f367a9d563a6ffda3b4a3cd")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results.values())
    
    # Convert dates to datetime
    df['model_date'] = pd.to_datetime(df['model_date'])
    
    # Define markers for each model family using clearly distinguishable shapes
    markers = {
        'qwen': '*',      # star
        'mistral': 'D',   # diamond
        'gemma': '^',     # triangle up
        'llama': 's',     # square
    }
    
    # Define colors for each size type using distinguishable colors
    colors = {
        'tiny': '#E69F00',     # orange
        'small': '#56B4E9',    # sky blue
        'medium': '#009E73',   # green
        'large': '#CC79A7'     # pink
    }
    
    plt.figure(figsize=(15, 10))  # Larger figure size
    
    # Add small random jitter to accuracy to better show overlapping points
    jitter = 0.2
    df['accuracy_jittered'] = df['accuracy'] + np.random.uniform(-jitter, jitter, len(df))
    
    # Sort by accuracy to plot higher accuracy points on top
    df = df.sort_values('accuracy')
    
    # Plot the actual data first with transparency
    for family in df['model_family'].unique():
        for size_type in df['model_size_type'].unique():
            mask = (df['model_family'] == family) & (df['model_size_type'] == size_type)
            if mask.any():
                # Plot points
                plt.scatter(df[mask]['model_date'], 
                          df[mask]['accuracy_jittered'],
                          marker=markers[family],
                          c=colors[size_type],
                          s=200,  # larger points
                          alpha=0.8,  # slightly less transparency
                          edgecolors='black',
                          linewidth=1.5)  # thicker edges
                
                # Add text labels for each point with smart positioning
                for idx in df[mask].index:
                    # Alternate between different positions around the point
                    position_idx = idx % 8
                    dx = 7 * (position_idx in [1, 2, 3]) - 7 * (position_idx in [5, 6, 7])
                    dy = 7 * (position_idx in [7, 0, 1]) - 7 * (position_idx in [3, 4, 5])
                    
                    plt.annotate(df.loc[idx, 'model'],
                               (df.loc[idx, 'model_date'], df.loc[idx, 'accuracy_jittered']),
                               xytext=(dx, dy), textcoords='offset points',
                               fontsize=9,  # slightly larger font
                               fontweight='bold',  # bold text
                               alpha=1.0,  # full opacity for text
                               bbox=dict(
                                   facecolor='white',
                                   edgecolor='gray',
                                   alpha=0.9,
                                   pad=2,
                                   boxstyle='round,pad=0.5'
                               ),
                               ha='left' if dx >= 0 else 'right',  # align text based on position
                               va='bottom' if dy >= 0 else 'top')
    
    # Create legend handles for shapes (model families)
    shape_handles = [plt.scatter([], [], marker=marker, c='black', s=150, label=family)
                    for family, marker in markers.items()]
    
    # Create legend handles for colors (size types)
    color_handles = [plt.scatter([], [], marker='o', c=color, s=150, label=size_type)
                    for size_type, color in colors.items()]
    
    plt.xlabel('Model Release Date', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Performance by Release Date', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add both legends with larger font
    first_legend = plt.legend(handles=shape_handles, title='Model Family', 
                            bbox_to_anchor=(1.05, 1), loc='upper left', 
                            title_fontsize=12, fontsize=10)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=color_handles, title='Model Size', 
              bbox_to_anchor=(1.05, 0.5), loc='upper left',
              title_fontsize=12, fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legends on the right
    
    # Save the plot
    plt.savefig('model_performance.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Create and save table
    table_df = pd.DataFrame(results.values())
    
    # Sort by accuracy (descending)
    table_df = table_df.sort_values('accuracy', ascending=False)
    
    # Format accuracy to 2 decimal places with % symbol after sorting
    table_df['accuracy'] = table_df['accuracy'].apply(lambda x: f"{x:.2f}%")
    
    # Reorder columns
    table_df = table_df[['model', 'model_family', 'model_size', 'model_size_type', 'model_date', 'accuracy']]
    
    # Save as CSV
    table_df.to_csv('model_performance_table.csv', index=False)
    
    # Create markdown table with a header indicating sorting
    markdown_table = "# Model Performance Results (Sorted by Accuracy)\n\n"
    markdown_table += tabulate(table_df.values, headers=table_df.columns, tablefmt="pipe")
    
    # Save markdown table
    with open('model_performance_table.md', 'w') as f:
        f.write(markdown_table)
    
    # Print the table to console
    print("\nModel Performance Table (Sorted by Accuracy):")
    print(table_df.to_string(index=False))
    