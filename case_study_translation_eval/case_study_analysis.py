import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Define constants
EXCLUDED_LANGUAGES = [
    "ckb-Arab", "kin", "tgk-Cyrl", "ewe", "ssw", "bem", "ltz", "mey", "dzo", 
    "nno", "ven", "gsw-ZH", "bel", "hmn", "wol", "gsw-BE", "shi", "fuc", 
    "tsn", "orm", "nde", "eng-IN", "eng-GB"
]

# Model information
MODELS = {
    "meta-llama/Llama-2-7b-hf": "Llama-2-7b",
    "MaLA-LM/emma-500-llama2-7b": "EMMA-500"
}

# Prompt strategies
PROMPT_STRATEGIES = {
    "multi": "Multi-prompt",
    "Eng": "English prompt",
    "Fin": "Finnish prompt",
    "Zho": "Chinese prompt"
}

# Translation tasks we want to analyze - reordered to group by language
TRANSLATION_TASKS = [
    "eng-US->", 
    "->eng-US", 
    "zho-CN->", 
    "->zho-CN", 
    "fin->", 
    "->fin"
]

# Split tasks into source and target categories
SOURCE_TASKS = ["eng-US->", "zho-CN->", "fin->"]
TARGET_TASKS = ["->eng-US", "->zho-CN", "->fin"]

# Colors for different prompt strategies
COLORS = {
    "multi": "blue",
    "Eng": "green",
    "Fin": "purple",
    "Zho": "orange"
}

# Metrics to use
METRICS = ["chrf_score", "bleu_score"]
METRIC_DISPLAY_NAMES = {"chrf_score": "ChrF Score", "bleu_score": "BLEU Score"}

def extract_model_from_filename(filename):
    """Extract model name from the filename."""
    if "Llama-2-7b-hf" in filename:
        return "meta-llama/Llama-2-7b-hf"
    elif "emma-500-llama2-7b" in filename:
        return "MaLA-LM/emma-500-llama2-7b"
    return None

def extract_prompt_strategy_from_filename(filename):
    """Extract prompt strategy from the filename."""
    if filename.startswith("scores_multi_"):
        return "multi"
    elif filename.startswith("scores_Eng_"):
        return "Eng"
    elif filename.startswith("scores_Fin_"):
        return "Fin"
    elif filename.startswith("scores_Zho_"):
        return "Zho"
    return None

def extract_translation_direction(results_key):
    """Determine the translation direction category based on the results key."""
    for task in TRANSLATION_TASKS:
        if task.startswith("->"):
            # Check if it ends with this target language
            if results_key.endswith(task[2:]):
                return task
        elif task.endswith("->"):
            # Check if it starts with this source language
            if results_key.startswith(task[:-2]):
                return task
    return None

def analyze_reorganized_data(data_dir, metric="chrf_score", verbose=True):
    """
    Analyze the reorganized data from JSON files.
    
    Args:
        data_dir: Directory containing the JSON files
        metric: Metric to use for evaluation
        verbose: Whether to print detailed logging information
    
    Returns:
        DataFrame of all processed data and dictionary of processed experiments
    """
    # Get all JSON files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if verbose:
        print(f"Found {len(json_files)} JSON files")
    
    # Dictionary to keep track of processed experiments
    processed_experiments = {}
    
    # List to store all data points
    all_data = []
    
    # Track statistics for logging
    stats = {
        'total_files': len(json_files),
        'processed_files': 0,
        'skipped_model_strategy': 0,
        'skipped_better_data': 0,
        'skipped_eng_x': 0,
        'skipped_errors': 0,
        'total_language_pairs': 0
    }
    
    # Process each JSON file
    for json_file in json_files:
        filename = os.path.basename(json_file)
        
        # Extract information from filename
        model = extract_model_from_filename(filename)
        prompt_strategy = extract_prompt_strategy_from_filename(filename)
        
        # Skip if we couldn't extract model or strategy
        if not (model and prompt_strategy):
            if verbose:
                print(f"Skipping {filename}: Could not extract model or strategy")
            stats['skipped_model_strategy'] += 1
            continue
        
        # Load and parse the JSON file
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get benchmark results
            results = data["benchmarks"]["ntrex128"]["results"]
            
            # Dictionary to keep track of translation tasks in this file
            file_tasks = {}
            
            # First pass: identify all translation tasks in this file
            for lang_pair in results.keys():
                # Skip excluded languages
                is_excluded = any(excluded in lang_pair for excluded in EXCLUDED_LANGUAGES)
                if is_excluded:
                    continue
                
                # Extract translation direction
                translation_task = extract_translation_direction(lang_pair)
                if not translation_task:
                    continue
                
                # Count this task
                file_tasks[translation_task] = file_tasks.get(translation_task, 0) + 1
            
            # Second pass: process data if this is the best file for each task
            file_processed = False
            for task, count in file_tasks.items():
                # Create a key for this experiment combination
                exp_key = (model, prompt_strategy, task)
                
                # Check if we've already processed this combination with more data
                if exp_key in processed_experiments and processed_experiments[exp_key]['count'] >= count:
                    if verbose:
                        print(f"Skipping {filename} for task {task}: Already have better data ({processed_experiments[exp_key]['filename']} with {processed_experiments[exp_key]['count']} items)")
                    stats['skipped_better_data'] += 1
                    continue
                
                # This is the best file for this combination so far
                processed_experiments[exp_key] = {
                    'filename': filename,
                    'count': count
                }
                
                # Remove any existing data for this combination
                all_data = [d for d in all_data if not (
                    d['model'] == model and 
                    d['prompt_strategy'] == prompt_strategy and 
                    d['translation_task'] == task
                )]
                
                # Process language pairs for this task
                pairs_processed = 0
                for lang_pair, metrics in results.items():
                    # Skip excluded languages
                    is_excluded = any(excluded in lang_pair for excluded in EXCLUDED_LANGUAGES)
                    if is_excluded:
                        continue
                        
                    # Extract translation direction
                    pair_task = extract_translation_direction(lang_pair)
                    if pair_task != task:
                        continue
                    
                    # Add to all data - include both metrics
                    all_data.append({
                        'model': model,
                        'model_name': MODELS[model],
                        'prompt_strategy': prompt_strategy,
                        'strategy_name': PROMPT_STRATEGIES.get(prompt_strategy, prompt_strategy),
                        'language_pair': lang_pair,
                        'translation_task': task,
                        'chrf_score': metrics.get("chrf_score", 0),
                        'bleu_score': metrics.get("bleu_score", 0)
                    })
                    pairs_processed += 1
                
                if verbose:
                    print(f"Processed {filename} for task {task}: {pairs_processed} language pairs")
                
                file_processed = True
                stats['total_language_pairs'] += pairs_processed
            
            if file_processed:
                stats['processed_files'] += 1
                
        except (json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"Error processing {json_file}: {e}")
            stats['skipped_errors'] += 1
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data)
    
    # Print summary statistics
    print(f"\nProcessing Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Files processed: {stats['processed_files']}")
    print(f"Files skipped (could not extract model/strategy): {stats['skipped_model_strategy']}")
    print(f"Files skipped (better file exists): {stats['skipped_better_data']}")
    print(f"Translation tasks skipped (eng-X): {stats['skipped_eng_x']}")
    print(f"Files skipped (errors): {stats['skipped_errors']}")
    print(f"Total language pairs processed: {stats['total_language_pairs']}")
    print(f"Unique experiment combinations: {len(processed_experiments)}")
    
    return df, processed_experiments

def reorganize_data(df, metric="chrf_score"):
    """
    Reorganize data to pair EMMA and Llama scores for the same language pair and prompt strategy.
    Returns a dataframe with columns for language_pair, prompt_strategy, translation_task, 
    emma_score, and llama_score.
    """
    # Create a new empty dataframe to store the paired scores
    paired_data = []
    
    # Get all unique combinations of language_pair, prompt_strategy, translation_task
    unique_combos = df[['language_pair', 'prompt_strategy', 'translation_task']].drop_duplicates()
    
    # For each unique combination
    for _, combo in unique_combos.iterrows():
        lang_pair = combo['language_pair']
        strategy = combo['prompt_strategy']
        task = combo['translation_task']
        
        # Get data for this specific combination
        combo_df = df[
            (df['language_pair'] == lang_pair) & 
            (df['prompt_strategy'] == strategy) & 
            (df['translation_task'] == task)
        ]
        
        # If we have data for both models
        if len(combo_df) == 2 and set(combo_df['model'].values) == set(MODELS.keys()):
            # Extract scores for each model
            emma_score = combo_df[combo_df['model'] == 'MaLA-LM/emma-500-llama2-7b'][metric].values[0]
            llama_score = combo_df[combo_df['model'] == 'meta-llama/Llama-2-7b-hf'][metric].values[0]
            
            # Add to paired data
            paired_data.append({
                'language_pair': lang_pair,
                'prompt_strategy': strategy,
                'translation_task': task,
                'emma_score': emma_score,
                'llama_score': llama_score
            })
    
    paired_df = pd.DataFrame(paired_data)
    print(f"\nSuccessfully paired {len(paired_df)} language pairs with {metric} scores from both models")
    return paired_df

def plot_customized_comparison(df, metric="chrf_score", output_dir="visualizations"):
    """
    Create customized plots comparing EMMA-500 vs Llama-2 scores for different translation tasks.
    Upper tasks (Eng->, Zho->, Fin->) only use English and task-specific prompts.
    Lower tasks (->Eng, ->Zho, ->Fin) use English, Multi, and task-specific prompts.
    
    Args:
        df: DataFrame with translation scores
        metric: Metric to use for evaluation (chrf_score or bleu_score)
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First reorganize the data to pair the scores
    paired_df = reorganize_data(df, metric)
    
    metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    
    print(f"Plotting customized comparison for {len(paired_df)} language pairs using {metric}")
    
    # Create a figure with 3x2 subplots (transposed from original 2x3)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()
    
    # Create a legend for prompt strategies
    legend_elements = []
    for strategy, color in COLORS.items():
        # Only include strategies that are present in the filtered data
        if strategy in paired_df["prompt_strategy"].values:
            strategy_name = PROMPT_STRATEGIES.get(strategy, strategy)
            legend_elements.append(
                Line2D([0], [0], marker='o', color=color, markerfacecolor=color, 
                       markersize=8, linestyle='', label=strategy_name)
            )
    
    # Find global min and max for consistent axes
    global_min = min(paired_df["emma_score"].min(), paired_df["llama_score"].min()) * 0.95  # Add 5% padding
    global_max = max(paired_df["emma_score"].max(), paired_df["llama_score"].max()) * 1.05
    
    # Create individual folders for each task
    task_folder = os.path.join(output_dir, "individual_tasks")
    os.makedirs(task_folder, exist_ok=True)
    
    # Process each translation task
    for i, task in enumerate(TRANSLATION_TASKS):
        ax = axes[i]
        
        # Filter task-specific data
        task_paired_df = paired_df[paired_df["translation_task"] == task]
        
        # Skip if no data for this task
        if task_paired_df.empty:
            ax.text(0.5, 0.5, f"No data for {task}", ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Apply custom filtering based on task type
        if task in SOURCE_TASKS:
            # For source tasks, only use English and task-specific prompts
            task_lang = task[:3]  # Extract language code from task
            if task_lang == "eng":
                # For English source, only show English prompt
                task_paired_df = task_paired_df[task_paired_df['prompt_strategy'] == 'Eng']
            elif task_lang == "zho":
                # For Chinese source, show English and Chinese prompts
                task_paired_df = task_paired_df[task_paired_df['prompt_strategy'].isin(['Eng', 'Zho'])]
            elif task_lang == "fin":
                # For Finnish source, show English and Finnish prompts
                task_paired_df = task_paired_df[task_paired_df['prompt_strategy'].isin(['Eng', 'Fin'])]
        else:
            # For target tasks, use English, Multi, and task-specific prompts
            task_lang = task[2:]  # Extract language code from task
            if task_lang == "eng-US":
                task_paired_df = task_paired_df[task_paired_df['prompt_strategy'].isin(['Eng', 'multi'])]
            elif task_lang == "zho-CN":
                task_paired_df = task_paired_df[task_paired_df['prompt_strategy'].isin(['Eng', 'Zho', 'multi'])]
            elif task_lang == "fin":
                task_paired_df = task_paired_df[task_paired_df['prompt_strategy'].isin(['Eng', 'Fin', 'multi'])]
        
        # Dictionary to store points by language pair and prompt strategy
        points_by_lang = {}
        
        # Group by prompt strategy
        for strategy, strategy_df in task_paired_df.groupby("prompt_strategy"):
            if strategy not in COLORS:
                continue
                
            # Plot all points for this strategy
            scatter = ax.scatter(
                strategy_df["llama_score"],
                strategy_df["emma_score"],
                color=COLORS.get(strategy, 'gray'),
                marker='o',
                s=50,
                alpha=0.7,
                label=PROMPT_STRATEGIES.get(strategy, strategy)
            )
            
            # Store points for connecting lines
            for _, row in strategy_df.iterrows():
                lang_pair = row['language_pair']
                if lang_pair not in points_by_lang:
                    points_by_lang[lang_pair] = {}
                
                points_by_lang[lang_pair][strategy] = (row["llama_score"], row["emma_score"])
        
        # Draw connecting lines between models for the same language pair
        for lang_pair, strategies in points_by_lang.items():
            # Find all strategies we have for this language pair
            present_strategies = list(strategies.keys())
            
            # Draw lines between all pairs of strategies
            for i in range(len(present_strategies)):
                for j in range(i+1, len(present_strategies)):
                    s1 = present_strategies[i]
                    s2 = present_strategies[j]
                    
                    point1 = strategies[s1]
                    point2 = strategies[s2]
                    
                    # Draw a line connecting the points
                    ax.plot(
                        [point1[0], point2[0]],
                        [point1[1], point2[1]],
                        color='gray',
                        linestyle='-',
                        alpha=0.3,
                        linewidth=0.5
                    )
        
        # Add diagonal line (y=x) to show which model is better
        ax.plot([global_min, global_max], [global_min, global_max], 'k--', alpha=0.5)
        
        # Create caption text
        title_text = task.replace("->", " → ")
        if title_text.startswith("→"):
            title_text = "X " + title_text
        if title_text.endswith("→"):
            title_text = title_text + " X"
        
        subplot_label = chr(97 + i)  # 97 is ASCII for 'a'
        # Remove title and add as text at the bottom
        caption_text = f"{subplot_label}) Translation: {title_text}"
        
        # Place caption at the bottom of the subplot
        ax.text(0.5, -0.15, caption_text, transform=ax.transAxes, 
                ha='center', va='center', fontsize=11)
        
        ax.set_xlabel(f"Llama-2-7b {metric_display_name}")
        ax.set_ylabel(f"EMMA-500 {metric_display_name}")
        
        # Set the same limits for both axes
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Annotate to explain the plot
        ax.text(0.05, 0.95, 
                "Points above the line: EMMA-500 outperforms Llama-2\nPoints below the line: Llama-2 outperforms EMMA-500", 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        
        # Save individual subplot
        individual_fig = plt.figure(figsize=(8, 7))  # Increase height for caption
        individual_ax = individual_fig.add_subplot(111)
        
        # Replicate the plot for the individual figure
        # Group by prompt strategy
        for strategy, strategy_df in task_paired_df.groupby("prompt_strategy"):
            if strategy not in COLORS:
                continue
                
            # Plot all points for this strategy
            scatter = individual_ax.scatter(
                strategy_df["llama_score"],
                strategy_df["emma_score"],
                color=COLORS.get(strategy, 'gray'),
                marker='o',
                s=50,
                alpha=0.7,
                label=PROMPT_STRATEGIES.get(strategy, strategy)
            )
        
        # Add the same elements as the main plot
        # Draw connecting lines between models for the same language pair
        for lang_pair, strategies in points_by_lang.items():
            present_strategies = list(strategies.keys())
            for i in range(len(present_strategies)):
                for j in range(i+1, len(present_strategies)):
                    s1 = present_strategies[i]
                    s2 = present_strategies[j]
                    point1 = strategies[s1]
                    point2 = strategies[s2]
                    individual_ax.plot(
                        [point1[0], point2[0]],
                        [point1[1], point2[1]],
                        color='gray',
                        linestyle='-',
                        alpha=0.3,
                        linewidth=0.5
                    )
        
        # Add diagonal line
        individual_ax.plot([global_min, global_max], [global_min, global_max], 'k--', alpha=0.5)
        
        # Create caption for individual plots
        subplot_label = chr(97 + i)  # 97 is ASCII for 'a'
        caption_text = f"{subplot_label}) Translation: {title_text} - {metric_display_name}"
        
        # Remove title and add caption at the bottom
        individual_ax.text(0.5, -0.15, caption_text, transform=individual_ax.transAxes, 
                ha='center', va='center', fontsize=11)
        
        individual_ax.set_xlabel(f"Llama-2-7b {metric_display_name}")
        individual_ax.set_ylabel(f"EMMA-500 {metric_display_name}")
        
        # Add grid and annotation
        individual_ax.grid(True, linestyle='--', alpha=0.3)
        individual_ax.text(0.05, 0.95, 
                "Points above the line: EMMA-500 outperforms Llama-2\nPoints below the line: Llama-2 outperforms EMMA-500", 
                transform=individual_ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        
        # Set axes limits
        individual_ax.set_xlim(global_min, global_max)
        individual_ax.set_ylim(global_min, global_max)
        
        # Add legend
        individual_ax.legend(loc='lower right')
        
        # Save individual figure
        safe_task_name = task.replace("->", "to").replace("-", "_")
        individual_path = f"{task_folder}/{metric}_{safe_task_name}.png"
        individual_fig.tight_layout()
        individual_fig.subplots_adjust(bottom=0.15)  # Make room for caption
        individual_fig.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close(individual_fig)
        print(f"Individual plot saved to {individual_path}")
    
    # Add a common legend to the figure
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(legend_elements), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.4, wspace=0.3)  # Increased bottom margin for legend
    
    # Save the full figure
    filename = f"{output_dir}/translation_model_comparison_{metric}_custom.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Combined plot saved to {filename}")
    return filename

def generate_summary_table(df, metrics=METRICS):
    """
    Generate a summary table of the results for all metrics.
    """
    summary_tables = {}
    
    for metric in metrics:
        # Calculate average scores by model, prompt strategy, and translation task
        summary = df.groupby(['model_name', 'prompt_strategy', 'translation_task'])[metric].agg(
            mean_score='mean',
            median_score='median',
            min_score='min',
            max_score='max',
            count='count'
        ).reset_index()
        
        # Sort the table
        summary = summary.sort_values(['translation_task', 'prompt_strategy', 'model_name'])
        
        # Save to CSV
        summary.to_csv(f"summary_results_{metric}.csv", index=False)
        
        # Create a pivot table for better readability
        pivot = summary.pivot_table(
            index=['model_name', 'prompt_strategy'],
            columns='translation_task',
            values='mean_score'
        )
        
        # Save the pivot table
        pivot.to_csv(f"summary_pivot_{metric}.csv")
        
        summary_tables[metric] = (summary, pivot)
    
    return summary_tables

def main():
    # Path to reorganized data
    data_dir = "results/reorganized"
    
    # Analyze reorganized data
    print(f"Analyzing data from {data_dir}...")
    df, processed_experiments = analyze_reorganized_data(data_dir, verbose=True)
    
    # Save the processed data
    df.to_csv("translation_analysis_results.csv", index=False)
    
    # Generate summary tables
    print("Generating summary tables...")
    summary_tables = generate_summary_table(df)
    
    # Create output directory for visualizations
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create customized visualizations for both metrics
    print("Creating customized visualizations...")
    
    # ChrF Score visualization
    print("\nCreating customized visualization for ChrF scores...")
    chrf_path = plot_customized_comparison(df, metric="chrf_score", output_dir=output_dir)
    
    # BLEU Score visualization
    print("\nCreating customized visualization for BLEU scores...")
    bleu_path = plot_customized_comparison(df, metric="bleu_score", output_dir=output_dir)
    
    print("\nAnalysis complete. Results saved to:")
    print("- translation_analysis_results.csv")
    print("- summary_results_chrf_score.csv and summary_pivot_chrf_score.csv")
    print("- summary_results_bleu_score.csv and summary_pivot_bleu_score.csv")
    
    # Print paths to the figures
    print("\nVisualizations saved to:")
    print(f"1. ChrF Score visualization: {chrf_path}")
    print(f"2. BLEU Score visualization: {bleu_path}")
    print(f"3. Individual task plots: {output_dir}/individual_tasks/")

if __name__ == "__main__":
    main()