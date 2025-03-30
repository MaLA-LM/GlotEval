import os
import csv
import json
import time
from typing import Dict, List, Tuple, Set, Any, Optional

def load_or_init_scores_data(output_dir, model_name, current_time):
    """
    Load the existing scores.json if present, otherwise initialize a new dictionary.
    """
    scores_file = os.path.join(output_dir, "scores.json")
    if os.path.exists(scores_file):
        with open(scores_file, "r", encoding="utf-8") as f:
            scores_data = json.load(f)
    else:
        scores_data = {
            "model_signature": model_name,
            "test_time": current_time,
            "benchmarks": {}
        }
    return scores_data


def update_scores_data(scores_data, benchmark_name, metric_name, benchmark_params, results):
    """
    Update the in-memory scores_data dictionary with new benchmark results.
    """
    scores_data["benchmarks"][benchmark_name] = {
        "metric": metric_name,
        "benchmark_params": benchmark_params,
        "results": results
    }


def save_scores_data(scores_data, output_dir):
    """
    Save the updated scores_data dictionary to scores.json.
    """
    scores_file = os.path.join(output_dir, "scores.json")
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=4)


def write_tsv_file(tsv_path, header, rows):
    """
    Write rows to a TSV file using a specified header.
    """
    with open(tsv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_efficiency_summary_file(summary_path, efficiency_data, item_key="languages"):
    """
    Write a JSON file containing efficiency summary data (e.g. for classification or translation).
    The default item_key is 'languages' for classification tasks. For translation, use 'language_pairs'.
    """
    with open(summary_path, "w", encoding="utf-8") as esf:
        json.dump({
            item_key: list(efficiency_data.values()),
            "average_samples_per_second": (
                sum(d["samples_per_second"] for d in efficiency_data.values()) / len(efficiency_data)
                if efficiency_data else 0
            ),
            "average_inference_time_ms": (
                sum(d["average_inference_time_ms"] for d in efficiency_data.values()) / len(efficiency_data)
                if efficiency_data else 0
            ),
            "average_inference_only_samples_per_second": (
                sum(d["inference_only_samples_per_second"] for d in efficiency_data.values()) / len(efficiency_data)
                if efficiency_data else 0
            )
        }, esf, indent=4)


def setup_benchmark_params(**kwargs):
    """
    Extract and process common benchmark parameters from kwargs.
    
    Args:
        **kwargs: Keyword arguments passed to the benchmark
        
    Returns:
        dict: Dictionary containing processed parameters
    """
    params = {
        # Output and file handling
        "store_details": kwargs.get("store_details", False),
        "efficiency_analysis": kwargs.get("efficiency_analysis", False),
        "output_dir": kwargs.get("output_dir", "results"),
        
        # Prompting strategy
        "prompt_library": kwargs.get("prompt_library", {}),
        "prompt_language_strategy": kwargs.get("prompt_language_strategy", "single"),
        "prompt_language": kwargs.get("prompt_language", "eng_Latn"),
        
        # Sampling parameters
        "n_shots": kwargs.get("n_shots", 0),
        "seed": kwargs.get("seed", 42),
        "dev_max_samples": kwargs.get("dev_max_samples", None),
        
        # Language filtering
        "filtered_lang_codes": kwargs.get("filtered_lang_codes", None),
        "center_lang": kwargs.get("center_lang", "eng_Latn"),
        "direction": kwargs.get("direction", "center-x"),
        
        # Model info
        "model_name": kwargs.get("model_name", "unknown_model"),
        "current_time": kwargs.get("current_time", time.strftime("%Y%m%d_%H%M%S")),
    }
    
    return params


def setup_benchmark_output(benchmark_name, output_dir):
    """
    Set up the benchmark output directory.
    
    Args:
        benchmark_name (str): Name of the benchmark
        output_dir (str): Base output directory
        
    Returns:
        str: Path to the benchmark output directory
    """
    benchmark_output_dir = os.path.join(output_dir, benchmark_name)
    os.makedirs(benchmark_output_dir, exist_ok=True)
    return benchmark_output_dir


def handle_language_config(benchmark_name, lang_config=None, filtered_lang_codes=None, center_lang=None):
    """
    Load and filter language configurations.
    
    Args:
        benchmark_name (str): Name of the benchmark
        lang_config (str, optional): Path to language configuration file
        filtered_lang_codes (set, optional): Set of ISO 639 language codes to filter by
        center_lang (str, optional): Center language code that must be included
        
    Returns:
        tuple: (prompt_langs_map, num_test_langs, num_benchmark_langs)
    """
    from benchmark_data_loader.data_loader import filter_language_config, load_full_language_config
    
    # Set default language config path if not provided
    if not lang_config:
        lang_config = f"benchmark_data_loader/data_langid/{benchmark_name}_langs.txt"
        if not os.path.exists(lang_config):
            raise ValueError(f"No language config file found for benchmark '{benchmark_name}'.")
    
    # Load language configuration (with filtering if requested)
    if filtered_lang_codes:
        print(f"[INFO] Filtering languages for benchmark '{benchmark_name}'")
        prompt_langs_map = filter_language_config(lang_config, filtered_lang_codes, center_lang)
        num_test_langs = len(prompt_langs_map)
        num_benchmark_langs = len(load_full_language_config(lang_config))
    else:
        prompt_langs_map = load_full_language_config(lang_config)
        num_test_langs = len(prompt_langs_map)
        num_benchmark_langs = num_test_langs
    
    print(f"[INFO] Selected {num_test_langs} / {num_benchmark_langs} languages for benchmark '{benchmark_name}'")
    
    return prompt_langs_map, num_test_langs, num_benchmark_langs


def get_available_prompt_languages(prompt_library, benchmark_name, task_key):
    """
    Get the available prompt languages from the prompt library.
    
    Args:
        prompt_library (dict): The prompt library
        benchmark_name (str): Name of the benchmark
        task_key (str): Key for the task in the prompt library
        
    Returns:
        tuple: (benchmark_prompt_langs, task_prompt_langs)
    """
    benchmark_prompt_langs = set()
    task_prompt_langs = set()
    
    if not prompt_library:
        return benchmark_prompt_langs, task_prompt_langs
    
    # Check for benchmark-specific prompts
    if benchmark_name in prompt_library:
        benchmark_prompt_langs = set(prompt_library[benchmark_name].keys())
    
    # Check for task-specific prompts
    if task_key in prompt_library:
        task_prompt_langs = set(prompt_library[task_key].keys())
    
    print(f"[INFO] Found {len(benchmark_prompt_langs)} benchmark-specific prompt languages and {len(task_prompt_langs)} task-specific prompt languages")
    
    return benchmark_prompt_langs, task_prompt_langs


def select_prompt_language(
    lang_code,
    strategy,
    prompt_language,
    prompt_langs_map,
    benchmark_prompt_langs,
    task_prompt_langs
):
    """
    Select the appropriate prompt language based on strategy and available languages.
    
    Args:
        lang_code (str): The language code being processed
        strategy (str): Strategy for prompt language selection ('single' or 'multi')
        prompt_language (str): Default prompt language
        prompt_langs_map (dict): Mapping of benchmark language codes to prompt language codes
        benchmark_prompt_langs (set): Set of available benchmark-specific prompt languages
        task_prompt_langs (set): Set of available task-specific prompt languages
        
    Returns:
        tuple: (actual_prompt_lang, prompt_source)
    """
    # Determine the appropriate prompt language based on strategy
    if strategy == "single":
        proposed_prompt_lang = prompt_language  # Use the configured global prompt language
    else:  # strategy == "multi"
        proposed_prompt_lang = prompt_langs_map.get(lang_code, lang_code)  # Use the mapped language or itself
    
    # Determine the actual prompt language and source to use with priorities
    
    # Priority 1: Benchmark-specific prompt in the proposed language
    if proposed_prompt_lang in benchmark_prompt_langs:
        return proposed_prompt_lang, "benchmark"
    
    # Priority 2: Task-specific prompt in the proposed language
    elif proposed_prompt_lang in task_prompt_langs:
        return proposed_prompt_lang, "task"
    
    # Priority 3: English benchmark-specific prompt
    elif "eng_Latn" in benchmark_prompt_langs:
        return "eng_Latn", "fallback benchmark"
    
    # Priority 4: English task-specific prompt
    elif "eng_Latn" in task_prompt_langs:
        return "eng_Latn", "fallback task"
    
    # Priority 5: Default to English as fallback
    else:
        return "eng_Latn", "fallback"


def update_benchmark_scores(output_dir, benchmark_name, model_name, current_time, results, metric_str, benchmark_params):
    """
    Update the scores.json file with benchmark results.
    
    Args:
        output_dir (str): Output directory
        benchmark_name (str): Name of the benchmark
        model_name (str): Name of the model
        current_time (str): Current time
        results (dict): Dictionary of benchmark results
        metric_str (str): String describing the metrics
        benchmark_params (dict): Dictionary of benchmark parameters
    """
    scores_file = os.path.join(output_dir, "scores.json")
    if os.path.exists(scores_file):
        with open(scores_file, "r", encoding="utf-8") as f:
            scores_data = json.load(f)
    else:
        scores_data = {
            "model_signature": model_name,
            "test_time": current_time,
            "benchmarks": {}
        }

    scores_data["benchmarks"][benchmark_name] = {
        "metric": metric_str,
        "benchmark_params": benchmark_params,
        "results": results
    }

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=4)


def update_efficiency_results(output_dir, benchmark_name, model_name, current_time, efficiency_stats, metric_str, benchmark_params):
    """
    Update the efficiency.json file with efficiency metrics.
    
    Args:
        output_dir (str): Output directory
        benchmark_name (str): Name of the benchmark
        model_name (str): Name of the model
        current_time (str): Current time
        efficiency_stats (dict): Dictionary of efficiency statistics
        metric_str (str): String describing the metrics
        benchmark_params (dict): Dictionary of benchmark parameters
    """
    efficiency_file = os.path.join(output_dir, "efficiency.json")
    if os.path.exists(efficiency_file):
        with open(efficiency_file, "r", encoding="utf-8") as f:
            efficiency_data = json.load(f)
    else:
        efficiency_data = {
            "model_signature": model_name,
            "test_time": current_time,
            "benchmarks": {}
        }

    efficiency_data["benchmarks"][benchmark_name] = {
        "metric": f"{metric_str}, Efficiency",
        "benchmark_params": benchmark_params,
        "results": efficiency_stats
    }
    
    with open(efficiency_file, "w", encoding="utf-8") as f:
        json.dump(efficiency_data, f, indent=4)