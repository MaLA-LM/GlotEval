import os
import csv
import json
import time
import random
from . import register_benchmark
from metrics.classification_metrics import compute_accuracy
from benchmark_data_loader.data_loader import (
    build_options_str,
    build_comprehension_prompt_multi,
    filter_language_config,
    load_full_language_config
)
from .benchmark_utils import (
    setup_benchmark_params,
    setup_benchmark_output,
    handle_language_config,
    get_available_prompt_languages,
    select_prompt_language,
    update_benchmark_scores,
    update_efficiency_results
)

def process_comprehension_benchmark(
    benchmark_name,
    model,
    load_data_func,
    has_dev=False,
    lang_config=None,
    **kwargs
):
    # Process common parameters
    params = setup_benchmark_params(**kwargs)
    store_details = params["store_details"]
    efficiency_analysis = params["efficiency_analysis"]
    prompt_library = params["prompt_library"]
    strategy = params["prompt_language_strategy"]
    prompt_language = params["prompt_language"]
    filtered_lang_codes = params["filtered_lang_codes"]
    n_shots = params["n_shots"]
    seed = params["seed"]
    dev_max_samples = params["dev_max_samples"]
    output_dir = params["output_dir"]
    model_name = params["model_name"]
    current_time = params["current_time"]

    # Setup benchmark output directory
    benchmark_output_dir = setup_benchmark_output(benchmark_name, output_dir)

    # Handle language configuration
    prompt_langs_map, num_test_langs, num_benchmark_langs = handle_language_config(
        benchmark_name, lang_config, filtered_lang_codes, None
    )
    
    # Get list of benchmark language codes
    lang_codes = list(prompt_langs_map.keys())
    
    # Get available prompt languages from the prompt library
    benchmark_prompt_langs, task_prompt_langs = get_available_prompt_languages(
        prompt_library, benchmark_name, "comprehension"
    )

    random.seed(seed)
    all_results = {}
    if efficiency_analysis:
        all_efficiency_statistics = {}

    for lang_code in lang_codes:
        print(f"[{benchmark_name}] Processing language: {lang_code}")
        
        # Select appropriate prompt language
        actual_prompt_lang, prompt_source = select_prompt_language(
            lang_code, strategy, prompt_language, prompt_langs_map,
            benchmark_prompt_langs, task_prompt_langs
        )
        
        print(f"[{benchmark_name}] Using {prompt_source} prompt in {actual_prompt_lang} for {lang_code}")

        # If has_dev & n_shots>0 => load dev
        few_shot_examples = []
        if has_dev and n_shots > 0:
            try:
                dev_data = load_data_func(lang_code, split="dev", limit_samples=None)
                if len(dev_data) >= n_shots:
                    chosen = random.sample(range(len(dev_data)), n_shots)
                    for idx in chosen:
                        ex = dev_data[idx]
                        q = ex["question"]
                        a = ex["answer"]
                        opts = build_options_str(ex["option_a"], ex["option_b"], ex["option_c"], ex["option_d"])
                        few_shot_examples.append({
                            "example_question": q,
                            "example_options": opts,
                            "example_answer": a
                        })
            except FileNotFoundError:
                print(f"No dev data for {lang_code}, skipping few-shot")
                few_shot_examples = []

        # Load test
        try:
            test_data = load_data_func(lang_code, split="test", limit_samples=dev_max_samples)
        except FileNotFoundError:
            print(f"[{benchmark_name}] No test data for '{lang_code}', skipping.")
            continue

        prompts, references = [], []
        for row in test_data:
            question = row["question"]
            gold = row["answer"]
            options_str = build_options_str(row["option_a"], row["option_b"], row["option_c"], row["option_d"])

            prompt = build_comprehension_prompt_multi(
                lang_code=lang_code,
                question=question,
                options_str=options_str,
                few_shot_examples=few_shot_examples,
                prompt_library=prompt_library,
                prompt_language=actual_prompt_lang,
                benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                task_key="comprehension"  # Always provide task_key
            )
            prompts.append(prompt)
            references.append(gold)

        candidate_labels = ["A","B","C","D"]
        
        # Predict with efficiency metrics
        predictions, efficiency_metrics = model.predict(prompts, candidate_labels)
        
        # Calculate accuracy
        acc = compute_accuracy(predictions, references)
        
        # Store results
        all_results[lang_code] = {
            "accuracy": acc, 
            "num_samples": len(test_data),
            "prompt_language": actual_prompt_lang,
            "prompt_source": prompt_source
        }
        
        # Store efficiency metrics if needed
        if efficiency_analysis:
            all_efficiency_statistics[lang_code] = {
                "prompt_language": actual_prompt_lang,
                "prompt_source": prompt_source,
                "num_samples": len(test_data),
                "accuracy": acc,
                "num_predictions": efficiency_metrics["total_samples"],
                "total_time_seconds": efficiency_metrics["total_time"],
                "inference_time_seconds": efficiency_metrics["inference_time"],
                "tokenize_time_seconds": efficiency_metrics.get("tokenize_time", 0),
                "samples_per_second": efficiency_metrics["samples_per_second"],
                "average_inference_time": efficiency_metrics["average_inference_time_per_sample"],
                "inference_only_samples_per_second": efficiency_metrics["inference_only_samples_per_second"],
            }

        if store_details:
            csv_path = os.path.join(benchmark_output_dir, f"{lang_code}.tsv")
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Tested Language","PromptLanguage","PromptSource","question","A","B","C","D","gold","prediction"])
                for row, pred in zip(test_data, predictions):
                    writer.writerow([
                        lang_code,
                        actual_prompt_lang,
                        prompt_source,
                        row["question"],
                        row["option_a"],
                        row["option_b"],
                        row["option_c"],
                        row["option_d"],
                        row["answer"],
                        pred
                    ])
            
            print(f"[{benchmark_name}] Saved results to {csv_path}, accuracy={acc:.4f}, speed={efficiency_metrics['samples_per_second']:.2f} samples/s")
        else:
            print(f"[{benchmark_name}] Results for {lang_code}: accuracy={acc:.4f}, speed={efficiency_metrics['samples_per_second']:.2f} samples/s")

    # Update scores.json using utility function
    benchmark_params = {
        "n_shots": n_shots,
        "prompt_strategy": strategy,
        "prompt_language": 'language-specific' if strategy == 'multi' else prompt_language,
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}'
    }
    update_benchmark_scores(
        output_dir, benchmark_name, model_name, current_time,
        all_results, "Accuracy", benchmark_params
    )

    # Update efficiency.json if needed
    if efficiency_analysis:
        update_efficiency_results(
            output_dir, benchmark_name, model_name, current_time,
            all_efficiency_statistics, "Accuracy", benchmark_params
        )

    # Log summary of results
    if all_results:
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_results)} languages over {num_test_langs} selected languages.")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] Done. Results saved.")
    return all_results


@register_benchmark("mmmlu")
def mmmlu_handler(model, **kwargs):
    from benchmark_data_loader.data_loader import load_mmmlu_data
    return process_comprehension_benchmark(
        benchmark_name="mmmlu",
        model=model,
        load_data_func=load_mmmlu_data,
        has_dev=False,
        **kwargs
    )

@register_benchmark("global_mmlu")
def global_mmlu_handler(model, **kwargs):
    from benchmark_data_loader.data_loader import load_global_mmlu_data
    return process_comprehension_benchmark(
        benchmark_name="global_mmlu",
        model=model,
        load_data_func=load_global_mmlu_data,
        has_dev=True,
        **kwargs
    )