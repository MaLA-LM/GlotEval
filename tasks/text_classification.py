import os
import csv
import json
import time
import random
from . import register_benchmark
from metrics.classification_metrics import compute_accuracy

from benchmark_data_loader.data_loader import (
    load_sib200_data,
    load_taxi1500_data,
    sample_few_shot_classification_examples,
    build_classification_prompt,
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

def process_classification_benchmark(
    benchmark_name, 
    model, 
    load_data_func, 
    candidate_labels,
    lang_config=None,
    **kwargs
):
    # Process common parameters
    params = setup_benchmark_params(**kwargs)
    global_sampling_params = kwargs.get("global_sampling_params", {})
    benchmark_sampling_params = params.get("sampling_params", {})
    final_sampling_params = global_sampling_params.copy()
    final_sampling_params.update(benchmark_sampling_params)
    print(f"[{benchmark_name}] Using final sampling params: {final_sampling_params}")
    
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
        prompt_library, benchmark_name, "classification"
    )

    random.seed(seed)
    all_accuracies = {}
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

        # load train/test data
        train_data = load_data_func(lang_code, split="train")
        test_data  = load_data_func(lang_code, split="test")
        if not test_data:
            print(f"[{benchmark_name}] No test data for '{lang_code}' => skipping.")
            continue

        # apply dev_max_samples if needed
        if dev_max_samples is not None and len(test_data) > dev_max_samples:
            test_data = test_data[:dev_max_samples]

        # few-shot
        few_shot_examples = []
        if train_data and n_shots > 0:
            few_shot_examples = sample_few_shot_classification_examples(train_data, n_shots, seed)

        # Build prompts
        prompts     = []
        references  = []
        texts       = []

        for example in test_data:
            text_  = example["text"]
            label_ = example["category"]  # ground truth

            # Use the modified build_classification_prompt function
            prompt_str = build_classification_prompt(
                text=text_,
                few_shot_examples=few_shot_examples,
                prompt_library=prompt_library,
                prompt_language=actual_prompt_lang,
                benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                task_key="classification"  # Always provide task_key
            )
            prompts.append(prompt_str)
            references.append(label_)
            texts.append(text_)

        # Predict with efficiency metrics
        predictions, efficiency_metrics = model.predict(prompts, candidate_labels)
        
        # Calculate accuracy
        accuracy = compute_accuracy(predictions, references)

        # Store results with efficiency metrics and prompt information
        all_accuracies[lang_code] = {
            "accuracy": accuracy,
            "num_samples": len(test_data),
            "prompt_language": actual_prompt_lang,
            "prompt_source": prompt_source,
        }

        if store_details:
            jsonl_path = os.path.join(benchmark_output_dir, f"{lang_code}.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as jf:
                for txt_, ref_, pred_, prompt_ in zip(texts, references, predictions, prompts):
                    record = {
                        "tested_language": lang_code,
                        "prompt_language": actual_prompt_lang,
                        "prompt_source": prompt_source,
                        "text": txt_,
                        "ground_truth": ref_,
                        "prediction": pred_,
                        "prompt_used": prompt_
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
        if efficiency_analysis:       
            all_efficiency_statistics[lang_code] = {
                "prompt_language": actual_prompt_lang,
                "prompt_source": prompt_source,
                "num_samples": len(test_data),
                "accuracy": accuracy,
                "num_predictions": efficiency_metrics["total_samples"],
                "total_time_seconds": efficiency_metrics["total_time"],
                "inference_time_seconds": efficiency_metrics["inference_time"],
                "tokenize_time_seconds": efficiency_metrics.get("tokenize_time", 0),
                "samples_per_second": efficiency_metrics["samples_per_second"],
                "average_inference_time": efficiency_metrics["average_inference_time_per_sample"], 
                "inference_only_samples_per_second": efficiency_metrics["inference_only_samples_per_second"],
            }

    # Update scores.json with benchmark results
    benchmark_params = {
        "n_shots": n_shots,
        "prompt_strategy": strategy,
        "prompt_language": 'language-specific' if strategy == 'multi' else prompt_language,
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}'
    }
    
    update_benchmark_scores(
        output_dir, benchmark_name, model_name, current_time,
        all_accuracies, "Accuracy", benchmark_params
    )

    # Update efficiency.json if needed
    if efficiency_analysis:
        update_efficiency_results(
            output_dir, benchmark_name, model_name, current_time,
            all_efficiency_statistics, "Accuracy", benchmark_params
        )

    # Log summary of results
    if all_accuracies:
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_accuracies)} languages over {num_test_langs} selected languages.")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] done. Results saved.")
    return all_accuracies


@register_benchmark("sib200")
def sib200_handler(model, **kwargs):
    candidate_labels = ["science/technology","travel","politics","sports","health","entertainment","geography"]
    return process_classification_benchmark(
        benchmark_name="sib200",
        model=model,
        load_data_func=load_sib200_data,
        candidate_labels=candidate_labels,
        **kwargs
    )

@register_benchmark("taxi1500")
def taxi1500_handler(model, **kwargs):
    candidate_labels = ["Recommendation","Faith","Description","Sin","Grace","Violence"]
    return process_classification_benchmark(
        benchmark_name="taxi1500",
        model=model,
        load_data_func=load_taxi1500_data,
        candidate_labels=candidate_labels,
        **kwargs
    )