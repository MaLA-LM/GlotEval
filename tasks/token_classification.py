import os
import csv
import json
import time
import random
import numpy as np
from . import register_benchmark

from metrics.classification_metrics import compute_accuracy
from benchmark_data_loader.data_loader import (
    load_wikiann_data,
    load_ud_data,
    build_token_classification_prompt,
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

def process_token_classification_benchmark(
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
        prompt_library, benchmark_name, "token_classification"
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

        few_shot_data = []
        if n_shots > 0:
            try:
                few_shot_data = load_data_func(lang_code, split="dev", limit_samples=None)
            except FileNotFoundError:
                print(f"[{benchmark_name}] No dev set for '{lang_code}', skipping few-shot.")
                few_shot_data = []

        few_shot_examples = []
        if few_shot_data and len(few_shot_data) >= n_shots:
            shot_indices = random.sample(range(len(few_shot_data)), n_shots)
            for idx in shot_indices:
                ex = few_shot_data[idx]
                tokens = ex["tokens"]
                t_idx = random.randint(0, len(tokens)-1)
                if "ner_tags" in ex:
                    label_gold = ex["ner_tags"][t_idx]
                elif "upos_tags" in ex:
                    label_gold = ex["upos_tags"][t_idx]
                else:
                    label_gold = None

                if isinstance(label_gold, (int, np.integer)) and label_gold < len(candidate_labels):
                    label_str = candidate_labels[label_gold]
                else:
                    label_str = label_gold

                few_shot_examples.append({
                    "tokens": tokens,
                    "idx": t_idx,
                    "label": label_str
                })

        try:
            test_data = load_data_func(lang_code, split="test", limit_samples=dev_max_samples)
        except FileNotFoundError:
            print(f"[{benchmark_name}] No test data for '{lang_code}'. Skipping.")
            continue

        if not test_data:
            print(f"[{benchmark_name}] No test data for '{lang_code}'. Skipping.")
            continue

        token_correct = 0
        token_total = 0
        csv_rows = []
        
        # Efficiency tracking
        total_inference_time = 0
        total_tokenize_time = 0
        total_time_start = time.time()
        total_samples = 0
        
        # Process each token in each test example
        for ex in test_data:
            tokens = ex["tokens"]
            if "ner_tags" in ex:
                labels = ex["ner_tags"]
            elif "upos_tags" in ex:
                labels = ex["upos_tags"]
            else:
                labels = []

            pred_seq, gold_seq = [], []

            for i, tok in enumerate(tokens):
                gold_label = labels[i]
                if isinstance(gold_label, (int, np.integer)) and gold_label < len(candidate_labels):
                    gold_label = candidate_labels[gold_label]

                prompt_str = build_token_classification_prompt(
                    tokens=tokens,
                    idx=i,
                    candidate_labels=candidate_labels,
                    few_shot_examples=few_shot_examples,
                    prompt_library=prompt_library,
                    prompt_language=actual_prompt_lang,
                    benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                    task_key="token_classification"  # Always provide task_key
                )

                # Get prediction with efficiency metrics
                preds, token_efficiency = model.predict([prompt_str], candidate_labels)
                pred_label = preds[0]
                
                # Accumulate efficiency metrics
                total_samples += 1
                total_inference_time += token_efficiency["inference_time"]
                if "tokenize_time" in token_efficiency:
                    total_tokenize_time += token_efficiency["tokenize_time"]

                pred_seq.append(pred_label)
                gold_seq.append(gold_label)

                if pred_label == gold_label:
                    token_correct += 1
                token_total += 1

            csv_rows.append([
                " ".join(tokens),
                " ".join(gold_seq),
                " ".join(pred_seq)
            ])
        
        total_time = time.time() - total_time_start
        
        # Calculate final efficiency metrics
        samples_per_second = total_samples / total_time if total_time > 0 else 0
        average_inference_time = total_inference_time / total_samples if total_samples > 0 else 0
        inference_only_samples_per_second = total_samples / total_inference_time if total_inference_time > 0 else 0
        
        accuracy = (token_correct / token_total) if token_total else 0.0
        
        # Store results
        all_accuracies[lang_code] = {
            "accuracy": accuracy,
            "num_samples": token_total,
            "prompt_language": actual_prompt_lang,
            "prompt_source": prompt_source
        }
        
        # Store efficiency metrics if needed
        if efficiency_analysis:
            all_efficiency_statistics[lang_code] = {
                "prompt_language": actual_prompt_lang,
                "prompt_source": prompt_source,
                "num_samples": token_total,
                "accuracy": accuracy,
                "num_predictions": total_samples,
                "total_time_seconds": total_time,
                "inference_time_seconds": total_inference_time,
                "tokenize_time_seconds": total_tokenize_time,
                "samples_per_second": samples_per_second,
                "average_inference_time": average_inference_time,
                "inference_only_samples_per_second": inference_only_samples_per_second,
            }

        if store_details:
            jsonl_file = os.path.join(benchmark_output_dir, f"{lang_code}.jsonl")
            with open(jsonl_file, "w", encoding="utf-8") as jf:
                for row in csv_rows:
                    record = {
                        "tested_language": lang_code,
                        "prompt_language": actual_prompt_lang,
                        "prompt_source": prompt_source,
                        "sentence": row[0],
                        "gold_labels": row[1],
                        "predicted_labels": row[2]
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"[{benchmark_name}] {lang_code}: wrote JSONL => accuracy={accuracy:.4f}, speed={samples_per_second:.2f} samples/s")
        else:
            print(f"[{benchmark_name}] {lang_code}: accuracy={accuracy:.4f}, speed={samples_per_second:.2f} samples/s")
            
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

    print(f"[{benchmark_name}] All results saved.")
    return all_accuracies


@register_benchmark("wikiann")
def wikiann_handler(model, **kwargs):
    wikiann_labels = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC"]
    return process_token_classification_benchmark(
        benchmark_name="wikiann",
        model=model,
        load_data_func=load_wikiann_data,
        candidate_labels=wikiann_labels,
        **kwargs
    )

@register_benchmark("ud_upos")
def ud_upos_handler(model, **kwargs):
    upos_labels = [
      "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM",
      "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
    ]
    return process_token_classification_benchmark(
        benchmark_name="ud_upos",
        model=model,
        load_data_func=load_ud_data,
        candidate_labels=upos_labels,
        **kwargs
    )