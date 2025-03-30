import os
import csv
import json
import time
from . import register_benchmark

from metrics.open_generation_metrics import compute_self_bleu
from benchmark_data_loader.data_loader import (
    load_aya_data,
    load_polywrite_data,
    build_open_generation_prompt,
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

def process_benchmark(benchmark_name, model, load_data_func, lang_config=None, **kwargs):
    # Process common parameters
    params = setup_benchmark_params(**kwargs)
    store_details = params["store_details"]
    efficiency_analysis = params["efficiency_analysis"]
    prompt_library = params["prompt_library"]
    strategy = params["prompt_language_strategy"]
    prompt_language = params["prompt_language"]
    filtered_lang_codes = params["filtered_lang_codes"]
    seed = params["seed"]
    dev_max_samples = params["dev_max_samples"]
    output_dir = params["output_dir"]
    model_name = params["model_name"]
    current_time = params["current_time"]

    # Setup benchmark output directory
    benchmark_output_dir = setup_benchmark_output(benchmark_name, output_dir)

    # Check if there's a language config file
    if not lang_config:
        lang_config_path = f"benchmark_data_loader/data_langid/{benchmark_name}_langs.txt"
        if os.path.exists(lang_config_path):
            lang_config = lang_config_path

    all_self_bleu_score = {}
    if efficiency_analysis:
        all_efficiency_statistics = {}

    # Variables to track language counts
    num_test_langs = 1
    num_benchmark_langs = 1

    # If we have a language config file, process languages based on that
    if lang_config:
        # Handle language configuration
        prompt_langs_map, num_test_langs, num_benchmark_langs = handle_language_config(
            benchmark_name, lang_config, filtered_lang_codes, None
        )
        
        # Get list of benchmark language codes
        lang_codes = list(prompt_langs_map.keys())
        
        # Get available prompt languages from the prompt library
        benchmark_prompt_langs, task_prompt_langs = get_available_prompt_languages(
            prompt_library, benchmark_name, "open_generation"
        )
        
        # multi-lingual approach
        for lang_code in lang_codes:
            print(f"[{benchmark_name}] Processing language: {lang_code}")
            
            # Select appropriate prompt language
            actual_prompt_lang, prompt_source = select_prompt_language(
                lang_code, strategy, prompt_language, prompt_langs_map,
                benchmark_prompt_langs, task_prompt_langs
            )
            
            print(f"[{benchmark_name}] Using {prompt_source} prompt in {actual_prompt_lang} for {lang_code}")
            
            try:
                src_texts = load_data_func(lang_code)
                if dev_max_samples is not None and len(src_texts) > dev_max_samples:
                    src_texts = src_texts[:dev_max_samples]

                prompts = []
                for text in src_texts:
                    p = build_open_generation_prompt(
                        lang_code=lang_code,
                        text=text,
                        prompt_library=prompt_library,
                        prompt_language=actual_prompt_lang,
                        benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                        task_key="open_generation"
                    )
                    prompts.append(p)

                # Generate with efficiency metrics
                hyp_texts, efficiency_metrics = model.generate(prompts)
                
                # Calculate self-BLEU score
                self_bleu_score = compute_self_bleu(hyp_texts)
                
                # Store basic results
                all_self_bleu_score[lang_code] = {
                    "self_bleu": self_bleu_score,
                    "num_samples": len(src_texts),
                    "prompt_language": actual_prompt_lang,
                    "prompt_source": prompt_source
                }
                
                # Store efficiency metrics if needed
                if efficiency_analysis:
                    all_efficiency_statistics[lang_code] = {
                        "prompt_language": actual_prompt_lang,
                        "prompt_source": prompt_source,
                        "num_samples": len(src_texts),
                        "self_bleu": self_bleu_score,
                        "generated_tokens": efficiency_metrics["generated_tokens"],
                        "total_time_seconds": efficiency_metrics["total_time"],
                        "prefill_time_seconds": efficiency_metrics["prefill_time"],
                        "decode_time_seconds": efficiency_metrics["decode_time"],
                        "tokens_per_second": efficiency_metrics["generated_tokens"] / efficiency_metrics["total_time"] if efficiency_metrics["total_time"] > 0 else 0,
                        "first_token_latency": efficiency_metrics["first_token_time"] / len(prompts) if len(prompts) > 0 else 0,
                        "remaining_tokens_per_second": efficiency_metrics["remaining_tokens_count"] / efficiency_metrics["remaining_tokens_time"] if efficiency_metrics["remaining_tokens_time"] > 0 else 0
                    }

                if store_details:
                    csv_file = os.path.join(benchmark_output_dir, f"{lang_code}.tsv")
                    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
                        writer = csv.writer(cf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(["Tested Language", "PromptLanguage", "PromptSource", "Src", "Hyp"])
                        for s, h in zip(src_texts, hyp_texts):
                            writer.writerow([lang_code, actual_prompt_lang, prompt_source, s, h])
                    
                    print(f"[{benchmark_name}] Saved results for {lang_code}")

            except Exception as e:
                print(f"Error processing language {lang_code}: {str(e)}")
                continue

    else:
        # single-lingual approach or pre-coded (no language config)
        print(f"[{benchmark_name}] No lang_config provided or it's empty. Attempting single load.")
        try:
            src_texts = load_data_func("")
            if dev_max_samples is not None and len(src_texts) > dev_max_samples:
                src_texts = src_texts[:dev_max_samples]

            prompts = []
            for text in src_texts:
                p = build_open_generation_prompt(
                    lang_code="",
                    text=text,
                    prompt_library=prompt_library,
                    prompt_language=prompt_language,
                    benchmark_name=None,
                    task_key="open_generation"
                )
                prompts.append(p)

            # Generate with efficiency metrics
            hyp_texts, efficiency_metrics = model.generate(prompts)
            
            # Calculate self-BLEU score
            self_bleu_score = compute_self_bleu(hyp_texts)
            
            # Store basic results
            all_self_bleu_score["default"] = {
                "self_bleu": self_bleu_score,
                "num_samples": len(src_texts),
                "prompt_language": prompt_language,
                "prompt_source": "default"
            }
            
            # Store efficiency metrics if needed
            if efficiency_analysis:
                all_efficiency_statistics["default"] = {
                    "prompt_language": prompt_language,
                    "prompt_source": "default",
                    "num_samples": len(src_texts),
                    "self_bleu": self_bleu_score,
                    "generated_tokens": efficiency_metrics["generated_tokens"],
                    "total_time_seconds": efficiency_metrics["total_time"],
                    "prefill_time_seconds": efficiency_metrics["prefill_time"],
                    "decode_time_seconds": efficiency_metrics["decode_time"],
                    "tokens_per_second": efficiency_metrics["generated_tokens"] / efficiency_metrics["total_time"] if efficiency_metrics["total_time"] > 0 else 0,
                    "first_token_latency": efficiency_metrics["first_token_time"] / len(prompts) if len(prompts) > 0 else 0,
                    "remaining_tokens_per_second": efficiency_metrics["remaining_tokens_count"] / efficiency_metrics["remaining_tokens_time"] if efficiency_metrics["remaining_tokens_time"] > 0 else 0
                }

            if store_details:
                csv_file = os.path.join(benchmark_output_dir, f"{benchmark_name}_info.tsv")
                with open(csv_file, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["Src", "Hyp"])
                    for s, h in zip(src_texts, hyp_texts):
                        writer.writerow([s, h])
                
                print(f"[{benchmark_name}] Saved default results")
        except Exception as e:
            print(f"Error in default processing: {str(e)}")

    # Update scores.json with benchmark results
    benchmark_params = {
        "prompt_strategy": strategy,
        "prompt_language": 'language-specific' if strategy == 'multi' else prompt_language,
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}' if 'num_test_langs' in locals() else 'default'
    }
    update_benchmark_scores(
        output_dir, benchmark_name, model_name, current_time,
        all_self_bleu_score, "self-BLEU", benchmark_params
    )
        
    # Update efficiency.json if needed
    if efficiency_analysis:
        update_efficiency_results(
            output_dir, benchmark_name, model_name, current_time,
            all_efficiency_statistics, "self-BLEU", benchmark_params
        )

    # Log summary of results
    if all_self_bleu_score:
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_self_bleu_score)} languages")
        if 'num_test_langs' in locals() and 'num_benchmark_langs' in locals():
            print(f"[INFO] Tested {num_test_langs} / {num_benchmark_langs} languages")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] All results saved.")
    return all_self_bleu_score


@register_benchmark("aya")
def aya_handler(model, **kwargs):
    return process_benchmark("aya", model, load_aya_data, **kwargs)

@register_benchmark("polywrite")
def polywrite_handler(model, **kwargs):
    return process_benchmark("polywrite", model, load_polywrite_data, **kwargs)