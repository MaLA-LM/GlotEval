import os
import csv
from . import register_benchmark

from benchmark_data_loader.data_loader import (
    load_benchmax_rule_based_data,
    build_chat_generation_prompt,
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

def process_benchmax_rule_based_benchmark(benchmark_name, model, load_data_func, lang_config=None, **kwargs):
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

    from metrics.benchmax_rule_based_metrics.compute_inst_follow_metrics import compute_inst_follow_acc

    # Check if there's a language config file
    if not lang_config:
        lang_config_path = f"benchmark_data_loader/data_langid/{benchmark_name}_langs.txt"
        if os.path.exists(lang_config_path):
            lang_config = lang_config_path

    all_inst_follow_score = {}
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
            prompt_library, benchmark_name, "benchmax_rule_based"
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
                dataset = load_data_func(lang_code)

                prompts = []
                for example in dataset:
                    p = build_chat_generation_prompt(
                        text=example["prompt"],
                        tokenizer=model.tokenizer,
                        prompt_library=prompt_library,
                        prompt_language=actual_prompt_lang,
                        benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                        task_key="benchmax_rule_based",
                    )
                    prompts.append(p)

                # Generate with efficiency metrics
                responses, efficiency_metrics = model.generate(prompts)
                
                inst_follow_accs = compute_inst_follow_acc(dataset, responses)

                all_inst_follow_score[lang_code] = inst_follow_accs.copy()
                all_inst_follow_score[lang_code].update({
                    "num_samples": len(dataset),
                    "prompt_language": actual_prompt_lang,
                    "prompt_source": prompt_source
                })
                
                # Store efficiency metrics if needed
                if efficiency_analysis:
                    all_efficiency_statistics[lang_code] = {
                        "prompt_language": actual_prompt_lang,
                        "prompt_source": prompt_source,
                        "num_samples": len(dataset),
                        **inst_follow_accs,
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
                        writer.writerow(["Tested Language", "PromptLanguage", "PromptSource", "Prompt", "Instruction_id_list", "Kwargs", "Response"])
                        for d, r in zip(dataset, responses):
                            writer.writerow([lang_code, actual_prompt_lang, prompt_source, d["prompt"], d["instruction_id_list"], d["kwargs"], r])
                    
                    print(f"[{benchmark_name}] Saved results for {lang_code}")

            except Exception as e:
                print(f"Error processing language {lang_code}: {str(e)}")
                continue

    else:
        raise RuntimeError(f"[{benchmark_name}] No lang_config provided or it's empty.")

    # Update scores.json with benchmark results
    benchmark_params = {
        "prompt_strategy": strategy,
        "prompt_language": 'language-specific' if strategy == 'multi' else prompt_language,
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}' if 'num_test_langs' in locals() else 'default'
    }
    update_benchmark_scores(
        output_dir, benchmark_name, model_name, current_time,
        all_inst_follow_score, "inst_follow_accs", benchmark_params
    )
        
    # Update efficiency.json if needed
    if efficiency_analysis:
        update_efficiency_results(
            output_dir, benchmark_name, model_name, current_time,
            all_efficiency_statistics, "inst_follow_accs", benchmark_params
        )

    # Log summary of results
    if all_inst_follow_score:
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_inst_follow_score)} languages")
        if 'num_test_langs' in locals() and 'num_benchmark_langs' in locals():
            print(f"[INFO] Tested {num_test_langs} / {num_benchmark_langs} languages")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] All results saved.")
    return all_inst_follow_score


@register_benchmark("benchmax_rule_based")
def benchmax_rule_based_handler(model, **kwargs):
    lang_config_path = "benchmark_data_loader/data_langid/benchmax_langs.txt"
    return process_benchmax_rule_based_benchmark(
        "benchmax_rule_based",
        model,
        load_benchmax_rule_based_data,
        lang_config_path,
        **kwargs
    )