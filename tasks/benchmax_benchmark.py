import os
import json
from . import register_benchmark

from benchmark_data_loader.data_loader import (
    load_benchmax_rule_based_data,
    load_benchmax_math_data,
    load_benchmax_science_data,
    build_chat_generation_prompt,
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

def process_benchmax_benchmark(benchmark_name, model, load_data_func, lang_config=None, **kwargs):
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

    all_scores = {}
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
            prompt_library, benchmark_name, "benchmax"
        )
        
        if benchmark_name == "benchmax_rule_based":
            build_prompt_kwargs = lambda x: {"text": x["prompt"]}
            from metrics.benchmax_rule_based_metrics.compute_inst_follow_metrics import compute_inst_follow_acc
            metric_fn = compute_inst_follow_acc
            metric_str = "inst_follow_accs"
        elif benchmark_name == "benchmax_math":
            build_prompt_kwargs = lambda x: {"text": x["question"]}
            from metrics.benchmax_math_metrics import compute_math_acc
            metric_fn = compute_math_acc
            metric_str = "accuracy"
        elif benchmark_name == "benchmax_science":
            build_prompt_kwargs = lambda x: {"question": x["Question"], "options": x["options"]}
            from metrics.benchmax_science_metrics import compute_science_acc
            metric_fn = compute_science_acc
            metric_str = "accuracy"

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
                        prompt_kwargs=build_prompt_kwargs(example),
                        tokenizer=model.tokenizer,
                        prompt_library=prompt_library,
                        prompt_language=actual_prompt_lang,
                        benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                        task_key="benchmax",
                    )
                    prompts.append(p)

                # Generate with efficiency metrics
                responses, efficiency_metrics = model.generate(prompts, **final_sampling_params)
                
                result_dict = metric_fn(dataset, responses, lang_code)

                all_scores[lang_code] = result_dict.copy()
                all_scores[lang_code].update({
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
                        **result_dict,
                        "generated_tokens": efficiency_metrics["generated_tokens"],
                        "total_time_seconds": efficiency_metrics["total_time"],
                        "prefill_time_seconds": efficiency_metrics["prefill_time"],
                        "decode_time_seconds": efficiency_metrics["decode_time"],
                        "tokens_per_second": efficiency_metrics["generated_tokens"] / efficiency_metrics["total_time"] if efficiency_metrics["total_time"] > 0 else 0,
                        "first_token_latency": efficiency_metrics["first_token_time"] / len(prompts) if len(prompts) > 0 else 0,
                        "remaining_tokens_per_second": efficiency_metrics["remaining_tokens_count"] / efficiency_metrics["remaining_tokens_time"] if efficiency_metrics["remaining_tokens_time"] > 0 else 0
                    }

                if store_details:
                    jsonl_file = os.path.join(benchmark_output_dir, f"{lang_code}.jsonl")
                    with open(jsonl_file, "w", encoding="utf-8") as jf:
                        if benchmark_name == "benchmax_rule_based":
                            for d, r in zip(dataset, responses):
                                record = {
                                    "Tested Language": lang_code,
                                    "PromptLanguage": actual_prompt_lang,
                                    "PromptSource": prompt_source,
                                    "Prompt": d["prompt"],
                                    "Instruction_id_list": d["instruction_id_list"],
                                    "Kwargs": d["kwargs"],
                                    "Response": r
                                }
                                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        elif benchmark_name == "benchmax_math":
                            for d, r in zip(dataset, responses):
                                record = {
                                    "Tested Language": lang_code,
                                    "PromptLanguage": actual_prompt_lang,
                                    "PromptSource": prompt_source,
                                    "Question": d["question"],
                                    "Answer_number": d["answer_number"],
                                    "Response": r
                                }
                                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        elif benchmark_name == "benchmax_science":
                            for d, r in zip(dataset, responses):
                                record = {
                                    "Tested Language": lang_code,
                                    "PromptLanguage": actual_prompt_lang,
                                    "PromptSource": prompt_source,
                                    "Question": d["Question"],
                                    "A": d["option_a"],
                                    "B": d["option_b"],
                                    "C": d["option_c"],
                                    "D": d["option_d"],
                                    "Answer": d["answer"],
                                    "Response": r
                                }
                                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        all_scores, metric_str, benchmark_params
    )
        
    # Update efficiency.json if needed
    if efficiency_analysis:
        update_efficiency_results(
            output_dir, benchmark_name, model_name, current_time,
            all_efficiency_statistics, metric_str, benchmark_params
        )

    # Log summary of results
    if all_scores:
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_scores)} languages")
        if 'num_test_langs' in locals() and 'num_benchmark_langs' in locals():
            print(f"[INFO] Tested {num_test_langs} / {num_benchmark_langs} languages")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] All results saved.")
    return all_scores


@register_benchmark("benchmax_rule_based")
def benchmax_rule_based_handler(model, **kwargs):
    lang_config_path = "benchmark_data_loader/data_langid/benchmax_langs.txt"
    return process_benchmax_benchmark(
        "benchmax_rule_based",
        model,
        load_benchmax_rule_based_data,
        lang_config_path,
        **kwargs
    )


@register_benchmark("benchmax_math")
def benchmax_math_handler(model, **kwargs):
    lang_config_path = "benchmark_data_loader/data_langid/benchmax_langs.txt"
    return process_benchmax_benchmark(
        "benchmax_math",
        model,
        load_benchmax_math_data,
        lang_config_path,
        **kwargs
    )


@register_benchmark("benchmax_science")
def benchmax_science_handler(model, **kwargs):
    lang_config_path = "benchmark_data_loader/data_langid/benchmax_langs.txt"
    return process_benchmax_benchmark(
        "benchmax_science",
        model,
        load_benchmax_science_data,
        lang_config_path,
        **kwargs
    )