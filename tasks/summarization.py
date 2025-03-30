import os
import csv
import json
import time
from . import register_benchmark
from transformers import AutoConfig

from metrics.summarization_metrics import calculate_rougeL_f1
from benchmark_data_loader.data_loader import (
    load_xlsum_data,
    sample_few_shot_examples,
    build_summarization_prompt,
    filter_language_config,
    load_full_language_config,
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


def get_max_context_len(model, fallback=2048):
    config = AutoConfig.from_pretrained(model)
    max_len = getattr(config, "max_position_embeddings", None)

    # Fallback
    if max_len == None:
        print(f"Warning: Could not detect context length. Using fallback of {fallback}")
        max_len = fallback

    return max_len


def process_summarization_benchmark(
    benchmark_name, model, load_data_func, lang_config=None, **kwargs
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
        prompt_library, benchmark_name, "summarization"
    )

    all_rouge_scores = {}
    if efficiency_analysis:
        all_efficiency_statistics = {}
        
    tokenizer = model.get_tokenizer()
    max_context_len = get_max_context_len(model_name)
    reserved_output_tokens = kwargs.get("reserved_output_tokens", 128)

    for lang_code in lang_codes:
        print(f"[{benchmark_name}] Processing language: {lang_code}")

        # Select appropriate prompt language
        actual_prompt_lang, prompt_source = select_prompt_language(
            lang_code, strategy, prompt_language, prompt_langs_map,
            benchmark_prompt_langs, task_prompt_langs
        )
        
        print(f"[{benchmark_name}] Using {prompt_source} prompt in {actual_prompt_lang} for {lang_code}")

        try:
            src_texts, tgt_texts = load_data_func(
                lang_code, split="test", limit_samples=dev_max_samples
            )
        except FileNotFoundError:
            print(f"No data for {lang_code} in test split.")
            continue

        # few-shot
        few_shot_examples = None
        if n_shots > 0:
            try:
                src_val, tgt_val = load_data_func(
                    lang_code, split="val", limit_samples=None
                )
                few_shot_examples = sample_few_shot_examples(
                    n_shots, src_val, tgt_val, seed
                )
            except FileNotFoundError:
                print(f"No train data for {lang_code}, skipping few-shot")

        # Build prompts
        prompts = []
        for text in src_texts:
            prompt_str = build_summarization_prompt(
                lang_code=lang_code,
                text=text,
                few_shot_examples=few_shot_examples,
                prompt_library=prompt_library,
                prompt_language=actual_prompt_lang,
                benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                task_key="summarization"
            )
            # tokenize and truncate
            max_prompt_tokens = max_context_len - reserved_output_tokens
            input_ids = tokenizer.encode(prompt_str)
            if len(input_ids) > max_prompt_tokens:
                input_ids = input_ids[:max_prompt_tokens]
                prompt_str = tokenizer.decode(input_ids)
            prompts.append(prompt_str)

        # Generate with efficiency metrics
        hyp_texts, efficiency_metrics = model.generate(prompts)

        # Evaluate
        rouge_score = calculate_rougeL_f1(hyp_texts, tgt_texts)

        # Store basic results
        all_rouge_scores[lang_code] = {
            "rougeL_score": rouge_score,
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
                "rougeL_score": rouge_score,
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
                writer.writerow(["Tested Language", "PromptLanguage", "PromptSource", "Src", "Tgt", "Hyp"])
                for s, t, h in zip(src_texts, tgt_texts, hyp_texts):
                    writer.writerow([lang_code, actual_prompt_lang, prompt_source, s, t, h])

            print(f"[{benchmark_name}] Saved results for {lang_code}")

    # Update scores.json with benchmark results
    benchmark_params = {
        "n_shots": n_shots,
        "prompt_strategy": strategy,
        "prompt_language": "language-specific" if strategy == "multi" else prompt_language,
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}'
    }
    
    update_benchmark_scores(
        output_dir, benchmark_name, model_name, current_time,
        all_rouge_scores, "ROUGE-L", benchmark_params
    )
        
    # Update efficiency.json if needed
    if efficiency_analysis:
        update_efficiency_results(
            output_dir, benchmark_name, model_name, current_time,
            all_efficiency_statistics, "ROUGE-L", benchmark_params
        )

    # Log summary of results
    if all_rouge_scores:
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_rouge_scores)} languages over {num_test_langs} selected languages.")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] All results saved.")
    return all_rouge_scores


@register_benchmark("xlsum")
def xlsum_handler(model, **kwargs):
    return process_summarization_benchmark(
        benchmark_name="xlsum", model=model, load_data_func=load_xlsum_data, **kwargs
    )