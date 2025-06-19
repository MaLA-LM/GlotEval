import os
import csv
import json
import time


from . import register_benchmark
from metrics.translation_metrics import compute_bleu_score, compute_chrf_score, compute_comet_score, compute_mmhb
from benchmark_data_loader.data_loader import (
    sample_few_shot_examples,
    build_translation_prompt,
    filter_language_config,
    load_full_language_config,
    load_flores200_data,
    load_flores_plus_data,
    load_americasnlp_data,
    load_in22_data,
    load_ntrex128_data,
    load_tatoeba_data,
    load_nteu_data,
    load_tico19_data,
    load_mafand_data,
    load_mmhb_data
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


def get_language_name(lang_code: str, prompt_lang: str, name_matrix) -> str:
    """
    Get the name of a language in another language using the name matrix.
    
    Args:
        lang_code (str): The language code to look up
        prompt_lang (str): The language code in which to get the name
        name_matrix (dict): A dictionary mapping language codes to dictionaries of language names
        
    Returns:
        str: The name of the language in the prompt language, or the original code if not found
    """
    if prompt_lang in name_matrix and lang_code in name_matrix[prompt_lang]:
        return name_matrix[prompt_lang][lang_code]
    return lang_code



def build_language_pairs(prompt_langs_map, center_lang: str, direction: str):
    """
    Build the list of language pairs to translate based on the language map.
    
    Args:
        prompt_langs_map (Dict[str, str]): Dictionary mapping benchmark codes to prompt codes
        center_lang (str): Center language code
        direction (str): Direction of translation ('center-x' or 'x-center')
        
    Returns:
        List[Tuple[str, str, bool]]: List of (src, tgt, is_multi_aligned) tuples
    """
    pairs_to_process = []
    multi_aligned_langs = set()
    pair_langs = []  # list of (src_code, tgt_code)
    
    # Parse language codes for multi-aligned versus pairs
    for benchmark_code in prompt_langs_map.keys():
        if '<->' in benchmark_code:
            # e.g. "por_Latn<->eng_Latn"
            parts = benchmark_code.split('<->')
            if len(parts) == 2:
                pair_langs.append((parts[0], parts[1]))
            else:
                print(f"[WARN] Unrecognized pair line: {benchmark_code}")
        else:
            # single language code
            multi_aligned_langs.add(benchmark_code)
    
    # Build language pairs to process
    if center_lang in multi_aligned_langs:
        # Process center language with other languages in multi-aligned set
        for lang_code in multi_aligned_langs:
            if lang_code == center_lang:
                continue
            if direction == "center-x":
                # center is src, lang_code is tgt
                pairs_to_process.append((center_lang, lang_code, True))
            else:
                # center is tgt, lang_code is src
                pairs_to_process.append((lang_code, center_lang, True))
    
    # Process explicit language pairs that include center_lang
    for (src, tgt) in pair_langs:
        if center_lang in (src, tgt):
            if direction == "center-x":
                # If center_lang is in src, do src->tgt, else do tgt->src
                if src == center_lang:
                    pairs_to_process.append((src, tgt, False))
                else:
                    pairs_to_process.append((src, tgt, False))
            else:
                # "x-center"
                # We reverse it so that the center ends up on the 'target' side
                if src == center_lang:
                    pairs_to_process.append((tgt, src, False))
                else:
                    pairs_to_process.append((tgt, src, False))
    
    # Make them unique (in case duplicates appear)
    return list(set(pairs_to_process))


def process_translation_benchmark(benchmark_name, model, load_data_func, lang_config=None, **kwargs):
    """
    Process a translation benchmark with support for different language codes for benchmarks and prompts,
    and with support for multi-language prompting using language names from a name matrix.
    
    This function now supports language filtering through the filtered_lang_codes parameter.
    
    Args:
        benchmark_name (str): Name of the benchmark
        model: Model object with a generate method
        load_data_func: Function to load the benchmark data
        lang_config (str, optional): Path to language configuration file
        **kwargs: Additional arguments including:
            - filtered_lang_codes (set): Set of ISO 639 language codes to filter by
            - center_lang (str): Center language for translation pairs
            - direction (str): Direction of translation ('center-x' or 'x-center')
            - n_shots (int): Number of few-shot examples
            - prompt_language (str): Language code for prompts
            - prompt_language_strategy (str): Strategy for prompting ('single' or 'multi')
    
    Returns:
        dict: Dictionary of scores for each language pair
    """
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
    direction = params["direction"]
    center_lang = params["center_lang"]
    output_dir = params["output_dir"]
    model_name = params["model_name"]
    current_time = params["current_time"]

    # Additional prompt/model config
    comet_model = kwargs.get("comet_model", "Unbabel/wmt22-comet-da")
    name_matrix = kwargs.get("name_matrix", {})  # Language names in different languages

    # Setup benchmark output directory
    benchmark_output_dir = setup_benchmark_output(benchmark_name, output_dir)

    # Handle language configuration
    prompt_langs_map, num_test_langs, num_benchmark_langs = handle_language_config(
        benchmark_name, lang_config, filtered_lang_codes, center_lang
    )
    
    # Get available prompt languages from the prompt library
    benchmark_prompt_langs, task_prompt_langs = get_available_prompt_languages(
        prompt_library, benchmark_name, "translation"
    )
    
    # 2) Build the list of language pairs to translate
    pairs_to_process = build_language_pairs(prompt_langs_map, center_lang, direction)
    
    if not pairs_to_process:
        print(f"[WARN] No valid translation pairs formed for benchmark '{benchmark_name}'. Check center language config.")
        return {}
        
    print(f"[INFO] Processing {len(pairs_to_process)} language pairs")
    
    # 3) Iterate over pairs_to_process, run translation, compute metrics
    all_scores = {}
    if efficiency_analysis:
        all_efficiency_statistics = {}

    # for multi-prompting strategy
    with open("benchmark_data_loader/prompt_library/language_matrix/language_name_matrix.json", "r", encoding="utf-8") as f: 
        name_matrix = json.load(f)

    for (src_bench_lang, tgt_bench_lang, is_multi_aligned) in pairs_to_process:
        print(f"[{benchmark_name}] Processing: {src_bench_lang} -> {tgt_bench_lang}, multi_aligned={is_multi_aligned}")

        # Map benchmark language codes to prompt language codes
        src_prompt_lang = prompt_langs_map.get(src_bench_lang, src_bench_lang)
        tgt_prompt_lang = prompt_langs_map.get(tgt_bench_lang, tgt_bench_lang)
        
        # Determine the actual prompt language based on strategy
        if strategy == "single":
            proposed_prompt_lang = prompt_language  # Use the configured global prompt language
        else:  # strategy == "multi"
            proposed_prompt_lang = src_prompt_lang  # Use the source language for prompting in multi strategy
        
        # Determine the actual prompt language and source using utility function
        actual_prompt_lang, prompt_source = select_prompt_language(
            src_prompt_lang, strategy, prompt_language, prompt_langs_map,
            benchmark_prompt_langs, task_prompt_langs
        )
        
        print(f"[{benchmark_name}] Using {prompt_source} prompt in {actual_prompt_lang} for {src_bench_lang}->{tgt_bench_lang}")
        
        try:
            src_texts, tgt_texts = load_data_func(src_bench_lang, tgt_bench_lang, split="test", limit_samples=dev_max_samples)
        except FileNotFoundError:
            print(f"No data found for pair {src_bench_lang}-{tgt_bench_lang}")
            continue
        except ValueError as e:
            print(e)
            continue

        # Few-shot examples (from dev set) if needed
        few_shot_examples = None
        if n_shots > 0:
            try:
                src_dev, tgt_dev = load_data_func(src_bench_lang, tgt_bench_lang, split="dev", limit_samples=None)
                few_shot_examples = sample_few_shot_examples(n_shots, src_dev, tgt_dev, seed)
            except FileNotFoundError:
                print(f"[WARN] No dev set for {src_bench_lang}-{tgt_bench_lang}, skipping few-shot.")
                few_shot_examples = None
            
        # Get language names for the prompt from the name matrix
        if name_matrix and actual_prompt_lang in name_matrix:
            src_lang_name = get_language_name(src_prompt_lang, actual_prompt_lang, name_matrix)
            tgt_lang_name = get_language_name(tgt_prompt_lang, actual_prompt_lang, name_matrix)
        else:
            # Fallback to prompt language codes if name matrix not available
            src_lang_name = src_prompt_lang
            tgt_lang_name = tgt_prompt_lang
            print(f"[WARN] No name matrix entry for {actual_prompt_lang}, using codes instead")

        # Build prompts with the appropriate language names/codes
        prompts = []
        for src_text in src_texts:
            prompt_text = build_translation_prompt(
                src_lang=src_lang_name,  # Use language name or code
                tgt_lang=tgt_lang_name,  # Use language name or code
                src_text=src_text,
                few_shot_examples=few_shot_examples,
                prompt_library=prompt_library,
                prompt_language=actual_prompt_lang,
                benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                task_key="translation"
            )
            prompts.append(prompt_text)

        # Generate predictions with efficiency metrics
        translations, efficiency_metrics = model.generate(prompts)
        
        pair_key = f"{src_bench_lang}->{tgt_bench_lang}"
        
        if benchmark_name == "mmhb":
            df = tgt_texts
            df["translation"] = translations
            mmhb_scores = compute_mmhb(df)
            
            # Store basic results
            all_scores[pair_key] = {
                "feminine_bleu": mmhb_scores.get("feminine_bleu", 0),
                "masculine_bleu": mmhb_scores.get("masculine_bleu", 0),
                "bias": mmhb_scores.get("bias", 0),
                "num_samples": len(src_texts),
                "multi_aligned": is_multi_aligned,
                "prompt_language": actual_prompt_lang,
                "prompt_source": prompt_source
            }
            
            # Store efficiency metrics if needed
            if efficiency_analysis:
                all_efficiency_statistics[pair_key] = {
                    "feminine_bleu": mmhb_scores.get("feminine_bleu", 0),
                    "masculine_bleu": mmhb_scores.get("masculine_bleu", 0),
                    "bias": mmhb_scores.get("bias", 0),
                    "prompt_language": actual_prompt_lang,
                    "prompt_source": prompt_source,
                    "num_samples": len(src_texts),
                    "multi_aligned": is_multi_aligned,
                    "generated_tokens": efficiency_metrics["generated_tokens"],
                    "total_time_seconds": efficiency_metrics["total_time"],
                    "prefill_time_seconds": efficiency_metrics["prefill_time"],
                    "decode_time_seconds": efficiency_metrics["decode_time"],
                    "tokens_per_second": efficiency_metrics["generated_tokens"] / efficiency_metrics["total_time"] if efficiency_metrics["total_time"] > 0 else 0,
                    "first_token_latency": efficiency_metrics["first_token_time"] / len(prompts) if len(prompts) > 0 else 0,
                    "remaining_tokens_per_second": efficiency_metrics["remaining_tokens_count"] / efficiency_metrics["remaining_tokens_time"] if efficiency_metrics["remaining_tokens_time"] > 0 else 0
                }
        else:
            # Compute metrics (using benchmark language codes)
            bleu_score = compute_bleu_score(translations, tgt_texts)
            chrf_score = compute_chrf_score(translations, tgt_texts)
            
            # Store basic results
            all_scores[pair_key] = {
                "bleu_score": bleu_score,
                "chrf_score": chrf_score,
                "num_samples": len(src_texts),
                "multi_aligned": is_multi_aligned,
                "prompt_language": actual_prompt_lang,
                "prompt_source": prompt_source
            }
            
            # Store efficiency metrics if needed
            if efficiency_analysis:
                all_efficiency_statistics[pair_key] = {
                    "bleu_score": bleu_score,
                    "chrf_score": chrf_score,
                    "prompt_language": actual_prompt_lang,
                    "prompt_source": prompt_source,
                    "num_samples": len(src_texts),
                    "multi_aligned": is_multi_aligned,
                    "generated_tokens": efficiency_metrics["generated_tokens"],
                    "total_time_seconds": efficiency_metrics["total_time"],
                    "prefill_time_seconds": efficiency_metrics["prefill_time"],
                    "decode_time_seconds": efficiency_metrics["decode_time"],
                    "tokens_per_second": efficiency_metrics["generated_tokens"] / efficiency_metrics["total_time"] if efficiency_metrics["total_time"] > 0 else 0,
                    "first_token_latency": efficiency_metrics["first_token_time"] / len(prompts) if len(prompts) > 0 else 0,
                    "remaining_tokens_per_second": efficiency_metrics["remaining_tokens_count"] / efficiency_metrics["remaining_tokens_time"] if efficiency_metrics["remaining_tokens_time"] > 0 else 0
                }

        # Optionally store TSV
        if store_details:
            tsv_file = os.path.join(benchmark_output_dir, f"{src_bench_lang}-{tgt_bench_lang}.tsv")
            with open(tsv_file, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf, delimiter="\t", quoting=csv.QUOTE_MINIMAL, escapechar="\\")
                # writer.writerow(["SourceLanguage", "TargetLanguage", "PromptLanguage", "PromptSource", "Source", "Reference", "Prediction","FullPrompt"])
                writer.writerow(["SourceLanguage", "TargetLanguage", "PromptLanguage", "PromptSource", "Source", "Reference", "Prediction"])

                for s, ref, hyp in zip(src_texts, tgt_texts, translations):
                    writer.writerow([src_bench_lang, tgt_bench_lang, actual_prompt_lang, prompt_source, s, ref, hyp])
                # for s, ref, hyp, prpt in zip(src_texts, tgt_texts, translations, prompts):
                #     writer.writerow([src_bench_lang, tgt_bench_lang, actual_prompt_lang, prompt_source, s, ref, hyp, prpt])

    # Update scores.json with benchmark results
    metric_str = "BLEU, ChrF++" if benchmark_name != "mmhb" else "Feminine/Masculine BLEU, Bias"
    benchmark_params = {
        "n_shots": n_shots,
        "prompt_strategy": strategy,
        "prompt_language": 'language-specific' if strategy == 'multi' else prompt_language,
        "central_lang": center_lang,
        "direction": direction,
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}'
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
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_scores)} language pairs over {num_test_langs} selected languages")
    else:
        print(f"[WARN] No scores were generated for benchmark '{benchmark_name}'")

    print(f"[{benchmark_name}] All results saved.")
    return all_scores

@register_benchmark("flores200_mt")
def flores200_handler(model, **kwargs):
    """
    FLORES-200 MT benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/flores200_langs.txt"
    return process_translation_benchmark(
        benchmark_name="flores200_mt",
        model=model,
        load_data_func=load_flores200_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("flores_plus_mt")
def flores_plus_handler(model, **kwargs):
    """
    FLORES-Plus MT benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/flores_plus_langs.txt"
    return process_translation_benchmark(
        benchmark_name="flores_plus_mt",
        model=model,
        load_data_func=load_flores_plus_data,
        lang_config=lang_config_path,
        **kwargs
    )


@register_benchmark("mmhb")
def mmhb_handler(model, **kwargs):
    """
    MMHB benchmark.
    Data at: https://dl.fbaipublicfiles.com/MMHB/mmhb_dataset.zip
    Place in data/mmhb
    """
    lang_config_path = "benchmark_data_loader/data_langid/mmhb_langs.txt"    
    return process_translation_benchmark(
        benchmark_name="mmhb",
        model=model,
        load_data_func=load_mmhb_data,
        lang_config=lang_config_path,
        **kwargs
    )


@register_benchmark("americasnlp")
def americasnlp_handler(model, **kwargs):
    """
    AmericasNLP benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/americasnlp_langs.txt"
    return process_translation_benchmark(
        benchmark_name="americasnlp",
        model=model,
        load_data_func=load_americasnlp_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("in22")
def in22_handler(model, **kwargs):
    """
    IN22 benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/in22_langs.txt"
    return process_translation_benchmark(
        benchmark_name="in22",
        model=model,
        load_data_func=load_in22_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("ntrex128")
def ntrex_handler(model, **kwargs):
    """
    NTeREX-128 benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/ntrex128_langs.txt"
    return process_translation_benchmark(
        benchmark_name="ntrex128",
        model=model,
        load_data_func=load_ntrex128_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("tatoeba")
def tatoeba_handler(model, **kwargs):
    """
    Tatoeba benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/tatoeba_langs.txt"
    return process_translation_benchmark(
        benchmark_name="tatoeba",
        model=model,
        load_data_func=load_tatoeba_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("nteu")
def nteu_handler(model, **kwargs):
    """
    NTEU benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/nteu_langs.txt"
    return process_translation_benchmark(
        benchmark_name="nteu",
        model=model,
        load_data_func=load_nteu_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("tico19")
def tico19_handler(model, **kwargs):
    """
    TICO-19 benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/tico19_langs.txt"
    return process_translation_benchmark(
        benchmark_name="tico19",
        model=model,
        load_data_func=load_tico19_data,
        lang_config=lang_config_path,
        **kwargs
    )

@register_benchmark("mafand")
def mafand_handler(model, **kwargs):
    """
    MAFAND benchmark.
    """
    lang_config_path = "benchmark_data_loader/data_langid/mafand_langs.txt"
    return process_translation_benchmark(
        benchmark_name="mafand",
        model=model,
        load_data_func=load_mafand_data,
        lang_config=lang_config_path,
        **kwargs
    )