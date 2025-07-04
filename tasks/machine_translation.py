import os
import csv
import json
import time
from typing import List, Tuple, Set, Dict, Optional


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
    load_mmhb_data,
    load_opensubtitles_data
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




def parse_pair_lang_config(lang_config_path: str, 
                          filtered_lang_codes: Optional[Set[str]] = None) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Parse language configuration file for pair-wise benchmarks.
    
    Args:
        lang_config_path: Path to language config file
        filtered_lang_codes: Optional set of language codes to filter by
        
    Returns:
        Tuple of:
        - prompt_langs_map: Mapping from benchmark codes to prompt codes
        - language_pairs: List of (src, tgt) pairs to process
    """
    prompt_langs_map = {}
    language_pairs = []
    
    if not os.path.exists(lang_config_path):
        print(f"[WARN] Language config file not found: {lang_config_path}")
        return prompt_langs_map, language_pairs
    
    with open(lang_config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse line format: "en-amh | eng_Latn-amh_Ethi"
            parts = line.split('|')
            if len(parts) != 2:
                continue
                
            benchmark_pair = parts[0].strip()
            prompt_pair = parts[1].strip()
            
            # Extract individual language codes
            if '-' in benchmark_pair and '-' in prompt_pair:
                bench_parts = benchmark_pair.split('-', 1)
                prompt_parts = prompt_pair.split('-', 1)
                
                if len(bench_parts) == 2 and len(prompt_parts) == 2:
                    src_bench, tgt_bench = bench_parts[0], bench_parts[1]
                    src_prompt, tgt_prompt = prompt_parts[0], prompt_parts[1]
                    
                    # Add to mapping (for individual languages)
                    prompt_langs_map[src_bench] = src_prompt
                    prompt_langs_map[tgt_bench] = tgt_prompt
                    
                    # Check if this pair should be included based on filter
                    if filtered_lang_codes:
                        # Include if any language in the pair matches filter
                        if (src_prompt.split('_')[0] in filtered_lang_codes or 
                            tgt_prompt.split('_')[0] in filtered_lang_codes):
                            # Add both directions for pair-wise benchmarks
                            language_pairs.append((src_bench, tgt_bench))
                            language_pairs.append((tgt_bench, src_bench))
                    else:
                        # No filter, include all pairs (both directions)
                        language_pairs.append((src_bench, tgt_bench))
                        language_pairs.append((tgt_bench, src_bench))
    
    # Remove duplicate pairs
    language_pairs = list(set(language_pairs))
    
    return prompt_langs_map, language_pairs


def build_language_pairs(prompt_langs_map, center_lang: str, direction: str, 
                        translation_mode: str = "center", 
                        language_pairs: Optional[List[Tuple[str, str]]] = None):
    """
    Build the list of language pairs to translate based on the language map.
    Modified to support both center and pairs modes cleanly.
    
    Args:
        prompt_langs_map (Dict[str, str]): Dictionary mapping benchmark codes to prompt codes
        center_lang (str): Center language code (for center mode)
        direction (str): Direction of translation (for center mode)
        translation_mode (str): "center" or "pairs"
        language_pairs (List[Tuple[str, str]]): Pre-computed pairs for pairs mode
        
    Returns:
        List[Tuple[str, str, bool]]: List of (src, tgt, is_multi_aligned) tuples
    """
    pairs_to_process = []
    
    if translation_mode == "pairs" and language_pairs is not None:
        # Use pre-computed pairs from config parsing
        for src, tgt in language_pairs:
            pairs_to_process.append((src, tgt, False))
        return pairs_to_process
    
    # Original center mode logic (unchanged)
    multi_aligned_langs = set()
    
    # Parse language codes for multi-aligned languages
    for benchmark_code in prompt_langs_map.keys():
        # Skip any codes that look like pairs (shouldn't happen in center mode)
        if '-' not in benchmark_code or len(benchmark_code.split('-')) > 2:
            multi_aligned_langs.add(benchmark_code)
    
    # Build language pairs for center mode
    if center_lang in multi_aligned_langs:
        for lang_code in multi_aligned_langs:
            if lang_code == center_lang:
                continue
            if direction == "center-x":
                pairs_to_process.append((center_lang, lang_code, True))
            else:
                pairs_to_process.append((lang_code, center_lang, True))
    
    return list(set(pairs_to_process))


def handle_language_config_enhanced(benchmark_name: str, 
                                   lang_config: Optional[str],
                                   filtered_lang_codes: Optional[Set[str]],
                                   center_lang: Optional[str],
                                   translation_mode: str = "center") -> Tuple[Dict[str, str], int, int, Optional[List[Tuple[str, str]]]]:
    """
    Enhanced language configuration handler that supports both center and pairs modes.
    
    Returns:
        Tuple of (prompt_langs_map, num_test_langs, num_benchmark_langs, language_pairs)
    """
    if translation_mode == "pairs":
        # For pairs mode, parse pairs from config
        prompt_langs_map, language_pairs = parse_pair_lang_config(lang_config, filtered_lang_codes)
        
        # Count unique languages in pairs
        unique_langs = set()
        for src, tgt in language_pairs:
            unique_langs.add(src)
            unique_langs.add(tgt)
        
        num_test_langs = len(unique_langs)
        num_benchmark_langs = len(unique_langs)  # Could be different if we had full list
        
        return prompt_langs_map, num_test_langs, num_benchmark_langs, language_pairs
    else:
        # Original center mode logic
        prompt_langs_map, num_test_langs, num_benchmark_langs = handle_language_config(
            benchmark_name, lang_config, filtered_lang_codes, center_lang
        )
        return prompt_langs_map, num_test_langs, num_benchmark_langs, None


def process_translation_benchmark(benchmark_name, model, load_data_func, lang_config=None, **kwargs):
    """
    Process a translation benchmark with support for different language codes for benchmarks and prompts.
    Minimally modified to support pair-wise translation.
    """
    # Process common parameters
    params = setup_benchmark_params(**kwargs)
    global_sampling_params = kwargs.get("global_sampling_params", {})
    benchmark_sampling_params = params.get("sampling_params", {})
    # print(f"[{benchmark_name}] Using benchmark sampling params: {benchmark_sampling_params}")
    final_sampling_params = global_sampling_params.copy()
    final_sampling_params.update(benchmark_sampling_params)
    # print(f"[{benchmark_name}] Using final sampling params: {final_sampling_params}")
    
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
    
    # New parameter for translation mode
    translation_mode = params.get("translation_mode", "center")

    # Additional prompt/model config
    comet_model = kwargs.get("comet_model", "Unbabel/wmt22-comet-da")
    name_matrix = kwargs.get("name_matrix", {})

    # Setup benchmark output directory
    benchmark_output_dir = setup_benchmark_output(benchmark_name, output_dir)

    # Handle language configuration based on mode
    if translation_mode == "pairs":
        prompt_langs_map, num_test_langs, num_benchmark_langs, language_pairs = handle_language_config_enhanced(
            benchmark_name, lang_config, filtered_lang_codes, center_lang, translation_mode
        )

    else:
        # Original center mode
        prompt_langs_map, num_test_langs, num_benchmark_langs = handle_language_config(
            benchmark_name, lang_config, filtered_lang_codes, center_lang
        )
        language_pairs = None


    # Get available prompt languages from the prompt library
    benchmark_prompt_langs, task_prompt_langs = get_available_prompt_languages(
        prompt_library, benchmark_name, "translation"
    )

    # Build the list of language pairs to translate
    pairs_to_process = build_language_pairs(
        prompt_langs_map, center_lang, direction, translation_mode, language_pairs
    )
    # print(f"[DEBUG] After building language pairs:")
    # print(f"[DEBUG]   pairs_to_process: {pairs_to_process}")

    if not pairs_to_process:
        print(f"[WARN] No valid translation pairs formed for benchmark '{benchmark_name}'. Check configuration.")
        return {}

    print(f"[INFO] Processing {len(pairs_to_process)} language pairs in {translation_mode} mode")
    
    # Rest of the function remains the same...
    all_scores = {}
    if efficiency_analysis:
        all_efficiency_statistics = {}

    # Load language name matrix
    with open("benchmark_data_loader/prompt_library/language_matrix/language_name_matrix.json", "r", encoding="utf-8") as f: 
        name_matrix = json.load(f)
        

    for (src_bench_lang, tgt_bench_lang, is_multi_aligned) in pairs_to_process:
        print(f"[{benchmark_name}] Processing: {src_bench_lang} -> {tgt_bench_lang}, multi_aligned={is_multi_aligned}")

        # Map benchmark language codes to prompt language codes
        src_prompt_lang = prompt_langs_map.get(src_bench_lang, src_bench_lang)
        tgt_prompt_lang = prompt_langs_map.get(tgt_bench_lang, tgt_bench_lang)
        
        # Determine the actual prompt language and source using utility function
        actual_prompt_lang, prompt_source = select_prompt_language(
            src_prompt_lang, strategy, prompt_language, prompt_langs_map,
            benchmark_prompt_langs, task_prompt_langs
        )
        
        print(f"[{benchmark_name}] Using {prompt_source} prompt in {actual_prompt_lang} for {src_bench_lang}->{tgt_bench_lang}")
        
        try:
            src_texts, tgt_texts = load_data_func(
                src_bench_lang, 
                tgt_bench_lang, 
                split="test", 
                limit_samples=dev_max_samples
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] Skipping {src_bench_lang}-{tgt_bench_lang}: {e}")
            # failed_pairs.append((src_bench_lang, tgt_bench_lang, str(e)))
            continue 
        except Exception as e:
            print(f"[ERROR] Unexpected error for {src_bench_lang}-{tgt_bench_lang}: {e}")
            continue

        # Few-shot examples (from dev set) if needed

        few_shot_examples = None
        if n_shots > 0:
            try:
                src_dev, tgt_dev = load_data_func(src_bench_lang, tgt_bench_lang, split="dev", limit_samples=None)
                few_shot_examples = sample_few_shot_examples(n_shots, src_dev, tgt_dev, seed)
            except (FileNotFoundError, ValueError) as e:
                print(f"[WARN] Cannot load dev set for {src_bench_lang}-{tgt_bench_lang}: {e}, skipping few-shot.")
                few_shot_examples = None
            except Exception as e:
                print(f"[ERROR] Unexpected error loading dev set for {src_bench_lang}-{tgt_bench_lang}: {e}")
                few_shot_examples = None
            
        # Get language names for the prompt from the name matrix
        if name_matrix and actual_prompt_lang in name_matrix:
            src_lang_name = get_language_name(src_prompt_lang, actual_prompt_lang, name_matrix)
            tgt_lang_name = get_language_name(tgt_prompt_lang, actual_prompt_lang, name_matrix)
        else:
            src_lang_name = src_prompt_lang
            tgt_lang_name = tgt_prompt_lang
            print(f"[WARN] No name matrix entry for {actual_prompt_lang}, using codes instead")

        # Build prompts with the appropriate language names/codes
        prompts = []
        for src_text in src_texts:
            prompt_text = build_translation_prompt(
                src_lang=src_lang_name,
                tgt_lang=tgt_lang_name,
                src_text=src_text,
                few_shot_examples=few_shot_examples,
                prompt_library=prompt_library,
                prompt_language=actual_prompt_lang,
                benchmark_name=benchmark_name if prompt_source == "benchmark" else None,
                task_key="translation"
            )
            prompts.append(prompt_text)
        # print(f"[DEBUG] final_sampling_params: {final_sampling_params}")
        # Generate predictions with efficiency metrics
        translations, efficiency_metrics = model.generate(prompts, **final_sampling_params)
        
        pair_key = f"{src_bench_lang}->{tgt_bench_lang}"
        # After loading data, check if it's MMHB
        if benchmark_name == "mmhb":
            # Special handling for MMHB
            # tgt_texts is actually a DataFrame for MMHB
            mmhb_df = tgt_texts
            # Add translation column to DataFrame
            mmhb_df['translation'] = translations
            
            # Compute MMHB-specific metrics
            mmhb_scores = compute_mmhb(mmhb_df)
            
            all_scores[pair_key] = {
                "chrfs_masculine": mmhb_scores['chrfs_masculine'],
                "chrfs_feminine": mmhb_scores['chrfs_feminine'],
                "chrfs_both": mmhb_scores['chrfs_both'],
                "num_samples": len(src_texts),
                "prompt_language": actual_prompt_lang,
                "prompt_source": prompt_source
            }
        else:
            # Compute metrics
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

        # Optionally store JSONL (compact version)
        if store_details:
            jsonl_file = os.path.join(benchmark_output_dir, f"{src_bench_lang}-{tgt_bench_lang}.jsonl")
            with open(jsonl_file, "w", encoding="utf-8") as jf:
                for s, ref, hyp, prompt in zip(src_texts, tgt_texts, translations, prompts):
                    record = {
                        "src_lang": src_bench_lang,
                        "tgt_lang": tgt_bench_lang,
                        "prompt_lang": actual_prompt_lang,
                        "prompt_src": prompt_source,
                        "src": s,
                        "ref": ref,
                        "pred": hyp,
                        "prompt": prompt
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update scores.json with benchmark results
    metric_str = "BLEU, ChrF++" if benchmark_name != "mmhb" else "Feminine/Masculine BLEU, Bias"
    benchmark_params = {
        "n_shots": n_shots,
        "prompt_strategy": strategy,
        "prompt_language": 'language-specific' if strategy == 'multi' else prompt_language,
        "translation_mode": translation_mode
    }
    
    # Add mode-specific parameters
    if translation_mode == "center":
        benchmark_params["central_lang"] = center_lang
        benchmark_params["direction"] = direction
        benchmark_params["tested_languages"] = f'{num_test_langs} / {num_benchmark_langs}'
    else:
        benchmark_params["tested_pairs"] = len(pairs_to_process) // 2  # Divide by 2 since we test both directions
    
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
        print(f"[INFO] Completed benchmark '{benchmark_name}' with {len(all_scores)} translation directions")
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

@register_benchmark("opensubtitles")
def opensubtitles_handler(model, **kwargs):
    """
    OpenSubtitles translation benchmark using pair-wise mode.
    """
    # Force pairs mode for OpenSubtitles    
    lang_config_path = "benchmark_data_loader/data_langid/opensubtitles_langs.txt"
    return process_translation_benchmark(
        benchmark_name="opensubtitles",
        model=model,
        load_data_func=load_opensubtitles_data,
        lang_config=lang_config_path,
        **kwargs
    )

