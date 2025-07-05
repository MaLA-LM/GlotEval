import json
import time
import os
import argparse
import torch
from iso639 import Lang, is_language

from models.model_loader import load_model
from tasks import BENCHMARK_HANDLERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="GlotEval: Unified multilingual benchmark runner (with automatic backend)."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path.")
    parser.add_argument("--benchmarks", nargs="+", required=True, help="List of benchmarks to run.")
    parser.add_argument("--params", type=str, default="config.json", help="Path to the config file.")
    parser.add_argument("--prompt_library", type=str, default="benchmark_data_loader/prompt_library/prompt_library.json", help="Path to the config file.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for results.")
    parser.add_argument("--store_details", action="store_true",
                        help="If set, writes CSV files. Otherwise skip CSV output.")
    parser.add_argument("--efficiency_analysis", action="store_true",
                        help="If set, tracks and records token generation efficiency metrics.")
    parser.add_argument("--langs", nargs="+", default=None, 
                        help="Filter languages by ISO 639-3 codes. If a macrolanguage is provided, all its individual languages will be included.")
    
    return parser.parse_args()

def get_required_method(benchmark_name):
    generation_benchmarks = [
        "flores200_mt","flores_plus_mt","xlsum","aya","polywrite",
        "americasnlp","in22","ntrex128","tatoeba","nteu","tico19","mafand","opensubtitles","mmhb", "massivesumm_long", "massivesumm_short",
        "benchmax_rule_based", "benchmax_math", "benchmax_science"
    ]
    non_generation_benchmarks = [
        "sib200","taxi1500","mmmlu","global_mmlu","wikiann","ud_upos","mala","pbc"
    ]
    if benchmark_name in generation_benchmarks:
        return "generate"
    elif benchmark_name in non_generation_benchmarks:
        return "predict"
    else:
        return "generate"  # default fallback

def expand_language_codes(lang_codes):
    """
    Expand a list of language codes to include individual languages if macrolanguages are provided.
    
    Args:
        lang_codes (list): List of ISO 639 language codes
        
    Returns:
        set: Expanded set of language codes
    """
    if not lang_codes:
        return None
        
    expanded_codes = set()
    
    for code in lang_codes:
        try:
            lg = Lang(code)
            # Add the language itself using ISO 639-3 code
            expanded_codes.add(lg.pt3)
            
            # If it's a macrolanguage, add all its individual languages
            if lg.scope() == "Macrolanguage":
                for individual in lg.individuals():
                    expanded_codes.add(individual.pt3)
                print(f"Expanded macrolanguage '{lg.name}' to include {len(lg.individuals())} individual languages")
        except Exception as e:
            print(f"Warning: Could not process language code '{code}': {str(e)}")
            # Still keep the original code in case it's a custom code used in the benchmark
            expanded_codes.add(code)
    
    return expanded_codes
    
def main():
    args = parse_args()
    if not os.path.exists(args.params):
        raise FileNotFoundError(f"Config file '{args.params}' not found.")

    with open(args.params, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(args.prompt_library, "r", encoding="utf-8") as f:
        prompt_library = json.load(f)

    model_args = config.get("model_args", {})
    global_sampling_params = model_args.get("sampling_params", {})

    benchmark_params = config.get("benchmark_params", {})
    dev_max_samples = config.get("dev_max_samples", None)
    prompt_language_strategy = config.get("prompt_language_strategy", "single")
    prompt_language = config.get("prompt_language", "eng_Latn")

    current_time = time.strftime("%Y%m%d_%H%M%S")
    two_level_name = '--'.join(args.model_name.rstrip(os.sep).split(os.sep)[-2:])

    base_output_dir = os.path.join(args.output_dir, two_level_name, current_time)
    os.makedirs(base_output_dir, exist_ok=True)

    # Create a log file for errors
    error_log_path = os.path.join(base_output_dir, "benchmark_errors.log")

    # Expand language codes if provided
    expanded_lang_codes = None
    if args.langs:
        expanded_lang_codes = expand_language_codes(args.langs)
        print(f"Filtering benchmarks to languages: {', '.join(expanded_lang_codes)}")

    # Partition benchmarks
    gen_benchmarks = []
    non_gen_benchmarks = []
    for b in args.benchmarks:
        if get_required_method(b)=="generate":
            gen_benchmarks.append(b)
        else:
            non_gen_benchmarks.append(b)

    hf_model = None
    if non_gen_benchmarks:
        print(f"[INFO] Loading HF model for: {non_gen_benchmarks}")
        try:
            hf_model = load_model(args.model_name, backend="hf", **model_args)
        except Exception as e:
            error_msg = f"[ERROR] Failed to load HF model: {str(e)}"
            print(error_msg)
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{error_msg}\n")
            # Skip all non-gen benchmarks if model loading fails
            non_gen_benchmarks = []
        
        for b in non_gen_benchmarks:
            handler = BENCHMARK_HANDLERS.get(b)
            if not handler:
                print(f"[WARN] No handler for benchmark '{b}'")
                continue
            params = benchmark_params.get(b, {})
            # add dev_max_samples
            params["dev_max_samples"] = dev_max_samples
            params["output_dir"] = base_output_dir
            params["model_name"] = args.model_name
            params["current_time"] = current_time
            params["prompt_library"] = prompt_library
            params["prompt_language_strategy"] = prompt_language_strategy
            params["prompt_language"] = prompt_language
            params["store_details"] = args.store_details
            params["efficiency_analysis"] = args.efficiency_analysis
            params["global_sampling_params"] = global_sampling_params
            params["filtered_lang_codes"] = expanded_lang_codes

            print(f"[INFO] Running benchmark '{b}' with HF.")
            try:
                handler(hf_model, **params)
                print(f"[INFO] Benchmark '{b}' completed successfully.")
            except Exception as e:
                error_msg = f"[ERROR] Benchmark '{b}' failed: {str(e)}"
                print(error_msg)
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{error_msg}\n")
                print(f"[INFO] Continuing with next benchmark...")

        if hf_model:
            del hf_model
            torch.cuda.empty_cache()

    if gen_benchmarks:
        print(f"[INFO] Loading vLLM model for: {gen_benchmarks}")
        vllm_model = None
        try:
            vllm_model = load_model(args.model_name, backend="vllm", **model_args)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        except Exception as e:
            error_msg = f"[ERROR] Failed to load vLLM model: {str(e)}"
            print(error_msg)
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{error_msg}\n")
            # Skip all gen benchmarks if model loading fails
            gen_benchmarks = []
            
        for b in gen_benchmarks:
            handler = BENCHMARK_HANDLERS.get(b)
            if not handler:
                print(f"[WARN] No handler for benchmark '{b}'")
                continue
                
            # Get benchmark parameters
            params = benchmark_params.get(b, {})
            params["dev_max_samples"] = dev_max_samples
            params["output_dir"] = base_output_dir
            params["model_name"] = args.model_name
            params["current_time"] = current_time
            params["prompt_library"] = prompt_library
            params["prompt_language_strategy"] = prompt_language_strategy
            params["prompt_language"] = prompt_language
            params["store_details"] = args.store_details
            params["efficiency_analysis"] = args.efficiency_analysis
            params["global_sampling_params"] = global_sampling_params
            # Add filtered language codes to parameters
            params["filtered_lang_codes"] = expanded_lang_codes

            print(f"[INFO] Running benchmark '{b}' with vLLM.")
            try:
                handler(vllm_model, **params)
                print(f"[INFO] Benchmark '{b}' completed successfully.")
            except Exception as e:
                error_msg = f"[ERROR] Benchmark '{b}' failed: {str(e)}"
                print(error_msg)
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{error_msg}\n")
                print(f"[INFO] Continuing with next benchmark...")

        if vllm_model:
            del vllm_model
            torch.cuda.empty_cache()

    # Print summary of errors if any
    try:
        with open(error_log_path, "r", encoding="utf-8") as f:
            errors = f.readlines()
        if errors:
            print(f"[INFO] {len(errors)} benchmarks encountered errors. See {error_log_path} for details.")
    except FileNotFoundError:
        # No errors were logged
        pass

    print("[INFO] All requested benchmarks completed.")

if __name__ == "__main__":
    main()