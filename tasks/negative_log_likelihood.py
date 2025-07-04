import os
import csv
import json
import time
import random
import torch
from tqdm import tqdm

from . import register_benchmark
from metrics.classification_metrics import compute_accuracy
from benchmark_data_loader.data_loader import load_mala_data, load_pbc_data
from .benchmark_utils import (
    setup_benchmark_output,
    update_benchmark_scores
)
def process_nll_benchmark(
    benchmark_name,
    model,
    load_data_func,
    has_dev=False,
    lang_config=None,
    **kwargs
):
    # Extract key parameters from kwargs
    seed = kwargs.get("seed", 42)
    dev_max_samples = kwargs.get("dev_max_samples", None)
    output_dir = kwargs.get("output_dir", "results")
    model_name = kwargs.get("model_name", "unknown_model")
    current_time = kwargs.get("current_time", time.strftime("%Y%m%d_%H%M%S"))
    max_length = kwargs.get("max_length", 2048)
    stride = kwargs.get("stride", 1024)
    filtered_lang_codes = kwargs.get("filtered_lang_codes", None)  # Add this line

    global_sampling_params = kwargs.get("global_sampling_params", {})
    benchmark_sampling_params = kwargs.get("sampling_params", {})
    final_sampling_params = global_sampling_params.copy()
    final_sampling_params.update(benchmark_sampling_params)
    print(f"[{benchmark_name}] Using final sampling params: {final_sampling_params}")
    

    # Setup benchmark output directory
    benchmark_output_dir = setup_benchmark_output(benchmark_name, output_dir)

    # Load language configuration
    if not lang_config:
        raise ValueError("No lang_config provided for NLL benchmark.")
    
    # Load all languages from config file
    with open(lang_config, "r", encoding="utf-8") as f:
        all_langs = [line.strip() for line in f if line.strip()]
    
    # Apply language filtering if requested
    if filtered_lang_codes:
        # Filter languages based on the expanded language codes
        langs = [lang for lang in all_langs if lang.split('_')[0] in filtered_lang_codes]
        num_test_langs = len(langs)
        num_benchmark_langs = len(all_langs)
        print(f"[INFO] Filtered {num_test_langs} / {num_benchmark_langs} languages for benchmark '{benchmark_name}'")
        
        if not langs:
            print(f"[WARN] No languages matched the filter for benchmark '{benchmark_name}'")
            return {}
    else:
        langs = all_langs
        num_test_langs = len(langs)
        num_benchmark_langs = num_test_langs
        print(f"[INFO] Processing all {num_test_langs} languages for benchmark '{benchmark_name}'")

    random.seed(seed)
    all_results = {}

    for lang_code in langs:
        print(f"Processing {lang_code} for {benchmark_name}")
        # load text data
        texts = load_data_func(lang_code)
        if not texts:
            print(f"No data found for {lang_code}")
            continue

        if dev_max_samples is not None and len(texts) > dev_max_samples:
            texts = texts[:dev_max_samples]

        encodings = model.tokenizer("\n\n".join(texts), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        total_nll = torch.stack(nlls).sum().item()
        all_results[lang_code] = {"nll": total_nll, "num_samples": len(texts)}

    # Update scores.json using utility function
    # Add tested_languages info to benchmark_params
    benchmark_params = {
        "tested_languages": f'{num_test_langs} / {num_benchmark_langs}'
    }
    
    update_benchmark_scores(
        output_dir, benchmark_name, model_name, current_time,
        all_results, "NLL", benchmark_params
    )

    print(f"[{benchmark_name}] Done. NLL results saved.")
    return all_results


@register_benchmark("mala")
def mala_handler(model, **kwargs):
    lang_config_path = "benchmark_data_loader/data_langid/mala_langs.txt"
    return process_nll_benchmark(
        benchmark_name="mala",
        model=model,
        load_data_func=load_mala_data,
        has_dev=False,
        lang_config=lang_config_path,
        **kwargs
    )


@register_benchmark("pbc")
def pbc_handler(model, **kwargs):
    lang_config_path = "benchmark_data_loader/data_langid/pbc_langs.txt"
    return process_nll_benchmark(
        benchmark_name="pbc",
        model=model,
        load_data_func=load_pbc_data,
        has_dev=True,
        lang_config=lang_config_path,
        **kwargs
    )