#!/usr/bin/env python3

import json
import argparse
import os
import re
import time
from datetime import datetime

# Import from our utility modules
from language_utils import MS_TO_ISO, ISO_TO_MS, get_all_supported_languages, iso_to_ms_code
from translation_utils import extract_placeholders, auto_translate_prompt

def guess_base_task_for_benchmark(benchmark_name):
    """
    A simple guess function or mapping from known benchmarks to a base task.
    """
    benchmarks_mapping = {
        "translation": ["flores200_mt", "flores_plus_mt", "americasnlp", "in22", "ntrex128",
                        "tatoeba", "nteu", "tico19", "mafand"],
        "classification": ["sib200", "taxi1500"],
        "summarization": ["xlsum"],
        "open_generation": ["aya", "polywrite"],
        "comprehension": ["mmmlu", "global_mmlu"],
        "token_classification": ["wikiann", "ud_upos"],
        "nll": ["mala", "pbc"]
    }
    
    for task, benchmarks in benchmarks_mapping.items():
        if benchmark_name in benchmarks:
            return task
    
    return "translation"  # Default to translation

def copy_prompt_to_all_languages(task_or_bench, src_lang_code, instruction, few_shot, library):
    """
    Copy the instruction and few-shot templates from the source language to all other languages
    in the library for the given task or benchmark.
    
    Args:
        task_or_bench (str): Task or benchmark name
        src_lang_code (str): Source language code
        instruction (str): Instruction template to copy (can be None)
        few_shot (str): Few-shot template to copy (can be None)
        library (dict): The prompt library
        
    Returns:
        dict: Updated library with copied templates
    """
    if task_or_bench not in library:
        print(f"Task or benchmark '{task_or_bench}' not found in the library.")
        return library
    
    # Get all language codes for this task/benchmark
    lang_codes = list(library[task_or_bench].keys())
    
    if not lang_codes:
        print(f"No languages found for task/benchmark '{task_or_bench}'.")
        return library
    
    copied_count = 0
    print(instruction)
    for lang_code in lang_codes:
        if lang_code != src_lang_code:  # Skip source language
            # Initialize if needed
            if lang_code not in library[task_or_bench]:
                library[task_or_bench][lang_code] = {}
                
            # Copy each field if provided
            if instruction is not None:
                library[task_or_bench][lang_code]["instruction"] = instruction
                copied_count += 1
            
            if few_shot is not None:
                library[task_or_bench][lang_code]["few_shot"] = few_shot
                copied_count += 1
    
    print(f"Copied templates to {len(lang_codes)-1} languages")
    return library

def main():
    parser = argparse.ArgumentParser(description="Prompt Multilingual Automatic Builder Tool")
    parser.add_argument("--task_or_benchmark", type=str, required=True,
                        help="Which top-level name to update: e.g. 'translation' or 'flores200_mt'")
    parser.add_argument("--lang_code", type=str, required=True,
                        help="Which language code is the new prompt intended for, e.g. 'fra_Latn'")
    parser.add_argument("--new_instruction", type=str, default=None,
                        help="New instruction template (must contain required placeholders).")
    parser.add_argument("--new_few_shot", type=str, default=None,
                        help="New few-shot template if desired.")
    parser.add_argument("--target_lang_codes", nargs="*", default=None,
                        help="Additional languages to auto-translate the new prompt into. If not specified, will translate to all supported languages.")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to the existing prompt library file to modify.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File path to write updated library. If not specified, will be determined based on input_file.")
    parser.add_argument("--ms_translator_key", type=str, default="",
                        help="Microsoft Translator API subscription key for auto-translation.")
    parser.add_argument("--ms_translator_region", type=str, default="northeurope",
                        help="Microsoft Translator API region (default: northeurope).")
    parser.add_argument("--copy_all", action="store_true", 
                        help="Copy the templates to all languages instead of translating. Requires input_file.")
    
    args = parser.parse_args()

    # Check if at least one of new_instruction or new_few_shot is provided
    if args.new_instruction is None and args.new_few_shot is None:
        print("[ERROR] At least one of --new_instruction or --new_few_shot must be provided.")
        return

    # Determine input/output files
    input_file = args.input_file
    output_file = args.output_file
    if output_file is None:
        if input_file is not None:
            user_choice = input(f"No output file specified. Do you want to:\n"
                                f"1) Directly modify the input file '{input_file}' (may overwrite existing entries)\n"
                                f"2) Create a new file prompt_library_{time.strftime('%Y%m%d_%H%M%S')}.json\n"
                                f"Enter 1 or 2: ")
            if user_choice.strip() == "1":
                output_file = input_file
                print(f"Will update '{input_file}' directly.")
            else:
                output_file = f"prompt_library_{time.strftime('%Y%m%d_%H%M%S')}.json"
                print(f"Will create new file: '{output_file}'")
        else:
            # No input or output file, create new with timestamp
            output_file = f"prompt_library_{time.strftime('%Y%m%d_%H%M%S')}.json"
            print(f"No input or output file specified. Will create: '{output_file}'")

    # Load existing prompt_library if provided
    if input_file and os.path.exists(input_file):
        print(f"Loading existing prompt library from '{input_file}'")
        with open(input_file, "r", encoding="utf-8") as f:
            library = json.load(f)
    else:
        print("Starting with an empty prompt library.")
        library = {}

    # Check config & guidelines
    if not os.path.exists("config.json"):
        print("config.json not found. Exiting.")
        return
    
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    guidelines = config.get("prompt_guidelines", {})

    # Determine placeholders from guidelines
    task_or_bench = args.task_or_benchmark
    if task_or_bench in guidelines:
        required_placeholders = guidelines[task_or_bench]["required_placeholders"]
    else:
        # Assume it's a benchmark
        base_task = guess_base_task_for_benchmark(task_or_bench)
        if base_task not in guidelines:
            print(f"No guidelines found for base_task='{base_task}'. Please define them.")
            return
        required_placeholders = guidelines[base_task]["required_placeholders"]

    # If new_instruction is provided, validate placeholders
    if args.new_instruction is not None:
        for ph in required_placeholders:
            if ph not in args.new_instruction:
                print(f"[ERROR] Missing required placeholder '{ph}' in new_instruction.")
                return

    # Determine final values to use
    instruction_to_use = args.new_instruction if args.new_instruction is not None else None
    few_shot_to_use = args.new_few_shot if args.new_few_shot is not None else None
    
    # Check if values are available for copy mode
    if args.copy_all and instruction_to_use is None and few_shot_to_use is None:
        print("[ERROR] Cannot copy templates: no instruction or few-shot provided and none exists for the source language.")
        return
    
    # Initialize task/benchmark if needed
    if task_or_bench not in library:
        library[task_or_bench] = {}
    
    # Initialize language if needed
    if args.lang_code not in library[task_or_bench]:
        library[task_or_bench][args.lang_code] = {}
    
    # Update the source language with new values
    if instruction_to_use is not None:
        library[task_or_bench][args.lang_code]["instruction"] = instruction_to_use
    
    if few_shot_to_use is not None:
        library[task_or_bench][args.lang_code]["few_shot"] = few_shot_to_use
    
    print(f"Updated {args.lang_code} template for '{task_or_bench}'.")

    # Process based on mode - copy or translate
    if args.copy_all:
        # Copy mode
        if not args.input_file:
            print("[ERROR] --copy_all requires an input file. Cannot copy to languages that don't exist yet.")
            return
        
        library = copy_prompt_to_all_languages(
            task_or_bench, 
            args.lang_code,
            instruction_to_use,
            few_shot_to_use,
            library
        )
    
    elif args.target_lang_codes is not None:
        # Translation mode
        target_langs = args.target_lang_codes or []  # Ensure it's a list
        
        if not args.ms_translator_key:
            print("Warning: No Microsoft Translator API key provided. Auto-translation disabled.")
        else:
            # If empty list, use all languages from our mapping
            if len(target_langs) == 0:
                print("No specific target languages provided. Using all supported languages...")
                target_langs = get_all_supported_languages()
                print(f"Found {len(target_langs)} supported languages for translation.")
            
            if target_langs:
                translations, missing_placeholders = auto_translate_prompt(
                    instruction_to_use,
                    few_shot_to_use,
                    args.lang_code,
                    target_langs,
                    args.ms_translator_key,
                    args.ms_translator_region
                )
                
                # Add translations to the library
                for tl, content in translations.items():
                    if tl != args.lang_code:  # Skip source language
                        if tl not in library[task_or_bench]:
                            library[task_or_bench][tl] = {}
                        
                        # Add instruction if it was translated
                        if "instruction" in content:
                            library[task_or_bench][tl]["instruction"] = content["instruction"]
                        
                        # Add few-shot if it was translated
                        if "few_shot" in content:
                            library[task_or_bench][tl]["few_shot"] = content["few_shot"]
                
                # Report missing placeholders at the end
                if missing_placeholders:
                    print("\nWARNING: Missing placeholders detected in some translations:")
                    for lang, missing in missing_placeholders.items():
                        if "instruction" in missing and missing["instruction"]:
                            print(f"  - {lang} instruction: Missing {', '.join(missing['instruction'])}")
                        if "few_shot" in missing and missing["few_shot"]:
                            print(f"  - {lang} few-shot: Missing {', '.join(missing['few_shot'])}")
                    print("Please review and fix these manually in the output file.")
            else:
                print("No target languages available for translation.")

    # 7. Save to output file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

    print(f"Successfully updated {output_file}")

if __name__ == "__main__":
    main()