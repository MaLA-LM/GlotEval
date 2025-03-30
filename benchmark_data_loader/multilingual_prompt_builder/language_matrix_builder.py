import json
import csv
import time
import os
import random
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import from our utility modules
from language_utils import MS_TO_ISO, ISO_TO_MS, LANGUAGE_NAMES, LANGUAGE_NAMES_WITH_SCRIPT
from translation_utils import AdaptiveRateLimiter, translate_text, process_language_parallel

def load_existing_results(file_path="language_matrix.json"):
    """Load existing results file if present."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return {}

def save_results(result_dict, filename="language_matrix.json"):
    """Save results to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")

def process_all_languages(iso_to_ms, ms_to_iso, language_names, subscription_key, output_dir=".", 
                         resume=True, languages_to_process=None, max_workers=3, 
                         initial_delay=1.0, adaptive_speed=True):
    """
    Process translations for all languages by source language, supporting adaptive rate limiting.
    
    Args:
        iso_to_ms (dict): Mapping from ISO codes to MS codes.
        ms_to_iso (dict): Mapping from MS codes to ISO codes.
        language_names (dict): Language name dictionary.
        subscription_key (str): Microsoft Translator API key.
        output_dir (str): Output directory.
        resume (bool): Whether to resume from previous results.
        languages_to_process (list): A list of languages to process. If None, process all.
        max_workers (int): Maximum number of parallel workers.
        initial_delay (float): Initial delay (in seconds).
        adaptive_speed (bool): Whether to enable adaptive rate limiting.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing matrix if present
    result_file = os.path.join(output_dir, "language_matrix.json")
    matrix = load_existing_results(result_file) if resume else {}
    
    # Determine which languages to process
    if languages_to_process is None:
        languages_to_process = list(iso_to_ms.keys())
    
    # Create a rate limiter
    rate_limiter = AdaptiveRateLimiter(initial_delay=initial_delay) if adaptive_speed else None
    
    # Process each language
    for source_iso in languages_to_process:
        source_ms = iso_to_ms.get(source_iso)
        
        if not source_ms:
            print(f"⚠️ Warning: Could not find MS code for {source_iso}, skipping")
            continue
            
        # Check if this language is already processed
        lang_file = os.path.join(output_dir, f"{source_iso}.json")
        if resume and os.path.exists(lang_file):
            print(f"📂 Loading existing results for {source_iso}...")
            lang_results = load_existing_results(lang_file)
            
            # Update the main matrix
            if source_iso not in matrix:
                matrix[source_iso] = {}
            matrix[source_iso].update(lang_results)
            
            # Save the updated matrix
            save_results(matrix, result_file)
            print(f"⏩ Skipped processing {source_iso}, using existing results")
            continue
            
        print(f"\n🔄 Starting processing for language: {source_iso} (MS code: {source_ms})")
        
        # Use parallel processing for efficiency
        lang_results = process_language_parallel(
            source_ms, 
            source_iso, 
            iso_to_ms, 
            language_names, 
            subscription_key,
            max_workers=max_workers,
            rate_limiter=rate_limiter
        )
        
        # Save this language's results
        save_results(lang_results, lang_file)
        
        # Update the main matrix
        if source_iso not in matrix:
            matrix[source_iso] = {}
        matrix[source_iso].update(lang_results)
        
        # Save the updated matrix
        save_results(matrix, result_file)
        
        print(f"✅ Completed processing for {source_iso}, results saved\n")
        
        # Rest for a short time after each language
        delay = 3 + (random.random() * 2)
        print(f"💤 Resting for {delay:.2f} seconds before continuing...")
        time.sleep(delay)
    
    # Save in other formats
    generate_output_formats(matrix, output_dir)
    print("🎉 All languages have been processed!")

def generate_output_formats(matrix, output_dir="."):
    """Generate different output file formats."""
    # Save as CSV
    csv_path = os.path.join(output_dir, "language_matrix.csv")
    save_as_csv(matrix, csv_path)
    
    # Attempt to save as Excel
    try:
        excel_path = os.path.join(output_dir, "language_matrix.xlsx")
        save_as_excel(matrix, excel_path)
    except Exception as e:
        print(f"⚠️ Error saving as Excel: {str(e)}")
        print("💡 Please install pandas and openpyxl: pip install pandas openpyxl")

def save_as_csv(matrix, filename):
    """Save the matrix as a CSV file."""
    all_languages = sorted(matrix.keys())
    
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        header = ['ISO_Code'] + all_languages
        writer.writerow(header)
        
        # Write each row
        for source_lang in all_languages:
            row = [source_lang]
            for target_lang in all_languages:
                row.append(matrix.get(source_lang, {}).get(target_lang, ''))
            writer.writerow(row)
    
    print(f"📄 CSV file saved: {filename}")

def save_as_excel(matrix, filename):
    """Save the matrix as an Excel file."""
    import pandas as pd
    
    # Convert to DataFrame
    all_languages = sorted(matrix.keys())
    data = {}
    
    for target_lang in all_languages:
        data[target_lang] = {}
        for source_lang in all_languages:
            data[target_lang][source_lang] = matrix.get(source_lang, {}).get(target_lang, '')
    
    df = pd.DataFrame(data)
    df.to_excel(filename, engine='openpyxl')
    print(f"📊 Excel file saved: {filename}")

# Usage example
if __name__ == "__main__":
    # Your API key
    subscription_key = "your-key"
    
    # Process all languages
    process_all_languages(
        iso_to_ms=ISO_TO_MS,
        ms_to_iso=MS_TO_ISO,
        language_names=LANGUAGE_NAMES_WITH_SCRIPT,
        subscription_key=subscription_key,
        output_dir="language_matrix_results",
        resume=True,  # Set to True to resume from previous partial results
        max_workers=10,  # Number of parallel threads
        initial_delay=0.8,  # Initial request delay
        adaptive_speed=True  # Enable adaptive rate limiting
    )
    
    # Or process only a few specified languages:
    # languages_to_process = ['eng_Latn', 'zho_Hans', 'jpn_Jpan']
    # process_all_languages(
    #     iso_to_ms=ISO_TO_MS,
    #     ms_to_iso=MS_TO_ISO,
    #     language_names=LANGUAGE_NAMES_WITH_SCRIPT,
    #     subscription_key=subscription_key,
    #     output_dir="language_matrix_results",
    #     resume=True,
    #     languages_to_process=languages_to_process,
    #     max_workers=10,
    #     initial_delay=0.8,
    #     adaptive_speed=True
    # )