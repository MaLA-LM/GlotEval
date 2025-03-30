"""
Utility module for translation functions and API rate limiting.
"""

import requests
import uuid
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class AdaptiveRateLimiter:
    """Adaptive rate limiter that automatically adjusts request rate based on API responses."""
    
    def __init__(self, initial_delay=1.0, min_delay=0.5, max_delay=10.0, backoff_factor=1.5, success_factor=0.9):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor  # Factor to increase delay on failure
        self.success_factor = success_factor  # Factor to reduce delay on success
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.success_threshold = 10  # Number of consecutive successes before decreasing delay
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0
        self.start_time = time.time()
    
    def success(self):
        """Record a successful request."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.total_requests += 1
        self.total_successes += 1
        
        # Decrease delay after multiple consecutive successes
        if self.consecutive_successes >= self.success_threshold:
            self.decrease_delay()
            self.consecutive_successes = 0
    
    def failure(self, status_code=None):
        """Record a failed request."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.total_requests += 1
        self.total_failures += 1
        
        # Immediately increase delay
        self.increase_delay(status_code)
    
    def increase_delay(self, status_code=None):
        """Increase waiting delay."""
        # For 429 error, use a more aggressive backoff
        if status_code == 429:
            self.current_delay = min(self.max_delay, self.current_delay * (self.backoff_factor * 1.5))
        else:
            self.current_delay = min(self.max_delay, self.current_delay * self.backoff_factor)
        print(f"⚠️ Increasing delay to {self.current_delay:.2f} seconds")
    
    def decrease_delay(self):
        """Decrease waiting delay."""
        if self.current_delay > self.min_delay:
            self.current_delay = max(self.min_delay, self.current_delay * self.success_factor)
            print(f"✓ Decreasing delay to {self.current_delay:.2f} seconds")
    
    def wait(self):
        """Wait for an appropriate amount of time."""
        # Add some random jitter to avoid fully synchronized request patterns
        jitter = random.uniform(-0.1, 0.1) * self.current_delay
        wait_time = max(0.1, self.current_delay + jitter)
        time.sleep(wait_time)
        return wait_time
    
    def get_stats(self):
        """Get statistics about the rate limiting."""
        elapsed = time.time() - self.start_time
        requests_per_second = self.total_requests / elapsed if elapsed > 0 else 0
        success_rate = (self.total_successes / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "success_rate": f"{success_rate:.1f}%",
            "requests_per_second": f"{requests_per_second:.2f}",
            "current_delay": f"{self.current_delay:.2f}s",
            "elapsed_time": f"{elapsed:.1f}s"
        }

def translate_text(text, target_language, subscription_key, source_language=None,
                   region="northeurope", endpoint="https://api.cognitive.microsofttranslator.com",
                   max_retries=5, rate_limiter=None):
    """
    Translate text into the specified target language, with adaptive rate limiting.
    
    Args:
        text (str): The text to be translated.
        target_language (str): The target language code, e.g., 'zh-Hans' for Simplified Chinese.
        subscription_key (str): The subscription key for Microsoft Translator API.
        source_language (str, optional): The source language code. If not specified, auto-detection is used.
        region (str): The region for the Microsoft Translator API.
        endpoint (str): The endpoint URL for the Microsoft Translator API.
        max_retries (int): Maximum number of retries.
        rate_limiter (AdaptiveRateLimiter): An instance of the rate limiter.
    
    Returns:
        str: The translated text.
    """
    translate_url = f"{endpoint}/translate?api-version=3.0"
    
    # Set request parameters
    params = f"&to={target_language}"
    if source_language:
        params += f"&from={source_language}"
    
    translation_url = f"{translate_url}{params}"
    
    headers = {
        'Content-type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    body = [{
        'text': text
    }]
    
    for retry in range(max_retries):
        try:
            # Use the rate limiter to wait for an appropriate time
            if rate_limiter:
                rate_limiter.wait()
            
            translation_response = requests.post(translation_url, headers=headers, json=body)
            
            if translation_response.status_code == 429:  # Rate limit exceeded
                if rate_limiter:
                    rate_limiter.failure(status_code=429)
                    
                retry_after = int(translation_response.headers.get('Retry-After', 1))
                print(f"⚠️ Rate limit exceeded, waiting {retry_after} seconds before retrying...")
                time.sleep(retry_after + 0.5)  # Add an extra 0.5s to avoid edge cases
                continue
                
            translation_data = translation_response.json()
            
            if 'error' in translation_data:
                error_code = translation_data['error'].get('code', 0)
                error_message = translation_data['error'].get('message', 'Unknown error')
                print(f"⚠️ Translation error: code={error_code}, message={error_message}")
                
                # Handle rate limiting error specifically
                if error_code == 429001:
                    if rate_limiter:
                        rate_limiter.failure(status_code=429)
                    wait_time = 2 ** retry
                    print(f"⚠️ API request limit, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                
                if rate_limiter:
                    rate_limiter.failure()
                return None
            
            # Extract the translation result
            translated_text = translation_data[0]['translations'][0]['text']
            
            # Record success
            if rate_limiter:
                rate_limiter.success()
                
            return translated_text
            
        except Exception as e:
            print(f"⚠️ Error during translation: {str(e)}, retrying...")
            if rate_limiter:
                rate_limiter.failure()
            time.sleep(1 * (2 ** retry))
    
    print(f"❌ Reached the maximum number of retries ({max_retries}), translation failed")
    return None

def batch_translate_text(texts, subscription_key, source_language, target_languages,
                         region="northeurope", endpoint="https://api.cognitive.microsofttranslator.com",
                         rate_limiter=None):
    """
    Translate a batch of texts to multiple target languages using Microsoft Translator API
    
    Args:
        texts (list): List of texts to translate
        subscription_key (str): Microsoft Translator API subscription key
        source_language (str): Source language code (Microsoft format)
        target_languages (list): List of target language codes (Microsoft format)
        region (str): API region
        endpoint (str): API endpoint URL
        rate_limiter (AdaptiveRateLimiter): An optional rate limiter instance
        
    Returns:
        dict: Dictionary mapping target language codes to translated texts
    """
    # API has limits on batch size, so we need to handle this
    batch_size = 100  # Max number of target languages per request
    max_texts_per_request = 100  # Max number of texts per request (API limit)
    
    results = {lang: [""] * len(texts) for lang in target_languages}
    
    # Create a rate limiter if none is provided
    if rate_limiter is None:
        rate_limiter = AdaptiveRateLimiter()
    
    # Process target languages in batches
    for i in range(0, len(target_languages), batch_size):
        batch_languages = target_languages[i:i+batch_size]
        print(f"Translating batch {i//batch_size + 1}/{(len(target_languages) + batch_size - 1) // batch_size} of languages...")
        
        # Process texts in batches if needed
        for j in range(0, len(texts), max_texts_per_request):
            batch_texts = texts[j:j+max_texts_per_request]
            
            # Prepare API URL with target languages
            translate_url = f"{endpoint}/translate?api-version=3.0&from={source_language}"
            for lang in batch_languages:
                translate_url += f"&to={lang}"
            
            headers = {
                'Content-type': 'application/json',
                'Ocp-Apim-Subscription-Key': subscription_key,
                'Ocp-Apim-Subscription-Region': region,
                'X-ClientTraceId': str(uuid.uuid4())
            }
            
            # Prepare request body with multiple texts
            body = [{'text': text} for text in batch_texts]
            
            max_retries = 5
            for retry in range(max_retries):
                try:
                    # Use the rate limiter
                    rate_limiter.wait()
                    
                    response = requests.post(translate_url, headers=headers, json=body)
                    
                    if response.status_code == 429:  # Rate limit exceeded
                        rate_limiter.failure(status_code=429)
                        retry_after = int(response.headers.get('Retry-After', 1))
                        print(f"⚠️ Rate limit exceeded, waiting {retry_after} seconds before retrying...")
                        time.sleep(retry_after + 0.5)
                        continue
                    
                    response.raise_for_status()
                    translation_data = response.json()
                    
                    # Process each text's translations
                    for idx, item in enumerate(translation_data):
                        for translation in item['translations']:
                            target_lang = translation['to']
                            results[target_lang][j + idx] = translation['text']
                    
                    # Record success
                    rate_limiter.success()
                    
                    # Success, break the retry loop
                    break
                    
                except Exception as e:
                    print(f"⚠️ Translation error for batch: {str(e)}")
                    rate_limiter.failure()
                    
                    if retry < max_retries - 1:
                        wait_time = 1 * (2 ** retry)
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed after {max_retries} retries")
    
    return results

def extract_placeholders(text):
    """
    Extract placeholders from the text.
    
    A placeholder is assumed to be in curly braces, e.g. {src_lang}, {text}, etc.
    
    Args:
        text (str): The text to search for placeholders.
        
    Returns:
        set: A set of all placeholders found (without braces).
    """
    if text is None:
        return set()
    
    placeholder_pattern = r'\{([a-zA-Z0-9_]+)\}'
    return set(re.findall(placeholder_pattern, text))

def process_language_parallel(source_ms, source_iso, target_languages, language_names, subscription_key,
                             max_workers=3, rate_limiter=None):
    """
    Use parallel processing to translate from one source language to all target languages.
    
    Args:
        source_ms (str): The Microsoft code of the source language.
        source_iso (str): The ISO code of the source language.
        target_languages (dict): Target languages mapping {iso_code: ms_code}.
        language_names (dict): Language name dictionary {iso_code: english_name}.
        subscription_key (str): Microsoft Translator API key.
        max_workers (int): Maximum number of parallel worker threads.
        rate_limiter (AdaptiveRateLimiter): The rate limiter instance.
    
    Returns:
        dict: Translation results for the source language, e.g. {target_iso: translated_name}.
    """
    results = {}
    target_items = list(target_languages.items())
    
    # Create a rate limiter if none is provided
    if rate_limiter is None:
        rate_limiter = AdaptiveRateLimiter()
        
    def translate_single_language(target_iso, target_ms):
        """Translate a single target language; suitable for thread pool usage."""
        # Get the English name of the target language
        english_name = language_names.get(target_iso, target_iso.split('_')[0])
        
        # Construct text to translate (add "language" if there's no parenthesis)
        if "(" in english_name:
            # Insert "language" before the parenthesis
            parts = english_name.split("(", 1)
            text_to_translate = f"{parts[0].strip()} language ({parts[1]}"
        else:
            text_to_translate = f"{english_name} language"
            
        # Translate the language name
        translated_name = translate_text(
            text_to_translate,
            target_language=source_ms,
            subscription_key=subscription_key,
            source_language="en",
            rate_limiter=rate_limiter
        )
        
        # Handle the translation result
        if translated_name:
            # For the parenthesis case
            if " language (" in translated_name:
                translated_name = translated_name.replace(" language (", " (")
            else:
                translated_name = translated_name.replace(" language", "").replace(" Language", "")
            
            # Remove common suffixes
            for suffix in [" 语言", " 語言", " idioma", " langue", " Sprache", " язык", " 言語", "语言", "語言", "言語"]:
                if translated_name.endswith(suffix):
                    translated_name = translated_name[:-len(suffix)]
            
            return target_iso, translated_name
        return target_iso, None
    
    # Use a thread pool for parallel translation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_target = {
            executor.submit(translate_single_language, target_iso, target_ms): target_iso
            for target_iso, target_ms in target_items
        }
        
        # Use tqdm to show progress
        for future in tqdm(future_to_target, total=len(target_items), desc=f"Processing {source_iso}"):
            try:
                target_iso, translated_name = future.result()
                if translated_name:
                    results[target_iso] = translated_name
                    
                    # Display rate limiter stats after processing every 10 targets
                    if len(results) % 10 == 0 and rate_limiter:
                        stats = rate_limiter.get_stats()
                        print(f"📊 Stats: {stats['total_requests']} requests processed, success rate {stats['success_rate']}, "
                              f"{stats['requests_per_second']} req/s, current delay {stats['current_delay']}")
                        
            except Exception as e:
                print(f"❌ Error processing language {future_to_target[future]}: {str(e)}")
    
    return results

def auto_translate_prompt(original_instruction, original_few_shot, src_lang_code, target_langs,
                          subscription_key=None, region="northeurope"):
    """
    Automatically translate prompt templates into multiple languages, and then verify
    that all placeholders in the source remain present in the translation.
    
    Args:
        original_instruction (str): Original instruction template
        original_few_shot (str): Original few-shot examples
        src_lang_code (str): Source language code (in iso639-3_Script format)
        target_langs (list): List of target language codes (iso639-3_Script) to translate into
        subscription_key (str): Microsoft Translator API subscription key
        region (str): Microsoft Translator API region
        
    Returns:
        dict: Dictionary with translations for each target language
        dict: Dictionary with languages and their missing placeholders
    """
    from language_utils import iso_to_ms_code
    
    translations = {}
    missing_placeholders_report = {}
    
    # Keep the source language in the dict as-is (no translation)
    translations[src_lang_code] = {}
    if original_instruction is not None:
        translations[src_lang_code]["instruction"] = original_instruction
    if original_few_shot is not None:
        translations[src_lang_code]["few_shot"] = original_few_shot
    
    # If there's no subscription key, we cannot translate anything else.
    if not subscription_key:
        print("No subscription key provided. Only returning source language.")
        return translations, missing_placeholders_report
    
    # If both strings are empty, there's nothing to translate
    if not original_instruction and not original_few_shot:
        print("No content provided for translation.")
        return translations, missing_placeholders_report
    
    # Convert source from iso639-3_Script to Microsoft code
    src_lang_ms = iso_to_ms_code(src_lang_code)
    
    if not src_lang_ms:
        print(f"Error: Could not find Microsoft language code for {src_lang_code}. Aborting translation.")
        return translations, missing_placeholders_report

    # Extract placeholders in the source templates (used for post-check)
    src_placeholders_instr = extract_placeholders(original_instruction) if original_instruction else set()
    src_placeholders_few_shot = extract_placeholders(original_few_shot) if original_few_shot else set()
    
    # Filter out the source language from target languages
    target_langs = [lang for lang in target_langs if lang != src_lang_code]
    
    # Convert target languages from ISO to Microsoft format
    target_langs_ms = []
    target_langs_mapping = []  # Keep track of ISO to MS mapping
    for target_lang in target_langs:
        target_ms = iso_to_ms_code(target_lang)
        if target_ms:
            target_langs_ms.append(target_ms)
            target_langs_mapping.append(target_lang)
        else:
            print(f"Warning: Could not find Microsoft language code for {target_lang}. Skipping.")
    
    print(f"Translating to {len(target_langs_ms)} languages...")
    
    # Prepare texts to translate (instruction and few_shot)
    texts_to_translate = []
    if original_instruction:
        texts_to_translate.append(original_instruction)
    if original_few_shot:
        texts_to_translate.append(original_few_shot)
    
    # Perform batch translation
    try:
        translations_result = batch_translate_text(
            texts_to_translate,
            subscription_key,
            src_lang_ms,
            target_langs_ms,
            region
        )
        
        # Process the results and check for missing placeholders
        for i, target_lang_ms in enumerate(target_langs_ms):
            # Get corresponding ISO code
            target_lang = target_langs_mapping[i]
            
            # Initialize the target language entry
            translations[target_lang] = {}
            
            # Get translated texts
            instruction_index = 0
            few_shot_index = 1 if original_instruction else 0
            
            # Get translated instruction if original was provided
            if original_instruction:
                translated_instruction = translations_result[target_lang_ms][instruction_index]
                
                # Skip if translation failed
                if not translated_instruction:
                    print(f"Failed to translate instruction to {target_lang}. Skipping.")
                    continue
                
                # Check placeholders
                instr_placeholders_after = extract_placeholders(translated_instruction)
                missing_instr = [p for p in src_placeholders_instr if p not in instr_placeholders_after]
                
                if missing_instr:
                    if target_lang not in missing_placeholders_report:
                        missing_placeholders_report[target_lang] = {"instruction": [], "few_shot": []}
                    missing_placeholders_report[target_lang]["instruction"] = missing_instr
                
                # Add to translations
                translations[target_lang]["instruction"] = translated_instruction
            
            # Get translated few-shot if original was provided
            if original_few_shot:
                translated_few_shot = translations_result[target_lang_ms][few_shot_index]
                
                # Skip if translation failed
                if not translated_few_shot:
                    print(f"Failed to translate few-shot to {target_lang}. Skipping.")
                    continue
                
                # Check placeholders
                fs_placeholders_after = extract_placeholders(translated_few_shot)
                missing_fs = [p for p in src_placeholders_few_shot if p not in fs_placeholders_after]
                
                if missing_fs:
                    if target_lang not in missing_placeholders_report:
                        missing_placeholders_report[target_lang] = {"instruction": [], "few_shot": []}
                    missing_placeholders_report[target_lang]["few_shot"] = missing_fs
                
                # Add to translations
                translations[target_lang]["few_shot"] = translated_few_shot
            
        print(f"Added translations for {len(translations) - 1} languages")
        
    except Exception as e:
        print(f"Error during batch translation: {str(e)}")
    
    return translations, missing_placeholders_report