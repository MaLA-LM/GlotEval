import os
import pandas as pd
import json
import random
from datasets import load_from_disk, load_dataset

from iso639 import Lang, is_language

### Language ID config related 


def filter_language_config(lang_config_path: str, filtered_lang_codes: set, center_lang: str):
    """
    Filter language configuration based on filtered language codes.
    
    Args:
        lang_config_path (str): Path to the language configuration file
        filtered_lang_codes (set): Set of ISO 639 language codes to filter by
        center_lang (str): Center language code that must be included
        
    Returns:
        Dict[str, str]: Dictionary mapping benchmark codes to prompt codes for filtered languages
    """
    # If no filtering requested, just load the full config
    if filtered_lang_codes is None:
        return load_full_language_config(lang_config_path)
        
    prompt_langs_map = {}  # Map benchmark language codes to prompt language codes
    center_lang_present = False
    
    with open(lang_config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse the mapping between benchmark lang code and prompt lang code
            benchmark_code = line
            prompt_code = line
            
            if '|' in line:
                benchmark_code, prompt_code = line.split('|', 1)
                benchmark_code = benchmark_code.strip()
                prompt_code = prompt_code.strip()
            else:
                # If no mapping is provided, use the same code for both
                prompt_code = benchmark_code
            
            # Check if this is the center language
            if benchmark_code == center_lang:
                prompt_langs_map[benchmark_code] = prompt_code
                center_lang_present = True
                continue
                
            # Check if this language should be included based on filtered codes
            try:
                # Try different matching strategies
                prompt_lang_base = prompt_code.split('_')[0]
                
                for code in filtered_lang_codes:
                    # Various matching strategies
                    if (code == prompt_lang_base or 
                        code.lower() == prompt_lang_base.lower() or
                        (is_language(prompt_lang_base) and Lang(prompt_lang_base).pt3 == code) or
                        (is_language(code) and Lang(code).pt3 == prompt_lang_base)):
                        prompt_langs_map[benchmark_code] = prompt_code
                        break
            except Exception as e:
                print(f"Warning: Error while matching language '{prompt_code}': {str(e)}")
    
    # If center language is not present, search for it specifically
    if not center_lang_present:
        print(f"[WARN] Center language '{center_lang}' not found in filtered languages.")
        print(f"[INFO] Adding center language to ensure proper translation pairs.")
        
        with open(lang_config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the mapping between benchmark lang code and prompt lang code
                if '|' in line:
                    benchmark_code, prompt_code = line.split('|', 1)
                    benchmark_code = benchmark_code.strip()
                    prompt_code = prompt_code.strip()
                else:
                    benchmark_code = line
                    prompt_code = line
                
                # Check if this is the center language
                if benchmark_code == center_lang:
                    prompt_langs_map[benchmark_code] = prompt_code
                    break
    
    if not prompt_langs_map:
        print(f"[WARN] No languages matched the filter criteria. Using full language config.")
        return load_full_language_config(lang_config_path)
        
    return prompt_langs_map


def load_full_language_config(lang_config_path: str):
    """
    Load the full language configuration without filtering.
    
    Args:
        lang_config_path (str): Path to the language configuration file
        
    Returns:
        Dict[str, str]: Dictionary mapping benchmark codes to prompt codes
    """
    prompt_langs_map = {}  # Map benchmark language codes to prompt language codes
    
    with open(lang_config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse the mapping between benchmark lang code and prompt lang code
            benchmark_code = line
            prompt_code = line
            
            if '|' in line:
                benchmark_code, prompt_code = line.split('|', 1)
                benchmark_code = benchmark_code.strip()
                prompt_code = prompt_code.strip()
            
            prompt_langs_map[benchmark_code] = prompt_code
            
    return prompt_langs_map


### Prompt related


def choose_prompt_text(
    task_key,
    library,
    lang_code,
    benchmark_name=None,
    fallback_lang="eng_Latn"
):
    """
    Retrieve the prompt text (instruction + few_shot) for a given task_key or benchmark.

    Args:
        task_key (str): e.g. 'translation', 'classification', etc.
        library (dict): The loaded JSON from prompt_library.json
        lang_code (str): e.g. 'spa_Latn'
        benchmark_name (str): e.g. 'flores200_mt', 'sib200'
        fallback_lang (str): e.g. 'eng_Latn'
    Returns:
        dict with keys {"instruction", "few_shot"} at minimum.
    """

    # 1) Check if there's a benchmark-specific entry
    if benchmark_name and benchmark_name in library:
        # if the requested lang_code is in that benchmark section
        if lang_code in library[benchmark_name]:
            return library[benchmark_name][lang_code]
        else:
            # fallback to English if it exists in the benchmark section
            if fallback_lang in library[benchmark_name]:
                return library[benchmark_name][fallback_lang]

    # 2) fallback to the task-level template
    if task_key in library:
        if lang_code in library[task_key]:
            return library[task_key][lang_code]
        else:
            if fallback_lang in library[task_key]:
                return library[task_key][fallback_lang]

    # 3) if truly nothing found, return a minimal
    return {
        "instruction": "No prompt found. Please define in prompt_library.json.",
        "few_shot": ""
    }


def build_translation_prompt(
    src_lang,
    tgt_lang,
    src_text,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key="translation"
):
    """
    Build a prompt for translation tasks.
    
    Args:
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        src_text (str): The source text to translate.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.
        
    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    # Get the prompt template
    prompt_template = None
    
    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]
    
    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]
    
    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "instruction": "Translate the following sentence from {src_lang} to {tgt_lang}\n[{src_lang}]: {src_text}\n[{tgt_lang}]:",
            "few_shot": "[{src_lang}]: {src_text}\n[{tgt_lang}]: {tgt_text}\n"
        }
    
    instruction_template = prompt_template.get("instruction", "")
    few_shot_template = prompt_template.get("few_shot", "")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            ex_src_text = ex.get("src_text", "")
            ex_tgt_text = ex.get("tgt_text", "")
            ex_src_lang = ex.get("src_lang", src_lang)
            ex_tgt_lang = ex.get("tgt_lang", tgt_lang)
            # fill the placeholders
            snippet = few_shot_template.format(
                src_lang=ex_src_lang,
                src_text=ex_src_text,
                tgt_text=ex_tgt_text,
                tgt_lang=ex_tgt_lang
            )
            few_shot_str += snippet

    final_str = instruction_template.format(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_text=src_text
    )
    
    final_prompt = ""
    if few_shot_str:
        final_prompt = few_shot_str + "\n" + final_str
    else:
        final_prompt = final_str

    return final_prompt



def build_classification_prompt(
    text,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key="classification"
):
    """
    Build a prompt for classification tasks.
    
    Args:
        text (str): The text to classify.
        lang_code (str): The language code of the text.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.
        
    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    # Get the prompt template
    prompt_template = None
    
    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]
    
    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]
    
    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "instruction": "The topic of the news \"{text}\" is",
            "few_shot": "The topic of the news \"{example_text}\" is {example_category}\n"
        }
    
    instruction_template = prompt_template.get("instruction", "")
    few_shot_template = prompt_template.get("few_shot", "")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            snippet = few_shot_template.format(
                example_text=ex["text"],
                example_category=ex["category"]
            )
            few_shot_str += snippet

    final_str = instruction_template.format(text=text)
    final_prompt = ""
    if few_shot_str:
        final_prompt = few_shot_str + "\n" + final_str
    else:
        final_prompt = final_str

    return final_prompt

def build_summarization_prompt(
    text,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key="summarization"
):
    """
    Build a prompt for summarization tasks.
    
    Args:
        text (str): The text to summarize.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.
        
    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    # Get the prompt template
    prompt_template = None
    
    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]
    
    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]
    
    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "title": "Document",
            "instruction": "Based on the previous text, provide a brief single summary"
        }
    
    instruction_template = prompt_template.get("instruction", "")
    # Some summarization prompts might have a 'title', though not strictly needed
    title = prompt_template.get("title", None)
    # For few-shot support
    few_shot_template = prompt_template.get("few_shot", "Original: {src_text}\nSummary: {tgt_text}\n\n")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            snippet = few_shot_template.format(
                src_text=ex.get("src_text", ""),
                tgt_text=ex.get("tgt_text", "")
            )
            few_shot_str += snippet

    final_str = instruction_template.format(text=text)
    
    final_prompt = ""
    if title:
        final_prompt = f"{title}\n\n{text}\n\n"
    else:
        final_prompt = f"{text}\n\n"
        
    if few_shot_str:
        final_prompt = few_shot_str + final_prompt + final_str
    else:
        final_prompt = final_prompt + final_str

    return final_prompt


def build_open_generation_prompt(
    text,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key="open_generation"
):
    """
    Build a prompt for open-ended generation tasks.
    
    Args:
        text (str): The text prompt for generation.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.
        
    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    # Get the prompt template
    prompt_template = None
    
    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]
    
    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]
    
    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "instruction": "Please produce a creative or relevant continuation for this prompt:\n{text}",
            "few_shot": "Example prompt: {example_text}\n---\n"
        }
    
    instruction_template = prompt_template.get("instruction", "")
    few_shot_template = prompt_template.get("few_shot", "")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            snippet = few_shot_template.format(example_text=ex.get("text", ""))
            few_shot_str += snippet

    final_str = instruction_template.format(text=text)
    
    final_prompt = ""
    if few_shot_str:
        final_prompt = few_shot_str + "\n" + final_str
    else:
        final_prompt = final_str

    return final_prompt


def build_comprehension_prompt_multi(
    question,
    options_str,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key="comprehension"
):
    """
    Build a prompt for multiple-choice comprehension tasks.
    
    Args:
        question (str): The question to answer.
        options_str (str): The formatted options string.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.
        
    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    # Get the prompt template
    prompt_template = None
    
    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]
    
    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]
    
    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "instruction": "Question: {question}\nOptions:\n{options}\nAnswer:",
            "few_shot": "Question: {example_question}\nOptions:\n{example_options}\nAnswer: {example_answer}\n"
        }
    
    instruction_template = prompt_template.get("instruction", "")
    few_shot_template = prompt_template.get("few_shot", "")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            snippet = few_shot_template.format(
                example_question=ex.get("example_question", ""),
                example_options=ex.get("example_options", ""),
                example_answer=ex.get("example_answer", "")
            )
            few_shot_str += snippet

    final_str = instruction_template.format(question=question, options=options_str)
    
    final_prompt = ""
    if few_shot_str:
        final_prompt = few_shot_str + "\n" + final_str
    else:
        final_prompt = final_str

    return final_prompt


def build_token_classification_prompt(
    tokens, 
    idx,
    candidate_labels,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key="token_classification"
):
    """
    Build a prompt for token classification tasks.
    
    Args:
        tokens (list): List of tokens in the sentence.
        idx (int): Index of the token to classify.
        candidate_labels (list): Possible labels for classification.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.
        
    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    token_str = tokens[idx]
    sentence_str = " ".join(tokens)

    # Get the prompt template
    prompt_template = None
    
    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]
    
    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]
    
    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "instruction": "In the sentence: {sentence}, classify the token: {token} among {candidate_labels}\nAnswer:",
            "few_shot": "Sentence: {example_sentence}\nToken: {example_token}\nLabel: {example_label}\n"
        }
    
    instruction_template = prompt_template.get("instruction", "")
    few_shot_template = prompt_template.get("few_shot", "")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            example_tokens_str = " ".join(ex.get("tokens", []))
            example_token_str = ex.get("tokens", [])[ex.get("idx", 0)]
            snippet = few_shot_template.format(
                example_sentence=example_tokens_str,
                example_token=example_token_str,
                example_label=ex.get("label", "")
            )
            few_shot_str += snippet

    final_str = instruction_template.format(
        sentence=sentence_str,
        token=token_str,
        candidate_labels=", ".join(candidate_labels)
    )

    final_prompt = ""
    if few_shot_str:
        final_prompt = few_shot_str + "\n" + final_str
    else:
        final_prompt = final_str

    return final_prompt


def build_chat_generation_prompt(
    text,
    tokenizer,
    few_shot_examples=None,
    prompt_library=None,
    prompt_language="eng_Latn",
    benchmark_name=None,
    task_key=None
):
    """
    Build a prompt for open-ended generation tasks.

    Args:
        text (str): The text prompt for generation.
        few_shot_examples (list, optional): Examples for few-shot learning.
        prompt_library (dict, optional): Library of prompt templates.
        prompt_language (str, optional): The language to use for the prompt.
        benchmark_name (str, optional): The name of the benchmark.
        task_key (str, optional): The key for the task in the prompt library.

    Returns:
        str: The formatted prompt.
    """
    if prompt_library is None:
        prompt_library = {}

    # Get the prompt template
    prompt_template = None

    # Try to get a benchmark-specific prompt if specified
    if benchmark_name and benchmark_name in prompt_library and prompt_language in prompt_library[benchmark_name]:
        prompt_template = prompt_library[benchmark_name][prompt_language]

    # Otherwise try to get a task-specific prompt
    elif task_key in prompt_library and prompt_language in prompt_library[task_key]:
        prompt_template = prompt_library[task_key][prompt_language]

    # If still no template, use a default
    if not prompt_template:
        prompt_template = {
            "instruction": "{text}",
            "few_shot": "{example_text}"
        }

    instruction_template = prompt_template.get("instruction", "")
    few_shot_template = prompt_template.get("few_shot", "")

    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            snippet = few_shot_template.format(example_text=ex.get("text", ""))
            few_shot_str += snippet

    final_str = instruction_template.format(text=text)

    final_prompt = ""
    if few_shot_str:
        final_prompt = few_shot_str + "\n" + final_str
    else:
        final_prompt = final_str

    message = [{"role": "user", "content": final_prompt}]
    final_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    return final_prompt


def sample_few_shot_examples(n_shots, src_texts_dev, tgt_texts_dev, seed=42):
    random.seed(seed)
    indices = range(len(src_texts_dev))
    chosen = random.sample(indices, min(n_shots, len(src_texts_dev)))
    few_shot_examples = []
    for idx in chosen:
        ex_src = src_texts_dev[idx]
        ex_tgt = tgt_texts_dev[idx]
        few_shot_examples.append({
            "src_text": ex_src,
            "tgt_text": ex_tgt
        })
    return few_shot_examples


def sample_few_shot_classification_examples(train_data, n_shots, seed=42):
    random.seed(seed)
    indices = range(len(train_data))
    chosen = random.sample(indices, min(n_shots, len(train_data)))
    few_shot_examples = []
    for idx in chosen:
        ex = train_data[idx]
        few_shot_examples.append({
            "text": ex["text"],
            "category": ex["category"]
        })
    return few_shot_examples

############################################################
## Data loading with unified pattern for local disk & HuggingFace
############################################################

def load_flores200_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/flores200"
    if split == 'test':
        split = 'devtest'
    src_file = os.path.join(base_path, split, f"{src_lang}.{split}")
    tgt_file = os.path.join(base_path, split, f"{tgt_lang}.{split}")

    with open(src_file, "r", encoding="utf-8") as f:
        src_texts = f.read().splitlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_texts = f.read().splitlines()

    if len(src_texts) != len(tgt_texts):
        raise ValueError("Source and target files have different numbers of lines.")

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts

def load_flores_plus_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads FLORES+ data from Hugging Face.
    """
    print(f"[INFO] Loading FLORES+ for '{src_lang}-{tgt_lang}' from Hugging Face.")
    if split == 'test':
        split = 'devtest'

    try:
        src_dataset = load_dataset("openlanguagedata/flores_plus", src_lang, split=split)
        tgt_dataset = load_dataset("openlanguagedata/flores_plus", tgt_lang, split=split)
        
        assert len(src_dataset) == len(tgt_dataset), "Source and target datasets have different lengths!"
        
        src_texts = src_dataset['text']
        tgt_texts = tgt_dataset['text']
        
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
            tgt_texts = tgt_texts[:limit_samples]
        
        return src_texts, tgt_texts
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"[ERROR] Failed to load FLORES+ for '{src_lang}-{tgt_lang}': {e}")
        if any(keyword in error_msg for keyword in ['unknown split', 'not found', '404', 'does not exist', 'no config']):
            raise FileNotFoundError(f"{split} dataset not found for {src_lang}-{tgt_lang}")
        elif 'gated' in error_msg or 'authentication' in error_msg:
            raise PermissionError(f"Authentication required for gated dataset: {src_lang}-{tgt_lang}")
        else:
            raise ValueError(f"Failed to load data for {src_lang}-{tgt_lang}: {e}")


def load_sib200_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads SIB-200 data from Hugging Face.
    This benchmark is loaded from the 'Davlan/sib200' dataset.
    
    Args:
        lang_code (str): The language code (e.g., 'eng_Latn').
        split (str): The data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.

    Returns:
        list: A list of data samples as dictionaries.
    """
    print(f"[INFO] Loading SIB-200 for '{lang_code}' from Hugging Face.")

    try:
        # SIB-200 uses the language code as the configuration name.
        dataset = load_dataset("Davlan/sib200", lang_code, split=split)
        data = []
        for item in dataset:
            data.append({
                "text": item["text"],
                "category": item["category"]
            })

        if limit_samples is not None:
            data = data[:limit_samples]

        return data
    except Exception as e:
        print(f"[ERROR] Failed to load SIB-200 for '{lang_code}' from Hugging Face: {e}")
        return []


def load_xlsum_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads XLSum data from Hugging Face.
    This benchmark is loaded from the 'csebuetnlp/xlsum' dataset.
    Note: The 'lang_code' must match one of the available configurations (e.g., 'english', 'french').
    
    Args:
        lang_code (str): Language configuration name (e.g., 'english').
        split (str): Data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.
        
    Returns:
        tuple: (src_texts, tgt_texts) - Lists of source articles and target summaries.
    """
    print(f"[INFO] Loading XLSum for '{lang_code}' from Hugging Face.")
    
    try:
        # XLSum uses the language name as the configuration.
        dataset = load_dataset("csebuetnlp/xlsum", lang_code, split=split)
        
        src_texts = dataset['text']
        tgt_texts = dataset['summary']
        
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
            tgt_texts = tgt_texts[:limit_samples]
            
        return src_texts, tgt_texts
        
    except Exception as e:
        print(f"[ERROR] Failed to load XLSum for '{lang_code}' from Hugging Face: {e}")
        print(f"[INFO] Make sure '{lang_code}' is a valid configuration for csebuetnlp/xlsum (e.g., 'english', 'hindi').")
        return [], []

def load_massivesumm_long_data(lang_code, split="train", limit_samples=None, **kwargs):
    """
    Loads MassiveSumm long data from Hugging Face Hub.
    Dataset: MaLA-LM/MassiveSumm_long
    """
    from datasets import load_dataset
    
    print(f"[INFO] Loading MassiveSumm long for '{lang_code}' from Hugging Face Hub.")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("MaLA-LM/MassiveSumm_long", split="train")
        
        # Filter by language
        filtered_dataset = dataset.filter(lambda x: x["language"] == lang_code)
        
        # Extract texts and summaries
        src_texts = []
        tgt_texts = []
        
        for item in filtered_dataset:
            src_texts.append(item["text"])
            tgt_texts.append(item["summary"])
        
        # Apply limit if specified
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
            tgt_texts = tgt_texts[:limit_samples]
        
        print(f"[INFO] Loaded {len(src_texts)} samples for language '{lang_code}'")
        return src_texts, tgt_texts
        
    except Exception as e:
        print(f"[ERROR] Failed to load MassiveSumm long from Hugging Face: {e}")
        return [], []


def load_massivesumm_short_data(lang_code, split="train", limit_samples=None, **kwargs):
    """
    Loads MassiveSumm short data from Hugging Face Hub.
    Dataset: MaLA-LM/MassiveSumm_short
    """
    from datasets import load_dataset
    
    print(f"[INFO] Loading MassiveSumm short for '{lang_code}' from Hugging Face Hub.")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("MaLA-LM/MassiveSumm_short", split="train")
        
        # Filter by language
        filtered_dataset = dataset.filter(lambda x: x["language"] == lang_code)
        
        # Extract texts and summaries
        src_texts = []
        tgt_texts = []
        
        for item in filtered_dataset:
            src_texts.append(item["text"])
            tgt_texts.append(item["summary"])
        
        # Apply limit if specified
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
            tgt_texts = tgt_texts[:limit_samples]
        
        print(f"[INFO] Loaded {len(src_texts)} samples for language '{lang_code}'")
        return src_texts, tgt_texts
        
    except Exception as e:
        print(f"[ERROR] Failed to load MassiveSumm short from Hugging Face: {e}")
        return [], []

def load_taxi1500_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads Taxi1500 data from local disk.
    """
    print(f"[INFO] Loading Taxi1500 for '{lang_code}' from local disk.")
    base_path = "benchmark_dataset/Taxi1500"
    file_path = os.path.join(base_path, lang_code, f"{lang_code}_{split}.tsv")
    
    try:
        df = pd.read_csv(
            file_path,
            delimiter="\t",
            names=["index_id", "category", "text"],
            index_col="index_id",
            on_bad_lines="skip",
            engine="python",
        )
        data = df.to_dict("records")
        if limit_samples is not None:
            data = data[:limit_samples]
        return data
    except FileNotFoundError:
        print(f"[ERROR] Local file not found for Taxi1500: {file_path}")
        return []

# Global cache for Aya dataset
_aya_cache = None
def load_aya_data(lang_code, limit_samples=None, **kwargs):
    """
    Loads Aya data from both subsets based on language_script format.
    
    Args:
        lang_code (str): Language code in format 'language_script' (e.g., 'eng_Latn', 'arb_Arab')
        limit_samples (int, optional): Maximum number of samples to load.
        
    Returns:
        list: A list of input texts from both subsets combined.
    """
    global _aya_cache
    
    # Parse language and script from input
    try:
        language, script = lang_code.split('_')
    except ValueError:
        raise ValueError(f"Invalid language code format: '{lang_code}'. Expected format: 'language_script' (e.g., 'eng_Latn')")
    
    print(f"[INFO] Loading Aya for language='{language}', script='{script}'")
    
    try:
        # Load and cache both subsets on first call
        if _aya_cache is None:
            print("[INFO] Loading and preprocessing Aya dataset (first time)...")
            _aya_cache = {}
            
            # Load both subsets
            subsets = ['aya_human_annotated', 'dolly_machine_translated']
            
            for subset in subsets:
                print(f"[INFO] Loading subset: {subset}")
                dataset = load_dataset("CohereLabs/aya_evaluation_suite", subset, split="test")
                
                # Group by language_script combination
                for item in dataset:
                    key = f"{item['language']}_{item['script']}"
                    if key not in _aya_cache:
                        _aya_cache[key] = []
                    _aya_cache[key].append(item['inputs'])
            
            print(f"[INFO] Cached data for {len(_aya_cache)} language-script combinations")
        
        # Check if the requested combination exists
        if lang_code not in _aya_cache:
            raise ValueError(f"No data found for language-script combination '{lang_code}'")
        
        src_texts = _aya_cache[lang_code]
        
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
        
        print(f"[INFO] Loaded {len(src_texts)} samples for '{lang_code}'")
        return src_texts
        
    except Exception as e:
        print(f"[ERROR] Failed to load Aya for '{lang_code}': {e}")
        raise ValueError(f"Failed to load Aya dataset for '{lang_code}': {e}")

# def load_polywrite_data_hf(lang_code, limit_samples=None, **kwargs):
#     """
#     Loads PolyWrite data from Hugging Face.
#     This benchmark is loaded from 'MaLA-LM/PolyWrite'.
    
#     Args:
#         lang_code (str): Language script code (e.g., 'aar_Latn', 'eng_Latn').
#         limit_samples (int, optional): Maximum number of samples to load.
        
#     Returns:
#         list: A list of translated prompts.
#     """
#     print(f"[INFO] Loading PolyWrite for '{lang_code}' from Hugging Face.")
    
#     try:
#         # Load the dataset - all languages are in a single dataset
#         dataset = load_dataset("MaLA-LM/PolyWrite", split="train", streaming=True)
        
#         # Filter for the specific language and collect prompts
#         src_texts = []
#         for item in dataset:
#             if item["lang_script"] == lang_code:
#                 # Use the translated prompt
#                 if item.get("prompt_translated"):
#                     src_texts.append(item["prompt_translated"])
                
#                 # Stop if we've collected enough samples
#                 if limit_samples is not None and len(src_texts) >= limit_samples:
#                     break
        
#         if not src_texts:
#             print(f"[WARN] No data found for language '{lang_code}' in PolyWrite")
#             raise FileNotFoundError(f"No data found for language '{lang_code}'")
            
#         return src_texts
        
#     except Exception as e:
#         error_msg = str(e).lower()
#         print(f"[ERROR] Failed to load PolyWrite for '{lang_code}': {e}")
#         if 'not found' in error_msg or 'does not exist' in error_msg:
#             raise FileNotFoundError(f"Dataset not found for {lang_code}")
#         else:
#             raise ValueError(f"Failed to load data for {lang_code}: {e}")


def load_polywrite_data(lang_code,limit_samples=None):
    base_path = "benchmark_dataset/PolyWrite"
    file_path = os.path.join(base_path, lang_code + ".jsonl")
    src_texts = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            src_texts.append(dat["prompt_translated"])
    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
    return src_texts


def build_options_str(a, b, c, d):
    return f"A) {a}\nB) {b}\nC) {c}\nD) {d}"


def load_mmmlu_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads MMLU data from Hugging Face.
    This function loads the English MMLU from 'cais/mmlu'.
    The 'lang_code' parameter is ignored as this dataset is in English.
    For multilingual versions, use 'load_global_mmlu_data'.
    
    Args:
        lang_code (str): Language code (ignored).
        split (str): Data split ('validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.
        
    Returns:
        list: A list of data samples.
    """
    print(f"[INFO] Loading MMLU (English) from Hugging Face for all subjects.")
    
    try:
        # MMLU on HF is best loaded by subject, but we can load 'all' and filter.
        # 'cais/mmlu' is the standard implementation.
        dataset = load_dataset("cais/mmlu", "all", split=split)
        
        data = []
        for item in dataset:
            example = {
                "question": item["question"],
                "option_a": item["choices"][0],
                "option_b": item["choices"][1],
                "option_c": item["choices"][2],
                "option_d": item["choices"][3],
                "answer": ["A", "B", "C", "D"][item["answer"]]
            }
            data.append(example)
        
        if limit_samples is not None and len(data) > limit_samples:
            # Note: Sampling might be better than truncating for MMLU
            data = data[:limit_samples]
            
        return data
        
    except Exception as e:
        print(f"[ERROR] Failed to load MMLU from Hugging Face: {e}")
        return []


def load_global_mmlu_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads Global MMLU data from Hugging Face.
    This benchmark is loaded from 'CohereLabs/Global-MMLU'.
    
    Args:
        lang_code (str): Language code (e.g., 'ar', 'bn').
        split (str): Data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.
        
    Returns:
        list: A list of data samples.
    """
    print(f"[INFO] Loading Global MMLU for '{lang_code}' from Hugging Face.")
    
    try:
        # Global MMLU uses the language code as the configuration.
        dataset = load_dataset("CohereLabs/Global-MMLU", lang_code, split=split)
        
        # Directly convert to list of dicts, columns match our format.
        data = dataset.to_dict()
        
        # The dataset is already in the desired list-of-dicts format.
        # But we need to re-format it to be safe.
        formatted_data = []
        for i in range(len(data['question'])):
            formatted_data.append({k: data[k][i] for k in data.keys()})

        if limit_samples is not None and len(formatted_data) > limit_samples:
            formatted_data = formatted_data[:limit_samples]
            
        return formatted_data
        
    except Exception as e:
        print(f"[ERROR] Failed to load Global MMLU for '{lang_code}' from Hugging Face: {e}")
        return []


def load_wikiann_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads WikiANN (PAN-X) data from Hugging Face.
    This benchmark is loaded from 'unimelb-nlp/wikiann'.
    
    Args:
        lang_code (str): Two-letter language code (e.g., 'en', 'de').
        split (str): Data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.
        
    Returns:
        list: A list of data samples.
    """
    print(f"[INFO] Loading WikiANN for '{lang_code}' from Hugging Face.")
    
    if split == 'dev':
        split = 'validation'  
    try:
        # WikiANN uses two-letter language codes as configurations.
        dataset = load_dataset("wikiann", lang_code, split=split)
        
        # Convert to a list of dicts.
        data = [row for row in dataset]
        
        if limit_samples is not None and len(data) > limit_samples:
            data = data[:limit_samples]
            
        return data
        
    except Exception as e:
        print(f"[ERROR] Failed to load WikiANN for '{lang_code}' from Hugging Face: {e}")
        print(f"[INFO] Make sure '{lang_code}' is a valid two-letter configuration for wikiann (e.g., 'en', 'de', 'zh').")
        return []


def parse_conllu_file(file_path):
    """Helper to parse a CoNLL-U file for Universal Dependencies."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        tokens, upos_tags = [], []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if tokens:
                    data.append({"tokens": tokens, "upos_tags": upos_tags})
                    tokens, upos_tags = [], []
                continue

            cols = line.split("\t")
            if len(cols) < 4: continue
            tokens.append(cols[1]) # form
            upos_tags.append(cols[3]) # upos
        if tokens:
            data.append({"tokens": tokens, "upos_tags": upos_tags})
    return data


def load_ud_data(treebank_code, split="test", limit_samples=None, **kwargs):
    """
    Loads Universal Dependencies data from local disk.
    UD datasets are distributed as individual treebanks and not available as a single unified HuggingFace dataset.
    """
    print(f"[INFO] Loading Universal Dependencies for '{treebank_code}' from local disk.")
    base_path = "benchmark_dataset/ud-treebanks-v2.15"
    folder_name = f"UD_{treebank_code}"
    folder_path = os.path.join(base_path, folder_name)
    conllu_file = None
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Treebank folder not found: {folder_path}")
        return []

    for f in os.listdir(folder_path):
        if f.endswith(f"ud-{split}.conllu"):
            conllu_file = os.path.join(folder_path, f)
            break

    if not conllu_file or not os.path.exists(conllu_file):
        print(f"[ERROR] No .conllu file found for {treebank_code} / {split}.")
        return []

    data = parse_conllu_file(conllu_file)
    if limit_samples is not None and len(data) > limit_samples:
        data = data[:limit_samples]
    return data


def load_mala_data(lang_code, split="validation"):
    """
    Evaluate perplexity (NLL). Only 'validation' or 'test' maybe.
    """
    base_path = "benchmark_dataset/mala"
    file_path = os.path.join(base_path, lang_code, f"{split}.jsonl")
    texts = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            texts.append(dat["text"])
    return texts


# def load_mala_data_hf(lang_code, split="validation", limit_samples=None, **kwargs):
#     """
#     Loads MaLA data for perplexity evaluation from Hugging Face.
#     """
#     print(f"[INFO] Loading MaLA for '{lang_code}' from Hugging Face.")
    
#     try:
#         dataset = load_dataset(
#             "MaLA-LM/mala-monolingual-split",
#             split="validation",
#         )
        
#         filtered_dataset = dataset.filter(lambda x: x["original_code"] == lang_code)
        
#         texts = filtered_dataset["text"]
        
#         if len(texts) == 0:
#             raise ValueError(f"No data found for language code: {lang_code}")
        
#         if limit_samples is not None:
#             texts = texts[:limit_samples]
        
#         print(f"[INFO] Loaded {len(texts)} samples for '{lang_code}'")
#         return texts
        
#     except Exception as e:
#         print(f"[ERROR] Failed to load MaLA data for '{lang_code}': {str(e)}")
#         raise


def load_pbc_data(lang_code, split="test", limit_samples=None, **kwargs):
    """
    Loads PBC data from local disk cache of a Hugging Face dataset.
    """
    print(f"[INFO] Loading PBC for '{lang_code}' from local disk.")
    base_path = "benchmark_dataset/pbc"
    file_path = os.path.join(base_path, lang_code)
    
    try:
        dataset = load_from_disk(file_path)
        if split not in dataset:
            print(f"[ERROR] Split '{split}' not in dataset: keys={list(dataset.keys())}")
            return []
        
        src_texts = dataset[split]['text']
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]

        return src_texts
    except Exception as e:
        print(f"[ERROR] Failed to load PBC from local disk: {e}")
        return []

def load_mmhb_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads MMHB data for bias evaluation from local disk.
    For MMHB, we return the source texts and the full dataframe 
    since it needs special handling.
    """
    print(f"[INFO] Loading MMHB for '{tgt_lang}' from local disk.")
    if split == 'test':
        split = 'devtest'  
    base_path = f"benchmark_dataset/mmhb/{tgt_lang}/{split}.csv"
    
    try:
        df = pd.read_csv(base_path, sep='\t', encoding='utf-8')
        selected_cols = ['sentence_eng', 'both', 'feminine', 'masculine', 'lang']
        df = df[selected_cols]
        if limit_samples is not None:
            df = df.head(limit_samples) 
        src_texts = df["sentence_eng"].tolist()  # Fixed: use actual column name
        return src_texts, df  # Return DataFrame for special MMHB handling
    except FileNotFoundError:
        print(f"[ERROR] Local file not found for MMHB: {base_path}")
        return [], None


def load_americasnlp_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads AmericasNLP data from local disk.
    """
    print(f"[INFO] Loading AmericasNLP for '{src_lang}-{tgt_lang}' from local disk.")
    base_path = "benchmark_dataset/americasnlp"
    src_code = src_lang.replace("_Latn", "")
    tgt_code = tgt_lang.replace("_Latn", "")
    # Logic assumes Spanish ('spa') is always one of the languages in the pair
    pair_code = f"{src_code}-spa" if tgt_code == "spa" else f"{tgt_code}-spa"
    
    if split == 'dev' or split == 'train':
        print(f"[INFO] AmericasNLP local setup only provides 'test' split (from original 'dev'). Requested '{split}' is not available.")
        return [], []
    if split == 'test':
        split = 'dev' # The local test set is the original dev set.
        
    src_file = os.path.join(base_path, f"{split}.{pair_code}.{src_code}")
    tgt_file = os.path.join(base_path, f"{split}.{pair_code}.{tgt_code}")

    try:
        with open(src_file, "r", encoding="utf-8") as f: src_texts = f.read().splitlines()
        with open(tgt_file, "r", encoding="utf-8") as f: tgt_texts = f.read().splitlines()
    except FileNotFoundError as e:
        print(f"[ERROR] Local file not found for AmericasNLP: {e}")
        return [], []

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_in22_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/in22"
    src_file = os.path.join(base_path, f"{split}.{src_lang}")
    tgt_file = os.path.join(base_path, f"{split}.{tgt_lang}")

    with open(src_file, "r", encoding="utf-8") as f:
        src_texts = f.read().splitlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_texts = f.read().splitlines()

    if len(src_texts) != len(tgt_texts):
        raise ValueError("Source and target files have different numbers of lines.")

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_ntrex128_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads NTREX-128 data from local disk.
    """
    print(f"[INFO] Loading NTREX-128 for '{src_lang}-{tgt_lang}' from local disk.")
    base_path = "benchmark_dataset/ntrex128"
    src_file = os.path.join(base_path, f"{split}.{src_lang}")
    tgt_file = os.path.join(base_path, f"{split}.{tgt_lang}")

    try:
        with open(src_file, "r", encoding="utf-8") as f: src_texts = f.read().splitlines()
        with open(tgt_file, "r", encoding="utf-8") as f: tgt_texts = f.read().splitlines()
    except FileNotFoundError as e:
        print(f"[ERROR] Local file not found for NTREX-128: {e}")
        return [], []

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]
        
    return src_texts, tgt_texts


def load_tatoeba_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads Tatoeba data from local disk.
    """
    print(f"[INFO] Loading Tatoeba for '{src_lang}-{tgt_lang}' from local disk.")
    base_path = f"benchmark_dataset/tatoeba/{split}"
    # Files are named with sorted language codes.
    sorted_langs = sorted([src_lang, tgt_lang])
    pair = "-".join(sorted_langs)
    file_path = os.path.join(base_path, f"tatoeba-{split}-v2023-09-26.{pair}.txt")
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['src', 'tgt'])
        lang_to_text = {
            sorted_langs[0]: df['src'].tolist(),
            sorted_langs[1]: df['tgt'].tolist()
        }
        src_texts = lang_to_text[src_lang]
        tgt_texts = lang_to_text[tgt_lang]
    except FileNotFoundError:
        print(f"[ERROR] Local file not found for Tatoeba: {file_path}. ")
        raise FileNotFoundError(f"Tatoeba file not found for language pair '{pair}'.")

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_nteu_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads NTEU data from local disk. This dataset is not on Hugging Face.
    """
    print(f"[INFO] Loading NTEU for '{src_lang}-{tgt_lang}' from local disk.")
    base_path = "benchmark_dataset/nteu"
    src_file = os.path.join(base_path, f"{split}.{src_lang}")
    tgt_file = os.path.join(base_path, f"{split}.{tgt_lang}")

    try:
        with open(src_file, "r", encoding="utf-8") as f: src_texts = f.read().splitlines()
        with open(tgt_file, "r", encoding="utf-8") as f: tgt_texts = f.read().splitlines()
    except FileNotFoundError as e:
        print(f"[ERROR] Local file not found for NTEU: {e}")
        return [], []

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_tico19_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads TICO-19 data from local disk.
    """
    print(f"[INFO] Loading TICO-19 for '{src_lang}-{tgt_lang}' from local disk.")
    base_path = "benchmark_dataset/tico19"
    src_file = os.path.join(base_path, split, f"{split}.{src_lang}")
    tgt_file = os.path.join(base_path, split, f"{split}.{tgt_lang}")

    try:
        with open(src_file, "r", encoding="utf-8") as f: src_texts = f.read().splitlines()
        with open(tgt_file, "r", encoding="utf-8") as f: tgt_texts = f.read().splitlines()
    except FileNotFoundError as e:
        print(f"[ERROR] Local file not found for TICO-19: {e}")
        return [], []

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_mafand_data(src_lang, tgt_lang, split="test", limit_samples=None, **kwargs):
    """
    Loads MAFAND data from Hugging Face.
    This benchmark is loaded from 'masakhane/mafand'.
    
    Args:
        src_lang (str): Source language code (e.g., 'en', 'fr', 'yor').
        tgt_lang (str): Target language code.
        split (str): Data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.
        
    Returns:
        tuple: (src_texts, tgt_texts) - Lists of source and target texts.
    """
    print(f"[INFO] Loading MAFAND for '{src_lang}-{tgt_lang}' from Hugging Face.")
    
    # MAFAND pairs are named with 'en' or 'fr' first.
    if src_lang in ["en", "fr"]:
        pair_name = f"{src_lang}-{tgt_lang}"
    else:
        pair_name = f"{tgt_lang}-{src_lang}"

    try:
        # MAFAND uses language pair as the configuration.
        dataset = load_dataset("masakhane/mafand", pair_name, split=split)
        
        translations = dataset['translation']
        src_texts = [t[src_lang] for t in translations]
        tgt_texts = [t[tgt_lang] for t in translations]
        
        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
            tgt_texts = tgt_texts[:limit_samples]
            
        return src_texts, tgt_texts
        
    except Exception as e:
        print(f"[ERROR] Failed to load MAFAND for '{src_lang}-{tgt_lang}' from Hugging Face: {e}")
        return [], []

def load_opensubtitles_data(src_lang, tgt_lang, split="test", limit_samples=None):
    """
    Loads a language pair from the Helsinki-NLP/OpenSubtitles2024 dataset using Hugging Face.

    This function handles the order of language pairs automatically (e.g., trying both
    'ar-de' and 'de-ar' if one fails) and extracts the text columns.

    Args:
        src_lang (str): The source language code (e.g., 'en', 'pt-BR').
        tgt_lang (str): The target language code (e.g., 'de', 'ar').
        split (str, optional): The dataset split to load. Defaults to "test".
        limit_samples (int, optional): If specified, limits the number of returned samples. Defaults to None.

    Raises:
        ValueError: If the specified language pair cannot be found in the dataset in either order.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists: (source_texts, target_texts).
    """
    print(f"[INFO] Loading OpenSubtitles2024 for '{src_lang}-{tgt_lang}' from Huggingface.")

    repo_id = "Helsinki-NLP/OpenSubtitles2024"
    sorted_langs = sorted([src_lang, tgt_lang])
    pair = "-".join(sorted_langs)

    # Try the first language order (src-tgt)
    data_file_path = f"{split}/{pair}/{pair}.parquet"
    # Files are named with sorted language codes.

    try:
        dataset = load_dataset(repo_id, data_files={'data': data_file_path})
        lang_to_text = {
            sorted_langs[0]: dataset['data']['src_text'],
            sorted_langs[1]: dataset['data']['tgt_text']
        }

        src_texts = lang_to_text[src_lang]
        tgt_texts = lang_to_text[tgt_lang]

    except Exception as e:  
        error_msg = str(e)

        if "Couldn't find cache" in error_msg:
            print(f"[INFO] Cache mismatch for {pair}, skipping this pair")
            raise ValueError(f"Cache configuration mismatch for {pair}")
        
        print(f"[ERROR] Failed to load {pair}: {error_msg}")
        raise ValueError(f"Failed to load language pair {pair}")

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_benchmax_rule_based_data(lang_code, split="train", limit_samples=None):
    print(f"[INFO] Loading new_benchmark for '{lang_code}'")
    try:
        dataset = load_dataset("LLaMAX/BenchMAX_Rule-based", lang_code, split=split).to_list()

        if limit_samples is not None:
            dataset = dataset[:limit_samples]

        return dataset

    except Exception as e:
        print(f"[ERROR] Failed to load new_benchmark for '{lang_code}' from Hugging Face: {e}")
        return []


def load_benchmax_math_data(lang_code, split="test", limit_samples=None):
    print(f"[INFO] Loading new_benchmark for '{lang_code}'")
    try:
        dataset = load_dataset("LLaMAX/BenchMAX_Math", lang_code, split=split).to_list()

        if limit_samples is not None:
            dataset = dataset[:limit_samples]

        return dataset

    except Exception as e:
        print(f"[ERROR] Failed to load new_benchmark for '{lang_code}' from Hugging Face: {e}")
        return []
