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
## Data loading with hardcoded paths & limit_samples
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



def load_flores_plus_data_hf(src_lang, tgt_lang, split="test", limit_samples=None):
    
    hf_split = "devtest" if split == "test" else split

    ds_src = load_dataset("openlanguagedata/flores_plus", src_lang, split=hf_split)
    ds_tgt = load_dataset("openlanguagedata/flores_plus", tgt_lang, split=hf_split)

    src_texts = ds_src["text"]
    tgt_texts = ds_tgt["text"]

    if len(src_texts) != len(tgt_texts):
        raise ValueError("Source and target texts have different lengths.")

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_flores_plus_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/flores_plus"
    if split == 'test':
        split = 'devtest'
    src_file = os.path.join(base_path, split, f"{split}.{src_lang}")
    tgt_file = os.path.join(base_path, split, f"{split}.{tgt_lang}")

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


def load_sib200_data(lang_code, split="test"):
    base_path = "benchmark_dataset/sib200"
    file_path = os.path.join(base_path, lang_code, f"{split}.tsv")
    df = pd.read_csv(file_path, delimiter="\t", index_col="index_id")
    data = df.to_dict("records")
    return data


def load_xlsum_data(lang_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/XLSum"
    src_texts, tgt_texts = [], []
    file_path = os.path.join(base_path, f"{lang_code}_{split}.jsonl")
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            src_texts.append(dat["text"])
            tgt_texts.append(dat["summary"])

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_massivesumm_long_data(lang_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/MassiveSumm_long"
    src_texts, tgt_texts = [], []
    file_path = os.path.join(base_path, f"{lang_code}_{split}.jsonl")
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            src_texts.append(dat["text"])
            tgt_texts.append(dat["summary"])

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_massivesumm_short_data(lang_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/MassiveSumm_short"
    src_texts, tgt_texts = [], []
    file_path = os.path.join(base_path, f"{lang_code}_{split}.jsonl")
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            src_texts.append(dat["text"])
            tgt_texts.append(dat["summary"])

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_taxi1500_data(lang_code, split="test"):
    base_path = "benchmark_dataset/Taxi1500"
    file_path = os.path.join(base_path, lang_code, f"{lang_code}_{split}.tsv")
    df = pd.read_csv(
        file_path,
        delimiter="\t",
        names=["index_id", "category", "text"],
        index_col="index_id",
        on_bad_lines="skip",
        engine="python",
    )
    data = df.to_dict("records")
    return data


def load_aya_data(lang_code):
    base_path = "benchmark_dataset/Aya"
    file_path = os.path.join(base_path, lang_code + ".jsonl")
    src_texts = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            src_texts.append(dat["inputs"])
    return src_texts


def load_polywrite_data(lang_code):
    base_path = "benchmark_dataset/PolyWrite"
    file_path = os.path.join(base_path, lang_code + ".jsonl")
    src_texts = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            dat = json.loads(line)
            src_texts.append(dat["prompt_translated"])
    return src_texts


def build_options_str(a, b, c, d):
    return f"A) {a}\nB) {b}\nC) {c}\nD) {d}"


def load_mmmlu_data(lang_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/MMMLU"
    file_name = f"mmlu_{lang_code}.csv"
    file_path = os.path.join(base_path, split, file_name)

    df = pd.read_csv(file_path, encoding='utf-8')
    data = []
    for _, row in df.iterrows():
        example = {
            "question": row["Question"],
            "option_a": row["A"],
            "option_b": row["B"],
            "option_c": row["C"],
            "option_d": row["D"],
            "answer":   row["Answer"]
        }
        data.append(example)

    if limit_samples is not None and len(data) > limit_samples:
        data = data[:limit_samples]

    return data


def load_global_mmlu_data(lang_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/Global_MMLU"
    file_path = os.path.join(base_path, lang_code, f"{split}-00000-of-00001.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)
    data = df.to_dict(orient="records")
    # Re-label the fields
    for row in data:
        row["question"]  = row["question"]
        row["option_a"]  = row["option_a"]
        row["option_b"]  = row["option_b"]
        row["option_c"]  = row["option_c"]
        row["option_d"]  = row["option_d"]
        row["answer"]    = row["answer"]

    if limit_samples is not None and len(data) > limit_samples:
        data = data[:limit_samples]

    return data


def load_wikiann_data(lang_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/wikiann"
    if split == 'dev':
        split = 'validation'
    file_name = f"{split}-00000-of-00001.parquet"
    file_path = os.path.join(base_path, lang_code, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)
    data = []
    for _, row in df.iterrows():
        example = {
            "tokens": row["tokens"],
            "ner_tags": row["ner_tags"],
            "langs": row.get("langs", None),
            "spans": row.get("spans", None),
        }
        data.append(example)

    if limit_samples is not None and len(data) > limit_samples:
        data = data[:limit_samples]
    return data


def parse_conllu_file(file_path):
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
            if len(cols) < 4:
                continue
            form = cols[1]
            upos = cols[3]
            tokens.append(form)
            upos_tags.append(upos)
        if tokens:
            data.append({"tokens": tokens, "upos_tags": upos_tags})
    return data

def load_ud_data(treebank_code, split="test", limit_samples=None):
    base_path = "benchmark_dataset/ud-treebanks-v2.15"
    folder_name = f"UD_{treebank_code}"
    folder_path = os.path.join(base_path, folder_name)
    conllu_file = None
    for f in os.listdir(folder_path):
        if f.endswith(f"ud-{split}.conllu"):
            conllu_file = os.path.join(folder_path, f)
            break

    if not conllu_file or not os.path.exists(conllu_file):
        print(f"[load_ud_data] No .conllu file found for {treebank_code} / {split}.")
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


def load_pbc_data(lang_code, split="test"):
    base_path = "benchmark_dataset/pbc"
    file_path = os.path.join(base_path, lang_code)
    dataset = load_from_disk(file_path)
    if split not in dataset:
        print(f"[load_pbc_data] split '{split}' not in dataset: keys={list(dataset.keys())}")
        return []
    data_split = dataset[split]
    src_texts = []
    for d in data_split:
        src_texts.append(d["text"])
    return src_texts

def load_mmhb_data(src_lang, tgt_lang, split="devtest", limit_samples=None):
    base_path = f"benchmark_dataset/mmhb/{tgt_lang}/{split}.csv"
    df = pd.read_csv(base_path ,  sep='\t', encoding='utf-8')
    selected_cols = ['sentence_eng', 'both', 'feminine', 'masculine', 'lang']
    df = df[selected_cols]
    if limit_samples is not None:
        df = df.head(limit_samples) 
    src_texts = df[f"sentence_{src_lang}"].tolist()
    return src_texts, df

def load_americasnlp_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/americasnlp"
    src_code=src_lang.replace("_Latn","")
    tgt_code=tgt_lang.replace("_Latn","")
    pair_code = f"{src_code}-spa" if tgt_code == "spa" else f"{tgt_code}-spa"
    # No few-shot support
    if split =='dev':
        return None, None
    # Dev set as test set
    if split == 'test':
        split = 'dev'
    src_file = os.path.join(base_path, f"{split}.{pair_code}.{src_code}") 
    tgt_file = os.path.join(base_path, f"{split}.{pair_code}.{tgt_code}")

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


def load_ntrex128_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/ntrex128"
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


def load_tatoeba_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/tatoeba"
    sorted_langs = sorted([src_lang, tgt_lang])
    pair="-".join(sorted_langs)

    file = os.path.join(base_path, split, f"tatoeba-{split}-v2023-09-26.{pair}.txt")
    with open(file, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()

    if len(texts) < 1000 and split=='test':
        raise ValueError(f"Not enough data for {pair}: only {len(texts)} sentences available.")

    src_idx = sorted_langs.index(src_lang)+2
    tgt_idx = sorted_langs.index(tgt_lang)+2

    src_texts = [line.split('\t')[src_idx] for line in texts]
    tgt_texts = [line.split('\t')[tgt_idx] for line in texts]

    if limit_samples is not None:
        src_texts = src_texts[:limit_samples]
        tgt_texts = tgt_texts[:limit_samples]

    return src_texts, tgt_texts


def load_nteu_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/nteu"
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


def load_tico19_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/tico19"
    src_file = os.path.join(base_path, split, f"{split}.{src_lang}")
    tgt_file = os.path.join(base_path, split, f"{split}.{tgt_lang}")

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


def load_mafand_data(src_lang, tgt_lang, split="test", limit_samples=None):
    base_path = "benchmark_dataset/mafand"
    if src_lang in ["en", "fr"]:
        src_file = os.path.join(base_path, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}")
        tgt_file = os.path.join(base_path, f"{src_lang}-{tgt_lang}", f"{split}.{tgt_lang}")
    else:
        src_file = os.path.join(base_path, f"{tgt_lang}-{src_lang}", f"{split}.{src_lang}")
        tgt_file = os.path.join(base_path, f"{tgt_lang}-{src_lang}", f"{split}.{tgt_lang}")

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