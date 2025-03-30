# Multilingual Prompt Builder

A set of tools for creating and managing prompts in multiple languages for language evaluation benchmarks.

## Overview

This repository contains two main tools:

1. **Multilingual Prompt Builder** - Creates and translates prompts for language tasks across multiple languages
2. **Language Matrix Builder** - Builds a translation matrix of language names for use in translation prompts

## Requirements

- Python 3.6+
- Required Python packages:
  - requests
  - tqdm
  - pandas (optional, for Excel output)
  - openpyxl (optional, for Excel output)

## Multilingual Prompt Builder

The Multilingual Prompt Builder allows you to create prompts for language tasks in multiple languages, either by translating from a source language or by copying the same prompt across languages.

### Usage

```bash
python multilingual_prompt_builder.py [arguments]
```

### Modes

The tool operates in two main modes:

#### Translation Mode

In this mode, the tool translates prompts from a source language to multiple target languages using the Microsoft Translator API.

```bash
python multilingual_prompt_builder.py \
  --task_or_benchmark "translation" \
  --lang_code "eng_Latn" \
  --target_lang_codes zho_Hans fra_Latn spa_Latn \
  --new_instruction "Translate the following sentence from {src_lang} to {tgt_lang}\n[{src_lang}]: {src_text}\n[{tgt_lang}]:" \
  --input_file "prompt_library.json" \
  --ms_translator_key "your_key_here" \
  --ms_translator_region "northeurope"
```

If `--target_lang_codes` is left empty, it will translate to all supported languages:

```bash
python multilingual_prompt_builder.py \
  --task_or_benchmark "translation" \
  --lang_code "eng_Latn" \
  --target_lang_codes \
  --new_instruction "Translate the following sentence from {src_lang} to {tgt_lang}\n[{src_lang}]: {src_text}\n[{tgt_lang}]:" \
  --input_file "prompt_library.json" \
  --ms_translator_key "your_key_here" \
  --ms_translator_region "northeurope"
```

#### Copy Mode

In this mode, the tool copies the same prompt across multiple languages without translation.

```bash
python multilingual_prompt_builder.py \
  --task_or_benchmark "translation" \
  --lang_code "eng_Latn" \
  --new_few_shot "[{src_lang}]: {src_text}\n[{tgt_lang}]: {tgt_text}\n" \
  --input_file "prompt_library.json" \
  --copy_all
```

### Placeholder Protection

When translating prompts, the tool includes a built-in placeholder protection mechanism. Prompts often contain placeholders like `{src_lang}`, `{tgt_lang}`, `{text}`, etc., which need to be preserved during translation. The tool:

1. Extracts all placeholders from the original templates before translation
2. Verifies that all placeholders still exist in the translated version
3. Reports any missing placeholders at the end of the process

Example of a missing placeholder report:
```
WARNING: Missing placeholders detected in some translations:
  - fra_Latn instruction: Missing src_lang
  - zho_Hans few-shot: Missing tgt_text
Please review and fix these manually in the output file.
```

If placeholders are missing or modified during translation, they may need manual inspection and correction. This helps ensure that the translated prompts maintain their functionality across languages.

### Input/Output Format

- **Input**: The tool can read an existing prompt library from a JSON file specified by `--input_file`.
- **Output**: The tool outputs a JSON file with the updated prompt library. If `--output_file` is not specified, it will either update the input file or create a new file with a timestamp.

The JSON structure is organized as follows:

```json
{
  "task_name": {
    "language_code": {
      "instruction": "Task instruction template with {placeholders}",
      "few_shot": "Few-shot examples template with {placeholders}"
    },
    ...
  },
  ...
}
```

### Key Arguments

- `--task_or_benchmark`: The task or benchmark name (e.g., "translation", "summarization")
- `--lang_code`: The source language code in ISO 639-3_Script format (e.g., "eng_Latn")
- `--new_instruction`: The instruction template (optional)
- `--new_few_shot`: The few-shot examples template (optional)
- `--target_lang_codes`: List of target language codes for translation (if empty, translates to all languages)
- `--input_file`: Path to the existing prompt library file (optional)
- `--output_file`: Path to write the updated library (optional)
- `--ms_translator_key`: Microsoft Translator API subscription key (required for translation mode)
- `--ms_translator_region`: Microsoft Translator API region (default: "northeurope")
- `--copy_all`: Flag to activate copy mode (copies templates to all languages without translation)

## Language Matrix Builder

The Language Matrix Builder creates a matrix of language name translations. This is particularly useful for translation tasks where the source and target languages need to be referenced by name in the prompt.

### Usage

```bash
python language_matrix_builder.py
```

The script will translate language names between all supported languages using the Microsoft Translator API.

### Configuration

Edit the main section of the script to configure your API key and other parameters:

```python
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
```

To process only specific languages, uncomment and modify the following section:

```python
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
```

### Output

The results are stored in the output directory (default: "language_matrix_results") in multiple formats:

1. Individual JSON files for each language
2. A combined "language_matrix.json" file with all translations
3. A CSV file "language_matrix.csv"
4. An Excel file "language_matrix.xlsx" (if pandas and openpyxl are installed)

The matrix format allows you to look up how to say any language name in any other language.

### Adaptive Rate Limiting

Both tools use adaptive rate limiting to optimize API usage and handle rate limits gracefully. The rate limiter automatically adjusts request rates based on API responses, backing off when needed and speeding up when possible.

## Utility Modules

The tools use two utility modules:

1. `language_utils.py` - Contains language code mappings and conversion functions
2. `translation_utils.py` - Contains translation functions and rate limiting logic

## Example Workflow

1. Use the Language Matrix Builder to create a matrix of language name translations
2. Use the Multilingual Prompt Builder to create prompts in multiple languages
3. Use the resulting prompt library in your language benchmarks