# GlotEval: Massively Multilingual Evaluation of Large Language Models 

GlotEval is a unified evaluation toolkit designed to benchmark Large Language Models (LLMs) across **multiple languages and tasks**. It supports text classification, machine translation, summarization, token classification, and open-ended generation, with a focus on **massively multilingual** coverage, evaluating models across over 1500 languages.

[GitHub Repository](https://github.com/MaLA-LM/GlotEval)

## Key Features

- **Consistent Multilingual Benchmarking**
  - Standardized ISO 639-3 language codes alignment across benchmarks
  - Language-specific evaluation that works across language groups (e.g., Bantu, Dravidian, Uralic)
  - Automatic language mapping to facilitate incorporation of new large-scale benchmarks

- **Language-Specific Prompt Templates**
  - Configure prompts for each language individually
  - Centralized prompt library supporting multilingual benchmarks
  - Microsoft Translator integration for automatic prompt template translation to 130+ languages

- **Non-English-Centered Machine Translation**
  - Break away from English-centric translation evaluation
  - Support for any language as the pivot in translation tasks
  - Flexible "any-to-pivot" and "pivot-to-any" translation directions

- **Multilingual Tasks**
  - **Text Classification**: SIB-200, Taxi-1500
  - **Machine Translation**: Flores-200, Flores+, AmericasNLP, IN22, NTREX-128, Tatoeba, NTEU, TICO-19, MAFAND, MMHB, OpenSubtitles
  - **Summarization**: XLSum
  - **Token Classification**: WikiANN, UD
  - **Comprehension**: MMLU-style tasks (MMMLU, Global-MMLU)
  - **Open-ended Generation**: Aya, PolyWrite
  - **Intrinsic Evaluation**: PBC, MaLA

- **Model Compatibility**
  - **Hugging Face (HF)**: Evaluate any model from the Transformers ecosystem, used for non-generation tasks
  - **vLLM**: Efficient, large-batch generation, used for generation tasks

- **Rich Metrics**
  - **Machine Translation**: BLEU, ChrF++, COMET, etc.
  - **Summarization**: ROUGE-L, etc.
  - **Classification**: Accuracy, F1, etc.
  - **Token Classification**: F1, token-level accuracy, etc.
  - **Open-ended**: Self-BLEU, other custom metrics

## Requirements

- Python 3.8+
- PyTorch (for HF)
- Additional libraries: `transformers`, `vllm`, `pandas`, `sacrebleu`, etc.
- Tasks require specific data files (.conllu, .tsv, .jsonl, etc.) placed in respective benchmark directories

## Quickstart

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MaLA-LM/GlotEval
   cd GlotEval
   ```

2. **Create/Activate Environment**
    
    ```bash
    conda create -n gloteval python=3.9
    conda activate gloteval
    pip install -r requirements.txt
    ```
    
3. **Obtain or Prepare Data**
    - Place relevant data under their respective benchmark directories (e.g., `benchmark_dataset/flores200`, `benchmark_dataset/wikiann`, etc.)
    - Update file paths in `config.json` if needed to point to your local data

4. **Run an Evaluation**
    
    ```bash
    python main.py \
        --model_name "Qwen/Qwen2-1.5B" \
        --benchmarks xlsum sib200 \
        --params config.json \
        --output_dir results \
        --langs zho gsw por fra fin \
        --store_details \
        --efficiency_analysis
    ```

    **Command Line Arguments Explained:**
    - `--model_name`: Hugging Face model name or your local model path
    - `--benchmarks`: Space-separated list of tasks or benchmarks to run (e.g., `xlsum sib200`)
    - `--params`: Path to the config file specifying prompts, shots, etc.
    - `--output_dir`: Directory to store results
    - `--langs`: Filter languages to evaluate (uses ISO 639-3 codes, can include macrolanguages)
    - `--store_details`: Save detailed output for each sample in CSV format (useful for error analysis)
    - `--efficiency_analysis`: Track and report token generation efficiency metrics

5. **Check Results**
    - A `scores.json` file will be created under `results/<model_name>/<timestamp>/scores.json`
    - If `--store_details` was specified, CSV files for each benchmark/language are also created in the same folder
    - Performance metrics are automatically calculated according to the task type

## Configuration & Customization

The central configuration is in `config.json`, which specifies:

### Model Arguments
```json
"model_args": {
  "device": "cuda",
  "tensor_parallel_size": 1,
  "batch_size": 1,
  "dtype": "auto",
  "max_num_seqs": 256,
  "sampling_params": {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 128,
    "stop": "\n"
  }
}
```

### Prompt Strategy
```json
"prompt_language_strategy": "single",
"prompt_language": "eng_Latn",
```
- `"single"`: Use the same prompt in one language for all datasets
- `"multi"`: Use language-specific prompts when available

### Benchmark-Specific Parameters
```json
"benchmark_params": {
  "flores200_mt": {
    "n_shots": 3,
    "seed": 42,
    "center_lang": "eng_Latn",
    "direction": "center-x"
  },
  "xlsum": {
    "n_shots": 0,
    "seed": 42
  }
}
```

### Task-Specific Prompt Guidelines
```json
"prompt_guidelines": {
  "translation": {
    "required_placeholders": ["{src_text}", "{tgt_lang}"],
    "optional_placeholders": ["{src_lang}"],
    "description": "For translation tasks, the instruction template must include {src_text} and {tgt_lang}."
  }
}
```

## Utility Tools

GlotEval includes two important utility tools that enhance its multilingual capabilities:

### 1. Language ID Alignment

The language alignment tool standardizes language codes from various benchmarks to the ISO 639-3 format with script information (e.g., `eng_Latn`, `zho_Hans`). This enables seamless cross-benchmark language-specific evaluation.

[Read more about Language ID Alignment](/benchmark_data_loader/data_langid_alignment/README.md)

Features:
- Processes inconsistent language codes from benchmarks (e.g., zh, zho, cmn, Chinese)
- Maps to standardized ISO 639-3 codes with script information
- Automatically detects scripts using GlotScript
- Handles special cases like CJK scripts with precise identification

### 2. Multilingual Prompt Builder

This tool helps create and manage prompts in multiple languages for all evaluation tasks.

[Read more about the Multilingual Prompt Builder](/benchmark_data_loader/multilingual_prompt_builder/README.md)

Features:
- Translates prompts from a source language to 130+ target languages
- Preserves placeholders during translation
- Supports various prompt formats for different tasks
- Creates a comprehensive prompt library for consistent multilingual evaluation

## Expected Output

After running an evaluation, GlotEval produces:

1. **scores.json**: Contains the overall performance metrics for each benchmark and language
   ```json
   {
     "xlsum": {
       "zho_Hans": {
         "rouge_l_f1": 0.342
       },
       "fra_Latn": {
         "rouge_l_f1": 0.387
       }
     },
     "sib200": {
       "zho_Hans": {
         "accuracy": 0.78
       }
     }
   }
   ```

2. **Detailed CSV files** (if `--store_details` is specified):
   - Contains each sample's prompt, model output, reference, and corresponding scores
   - Useful for fine-grained error analysis and qualitative evaluation

3. **Performance metrics**:
   - If `--efficiency_analysis` is specified, you'll get statistics on token generation efficiency
   - Metrics include tokens per second, prefill and decode times, etc.

## Contributing

Please refer to the GlotEval GitHub repository for contribution guidelines.

## License

See the LICENSE file in the repository.