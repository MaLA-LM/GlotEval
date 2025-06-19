We highly love your contributions! 

# **Did you find a bug?**

- **Ensure the bug was not already reported** by searching on GitHub under `Issues`.
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/MaLA-LM/GlotEval/issues). Be sure to include **a title and clear description**, as much relevant information as possible.
- Sometimes, an issue might take a long to solve. You can also join our [Discord server](https://discord.com/invite/F5mEb7U6we) for discussion.

# **Add a New Benchmark to the GlotEval Toolkit**

This guide outlines the process for integrating a new benchmark into the GlotEval framework. It covers data loading, language configuration, prompt creation, handler implementation, and testing.

---

## 1. Implement the Data Loading Function

The first step is to implement a function that loads your benchmark's data.

**Recommendation:** The preferred method is to load data directly from the **Hugging Face Hub** using the `datasets` library. If your dataset is not on the Hub, you can implement a fallback to load it from a local disk. In this case, you will need to test your loader and submit the dataset files to the GlotEval GitHub release along with your code PR.

The function should adhere to the following standards:

- **Name:** `load_<benchmark_name>_data`
- **Location:** `benchmark_data_loader/data_loader.py`
- **Signature:** It must accept at least `lang_code` and `split` arguments.

Below is a template that handles both Hugging Face loading and a local fallback.

### Template (with HF datasets loading)

```python
import os
import json
import pandas as pd
from datasets import load_dataset

def load_new_benchmark_data(lang_code: str, split: str = "test", limit_samples: int = None, **kwargs):
    """
    Loads data for Your Benchmark for the requested language and split.

    Args:
        lang_code (str): Language code.
        split (str): Data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.

    
    Returns:
        tuple: A tuple of lists (e.g., src_texts, tgt_texts) or a list of dictionaries.
    """
    print(f"[INFO] Loading new_benchmark for '{lang_code}'")

      # --- Hugging Face loading logic ---
    try:
        # TODO: Replace with the correct HF repo name and configuration
        dataset = load_dataset("your-org/your-benchmark", lang_code, split=split)
        
        # TODO: Adapt this code to extract the correct data from the dataset object
        # For example, for a translation task:
        src_texts = [item['source_sentence'] for item in dataset]
        tgt_texts = [item['target_sentence'] for item in dataset]

        if limit_samples is not None:
            src_texts = src_texts[:limit_samples]
            tgt_texts = tgt_texts[:limit_samples]
            
        return src_texts, tgt_texts
        
    except Exception as e:
        print(f"[ERROR] Failed to load new_benchmark for '{lang_code}' from Hugging Face: {e}")
        # Optional: add a note if the dataset is gated
        print("[INFO] Note: 'your-org/your-benchmark' may be a gated dataset and require authentication.")
        return [], []


    
```

### Template (with local disk loading)

```python
import os
import json
import pandas as pd
from datasets import load_dataset

def load_new_benchmark_data(lang_code: str, split: str = "test", limit_samples: int = None, **kwargs):
    """
    Loads data for Your Benchmark for the requested language and split.

    Args:
        lang_code (str): Language code.
        split (str): Data split ('train', 'validation', 'test').
        limit_samples (int, optional): Maximum number of samples to load.
    
    Returns:
        tuple: A tuple of lists (e.g., src_texts, tgt_texts) or a list of dictionaries.
    """
    print(f"[INFO] Loading new_benchmark for '{lang_code}'")

    # --- Local disk loading logic ---
    # TODO: Adjust the base path as needed
    base_path = f"benchmark_dataset/new_benchmark"
    file_path = os.path.join(base_path, lang_code, f"{split}.jsonl") # or .tsv, .csv, etc.

    try:
        # TODO: Add your file reading logic here
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        
        if limit_samples is not None:
            lines = lines[:limit_samples]
        
        return lines, None # Adjust return value as needed
        
    except FileNotFoundError:
        print(f"[ERROR] Local file not found for new_benchmark: {file_path}")
        return [], []
    
```



## 2. Declare the Language List

Create a language configuration file that maps your benchmark's internal language identifiers to GlotEval's standard language codes (ISO 639-3 language id_ISO15924 script code)

- **Location:** `data/data_config/<benchmark_name>_langs.txt`
- **Format:** Each line should contain `benchmark_internal_code | standard_code`.

### Example: `new_benchmark_langs.txt`

```python
fr | fra_Latn
en | eng_Latn
amh | amh_Ethi
bam | bam_Latn
ewe | ewe_Latn
fon | fon_Latn
hau | hau_Latn
kin | kin_Latn
# ...and so on
```

If you find trouble when creating this file, you can run the alignment tool to automatically fill in any missing standard language codes. For more details, see the tool's documentation: [`data_langid_alignment/README.md`](https://github.com/MaLA-LM/GlotEval/blob/main/benchmark_data_loader/data_langid_alignment/README.md).



## 3. Add (or Reuse) Prompt Templates

Next, define the prompts for your benchmark.

- **If your benchmark fits a standard task** (e.g., `summarization`, `translation`, `classification`, `token_classification`, `open_generation`), you can **skip this step**. GlotEval will automatically use a default prompt for that task type.
- **If your task is new or requires specific wording**, you must add a prompt template to `prompt_library.json`.

The library can match prompts based on the general **task key** (e.g., `"classification"`) or the specific **benchmark name** (e.g., `"xlsum"`).

### Example: `prompt_library.json`

```json
{
  "classification": {
    "eng_Latn": {
      "instruction": "The topic of the news \"{text}\" is",
      "few_shot": "The topic of the news \"{example_text}\" is {example_category}\n"
    },
    "zho_Hans": {
      "instruction": "以下新闻的主题是：\"{text}\"",
      "few_shot": "示例新闻：\"{example_text}\"，其主题为 {example_category}\n"
    }
  },
  "xlsum": {
    "arb_Arab": {
      "title": "وثيقة",
      "instruction": "استناداً إلى النص السابق، قم بتقديم ملخص واحد موجز"
    },
    "zho_Hans": {
      "title": "文档",
      "instruction": "根据前面的文字，提供一个简短的单一摘要"
    }
  }
}
```



## 4. Implement the Benchmark Handler

The handler is the core logic that connects your data and prompts, runs the model, and evaluates the results.

- **Location:** In the `tasks/` directory, either add your handler to an existing file (e.g., `summarization.py` if it's a summarization task) or create a new file for a new task type.
- **Implementation:** Follow the pattern of existing handlers (like `process_summarization_benchmark` in `summarization.py`). Use the `@register_benchmark("new_benchmark_name")` decorator to make it discoverable by the main script.
- **Output:** Ensure your handler saves detailed results to a TSV file and updates `scores.json` with the final metrics.



## 5. Configure and Test the Benchmark

### A. Add Parameters to `config.json`

Update the `config.json` file to include parameters for your new benchmark. This tells GlotEval where to find its configuration and how to run it.

```json
{
  "benchmark_params": {
    "new_benchmark": {
      "langs_path": "data/data_config/new_benchmark_langs.txt",
      "n_shots": 3,
      "seed": 42
    }
  }
}
```

### B. Run the Benchmark

Execute the main script to run your benchmark. You can specify the model, benchmarks, and languages to test.

Bash

```bash
python main.py \
    --model_name "YourModelName" \
    --benchmarks new_benchmark \
    --langs zho gsw por fra fin \
    --params config.json \
    --output_dir results
    --store_details
```

### C. Verify Outputs

After the run is complete, verify that the outputs have been generated correctly:

- Check that the `scores.json` file in the output directory includes an entry for your benchmark.
- Ensure that detailed TSV results are saved under `results/[model_signature]/[timestamp]/new_benchmark/`.