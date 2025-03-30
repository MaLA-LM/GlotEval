# GlotEval Language ID and Script Alignment Tool

This tool is designed to align language codes from various benchmarks to standardized language identifiers with ISO 639-3 codes and script information (e.g., `eng_Latn`). It processes language code files, detects scripts from sample texts, and generates aligned language code mappings.

## What it does

The alignment tool performs the following three main tasks:

### 1. Language Code Identification

The tool parses language codes from benchmark datasets (which often use inconsistent formats) and matches them to standardized ISO 639 language codes:

- Extracts language and script components from codes like "eng-US", "spa_Latn", or "zh-CN"
- Performs exact matching against ISO 639 standards (ISO 639-1, 639-2, 639-3)
- Falls back to fuzzy matching for non-standard or ambiguous codes
- Identifies macrolanguages and their relationships
- Handles deprecated language codes with appropriate mappings

### 2. Script Detection

The tool identifies the writing system (script) used in language samples:

- Uses GlotScript for initial script detection
- Loads sample text from corresponding benchmark datasets
- Determines confidence scores for script identification
- Handles special cases and ambiguities

#### Special Handling for CJK Scripts

One significant challenge is properly identifying East Asian writing systems:

- GlotScript typically outputs "Hani" for text containing Han characters (hanzi/kanji/hanja)
- However, we need more specific designations:
  - **Hans**: Simplified Chinese
  - **Hant**: Traditional Chinese
  - **Jpan**: Japanese (mixture of kanji and kana)
  - **Hani**: Generic Han characters (when specific script can't be determined)

The tool implements a custom character-based detection system that analyzes the presence of:
- Unique simplified Chinese characters
- Unique traditional Chinese characters
- Japanese-specific characters (hiragana, katakana, and unique kanji)

This enables more accurate classification of CJK writing systems beyond the capabilities of the underlying GlotScript library.

### 3. Alignment and Output Generation

Finally, the tool combines language and script information to create standardized, aligned outputs:

- Generates language mappings in the format `original | iso3_script` (e.g., `en | eng_Latn`)
- Flags uncertain matches with question marks (e.g., `ja | jpn_Jpan?`)
- Creates detailed CSV reports with confidence scores and metadata
- Handles errors and edge cases gracefully

## Installation

This tool requires several dependencies:

```bash
pip install iso639-lang
pip install GlotScript
```

Additionally, ensure you have access to the benchmark datasets referenced in `data_loader.py`.

## Usage

### Basic Usage

To process all language code files in the default directory:

```bash
python langid_alignment.py
```

### Process a Specific File

To process a single language code file:

```bash
python langid_alignment.py --file wikiann_langs.txt
```

### Additional Options

```bash
python langid_alignment.py --help
```

Options include:
- `--data_config_dir`: Directory containing language config files
- `--max_lines`: Maximum number of sample lines to load for script detection
- `--file`: Process a specific file only
- `--no-aligned`: Skip generation of aligned files
- `--verbose`: Enable verbose logging

## Adding a New Benchmark

To add support for a new benchmark, follow these steps:

1. **Create a data loader function**
   
   Add a new loader function in `data_loader.py`. The function should return sample texts from your benchmark:

   ```python
   def load_my_benchmark_data(lang_code, split="test", limit_samples=None):
       # Load data for the given language code
       # Return a list of text samples
       return text_samples
   ```

2. **Update the BENCHMARK_LOADERS dictionary**
   
   Add your new loader to the `BENCHMARK_LOADERS` dictionary in `langid_alignment.py`:

   ```python
   BENCHMARK_LOADERS = {
       # Existing loaders...
       "my_benchmark": load_my_benchmark_data,
   }
   ```

3. **Create a language code file**
   
   Create a file named `my_benchmark_langs.txt` in the `data_langid_alignment` directory.
   Add one language code per line, using the benchmark's native language code format.

4. **Run the alignment**
   
   ```bash
   python langid_alignment.py --file my_benchmark_langs.txt
   ```

5. **Check the output**
   
   - The detailed report will be in `alignment_reports/my_benchmark_langs_script_report.csv`
   - The aligned language codes will be in `data_langid_aligned/my_benchmark_langs.txt`

### Important Considerations for New Benchmarks

When adding a new benchmark, keep these points in mind:

1. **Consistent language pairing**: For translation benchmarks, ensure your loader function handles language pairing logically. Many loaders use English as a pivot language.

2. **Sample text quality**: The script detection depends heavily on the quality of text samples. Include multiple samples (at least 3) where possible for more reliable script detection.

3. **Special character handling**: Be mindful of text encoding issues, especially with scripts like Arabic, CJK languages, and Indic scripts.

4. **Benchmark-specific parsing**: Some benchmarks have unique formatting or metadata. Ensure your loader extracts clean text samples without markup or metadata.

5. **Error handling**: Implement robust error handling in your data loader. The alignment tool will catch exceptions, but proper error handling in the loader improves debugging.

6. **Language variants**: Be attentive to language variants (e.g., "eng-US" vs "eng-GB"). The tool will process these as separate entries.

7. **Script confidence**: Review the CSV reports to identify low-confidence script detections that might need manual verification.

## Output Format

### CSV Report

The CSV report contains the following columns:

- `original_label`: Original language label from the input file
- `parsed_lang_part`, `parsed_script_part`: Parsed components of the language label
- `iso639_match_status`: Status of the ISO 639 language matching
- `iso639_pt1`, `iso639_pt2b`, `iso639_pt2t`, `iso639_pt3`: Various ISO 639 codes
- `iso639_name`: Full language name
- `macro_status`, `macro_language`: Information about macrolanguages
- `detected_script`: Detected script code
- `script_confidence`: Confidence score of script detection
- `sample_text`: Sample of the text used for detection
- `data_load_error`: Any errors encountered when loading data
- `aligned_key`: The final aligned key in the format `original | iso3_script`

### Aligned Language Codes

The aligned language code file contains lines in the format:

```
original_code | iso3_script
```

For example:
```
en | eng_Latn
zh-CN | zho_Hans
ja | jpn_Jpan
```

Lines with uncertain matches are prefixed with `# UNCERTAIN:`.

This format creates a standardized mapping that can be used consistently across benchmarks, making it easier to integrate multilingual evaluation regardless of the original language code formats.

## Troubleshooting

If uncertain matches are found, they will be logged in `alignment_reports/uncertain_matches.csv`. You may need to manually verify and correct these mappings.

Common issues include:
- Missing language codes in ISO 639 standards
- Insufficient sample texts for script detection
- Ambiguous script detection for mixed-script texts