# python prompt_multilingual_automatic_builder.py \
#   --task_or_benchmark "translation" \
#   --lang_code "eng_Latn" \
#   --new_few_shot "[{src_lang}]: {src_text}\n[{tgt_lang}]: {tgt_text}\n" \
#   --input_file "prompt_library.json" \
#   --copy_all

python multilingual_prompt_builder.py \
  --config_file "../../config.json" \
  --task_or_benchmark "translation" \
  --lang_code "eng_Latn" \
  --target_lang_codes zho_Hans fra_Latn \
  --new_instruction $'Translate the following sentence from {src_lang} to {tgt_lang}\n[{src_lang}]: {src_text}\n[{tgt_lang}]:' \
  --input_file "prompt_library.json" \
  --ms_translator_key "your-key" \
  --ms_translator_region "northeurope"