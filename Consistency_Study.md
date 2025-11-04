# Consistency Study

## 1. Motivation
GlotEval targets massively multilingual MT evaluation, so we need to show it produces trustworthy scores that agree with widely used toolkits. The goal of this study is to audit GlotEval against the LM Evaluation Harness (LM-Eval) on the FLORES-200 benchmark and verify that both frameworks yield comparable ChrF++ scores under matched settings. A collaborating teammate @[Tianxiang Wang](https://github.com/congc100) independently ran the LM-Eval pipeline, providing reference outputs used in this comparison.

## 2. Experimental Setup
- **Experiment grid**: 6 Llama-family checkpoints (MaLA-LM `emma-500` L3/L3.1 in mono/bi, `Llama-3.1-8B`, `Meta-Llama-3-8B`) × 2 directions (Eng→X, X→Eng) × 2 prompt regimes (0-shot, 3-shot) → 24 runs, each covering all 203 FLORES-200 language pairs.
- **GlotEval configuration**: `flores200_mt` task with single-language prompts (`eng_Latn`), temperature `0.6`, top-p `0.9`, `max_tokens` 128, newline stop token, tensor-parallel 1, batch size 1, seed 42 (see `config_flores200_*.json`).
- **LM-Eval configuration**: Inspection of the shared LM-Eval JSON/JSONL artifacts confirmed identical settings:
  - Prompt template matches GlotEval (`Translate the following sentence…`), with three in-context examples for the 3-shot runs.
  - Generation args (`temperature`: 0.6, `top_p`: 0.9, `max_gen_toks`: 128, `until`: `"\n"`) and shot counts align with our configs.
  - Harness identifiers (`flores200_lm_eval_*`) confirm LM-Eval was used.
- **Comparison method**: Consolidated 4,872 language–direction–shot–model tuples from both pipelines, then computed ChrF++ deltas, MAE, RMSE, and Pearson correlation. Scatter visualizations live in `consistency_plots/`.

## 3. Results & Observations
- **Coverage**: All 4,872 tuples aligned perfectly (no missing languages or models).
- **ChrF++ agreement**: Mean MAE 4.11, RMSE 4.69, Pearson 0.978 across the 24 experiments. Twenty-one experiments land at MAE ≤ 5 with Pearson ≥ 0.99.
- **Stable directions**: X→Eng runs are particularly tight (ChrF++ MAE between 1.7–3.0, Pearson ≥ 0.99 for every non-EMMA mono case).
- **Higher-variance pocket**: EMMA-500 L3.1-8B mono at 0-shot shows larger offsets (ChrF++ MAE ≈16 for both directions, Pearson ≈0.84). Its 3-shot X→Eng run drops back to MAE ≈5, suggesting prompt demonstrations mitigate drift.
- **Visual confirmation**: The ChrF++ combined plots (`consistency_plots/*.png`) highlight near-diagonal clustering for every model/setting except the noted outlier.

<div align="center">

![EMMA-500 L3-8B bi](consistency_plots/EMMA-500_L3-8B_bi-combined.png)
![EMMA-500 L3-8B mono](consistency_plots/EMMA-500_L3-8B_mono-combined.png)
![EMMA-500 L3.1-8B bi](consistency_plots/EMMA-500_L3.1-8B_bi-combined.png)
![EMMA-500 L3.1-8B mono](consistency_plots/EMMA-500_L3.1-8B_mono-combined.png)
![Llama-3.1-8B](consistency_plots/Llama-3.1-8B-combined.png)
![Meta-Llama-3-8B](consistency_plots/Meta-Llama-3-8B-combined.png)

</div>

## 4. Conclusion
Under matched prompts and decoding parameters, GlotEval’s FLORES-200 ChrF++ scores track the LM Evaluation Harness extremely closely. The pipelines agree on relative model ranking, and absolute differences stay within ~3–5 ChrF++ points for 21/24 experiments. The sole exception (EMMA-500 L3.1-8B mono, 0-shot) merits targeted follow-up but does not undermine the broader finding: GlotEval delivers reproducible MT measurements consistent with the community-standard harness.

_Acknowledgement: Many thanks to our collaborator @[Tianxiang Wang](https://github.com/congc100) for running the LM Evaluation Harness reference experiments that enabled this study._
