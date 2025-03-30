from rouge_score import rouge_scorer

def calculate_rougeL_f1(hyp_texts, tgt_texts):
    """
    Calculate overall ROUGE-L F1 score between two lists of texts.

    Args:
        hyp_texts (list of str): List of hypothesis texts.
        tgt_texts (list of str): List of target texts.

    Returns:
        float: Overall average ROUGE-L F1 score.
    """
    if len(hyp_texts) != len(tgt_texts):
        raise ValueError("The length of hyp_texts and tgt_texts must be the same.")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_f1 = 0

    for hyp, tgt in zip(hyp_texts, tgt_texts):
        score = scorer.score(tgt, hyp)
        total_f1 += score['rougeL'].fmeasure

    overall_f1 = total_f1 / len(hyp_texts)
    return overall_f1