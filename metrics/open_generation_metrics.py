import sacrebleu
from statistics import mean


def compute_self_bleu(texts):
    """
    Compute the Self-BLEU score for a list of generated texts using SacreBLEU.

    Each text in the list is treated as a candidate, while all the other texts
    are treated as references. The final score is the average BLEU score across
    all texts.

    Args:
        texts (List[str]): A list of generated text strings.
    Returns:
        float: The average BLEU score (0-100 range, where higher indicates greater
               similarity among the texts).
    """
    # Create a BLEU metric instance with effective_order=True to handle
    # edge cases when certain n-grams are missing.
    bleu_scorer = sacrebleu.metrics.BLEU(effective_order=True)

    scores = []
    num_texts = len(texts)

    for i in range(num_texts):
        # Candidate text
        candidate = texts[i]

        # Use all other texts as references
        # SacreBLEU's sentence_score expects references in the format: List[List[str]]
        # The outer list contains multiple reference sets, each of which can
        # contain one or more references. Here, we combine all other texts into
        # a single reference set.
        references = texts[:i] + texts[i + 1 :]

        # Calculate the sentence-level BLEU score
        result = bleu_scorer.sentence_score(candidate, references)

        # result.score is the BLEU value in the 0-100 range
        scores.append(result.score)

    # Return the average of the BLEU scores for all texts
    return mean(scores)
