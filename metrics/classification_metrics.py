# metrics/classification_metrics.py

def compute_accuracy(predictions, references):
    """
    Computes accuracy for classification tasks.

    Args:
        predictions (list): List of predicted labels.
        references (list): List of true labels.

    Returns:
        accuracy (float): Accuracy score.
    """
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0
