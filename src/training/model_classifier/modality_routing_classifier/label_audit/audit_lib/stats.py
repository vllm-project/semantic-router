"""Agreement and significance statistics."""

from collections import Counter
from math import comb


def accuracy_against(predicted: list[str], reference: list[str]) -> float:
    """Compute the share of predictions equal to the reference labels.

    Args:
        predicted: Predicted label names.
        reference: Reference label names, aligned with predicted.

    Returns:
        Accuracy in [0, 1].
    """
    return sum(x == y for x, y in zip(predicted, reference, strict=True)) / len(
        reference
    )


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Compute Cohen's kappa between two label sequences.

    Args:
        a: Labels from the first rater.
        b: Labels from the second rater, aligned with a.

    Returns:
        Kappa, or 1.0 if chance agreement is already total.
    """
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[label] / n) * (cb[label] / n) for label in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Compute the two-sided exact McNemar p-value.

    Args:
        only_a: Rows where only the first model is right.
        only_b: Rows where only the second model is right.

    Returns:
        The p-value, 1.0 if the models never differ.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def discordant_counts(
    preds_a: list[str], preds_b: list[str], reference: list[str]
) -> tuple[int, int]:
    """Count the rows where exactly one of two models matches the reference.

    Args:
        preds_a: Predictions of the first model.
        preds_b: Predictions of the second model, aligned with preds_a.
        reference: Reference labels, aligned with both.

    Returns:
        (rows only the first model gets right, rows only the second gets right).
    """
    right_a = [p == r for p, r in zip(preds_a, reference, strict=True)]
    right_b = [p == r for p, r in zip(preds_b, reference, strict=True)]
    only_a = sum(x and not y for x, y in zip(right_a, right_b, strict=True))
    only_b = sum(y and not x for x, y in zip(right_a, right_b, strict=True))
    return only_a, only_b
