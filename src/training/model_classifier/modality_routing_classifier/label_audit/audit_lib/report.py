"""The text report: agreement with the original labels, model re-scoring and tests.

Every function returns lines of text and prints nothing, so the report can be built
and checked without capturing stdout.
"""

import json
from collections import Counter

from audit_lib.checkpoint import Records
from audit_lib.constants import CONFIDENCE_ORDER, LABELS
from audit_lib.dataset import clip, text_sha256
from audit_lib.human_review import HumanReview
from audit_lib.judgment import inclusive_label
from audit_lib.stats import (
    accuracy_against,
    cohen_kappa,
    discordant_counts,
    mcnemar_exact,
)

BASELINE_MODEL_KEYS = ("published_baseline", "clean_baseline", "candidate")


class PredictionIdentityError(ValueError):
    """Predictions cannot be tied to the rows of the split they are scored against."""


DEFAULT_COMPARE = (("clean_baseline", "candidate"),)


def align_predictions(
    name: str, rows: list[dict], indexed: dict[int, tuple[str, str]]
) -> list[str]:
    """Check that predictions belong to the rows of the split, and put them in row order.

    Every row must be covered exactly once and its prompt hash must match, so a report
    from another export, another split or another row order is rejected instead of
    being joined by position.

    Args:
        name: Model name, used in error messages.
        rows: The split's rows.
        indexed: Row index to (prompt hash, predicted label).

    Returns:
        The predicted labels in row order.

    Raises:
        PredictionIdentityError: If the predictions do not cover exactly these rows.
    """
    if set(indexed) != set(range(len(rows))):
        raise PredictionIdentityError(
            f"{name}: predictions cover {len(indexed)} rows but the split has "
            f"{len(rows)}; they come from a different split or export"
        )
    labels = []
    for i, row in enumerate(rows):
        digest, label = indexed[i]
        if digest != text_sha256(row["text"]):
            raise PredictionIdentityError(
                f"{name}: row {i} was predicted for a different prompt (input hash "
                f"mismatch); the predictions come from a different export or row order"
            )
        labels.append(label)
    return labels


def eval_report_predictions(path: str, rows: list[dict]) -> dict[str, list[str]]:
    """Load the model predictions in an evaluation report, checked against the rows.

    Args:
        path: Path to modality_candidate_eval_report.json.
        rows: The split the predictions are scored against.

    Returns:
        Predicted label names per model name, in row order.

    Raises:
        PredictionIdentityError: If the report's rows are not these rows.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)["per_example_records"]
    preds: dict[str, list[str]] = {}
    for name in BASELINE_MODEL_KEYS:
        if not records or f"{name}_pred" not in records[0]:
            continue
        indexed = {
            r["row_index"]: (r["input_hash_sha256"], r[f"{name}_pred"]) for r in records
        }
        if len(indexed) != len(records):
            raise PredictionIdentityError(f"{name}: {path} repeats a row_index")
        preds[name] = align_predictions(name, rows, indexed)
    return preds


def file_predictions(name: str, path: str, rows: list[dict]) -> list[str]:
    """Load one model's predictions from a JSON file, checked against the rows.

    The file needs "preds" and "input_hashes" (the sha256 of each prompt, in the same
    order), because a bare list of labels cannot show which prompts it was made for.

    Args:
        name: Model name, used in error messages.
        path: Path to the JSON file.
        rows: The split the predictions are scored against.

    Returns:
        Predicted label names in row order.

    Raises:
        PredictionIdentityError: If the file has no hashes or they do not match.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    hashes = data.get("input_hashes")
    if not hashes or len(hashes) != len(data["preds"]):
        raise PredictionIdentityError(
            f"{name}: {path} has no input_hashes, one per prediction, so it cannot be "
            f"tied to these prompts; regenerate it with the eval script that wrote it"
        )
    indexed = {
        i: (h, p) for i, (h, p) in enumerate(zip(hashes, data["preds"], strict=True))
    }
    return align_predictions(name, rows, indexed)


def load_predictions(
    eval_report: str | None, pred_specs: list[str] | None, rows: list[dict]
) -> dict[str, list[str]]:
    """Load model predictions for re-scoring against the judged labels.

    Nothing is joined by position alone: every prediction is checked against the
    prompt hash of the row it is scored on.

    Args:
        eval_report: Path to modality_candidate_eval_report.json, or None.
        pred_specs: "NAME=PATH" entries, each a JSON file with "preds" and "input_hashes".
        rows: The split the predictions are scored against.

    Returns:
        Predicted label names per model name, in row order.

    Raises:
        PredictionIdentityError: If any predictions cannot be tied to these rows.
    """
    preds = eval_report_predictions(eval_report, rows) if eval_report else {}
    for spec in pred_specs or []:
        name, path = spec.split("=", 1)
        preds[name] = file_predictions(name, path, rows)
    return preds


def agreement_lines(
    split: str,
    rows: list[dict],
    original: list[str],
    records: dict[int, dict],
) -> tuple[list[str], list[int]]:
    """Report how often the judge agrees with the original labels.

    Args:
        split: Split name.
        rows: The split's rows.
        original: Original label name per row id.
        records: Judgment records by row id.

    Returns:
        (report lines, ids of the rows where judge and original disagree).
    """
    ids = sorted(records)
    o = [original[i] for i in ids]
    j = [records[i]["label"] for i in ids]
    ji = [inclusive_label(records[i]) for i in ids]
    agree = sum(a == b for a, b in zip(o, j, strict=True))
    agree_i = sum(a == b for a, b in zip(o, ji, strict=True))
    lines = [
        f"judged {len(ids)}/{len(rows)} rows of {split} | rubric hashes "
        f"{dict(Counter(r['rubric'] for r in records.values()))}",
        f"agreement original vs judge (strict):    {agree}/{len(ids)} = {agree / len(ids):.4f}  "
        f"kappa={cohen_kappa(o, j):.3f}",
        f"agreement original vs judge (inclusive): {agree_i}/{len(ids)} = {agree_i / len(ids):.4f}  "
        f"(inclusive = strict, but AR+vh counts as BOTH)",
        "\nconfusion, rows=original cols=judge (strict), order AR/DIFFUSION/BOTH:",
    ]
    cm = Counter(zip(o, j, strict=True))
    lines += [f"  {a:10s} {[cm.get((a, b), 0) for b in LABELS]}" for a in LABELS]
    disagreements = [i for i in ids if original[i] != records[i]["label"]]
    lines.append(
        f"\ndisagreements: {len(disagreements)} | by judge confidence "
        f"{dict(Counter(records[i]['conf'] for i in disagreements))} | vh among judge=AR: "
        f"{sum(records[i]['vh'] for i in ids)}/{sum(1 for i in ids if records[i]['label'] == 'AR')}"
    )
    lines.append(
        f"tags: {dict(Counter(t for r in records.values() for t in r['tags']))}"
    )
    return lines, disagreements


def model_lines(
    preds: dict[str, list[str]],
    pairs,
    views: dict[str, list[str]],
    ids: list[int],
) -> list[str]:
    """Re-score models against the original, strict and inclusive labels.

    Args:
        preds: Predicted label names per model name.
        pairs: Model name pairs for the McNemar tests.
        views: Reference labels per view name: "original", "strict", "inclusive".
        ids: Row ids that were judged, aligned with each view.

    Returns:
        The accuracy table and the paired significance tests.
    """
    lines = [
        "\nmodel accuracy on the judged rows: vs original | vs judge strict | vs judge inclusive"
    ]
    for name, p in preds.items():
        picked = [p[i] for i in ids]
        lines.append(
            f"  {name:22s} {accuracy_against(picked, views['original']):.4f} | "
            f"{accuracy_against(picked, views['strict']):.4f} | "
            f"{accuracy_against(picked, views['inclusive']):.4f}"
        )
    lines.append(
        "\npaired significance (exact McNemar, two-sided): rows where exactly one of the two models is right"
    )
    for a, b in pairs:
        if a not in preds or b not in preds:
            lines.append(
                f"  {a} vs {b}: no predictions for one of them (see --eval-report / --preds)"
            )
            continue
        for view, ref in views.items():
            only_a, only_b = discordant_counts(
                [preds[a][i] for i in ids], [preds[b][i] for i in ids], ref
            )
            lines.append(
                f"  {a} vs {b} | {view:9s}: only {a} right {only_a} | only {b} right {only_b} | "
                f"p={mcnemar_exact(only_a, only_b):.3f}"
            )
    return lines


def human_lines(review: HumanReview) -> list[str]:
    """Format a human spot-check comparison.

    Args:
        review: Result of estimate_error_rates.

    Returns:
        The report lines.
    """
    lines = [
        f"\n== human review: {review.n_labeled} rows labeled "
        f"({review.n_disagreement_rows} disagreement rows, {review.n_agreement_rows} agreement rows) =="
    ]
    if not review.n_disagreement_rows or not review.n_agreement_rows:
        return [
            *lines,
            "  need at least one labeled row in each stratum for the estimate",
        ]
    side = review.sides
    return [
        *lines,
        f"  on disagreement rows the human sides with: judge {side['judge']} | "
        f"original {side['original']} | neither {side['neither']}",
        f"  on agreement rows the human agrees with the shared label: "
        f"{review.agreement_ok}/{review.n_agreement_rows}",
        f"  stratified estimate over all {review.n_rows} rows: original-label error "
        f"~{review.original_error:.1%} | judge error ~{review.judge_error:.1%}"
        f" (small samples: treat as rough)",
    ]


def self_consistency_lines(
    records: dict[int, dict], rejudged: dict[int, dict]
) -> list[str]:
    """Report how often a second pass by the same judge gives the same label.

    Args:
        records: Primary judgment records by row id.
        rejudged: Re-judge records by row id.

    Returns:
        The report lines, empty if no row was judged twice.
    """
    both = sorted(set(rejudged) & set(records))
    if not both:
        return []
    a = [records[i]["label"] for i in both]
    b = [rejudged[i]["label"] for i in both]
    flips = [i for i, x, y in zip(both, a, b, strict=True) if x != y]
    return [
        f"\nself-consistency (re-judged {len(both)} rows): agreement "
        f"{(len(both) - len(flips)) / len(both):.4f} kappa={cohen_kappa(a, b):.3f} | flips {flips[:15]}"
    ]


def disagreement_lines(
    rows: list[dict],
    original: list[str],
    records: dict[int, dict],
    disagreements: list[int],
    show: int,
) -> list[str]:
    """List the disagreements, confident ones first.

    Args:
        rows: The split's rows.
        original: Original label name per row id.
        records: Judgment records by row id.
        disagreements: Ids where the judge and the original differ.
        show: How many to list.

    Returns:
        The report lines.
    """
    lines = [
        f"\n--- disagreements (original -> judge), confident ones first, showing {min(show, len(disagreements))} ---"
    ]
    ordered = sorted(
        disagreements, key=lambda i: (CONFIDENCE_ORDER[records[i]["conf"]], i)
    )
    for i in ordered[:show]:
        r = records[i]
        vh = " vh" if r["vh"] else ""
        lines.append(
            f"{i:4d} {original[i]}->{r['label']} {r['conf']}{vh} {','.join(r['tags'])} | {clip(rows[i]['text'])[0][:110]}"
        )
    return lines


def build_report_lines(
    split: str,
    rows: list[dict],
    records: Records,
    *,
    preds: dict[str, list[str]],
    pairs=DEFAULT_COMPARE,
    human: HumanReview | None = None,
    rejudged: Records | None = None,
    show: int = 40,
) -> list[str]:
    """Build the full audit report for one split.

    Args:
        split: Split name.
        rows: The split's rows.
        records: Primary checkpoint records of that split, by row id.
        preds: Predicted label names per model name; may be empty.
        pairs: Model pairs for the McNemar tests.
        human: Result of a human spot-check, or None.
        rejudged: Self-consistency records of that split, by row id, or None.
        show: How many disagreements to list.

    Returns:
        The report lines.
    """
    original = [r["label_name"] for r in rows]
    ids = sorted(records)
    lines, disagreements = agreement_lines(split, rows, original, records)
    if preds:
        views = {
            "original": [original[i] for i in ids],
            "strict": [records[i]["label"] for i in ids],
            "inclusive": [inclusive_label(records[i]) for i in ids],
        }
        lines += model_lines(preds, pairs, views, ids)
    if human is not None:
        lines += human_lines(human)
    if rejudged:
        lines += self_consistency_lines(records, rejudged)
    return lines + disagreement_lines(rows, original, records, disagreements, show)
