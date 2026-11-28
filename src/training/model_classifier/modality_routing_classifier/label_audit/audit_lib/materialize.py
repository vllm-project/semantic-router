"""Join dataset rows with their judgments into a reviewable artifact.

The judge's label is never used to overwrite the dataset's own label. Instead each
row gets both: `label_name` stays the original, and `judged_label_name` is the
judge's strict label. See DECISION_RECORD.md section 11.4 for why: a second column
anyone can diff is what lets a policy decision be checked rather than taken on
trust.
"""

from audit_lib.judgment import inclusive_label


def judged_dataset_rows(rows: list[dict], records: dict[int, dict]) -> list[dict]:
    """Pair each judged row with its judgment, keeping the original fields untouched.

    Args:
        rows: The split's rows, in original order, as loaded from its JSONL file.
        records: Judgment records by row id (from checkpoint.records_for_split).

    Returns:
        One dict per judged row id, in ascending id order. The original "label"
        and "label_name" are carried over unchanged; the judgment is added under
        separate "judged_*" names, so nothing here can be mistaken for the
        dataset's own column.
    """
    out = []
    for row_id in sorted(records):
        row = rows[row_id]
        judgment = records[row_id]
        out.append(
            {
                "id": row_id,
                "text": row["text"],
                "label": row["label"],
                "label_name": row["label_name"],
                "judged_label_name": judgment["label"],
                "judged_label_name_inclusive": inclusive_label(judgment),
                "judge_confidence": judgment["conf"],
                "judge_vh": bool(judgment["vh"]),
                "judge_tags": judgment["tags"],
            }
        )
    return out
