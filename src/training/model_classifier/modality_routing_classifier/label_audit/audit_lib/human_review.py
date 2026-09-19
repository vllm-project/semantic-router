"""The blinded sheet for a human spot-check, and the error estimate it gives."""

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audit_lib.constants import LABEL_CODES, MIN_SHEET_COLUMNS
from audit_lib.dataset import clip


class SheetSplitError(Exception):
    """A review sheet was made for a different split than the report."""


@dataclass(frozen=True)
class HumanReview:
    """What a human's labels say about the original labels and the judge.

    Attributes:
        n_labeled: Rows the human labelled.
        n_disagreement_rows: Labelled rows where the judge and the original differ.
        n_agreement_rows: Labelled rows where they agree.
        sides: On disagreement rows, how often the human sided with the "judge",
            the "original" or "neither".
        agreement_ok: Agreement rows where the human agrees with the shared label.
        original_error: Estimated share of wrong original labels, or None if a
            stratum is empty.
        judge_error: Estimated share of wrong judge labels, or None likewise.
        n_rows: Rows judged in the split, which the estimate is scaled to.
    """

    n_labeled: int
    n_disagreement_rows: int
    n_agreement_rows: int
    sides: Counter
    agreement_ok: int
    original_error: float | None
    judge_error: float | None
    n_rows: int


def pick_sheet_rows(
    rows: list[dict], records: dict[int, dict], n_agree: int, seed: int
) -> tuple[list[int], int]:
    """Pick every judge/original disagreement plus a random sample of agreements.

    Args:
        rows: The split's rows.
        records: Judgment records by row id.
        n_agree: Number of agreeing rows to add.
        seed: Seed for the sample and the shuffle.

    Returns:
        (shuffled row ids, number of disagreements among them).
    """
    disagreements = [
        i for i, r in records.items() if rows[i]["label_name"] != r["label"]
    ]
    disagreement_set = set(disagreements)
    agreements = [i for i in records if i not in disagreement_set]
    rng = random.Random(seed)
    picked = disagreements + rng.sample(agreements, min(n_agree, len(agreements)))
    rng.shuffle(picked)
    return picked, len(disagreements)


def write_sheet(path: Path, rows: list[dict], picked: list[int], split: str) -> None:
    """Write the blinded review sheet.

    Args:
        path: Where to write the tab-separated sheet.
        rows: The split's rows.
        picked: Row ids to include, in sheet order.
        split: Split name, recorded in the header.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# split={split}\n")
        f.write(
            "# Blind human labels: put A, D or B in the last column, following RUBRIC.md.\n"
        )
        f.write(
            "# Rows left blank are dropped from the estimate, which biases it toward easy rows, so guess when unsure.\n"
        )
        f.write(
            "# Do not open checkpoint*.jsonl or audit_report_*.txt before finishing: they contain the labels.\n"
        )
        f.write("id\ttext\tlabel\n")
        for i in picked:
            f.write(f"{i}\t{clip(rows[i]['text'])[0]}\t\n")


def load_human(path: str, split: str) -> dict[int, str]:
    """Load the labels a human filled into the review sheet.

    Args:
        path: Path to the sheet written by write_sheet.
        split: Split the report is for.

    Returns:
        Label name per row id, for the rows that were filled in.

    Raises:
        SheetSplitError: If the sheet was made for a different split.
    """
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("# split=") and line[len("# split=") :].strip() != split:
                raise SheetSplitError(
                    f"{path} was made for split {line[len('# split='):].strip()!r}, "
                    f"but the report is for {split!r}; pass --split"
                )
            if not line.strip() or line.startswith("#") or line.startswith("id\t"):
                continue
            parts = line.split("\t")
            label = (
                LABEL_CODES.get(parts[-1].strip().upper())
                if len(parts) >= MIN_SHEET_COLUMNS
                else None
            )
            if label and parts[0].strip().isdigit():
                out[int(parts[0])] = label
    return out


def estimate_error_rates(
    human: dict[int, str], original: list[str], records: dict[int, dict]
) -> HumanReview:
    """Compare a human's labels with the original and the judge.

    The error estimates weight the disagreement and agreement samples back up to the
    full split, so they are rough on a small sheet.

    Args:
        human: Label name per row id from load_human.
        original: Original label name per row id.
        records: Judgment records by row id.

    Returns:
        The comparison and, when both strata have labelled rows, the estimates.
    """
    human = {i: h for i, h in human.items() if i in records}
    dis_all = [i for i in records if original[i] != records[i]["label"]]
    agree_all = [i for i in records if original[i] == records[i]["label"]]
    dis = [i for i in human if original[i] != records[i]["label"]]
    agr = [i for i in human if original[i] == records[i]["label"]]
    sides = Counter(
        (
            "judge"
            if human[i] == records[i]["label"]
            else "original" if human[i] == original[i] else "neither"
        )
        for i in dis
    )
    agreement_ok = sum(human[i] == original[i] for i in agr)
    original_error = judge_error = None
    n = len(records)
    if dis and agr:
        q_wrong = 1 - agreement_ok / len(agr)
        original_error = (
            len(dis_all) * (sides["judge"] + sides["neither"]) / len(dis)
            + len(agree_all) * q_wrong
        ) / n
        judge_error = (
            len(dis_all) * (sides["original"] + sides["neither"]) / len(dis)
            + len(agree_all) * q_wrong
        ) / n
    return HumanReview(
        len(human),
        len(dis),
        len(agr),
        sides,
        agreement_ok,
        original_error,
        judge_error,
        n,
    )
