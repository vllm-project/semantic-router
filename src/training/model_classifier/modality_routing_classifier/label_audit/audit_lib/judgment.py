"""Judgment lines: parsing them and turning them into checkpoint records."""

from pathlib import Path
from typing import NamedTuple

from audit_lib.checkpoint import Records, append_records
from audit_lib.constants import LABEL_CODES, MIN_LINE_TOKENS, TAG_CODES
from audit_lib.dataset import clip


class Judgment(NamedTuple):
    """One parsed judgment line.

    Attributes:
        id: Row id within the split.
        label: AR, DIFFUSION or BOTH.
        vh: 1 if the judge flagged that visuals would clearly help, else 0.
        confidence: H, M or L.
        tags: Sorted tag names.
    """

    id: int
    label: str
    vh: int
    confidence: str
    tags: list[str]


def parse_line(line: str) -> Judgment:
    """Parse one judgment line of the form "ID LABEL [flags]".

    Args:
        line: The line, for example "14 B M" or "15 A vh dvis".

    Returns:
        The parsed judgment.

    Raises:
        ValueError: If the id, label or a flag is not recognised, or vh is used
            with a label other than A.
    """
    tok = line.split()
    if len(tok) < MIN_LINE_TOKENS:
        raise ValueError("need 'ID LABEL'")
    try:
        rid = int(tok[0])
    except ValueError:
        raise ValueError(f"bad id {tok[0]!r}") from None
    label = LABEL_CODES.get(tok[1].upper())
    if label is None:
        raise ValueError(f"bad label {tok[1]!r}")
    vh, conf, tags = 0, "H", []
    for flag in tok[2:]:
        if flag.lower() == "vh":
            vh = 1
        elif len(flag) == 1 and flag.upper() in "MLH":
            conf = flag.upper()
        elif flag.lower() in TAG_CODES:
            tags.append(TAG_CODES[flag.lower()])
        else:
            raise ValueError(f"unknown flag {flag!r}")
    if vh and label != "AR":
        raise ValueError("vh is only valid with label A")
    return Judgment(rid, label, vh, conf, sorted(set(tags)))


def parse_lines(
    text: str, allowed: set[int] | None = None
) -> tuple[list[Judgment], list[str], int]:
    """Parse a block of judgment lines.

    Args:
        text: Lines of "ID LABEL [flags]"; blank and "#" lines are skipped.
        allowed: Ids that may appear, or None to allow any.

    Returns:
        (parsed judgments, error strings, count of ignored non-row lines).
    """
    parsed, errors, ignored, seen = [], [], 0, set()
    for raw in text.splitlines():
        line = raw.strip().strip("`")
        if not line or line.startswith("#"):
            continue
        if not line[0].isdigit():
            ignored += 1
            continue
        try:
            judgment = parse_line(line)
        except ValueError as e:
            errors.append(f"{line!r}: {e}")
            continue
        if allowed is not None and judgment.id not in allowed:
            errors.append(f"{line!r}: id out of range or not in this batch")
        elif judgment.id in seen:
            errors.append(f"{line!r}: duplicate id in input")
        else:
            seen.add(judgment.id)
            parsed.append(judgment)
    return parsed, errors, ignored


def make_record(
    split: str, judgment: Judgment, rows: list[dict], judge: str, rubric: str
) -> dict:
    """Build a checkpoint record from a parsed judgment.

    Adds the "truncated" tag itself when the judge only saw a clipped prompt.

    Args:
        split: Split name.
        judgment: The parsed judgment.
        rows: The split's rows.
        judge: Name of the judge, recorded on the record.
        rubric: Rubric hash, recorded on the record.

    Returns:
        The record to append to the checkpoint.
    """
    tags = judgment.tags
    if clip(rows[judgment.id]["text"])[1] and "truncated" not in tags:
        tags = sorted([*tags, "truncated"])
    return {
        "split": split,
        "id": judgment.id,
        "label": judgment.label,
        "vh": judgment.vh,
        "conf": judgment.confidence,
        "tags": tags,
        "judge": judge,
        "rubric": rubric,
    }


def inclusive_label(record: dict) -> str:
    """Return the label under the inclusive policy.

    Under it a text-only judgment that flagged visuals as clearly helpful (vh)
    counts as BOTH.

    Args:
        record: A judgment record.

    Returns:
        The judge's label, or BOTH for AR with vh set.
    """
    if record["label"] == "BOTH" or (record["label"] == "AR" and record["vh"]):
        return "BOTH"
    return record["label"]


def build_user_message(rows: list[dict], ids: list[int]) -> str:
    """Build the message that asks a judge to label one batch of rows.

    Args:
        rows: The split's rows.
        ids: Ids of the rows in this batch.

    Returns:
        One "id text" line per row, after a short instruction.
    """
    lines = "\n".join(f"{i} {clip(rows[i]['text'])[0]}" for i in ids)
    return f"Judge these {len(ids)} rows per the rubric. Reply with exactly one line per row.\n\n{lines}"


def save_judgments(
    judgments: list[Judgment],
    rows: list[dict],
    *,
    split: str,
    judge: str,
    rubric: str,
    path: Path,
    existing: Records,
    overwrite: bool = False,
) -> list[dict]:
    """Write judgments to a checkpoint, skipping rows already judged.

    Args:
        judgments: Parsed judgments.
        rows: The split's rows.
        split: Split name.
        judge: Name of the judge.
        rubric: Rubric hash.
        path: Checkpoint file to append to.
        existing: Records already in that checkpoint.
        overwrite: Whether to re-judge rows that are already saved.

    Returns:
        The records that were written.
    """
    fresh = [j for j in judgments if overwrite or (split, j.id) not in existing]
    records = [make_record(split, j, rows, judge, rubric) for j in fresh]
    if records:
        append_records(path, records)
    return records
