"""Loading a dataset split and keeping judgments tied to the file they were made on."""

import hashlib
import json
import re
from pathlib import Path

from audit_lib.constants import CLIP_HEAD, CLIP_LIMIT, CLIP_TAIL, RUBRIC


class DatasetMismatchError(Exception):
    """A split file differs from the one its judgments were made on."""


def clip(text: str) -> tuple[str, bool]:
    """Collapse whitespace and cut long text to its head and tail.

    Keeps batches small. The cut is marked with the number of characters removed.

    Args:
        text: Raw prompt text.

    Returns:
        (text to show, whether it was cut).
    """
    norm = re.sub(r"\s+", " ", text).strip()
    if len(norm) <= CLIP_LIMIT:
        return norm, False
    cut = len(norm) - CLIP_HEAD - CLIP_TAIL
    return f"{norm[:CLIP_HEAD]} [...{cut} chars cut...] {norm[-CLIP_TAIL:]}", True


def text_sha256(text: str) -> str:
    """Hash a prompt so predictions can be tied to the row they were made for.

    This matches the `input_hash_sha256` in the evaluation report.

    Args:
        text: Raw prompt text.

    Returns:
        Hex sha256 of the UTF-8 encoded text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rubric_hash(rubric_path: Path = RUBRIC) -> str:
    """Return the first 8 hex characters of the sha256 of the rubric.

    It is stored on every judgment, so a mix of rubric versions is detectable.

    Args:
        rubric_path: Path to RUBRIC.md.

    Returns:
        The short hash.
    """
    return hashlib.sha256(rubric_path.read_bytes()).hexdigest()[:8]


def load_rows(data_dir: Path, split: str) -> list[dict]:
    """Load one split's rows.

    Args:
        data_dir: Directory holding the split files.
        split: Split name, "train", "validation" or "test".

    Returns:
        Rows with "text", "label" and "label_name", in file order.
    """
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_pinned(
    path: Path, split: str, manifest_path: Path, *, require_existing: bool = False
) -> None:
    """Pin a dataset split by hash, or check it against the pinned hash.

    Judgments are keyed by row position, so a re-exported dataset would silently
    misalign them. The export is not reproducible across runs, so the file is
    pinned in the manifest on first use.

    Args:
        path: Path to the split's JSONL file.
        split: Split name, "train", "validation" or "test".
        manifest_path: JSON file holding the pinned hash per split.
        require_existing: Whether a split with no pinned hash is an error, because
            judgments for it already exist and there is nothing to check them against.

    Raises:
        DatasetMismatchError: If the file differs from the one that was pinned.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    known = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if split not in known and require_existing:
        raise DatasetMismatchError(
            f"judgments exist for the {split} split but {manifest_path.name} has no "
            f"recorded hash for it, so {path.name} cannot be checked against them"
        )
    if split not in known:
        known[split] = digest
        manifest_path.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
    elif known[split] != digest:
        raise DatasetMismatchError(
            f"{path.name} does not match the file the {split} judgments were made on "
            f"(sha256 {digest[:12]} vs recorded {known[split][:12]}); refusing to continue"
        )
