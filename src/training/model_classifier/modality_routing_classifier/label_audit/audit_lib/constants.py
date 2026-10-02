"""Paths, label codes and limits shared by the audit modules."""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA_DIR = HERE.parent / "exported_modality_routing_dataset"
RUBRIC = HERE / "RUBRIC.md"
MANIFEST = HERE / "dataset_sha256.json"
CHECKPOINT = HERE / "checkpoint.jsonl"
REJUDGE = HERE / "checkpoint_rejudge.jsonl"

LABELS = ["AR", "DIFFUSION", "BOTH"]
LABEL_CODES = {
    "A": "AR",
    "D": "DIFFUSION",
    "B": "BOTH",
    "AR": "AR",
    "DIFFUSION": "DIFFUSION",
    "BOTH": "BOTH",
}
TAG_CODES = {
    "img": "about_images",
    "prm": "prompt_writing",
    "nor": "no_request",
    "frag": "fragment",
    "nen": "non_english",
    "dvis": "deliverable_visual",
    "amb": "ambiguous",
}
CONFIDENCE_ORDER = {"H": 0, "M": 1, "L": 2}

# Rows longer than CLIP_LIMIT are shown as head + tail so batches stay cheap; the
# tooling adds the `truncated` tag itself (deterministically) instead of relying on
# the judge.
CLIP_LIMIT, CLIP_HEAD, CLIP_TAIL = 400, 280, 120
MIN_LINE_TOKENS = 2  # a judgment line is at least "ID LABEL"
MIN_SHEET_COLUMNS = 3  # id, text, label
MAX_API_ATTEMPTS = 4
LEGEND = "reply lines: ID A|D|B [vh] [M|L] [" + " ".join(TAG_CODES) + "]"
