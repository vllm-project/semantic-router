"""Stance of Italian Instagram comments toward the politician who wrote the post."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import (
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    input_chars,
    normalized,
)

SPEC = SourceSpec(
    key="stance_it",
    dataset_id="giuseppe-aiello/stance-detection-it-dataset",
    revision="2e4c575ae73144567335066d5d383c4b1e969bdf",
    licence="mit",
    licence_flag=None,
    first_release="2026-08-31",
    evidence=(
        "https://huggingface.co/api/datasets/giuseppe-aiello/stance-detection-it-dataset"
        "/commits/main: single upload 2026-08-31. The linked GitHub project "
        "(giuseppe-aiello/political-stance-detection-it) held only README, LICENSE and "
        "requirements before 2026-08-31, when data, labels and results were added. "
        "Comments were collected from public Instagram posts in February 2026."
    ),
    label_provenance=(
        "One annotator (the author) labelled 500 sampled comments by hand "
        "(labeling/train_to_label.xlsx 400 + test_to_label.xlsx 100) with written class "
        "definitions and edge-case rules (project report sec. 3.6). A later AI-assisted "
        "batch is not in this dataset; the master file's label column is empty."
    ),
    languages=("it",),
    tasks=("stance_it/stance",),
    notes=(
        "Reads labeling/*.xlsx plus processed/*_READY.xlsx (to find each comment's "
        "post); likes, NLI pairs and the zero-shot exploration are never read. Exact "
        "duplicate (comment, target, topic) rows are kept once. group_id = the post when "
        "the comment occurs under exactly one post of its target and topic, else the "
        "normalized comment. Risks: single annotator; comment text public on Instagram "
        "before the cutoff; disagree comments are much longer than the others."
    ),
)

TASK = "stance_it/stance"
LABELS = {"Consenso": "agree", "Dissenso": "disagree", "Altro": "other"}
INSTRUCTIONS = (
    "The state holds a comment, in Italian, posted under an Instagram post by an "
    "Italian politician. `target` says whether the post's author belongs to the "
    "government (Governo: Giorgia Meloni or Matteo Salvini) or to the opposition "
    "(Opposizione: Giuseppe Conte or Elly Schlein); `topic` is the subject of the post "
    "(Emergenza Sicurezza: public security; Emergenza Sud Italia: the problems of "
    "southern Italy). What stance does the comment take toward the post's author?"
)
CRITERIA = {
    "agree": (
        "The comment supports or agrees with the post's author or their position, "
        "including by attacking the author's political opponents."
    ),
    "disagree": "The comment criticises or disagrees with the post's author or their position.",
    "other": (
        "The comment takes no stance toward the post's author: it is off-topic, spam, "
        "ambiguous or unclassifiable, or it reacts to the situation in the post "
        "without judging the author."
    ),
}

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DOC_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _text(node: ET.Element) -> str:
    parts = [t.text or "" for t in node.findall(f"{MAIN}t")]
    for run in node.findall(f"{MAIN}r"):
        parts.extend(t.text or "" for t in run.findall(f"{MAIN}t"))
    return "".join(parts)


def _column(ref: str) -> int:
    index = 0
    for char in re.match(r"[A-Z]+", ref).group(0):
        index = index * 26 + ord(char) - 64
    return index - 1


def _sheet(path: Path) -> list[dict[str, str | None]]:
    """Rows of the first worksheet as header -> cell text."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_text(item) for item in strings.findall(f"{MAIN}si")]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        links = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            link.get("Id"): link.get("Target")
            for link in links.findall(f"{PKG_REL}Relationship")
        }
        first = workbook.find(f"{MAIN}sheets").find(f"{MAIN}sheet")
        target = targets[first.get(f"{DOC_REL}id")]
        member = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        grid = []
        for row in ET.fromstring(archive.read(member)).iter(f"{MAIN}row"):
            cells: dict[int, str | None] = {}
            for cell in row.findall(f"{MAIN}c"):
                value = cell.find(f"{MAIN}v")
                if cell.get("t") == "s":
                    text = shared[int(value.text)]
                elif cell.get("t") == "inlineStr":
                    inline = cell.find(f"{MAIN}is")
                    text = _text(inline) if inline is not None else ""
                else:
                    text = value.text if value is not None else None
                cells[_column(cell.get("r"))] = text
            grid.append(cells)
    if not grid:
        return []
    header = {index: (name or "").strip() for index, name in grid[0].items()}
    return [
        {name: cells.get(index) for index, name in header.items()}
        for cells in grid[1:]
        if any(value not in (None, "") for value in cells.values())
    ]


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _posts(root: Path) -> dict[tuple[str, str, str], set[str]]:
    posts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for path in sorted((root / "processed").glob("*_READY.xlsx")):
        post = path.name.removesuffix("_READY.xlsx")
        for row in _sheet(path):
            key = (
                normalized(_clean(row.get("text_clean"))),
                _clean(row.get("target")),
                _clean(row.get("topic")),
            )
            posts[key].add(post)
    return posts


def candidates(root: Path) -> Iterator[Candidate]:
    posts = _posts(root)
    seen: set[tuple[str, str, str]] = set()
    for split in ("train", "test"):
        rows = _sheet(root / "labeling" / f"{split}_to_label.xlsx")
        for index, row in enumerate(rows):
            comment = _clean(row.get("text_clean"))
            target = _clean(row.get("target"))
            topic = _clean(row.get("topic"))
            label = _clean(row.get("label"))
            key = (normalized(comment), target, topic)
            if not comment or not label or key in seen:
                continue
            seen.add(key)
            found = posts.get(key, set())
            if len(found) == 1:
                group = f"post:{next(iter(found))}"
            else:
                group = "text:" + hashlib.sha256(key[0].encode()).hexdigest()[:16]
            gold = LABELS[label]
            state = {"target": target, "topic": topic, "comment": comment}
            question = {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": dict(CRITERIA),
            }
            if input_chars(state, question) > MAX_INPUT_CHARS:
                continue
            yield Candidate(
                source=SPEC.key,
                task=TASK,
                source_item_id=f"{split}-{index:03d}",
                group_id=group,
                balance_label=gold,
                language="it",
                state=state,
                question=question,
                gold=gold,
                overlap_texts=[comment],
            )
