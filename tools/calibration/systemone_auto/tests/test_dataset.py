"""Source format adapters retain human votes and split original groups together."""

from __future__ import annotations

import csv
import json
import zipfile

import pytest
from systemone_auto.artifacts import file_digest, write_json
from systemone_auto.dataset import prepare
from systemone_auto.sources import SOURCES


def sources(directory):
    with (directory / "banking77.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["text", "category"])
        writer.writerows(
            (f"independent request {i}", f"intent_{i % 3}") for i in range(60)
        )
    with zipfile.ZipFile(directory / "boolq.zip", "w") as archive:
        rows = [
            {
                "passage": f"article{i // 2} -- paragraph{i}",
                "question": "Is it true?",
                "label": bool(i % 2),
            }
            for i in range(60)
        ]
        archive.writestr(
            "BoolQ/train.jsonl", "\n".join(json.dumps(row) for row in rows)
        )
    with zipfile.ZipFile(directory / "dynasent.zip", "w") as archive:
        rows = [
            {
                "sentence": f"opinion{i}",
                "text_id": str(i),
                "gold_label": "positive",
                "prompt_data": {"review_id": str(i // 2)},
                "label_distribution": {
                    "positive": ["a", "b", "c"],
                    "negative": ["d"],
                    "mixed": ["e"],
                },
            }
            for i in range(60)
        ]
        archive.writestr(
            "dataset/round02-dynabench-train.jsonl",
            "\n".join(json.dumps(row) for row in rows),
        )
        archive.writestr("__MACOSX/._round02-dynabench-train.jsonl", "not data")
    write_json(
        directory / "sources.json",
        {
            "sources": {
                name: {"sha256": file_digest(directory / source["filename"])}
                for name, source in SOURCES.items()
            }
        },
    )


def test_deterministic_unique_source_partition_preserves_annotation_counts(tmp_path):
    sources(tmp_path)
    first = prepare(tmp_path, tmp_path / "first.json", per_source=12)
    second = prepare(tmp_path, tmp_path / "second.json", per_source=12)
    assert first == second
    public = [row for row in first["records"] if row["cohort"] == "public"]
    assert len(public) == len({row["group_id"] for row in public}) == 36
    assert first["counts"]["public_splits"] == {
        "train": 18,
        "calibration": 9,
        "held_out": 9,
    }
    sentiment = next(row for row in public if row["source"] == "dynasent")
    label = sentiment["labels"]["sentiment"]
    assert label["annotation_counts"] == {"positive": 3, "negative": 1, "mixed": 1}
    assert label["distribution"] == {"0": 0.25, "1": 0.0, "2": 0.75}
    assert '"a"' not in (tmp_path / "first.json").read_text()
    with (tmp_path / "banking77.csv").open("a") as stream:
        stream.write("modified,modified\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare(tmp_path, tmp_path / "bad.json", per_source=12)
