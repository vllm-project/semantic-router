from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import hitgroups


class GroupTest(unittest.TestCase):
    def test_groups(self) -> None:
        cases = {
            "/data/dev2/private/sources/tydiqa@da78/primary_task/train-0.jsonl": "/data/dev2/private/sources/tydiqa@da78",
            "/data/dev2/private/sources/m3b/nq/default/train-1.parquet": "/data/dev2/private/sources/m3b/nq",
            "/data/dev2/private/a7/sources/enc10/massive/x.jsonl": "/data/dev2/private/a7/sources/enc10",
            "/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/r/v2/a7/arms/A7q/train.jsonl": "training-data:v2/a7/arms/A7q",
            "/data/dev2/private/c1-rescan-hf-delta/v1_3-x/files/0123456789ab/m2/arms/H1/train.jsonl": "training-data:m2/arms/H1",
            "/data/dev2/hf-cache/datasets--demelin--moral_stories/snapshots/r/data/t.jsonl": "hf:datasets--demelin--moral_stories",
            "/data/dev2/hf-cache/blobs/ab/abcdef.json": "hf-blobs",
            "/data/dev2/runs/dec/m4/data/mix.jsonl": "/data/dev2/runs/dec/m4",
            "/data/decision20-20260926/external/LLMs_for_CSS/css_data/t.csv": "/data/decision20-20260926/external/LLMs_for_CSS/css_data",
        }
        for path, expected in cases.items():
            self.assertEqual(hitgroups.group(path), expected, path)

    def test_work_paths_map_back(self) -> None:
        roots = {"dev2-private": "/data/dev2/private"}
        work = "/w/work"
        self.assertEqual(
            hitgroups.source_path(
                "/w/work/dev2-private/sources/x.tar.gz.d/a/b.jsonl", work, roots
            ),
            "/data/dev2/private/sources/x.tar.gz",
        )
        self.assertEqual(
            hitgroups.source_path(
                "/w/work/dev2-private/sources/t/notes.txt.jsonl", work, roots
            ),
            "/data/dev2/private/sources/t/notes.txt",
        )
        self.assertEqual(
            hitgroups.source_path("/data/x.jsonl", work, roots), "/data/x.jsonl"
        )


class ToolTest(unittest.TestCase):
    def test_subset_and_table(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            rows = [
                {"task": "s/f", "source_item_id": str(i), "overlap_texts": ["t"]}
                for i in range(3)
            ]
            (tmp / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (tmp / "ids.txt").write_text("s/f|0\ns/f|2\n")
            with contextlib.redirect_stdout(io.StringIO()):
                hitgroups.main(
                    [
                        "subset",
                        "--protected",
                        str(tmp / "p.jsonl"),
                        "--ids",
                        str(tmp / "ids.txt"),
                        "--output",
                        str(tmp / "s.jsonl"),
                    ]
                )
            self.assertEqual(len((tmp / "s.jsonl").read_text().splitlines()), 2)
            hit = {
                "id": "s/f|0",
                "shingles": 4,
                "labels": {
                    "g1": {"verdict": "REVIEW", "containment": 0.3, "exact": None},
                    "g2": {"verdict": "CLEAN", "containment": 0.1, "exact": None},
                },
            }
            other = {
                **hit,
                "labels": {
                    "g1": {"verdict": "OVERLAP", "containment": 0.9, "exact": 12}
                },
            }
            (tmp / "a.jsonl").write_text(json.dumps(hit) + "\n")
            (tmp / "b.jsonl").write_text(json.dumps(other) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                hitgroups.main(
                    [
                        "table",
                        "--hits",
                        str(tmp / "a.jsonl"),
                        "--hits",
                        str(tmp / "b.jsonl"),
                        "--output",
                        str(tmp / "t.json"),
                    ]
                )
            table = json.loads((tmp / "t.json").read_text())["rows"]["s/f|0"]
            self.assertEqual(
                table["groups"],
                {"g1": {"verdict": "OVERLAP", "containment": 0.9, "exact": 12}},
            )


if __name__ == "__main__":
    unittest.main()
