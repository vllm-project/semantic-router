from __future__ import annotations

import collections
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from v2.data.a7 import build_enc


def _entry(path: Path, name: str) -> dict:
    return {
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class LevelTest(unittest.TestCase):
    def test_rounding_and_tie_band(self) -> None:
        self.assertEqual(build_enc.level_or_none(3.714, 5), 4)
        self.assertEqual(build_enc.level_or_none(0.0, 5), 0)
        self.assertEqual(build_enc.level_or_none(5.0, 5), 5)
        self.assertIsNone(build_enc.level_or_none(2.5, 5))
        self.assertIsNone(build_enc.level_or_none(2.45, 5))
        self.assertEqual(build_enc.level_or_none(2.35, 5), 2)
        self.assertEqual(build_enc.level_or_none(4 * (2 / 3), 4), 3)
        self.assertIsNone(build_enc.level_or_none(5.2, 5))

    def test_balance_caps_each_level_at_the_ratio(self) -> None:
        rows = []
        for level, count in ((0, 10), (1, 3), (2, 7)):
            for index in range(count):
                rows.append(
                    {
                        "id": f"r{level}-{index}",
                        "task_type": "score",
                        "family": "f",
                        "label": level,
                        "options": [{}, {}, {}],
                    }
                )
        kept, dropped = build_enc.balance_levels(rows)
        counts = collections.Counter(row["label"] for row in kept)
        self.assertEqual(counts, {0: 3, 1: 3, 2: 3})
        self.assertEqual(dropped, {"f|0": 7, "f|2": 4})


class SourceRowsTest(unittest.TestCase):
    def test_afrisenti_rows_drop_conflicts_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sw.tsv")
            path.write_text(
                "ID\ttweet\tlabel\n"
                "a\tHabari njema sana\tpositive\n"
                "b\tHabari njema sana\tpositive\n"
                "c\tSipendi hii\tnegative\n"
                "d\tSipendi hii\tneutral\n"
                "e\tLeo ni jumatatu\tneutral\n",
                encoding="utf-8",
            )
            rows, skipped = build_enc.afrisenti_rows(_entry(path, "afrisenti_sw"))
        self.assertEqual([row["label"] for row in rows], [2, 1])
        self.assertEqual(skipped, {"conflicting_labels": 2, "duplicate_text": 1})
        self.assertEqual(rows[0]["options"][2]["key"], "2")
        self.assertTrue(rows[0]["state"].startswith("Message: "))

    def test_sts_rows_exclude_data_track_pairs_and_their_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            klue = Path(tmp, "klue.json")
            klue.write_text(
                json.dumps(
                    [
                        {
                            "guid": "k0",
                            "source": "s",
                            "sentence1": "A",
                            "sentence2": "B",
                            "labels": {"real-label": 4.9},
                        },
                        {
                            "guid": "k1",
                            "source": "s",
                            "sentence1": "B",
                            "sentence2": "C",
                            "labels": {"real-label": 1.2},
                        },
                        {
                            "guid": "k2",
                            "source": "s",
                            "sentence1": "D",
                            "sentence2": "E",
                            "labels": {"real-label": 2.5},
                        },
                        {
                            "guid": "k3",
                            "source": "s",
                            "sentence1": "F",
                            "sentence2": "G",
                            "labels": {"real-label": 3.1},
                        },
                        {
                            "guid": "k4",
                            "source": "s",
                            "sentence1": "G",
                            "sentence2": "H",
                            "labels": {"real-label": 0.0},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            jsts = Path(tmp, "jsts.json")
            jsts.write_text(
                json.dumps({"sentence_pair_id": "7", "yjcaptions_id": "x", "sentence1": "犬", "sentence2": "猫", "label": 0.4}) + "\n",
                encoding="utf-8",
            )  # fmt: skip
            rows, skipped = build_enc.sts_rows(
                [_entry(klue, "klue_sts"), _entry(jsts, "jsts")], {"klue_sts": {"k0"}}
            )
        by_id = {row["audit_metadata"]["a7"]["upstream_id"]: row for row in rows}
        self.assertEqual(sorted(by_id), ["7", "k3", "k4"])
        self.assertEqual(by_id["k3"]["label"], 3)
        self.assertEqual(by_id["k3"]["group_id"], by_id["k4"]["group_id"])
        self.assertEqual(by_id["7"]["language"], "ja")
        self.assertEqual(skipped["klue_sts_used_by_data_track"], 1)
        self.assertEqual(skipped["klue_sts_shares_sentence_with_data_track"], 1)
        self.assertEqual(skipped["klue_sts_tie"], 1)

    def test_massive_rows_balance_rotate_and_keep_train_only(self) -> None:
        items = []
        for index in range(12):
            intent = ["alarm_set", "alarm_query", "alarm_remove", "calendar_set"][
                index % 4
            ]
            scenario = "calendar" if intent.startswith("calendar") else "alarm"
            items.append(
                {
                    "id": str(index),
                    "locale": "ja-JP",
                    "partition": "train",
                    "scenario": scenario,
                    "intent": intent,
                    "utt": f"u{index}",
                }
            )
        items.append(
            {
                "id": "99",
                "locale": "ja-JP",
                "partition": "test",
                "scenario": "alarm",
                "intent": "alarm_set",
                "utt": "t",
            }
        )
        items.append(
            {
                "id": "5",
                "locale": "fr-FR",
                "partition": "train",
                "scenario": "alarm",
                "intent": "alarm_set",
                "utt": "f",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "massive.tar.gz")
            with tarfile.open(path, "w:gz") as archive:
                for locale in ("ja-JP", "fr-FR"):
                    data = "".join(
                        json.dumps(i) + "\n" for i in items if i["locale"] == locale
                    ).encode()
                    info = tarfile.TarInfo(f"1.1/data/{locale}.jsonl")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            rows, skipped = build_enc.massive_rows(_entry(path, "massive"))
        self.assertEqual(len(rows), 9)
        self.assertEqual(skipped["single_intent_scenario"], 3)
        self.assertTrue(all(row["language"] == "ja" for row in rows))
        for row in rows:
            gold = row["options"][row["label"]]["description"]
            self.assertEqual(
                gold, row["audit_metadata"]["a7"]["intent"].replace("_", " ")
            )
            self.assertEqual(len(row["options"]), 3)
            self.assertTrue(
                all(o["description"].startswith("alarm") for o in row["options"])
            )
            self.assertEqual(
                row["group_id"],
                f"a7:A7x:massive:{row['audit_metadata']['a7']['upstream_id'].split(':')[1]}",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow"), "needs pyarrow (runtime image)"
    )
    def test_oasst_rows_render_conversation_and_levels(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        def message(mid, parent, role, text, labels=None, **extra):
            return {
                "message_id": mid, "parent_id": parent, "role": role, "text": text,
                "lang": "pt-BR", "deleted": False, "synthetic": False, "review_result": True,
                "message_tree_id": "t1",
                "labels": labels or {"name": [], "value": [], "count": []}, **extra,
            }  # fmt: skip

        table = [
            message("p", None, "prompter", "Olá?"),
            message("a", "p", "assistant", "Oi!", {"name": ["quality", "humor", "pii"], "value": [0.75, 0.625, 0.0], "count": [3, 2, 3]}),
            message("b", "p", "assistant", "Tchau", {"name": ["quality", "pii"], "value": [1.0, 0.5], "count": [3, 3]}),
        ]  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "oasst.parquet")
            pq.write_table(pa.Table.from_pylist(table), path)
            rows, skipped = build_enc.oasst_rows(_entry(path, "oasst1"))
        self.assertEqual(
            [(r["family"], r["label"]) for r in rows], [("oasst1_quality", 3)]
        )
        self.assertEqual(rows[0]["language"], "pt")
        self.assertIn("User: Olá?", rows[0]["state"])
        self.assertTrue(rows[0]["state"].endswith("Final assistant reply:\nOi!"))
        self.assertEqual(skipped, {"pii_or_spam": 1, "tie_humor": 1})


if __name__ == "__main__":
    unittest.main()
