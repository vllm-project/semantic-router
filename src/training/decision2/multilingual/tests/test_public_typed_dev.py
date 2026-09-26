import json
import tempfile
import unittest
from pathlib import Path

from inference.run import file_digest
from multilingual import public_typed_dev as public


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class PublicTypedDevTest(unittest.TestCase):
    def test_native_target_mapping_preserves_option_order_and_score_index(self):
        choice = {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"B": "Beta", "A": "Alpha"},
        }
        noul = {
            "type": "noul",
            "instructions": "Is it?",
            "criteria": {"true": "Yes", "false": "No"},
        }
        score = {
            "type": "score",
            "instructions": "Rate",
            "criteria": ["Low", "Medium", "High"],
        }
        _, choice_target = public._make_row(
            "c", "g", "zh/x", "zh-CN", "Context", choice, "A"
        )
        _, noul_target = public._make_row(
            "n", "g", "zh/x", "zh-CN", "Context", noul, False
        )
        prompt, score_target = public._make_row(
            "s", "g", "zh/x", "zh-CN", "Context", score, "Medium"
        )
        self.assertEqual(choice_target["options"], ["B", "A"])
        self.assertEqual(noul_target["gold"], False)
        self.assertEqual(score_target["gold"], 1)
        self.assertEqual(score_target["options"], ["0", "1", "2"])
        self.assertNotIn("gold", prompt)
        self.assertNotIn("answer", prompt["questions"]["decision"])
        _, bare_target = public._make_row(
            "bare",
            "g",
            "ru/x",
            "ru-RU",
            "Context",
            {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"alarm": None, "audio": None},
            },
            "alarm",
        )
        self.assertEqual(bare_target["options"], ["alarm", "audio"])

    def test_serialized_choice_preserves_input_fingerprint_and_option_order(self):
        question = {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"z_last": "Last", "a_first": "First"},
        }
        prompt, target = public._make_row(
            "item", "group", "zh/test", "zh-CN", "State", question, "a_first"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            public._write_jsonl(path, [prompt])
            serialized = public._read_jsonl(path)
            public._validate_serialized_inputs(serialized, [target])
            self.assertEqual(
                list(serialized[0]["questions"]["decision"]["criteria"]),
                ["z_last", "a_first"],
            )
            path.write_text(
                json.dumps(prompt, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                public._validate_serialized_inputs(public._read_jsonl(path), [target])

    def test_malformed_native_semantics_fail_closed(self):
        with self.assertRaises(ValueError):
            public._question_target(
                {"type": "choice", "instructions": "?", "criteria": {"a": "A"}}, "a"
            )
        with self.assertRaises(ValueError):
            public._question_target(
                {"type": "noul", "instructions": "?", "criteria": None}, {"true": 1}
            )
        with self.assertRaises(ValueError):
            public._question_target(
                {"type": "score", "instructions": "?", "criteria": ["same", "same"]}, 0
            )
        target = {"task_type": "score", "options": ["0", "1", "2"], "gold": 1}
        self.assertEqual(
            public._native_choice(
                {"type": "score", "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}},
                target,
            ),
            (True, True),
        )
        self.assertEqual(
            public._native_choice(
                {"type": "score", "probabilities": {"0": 0.1, "1": 0.9}}, target
            ),
            (False, False),
        )
        self.assertEqual(
            public._native_choice(
                {"type": "score", "probabilities": {"0": 0.1, "1": 0.8, "2": 0.8}},
                target,
            ),
            (False, False),
        )
        noul = {"task_type": "noul", "options": ["false", "true"], "gold": True}
        self.assertEqual(
            public._native_choice({"type": "noul", "noul": 0.5}, noul), (True, True)
        )
        self.assertEqual(
            public._native_choice({"type": "noul", "noul": float("nan")}, noul),
            (False, False),
        )

    def test_massive_overlap_and_malformed_lineage_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1, v2 = root / "v1.jsonl", root / "v2.jsonl"
            write_rows(
                v1,
                [
                    {
                        "group_id": "massive-1.1:200",
                        "audit_metadata": {"source_id": "200"},
                    }
                ],
            )
            write_rows(
                v2,
                [
                    {
                        "group_id": "massive-1.1:300",
                        "audit_metadata": {"source_id": "300"},
                    }
                ],
            )
            ids = {str(i) for i in range(1, 180)}
            ru_ids = {str(i) for i in range(1000, 1290)}
            result = public.audit_massive_ids(ids, ru_ids, {"v1": v1, "v2": v2})
            self.assertEqual(result["v1"]["zh_overlap_source_groups"], 0)
            write_rows(
                v2,
                [{"group_id": "massive-1.1:1", "audit_metadata": {"source_id": "1"}}],
            )
            with self.assertRaisesRegex(ValueError, "overlaps protected v2 TRAIN"):
                public.audit_massive_ids(ids, ru_ids, {"v1": v1, "v2": v2})
            write_rows(
                v2,
                [
                    {
                        "group_id": "massive-1.1:1000",
                        "audit_metadata": {"source_id": "1000"},
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "overlaps protected v2 TRAIN"):
                public.audit_massive_ids(ids, ru_ids, {"v1": v1, "v2": v2})
            write_rows(
                v2, [{"group_id": "wrong:300", "audit_metadata": {"source_id": "300"}}]
            )
            with self.assertRaisesRegex(ValueError, "lacks matching source ID"):
                public.audit_massive_ids(ids, ru_ids, {"v1": v1, "v2": v2})
            with self.assertRaisesRegex(ValueError, "Both known MASSIVE"):
                public.audit_massive_ids(ids, ru_ids, {"v1": v1})

    def test_score_counts_missing_and_invalid_as_wrong(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory)
            cases = [
                (
                    "c",
                    {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {"A": "Alpha", "B": "Beta"},
                    },
                    "B",
                ),
                ("n", {"type": "noul", "instructions": "Yes?", "criteria": None}, True),
                (
                    "s",
                    {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": ["Low", "Mid", "High"],
                    },
                    2,
                ),
            ]
            pairs = [
                public._make_row(id_, id_, "zh/test", "zh-CN", "State", question, gold)
                for id_, question, gold in cases
            ]
            prompts, targets = zip(*pairs, strict=True)
            write_rows(panel / "prompts.jsonl", list(prompts))
            write_rows(panel / "targets.private.jsonl", list(targets))
            manifest = {
                "schema_version": public.VERSION,
                "adapter_source_sha256": file_digest(Path(public.__file__)),
                "release_panel_eligible": False,
                "files_sha256": {
                    name: file_digest(panel / name)
                    for name in ("prompts.jsonl", "targets.private.jsonl")
                },
            }
            (panel / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            identity = {
                "backend": "fixture",
                "model_id": "model",
                "model_revision": "rev",
                "adapter_version": "adapter",
            }
            predictions = panel / "predictions.jsonl"
            write_rows(
                predictions,
                [
                    {
                        "id": "c",
                        "source_input_sha256": targets[0]["source_input_sha256"],
                        "answers": {"decision": {"type": "choice", "choice": "B"}},
                        **identity,
                    },
                    {
                        "id": "s",
                        "source_input_sha256": targets[2]["source_input_sha256"],
                        "answers": {
                            "decision": {
                                "type": "score",
                                "probabilities": {"0": 0.4, "1": 0.4},
                            }
                        },
                        **identity,
                    },
                ],
            )
            report = public.score(panel, predictions)
            self.assertEqual(report["by_task"]["zh/test"]["questions"], 3)
            self.assertEqual(report["by_task"]["zh/test"]["correct"], 1)
            self.assertEqual(report["by_task"]["zh/test"]["invalid_or_missing"], 2)
            self.assertEqual(report["by_task"]["zh/test"]["accuracy"], 1 / 3)
            self.assertEqual(report["by_language_type"]["zh-CN/choice"]["correct"], 1)
            self.assertEqual(
                report["by_task_type"]["zh/test/score"]["invalid_or_missing"], 1
            )
            write_rows(
                predictions,
                [
                    {
                        "id": "c",
                        "source_input_sha256": "wrong",
                        "answers": {"decision": {"type": "choice", "choice": "B"}},
                        **identity,
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                public.score(panel, predictions)
            predictions.write_text("", encoding="utf-8")
            report = public.score(panel, predictions)
            self.assertEqual(report["by_task"]["zh/test"]["invalid_or_missing"], 3)


if __name__ == "__main__":
    unittest.main()
