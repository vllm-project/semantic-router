"""Native type conversion, leakage and derived checkpoint identity tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.run import file_digest

from training.kev import prepare as converter
from training.kev.infer import collect, verify_checkpoint
from training.kev.prepare import FORMAT, audit_overlap, convert_row, prepare
from training.model.data import digest


def flat(
    item_id: str,
    kind: str,
    options: list[tuple[str, str]],
    label: int,
    split: str = "train",
    state: str | None = None,
) -> dict:
    choices = [{"key": key, "description": description} for key, description in options]
    row = {
        "id": item_id,
        "state": state or f"state-{item_id}",
        "instructions": f"decide {item_id}",
        "options": choices,
        "label": label,
        "task_type": kind,
        "family": "fixture",
        "group_id": f"group-{item_id}",
        "language": "en",
        "split": split,
        "source": "fixture_original",
        "evaluation_role": "calibrate" if split == "cal" else split,
        "render_template": "fixture_v1",
        "audit_metadata": {},
    }
    row["input_sha256"] = digest(
        {
            field: row[field]
            for field in ("state", "instructions", "options", "task_type")
        }
    )
    return row


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class PrepareKevTest(unittest.TestCase):
    def test_native_choice_noul_score_gold_and_order(self):
        choice = flat("c", "choice", [("z", "last"), ("a", "first")], 0)
        noul = flat("n", "noul", [("true", "Yes"), ("false", "No")], 0)
        score = flat("s", "score", [("2", "High"), ("0", "Low"), ("1", "Middle")], 0)
        c, n, s = map(convert_row, (choice, noul, score))
        self.assertEqual(list(c["questions"]["decision"]["criteria"]), ["z", "a"])
        self.assertEqual(c["questions"]["decision"]["label"], "z")
        self.assertIs(n["questions"]["decision"]["label"], True)
        self.assertEqual(
            s["questions"]["decision"]["criteria"], ["Low", "Middle", "High"]
        )
        self.assertEqual(s["questions"]["decision"]["label"], 2)

    def test_cross_partition_state_and_gold_free_prompt_overlap_fail(self):
        train = [flat("t", "choice", [("a", "A"), ("b", "B")], 0)]
        select = [
            flat("s", "choice", [("c", "C"), ("d", "D")], 0, "select", state="state-t")
        ]
        cal = [flat("k", "noul", [("false", "No"), ("true", "Yes")], 1, "cal")]
        with self.assertRaisesRegex(ValueError, "state overlaps"):
            audit_overlap(train, select, cal, {})
        select[0] = flat("s", "choice", [("c", "C"), ("d", "D")], 0, "select")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.prompts.jsonl"
            prompt = {
                "id": "p",
                "state": "state-t",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "other",
                        "criteria": {"a": "A", "b": "B"},
                    }
                },
            }
            write_rows(path, [prompt])
            with self.assertRaisesRegex(
                ValueError, "overlaps a train/select/cal partition"
            ):
                audit_overlap(train, select, cal, {path: [prompt]})

    def test_conversion_requires_every_native_record_and_preserves_choice_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, select, cal = (
                root / f"{name}.jsonl" for name in ("train", "select", "cal")
            )
            write_rows(train, [flat("t", "choice", [("z", "Z"), ("a", "A")], 0)])
            write_rows(
                select,
                [flat("s", "noul", [("false", "No"), ("true", "Yes")], 0, "select")],
            )
            write_rows(cal, [flat("k", "score", [("0", "L"), ("1", "H")], 1, "cal")])
            output = root / "converted.jsonl"
            parent = {
                "model_revision": "pinned",
                "adapter_sha256": "a",
                "head_sha256": "h",
            }

            def native(path, converted, _source, max_state):
                self.assertEqual(max_state, 7552)
                raw = json.loads(path.read_text().strip())
                self.assertEqual(
                    list(raw["questions"]["decision"]["criteria"]), ["z", "a"]
                )
                self.assertEqual(raw, converted[0])
                return {
                    "records": 1,
                    "questions": 1,
                    "training_context": {"max_state": max_state},
                    "native_source_revision": "pinned",
                }

            manifest = prepare(
                train,
                select,
                cal,
                root,
                root,
                output,
                native_check=native,
                parent_check=lambda _m, _s: parent,
            )
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["kev_train_sha256"], file_digest(output))
            self.assertEqual(
                json.loads((root / "converted.jsonl.manifest.json").read_text()),
                manifest,
            )
            with self.assertRaises(FileExistsError):
                prepare(
                    train,
                    select,
                    cal,
                    root,
                    root,
                    output,
                    native_check=native,
                    parent_check=lambda _m, _s: parent,
                )
            second = root / "rejected.jsonl"
            with self.assertRaisesRegex(ValueError, "every train row"):
                prepare(
                    train,
                    select,
                    cal,
                    root,
                    root,
                    second,
                    native_check=lambda *_: {"records": 0, "questions": 0},
                    parent_check=lambda _m, _s: parent,
                )
            self.assertFalse(second.exists())


class DerivedCheckpointTest(unittest.TestCase):
    def fixture(self, root: Path):
        parent = {
            "model_id": "jaredpalmer/kev-4b",
            "model_revision": "pinned",
            "adapter_sha256": "a" * 64,
            "head_sha256": "b" * 64,
        }
        data = root / "train.jsonl"
        data.write_text("{}\n")
        manifest_file = root / "train.jsonl.manifest.json"
        manifest_file.write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "parent": parent,
                    "kev_train_sha256": file_digest(data),
                    "conversion_code_sha256": file_digest(Path(converter.__file__)),
                    "rows": 2,
                    "native_preflight": {
                        "records": 2,
                        "questions": 2,
                        "native_source_revision": "6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63",
                        "training_context": {"max_state": 7552},
                    },
                }
            )
        )
        checkpoint = root / "run"
        checkpoint.mkdir()
        config = {
            "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
            "init_source": {
                "adapter_sha256": parent["adapter_sha256"],
                "head_sha256": parent["head_sha256"],
            },
            "args": {
                "base": "Qwen/Qwen3.5-4B-Base",
                "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                "lora": 16,
                "head_dim": 256,
                "option_isolation": 0,
                "special_embeddings": 0,
                "weights_dtype": "fp32",
                "lora_placement": "full",
                "replay": 0,
                "lora_targets": "all",
                "suite": None,
                "anchor": "",
                "anchor_w": 0.0,
                "p_none": 0.0,
                "p_none_distract": 0.0,
                "p_distract": 0.0,
                "p_none_pair": 0.0,
                "public_frac": 1.0,
                "synthetic_repeat": 1,
                "train_sources": "",
                "max_state": 7552,
                "data": str(data),
                "epochs": 2,
            },
        }
        metrics = {
            "requested_records": 4,
            "truncated_records": 0,
            "rejected_records": 0,
        }
        adapter = {
            "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }
        for name, value in (
            ("training_config.json", config),
            ("training_metrics.json", metrics),
            ("adapter_config.json", adapter),
        ):
            (checkpoint / name).write_text(json.dumps(value))
        for name in ("adapter_model.safetensors", "head.pt", "tokenizer.json"):
            (checkpoint / name).write_bytes(name.encode())
        return checkpoint, data, manifest_file, parent

    def test_checkpoint_requires_full_train_count_and_parent_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, data, manifest_file, parent = self.fixture(root)
            check = lambda _m, _s: parent
            result = verify_checkpoint(
                checkpoint, root, root, data, manifest_file, parent_check=check
            )
            self.assertEqual(len(result["model_sha256"]), 64)
            metrics = json.loads((checkpoint / "training_metrics.json").read_text())
            metrics["requested_records"] = 3
            (checkpoint / "training_metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "dropped or truncated"):
                verify_checkpoint(
                    checkpoint, root, root, data, manifest_file, parent_check=check
                )

    def test_mock_collect_counts_invalid_and_rejects_stale_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, data, manifest_file, parent = self.fixture(root)
            verified = verify_checkpoint(
                checkpoint,
                root,
                root,
                data,
                manifest_file,
                parent_check=lambda _m, _s: parent,
            )
            prompts = root / "prompts.jsonl"
            write_rows(
                prompts,
                [
                    {
                        "id": "a",
                        "state": "s",
                        "questions": {
                            "q": {
                                "type": "choice",
                                "instructions": "pick",
                                "criteria": {"x": "X", "y": "Y"},
                            }
                        },
                    },
                    {
                        "id": "b",
                        "state": "s2",
                        "questions": {"q": {"type": "noul", "instructions": "yes?"}},
                    },
                ],
            )
            output = root / "predictions.jsonl"

            def decide(state, questions):
                return {
                    "answers": {
                        "q": (
                            {
                                "type": "choice",
                                "choice": "x",
                                "probabilities": {"x": 0.8, "y": 0.2},
                            }
                            if state == "s"
                            else {"type": "noul", "invalid_reason": "context_overflow"}
                        )
                    },
                    "status": "ok" if state == "s" else "context_overflow",
                }

            first = collect(
                prompts,
                output,
                verified,
                model_id="d2-kev",
                decide=decide,
                sync=lambda _: None,
                max_items=1,
            )
            self.assertFalse(first["complete"])
            final = collect(
                prompts,
                output,
                verified,
                model_id="d2-kev",
                decide=decide,
                sync=lambda _: None,
                resume=True,
            )
            self.assertEqual(
                (final["valid_questions"], final["invalid_questions"]), (1, 1)
            )
            self.assertTrue(final["complete"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(
                all(row["model_sha256"] == verified["model_sha256"] for row in rows)
            )
            rows[0]["model_sha256"] = "stale"
            write_rows(output, rows)
            with self.assertRaisesRegex(ValueError, "stale"):
                collect(
                    prompts,
                    output,
                    verified,
                    model_id="d2-kev",
                    decide=decide,
                    sync=lambda _: None,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
