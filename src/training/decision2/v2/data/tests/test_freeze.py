from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import stat
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from training.model.data import INPUT_FIELDS, canonical, digest
from v2.data import freeze

REGISTRY = {
    source: {
        "license": "CC-BY-4.0",
        "attribution": f"{source} authors",
        "evidence": f"https://example.invalid/{source}/LICENSE",
        "redistribution": "allowed with attribution",
    }
    for source in ("alpha", "beta")
}


def make_row(
    row_id: str,
    task_type: str,
    keys: list[str],
    label: int,
    *,
    group: str | None = None,
    source: str = "alpha",
    language: str = "en",
    split: str = "train",
    state: object = "some context",
) -> dict:
    row = {
        "id": row_id,
        "state": state,
        "instructions": {"question": f"question for {row_id}"},
        "options": [{"key": key, "description": f"option {key}"} for key in keys],
        "label": label,
        "task_type": task_type,
        "family": f"{task_type}_family",
        "group_id": group or f"group-{row_id}",
        "language": language,
        "split": split,
        "source": source,
        "evaluation_role": split,
        "render_template": "test",
        "audit_metadata": {"note": "x"},
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return row


def arm_rows(split: str = "train") -> list[dict]:
    levels3, levels5 = ["0", "1", "2"], ["0", "1", "2", "3", "4"]
    return [
        make_row("c1", "choice", ["A", "B"], 1, split=split, group="shared"),
        make_row("c2", "choice", ["A", "B", "C", "D"], 3, split=split, group="shared"),
        make_row(
            "c3", "choice", [f"k{i}" for i in range(6)], 0, split=split, source="beta"
        ),
        make_row(
            "n1", "noul", ["false", "true"], 1, split=split, language="zh", state="短"
        ),
        make_row("n2", "noul", ["true", "false"], 1, split=split),
        make_row(
            "n3", "noul", ["false", "true"], 1, split=split, state={"k": "v" * 40}
        ),
        make_row("s1", "score", levels3, 2, split=split),
        make_row("s2", "score", levels5, 4, split=split, source="beta"),
        make_row("s3", "score", list(reversed(levels5)), 0, split=split),
    ]


def write_jsonl(path: Path, rows: list[dict], *, pretty: bool = True) -> Path:
    lines = [json.dumps(row) if pretty else canonical(row) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def native_encode_importable() -> bool:
    try:
        importlib.import_module("training.model.decision_model")
    except ImportError:
        return False
    return True


class FreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_freeze(self, rows_path: Path, name: str, **overrides: object) -> dict:
        arguments = {
            "arm_id": "A6",
            "role": "train",
            "license_registry": REGISTRY,
            "tokenizers": [],
            "out_manifest": self.root / name,
        }
        arguments.update(overrides)
        return freeze.freeze(rows_path, **arguments)

    def test_content_hash_is_stable_across_order_and_formatting(self) -> None:
        rows = arm_rows()
        shuffled = list(rows)
        random.Random(4).shuffle(shuffled)
        source = write_jsonl(self.root / "arm.jsonl", shuffled)
        manifest = self.run_freeze(source, "m1.json")
        written = self.root / "arm.canonical.jsonl"
        expected = "".join(
            canonical(row) + "\n" for row in sorted(rows, key=lambda r: r["id"])
        )
        self.assertEqual(written.read_bytes(), expected.encode("utf-8"))
        self.assertEqual(
            manifest["content_sha256"], hashlib.sha256(written.read_bytes()).hexdigest()
        )
        self.assertFalse(manifest["input"]["is_canonical"])
        self.assertEqual(manifest["canonical_path"], str(written))
        self.assertEqual(stat.S_IMODE(os.stat(written).st_mode), 0o600)

        again = self.run_freeze(written, "m2.json")
        self.assertTrue(again["input"]["is_canonical"])
        self.assertEqual(again["content_sha256"], manifest["content_sha256"])
        self.assertEqual(again["input"]["sha256"], manifest["content_sha256"])
        self.assertFalse((self.root / "arm.canonical.canonical.jsonl").exists())

        other = write_jsonl(
            self.root / "other.jsonl", list(reversed(rows)), pretty=False
        )
        self.assertEqual(
            self.run_freeze(other, "m3.json")["content_sha256"],
            manifest["content_sha256"],
        )
        repeat = self.run_freeze(source, "m4.json")
        self.assertEqual(repeat["content_sha256"], manifest["content_sha256"])

    def test_manifest_counts(self) -> None:
        manifest = self.run_freeze(
            write_jsonl(self.root / "arm.jsonl", arm_rows()), "m.json"
        )
        self.assertEqual((manifest["rows"], manifest["groups"]), (9, 8))
        self.assertEqual(
            manifest["counts"]["task_type"], {"choice": 3, "noul": 3, "score": 3}
        )
        self.assertEqual(manifest["counts"]["language"], {"en": 8, "zh": 1})
        self.assertEqual(manifest["counts"]["source"], {"alpha": 7, "beta": 2})
        self.assertEqual(manifest["score"]["by_levels"], {"3": 1, "5": 2})
        self.assertEqual(
            manifest["score"]["by_levels_grade"], {"3": {"2": 1}, "5": {"4": 2}}
        )
        self.assertEqual(
            manifest["choice"]["by_option_bucket"], {"2": 1, "4": 1, "5-8": 1}
        )
        self.assertEqual(
            manifest["choice"]["gold_position_by_bucket"],
            {"2": {"1": 1}, "4": {"3": 1}, "5-8": {"0": 1}},
        )
        self.assertEqual(manifest["noul"]["labels"], {"false": 1, "true": 2})
        self.assertEqual(manifest["rows_per_group"], {"1": 7, "2": 1})
        self.assertEqual(manifest["state_chars"]["p0"], 1)
        self.assertEqual(
            manifest["state_chars"]["p100"], len(canonical({"k": "v" * 40}))
        )
        self.assertEqual(sorted(manifest["licenses"]), ["alpha", "beta"])
        self.assertEqual(stat.S_IMODE(os.stat(self.root / "m.json").st_mode), 0o600)

    def test_refuses_overwrite(self) -> None:
        source = write_jsonl(self.root / "arm.jsonl", arm_rows())
        self.run_freeze(source, "m.json")
        with self.assertRaises(FileExistsError):
            self.run_freeze(source, "m.json")
        (self.root / "arm.canonical.jsonl").write_text("different\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_freeze(source, "fresh.json")
        self.assertFalse((self.root / "fresh.json").exists())

    def test_validation_failures(self) -> None:
        cases = {
            "duplicate": arm_rows() + [arm_rows()[0]],
            "hash": [dict(arm_rows()[0], input_sha256="0" * 64)],
            "group": [dict(arm_rows()[0], group_id="")],
            "teacher": [dict(arm_rows()[0], teacher_probs={"A": 0.5, "B": 0.5})],
        }
        for name, rows in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.run_freeze(
                    write_jsonl(self.root / f"{name}.jsonl", rows), f"{name}.json"
                )
        with self.assertRaisesRegex(ValueError, "split=select"):
            self.run_freeze(
                write_jsonl(self.root / "aho.jsonl", arm_rows()), "aho.json", role="aho"
            )
        aho = self.run_freeze(
            write_jsonl(self.root / "aho_ok.jsonl", arm_rows("select")),
            "aho_ok.json",
            role="aho",
        )
        self.assertEqual(aho["partition"], "select")
        with self.assertRaisesRegex(ValueError, "teacher_probs"):
            self.run_freeze(
                write_jsonl(self.root / "replay.jsonl", arm_rows()),
                "r.json",
                role="replay",
            )
        replay = [
            dict(
                row,
                teacher_probs={
                    o["key"]: 1 / len(row["options"]) for o in row["options"]
                },
            )
            for row in arm_rows()
        ]
        manifest = self.run_freeze(
            write_jsonl(self.root / "replay_ok.jsonl", replay),
            "r_ok.json",
            role="replay",
        )
        self.assertEqual(manifest["role"], "replay")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            path = self.root / "dupkey.jsonl"
            path.write_text('{"id": "a", "id": "b"}\n', encoding="utf-8")
            self.run_freeze(path, "dupkey.json")

    def test_license_registry_must_cover_every_source(self) -> None:
        source = write_jsonl(self.root / "arm.jsonl", arm_rows())
        with self.assertRaisesRegex(ValueError, "beta"):
            self.run_freeze(
                source, "a.json", license_registry={"alpha": REGISTRY["alpha"]}
            )
        partial = {
            "alpha": REGISTRY["alpha"],
            "beta": dict(REGISTRY["beta"], evidence=""),
        }
        with self.assertRaisesRegex(ValueError, "beta"):
            self.run_freeze(source, "b.json", license_registry=partial)
        registry = self.root / "registry.json"
        registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
        manifest = self.run_freeze(source, "c.json", license_registry=registry)
        self.assertEqual(
            manifest["license_registry_sha256"],
            hashlib.sha256(registry.read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["licenses"]["beta"], REGISTRY["beta"])

    def test_raw_tokenizer_counts(self) -> None:
        rows = arm_rows()
        spec = {"name": "ws", "path": str(self.root), "revision": "r1", "kind": "raw"}
        manifest = self.run_freeze(
            write_jsonl(self.root / "arm.jsonl", rows),
            "m.json",
            tokenizers=[spec],
            tokenizer_loader=lambda _: WhitespaceTokenizer(),
        )
        counts = {row["id"]: len(freeze.raw_text(row).split()) for row in rows}
        tokens = manifest["tokens"][0]
        self.assertEqual(tokens["total"], sum(counts.values()))
        self.assertEqual(
            tokens["padded8_total"], sum(-(-n // 8) * 8 for n in counts.values())
        )
        self.assertEqual(tokens["by_language"]["zh"], counts["n1"])
        self.assertEqual(
            tokens["by_task_type"]["score"], counts["s1"] + counts["s2"] + counts["s3"]
        )
        self.assertEqual(
            (tokens["name"], tokens["revision"], tokens["kind"]), ("ws", "r1", "raw")
        )
        with self.assertRaises(ValueError):
            self.run_freeze(
                self.root / "arm.jsonl", "bad.json", tokenizers=[dict(spec, kind="bpe")]
            )

    def test_native_decoder_uses_repo_encode(self) -> None:
        calls = []

        def encode(row: dict, tokenizer: object, max_length: int) -> dict:
            calls.append((row["id"], max_length))
            return {"ids": list(range(len(row["options"]) + 10))}

        fake = types.ModuleType("training.model.decision_model")
        fake.encode = encode
        with mock.patch.dict(sys.modules, {"training.model.decision_model": fake}):
            manifest = self.run_freeze(
                write_jsonl(self.root / "arm.jsonl", arm_rows()),
                "m.json",
                tokenizers=[
                    {
                        "name": "native",
                        "path": "/nonexistent",
                        "revision": None,
                        "kind": "native_decoder",
                    }
                ],
                tokenizer_loader=lambda _: object(),
            )
        self.assertEqual(
            sorted(row_id for row_id, _ in calls),
            sorted(row["id"] for row in arm_rows()),
        )
        self.assertTrue(all(limit >= 2**31 for _, limit in calls))
        self.assertEqual(
            manifest["tokens"][0]["total"],
            sum(len(row["options"]) + 10 for row in arm_rows()),
        )

    @unittest.skipUnless(native_encode_importable(), "torch is not installed")
    def test_native_decoder_with_real_encode(self) -> None:
        from training.model.decision_model import encode

        rows = arm_rows()
        manifest = self.run_freeze(
            write_jsonl(self.root / "arm.jsonl", rows),
            "m.json",
            tokenizers=[
                {"name": "ws", "path": "x", "revision": None, "kind": "native_decoder"}
            ],
            tokenizer_loader=lambda _: WhitespaceTokenizer(),
        )
        expected = sum(
            len(encode(row, WhitespaceTokenizer(), 10**9)["ids"]) for row in rows
        )
        self.assertEqual(manifest["tokens"][0]["total"], expected)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is not None, "torch is installed"
    )
    def test_native_decoder_without_torch_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "torch"):
            self.run_freeze(
                write_jsonl(self.root / "arm.jsonl", arm_rows()),
                "m.json",
                tokenizers=[
                    {
                        "name": "native",
                        "path": "x",
                        "revision": None,
                        "kind": "native_decoder",
                    }
                ],
                tokenizer_loader=lambda _: object(),
            )
        self.assertFalse((self.root / "m.json").exists())

    @unittest.skipIf(
        importlib.util.find_spec("transformers") is not None,
        "transformers is installed",
    )
    def test_default_loader_without_transformers_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "transformers"):
            self.run_freeze(
                write_jsonl(self.root / "arm.jsonl", arm_rows()),
                "m.json",
                tokenizers=[
                    {"name": "raw", "path": "x", "revision": None, "kind": "raw"}
                ],
            )


class IsolationTest(unittest.TestCase):
    def test_disjoint_partitions_pass(self) -> None:
        train = [
            make_row("t1", "noul", ["false", "true"], 0),
            make_row("t2", "noul", ["false", "true"], 1),
        ]
        held = [make_row("h1", "noul", ["false", "true"], 0, split="select")]
        report = freeze.isolation_check({"train/A6": train, "aho/A6": held})
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["intersections"], [])

    def test_each_kind_of_intersection_fails(self) -> None:
        base = make_row("t1", "noul", ["false", "true"], 0, group="g1")
        cases = {
            "id": make_row(
                "t1", "noul", ["false", "true"], 1, group="other", state="changed"
            ),
            "group_id": make_row(
                "s9", "noul", ["false", "true"], 0, group="g1", state="changed"
            ),
            "input_sha256": dict(base, id="s8", group_id="g8", label=1, split="select"),
        }
        for field, row in cases.items():
            with self.subTest(field):
                with self.assertRaises(freeze.IsolationError) as caught:
                    freeze.isolation_check(
                        {"train/A6": [base], "select/SELECT700": [row]}
                    )
                report = caught.exception.report
                self.assertEqual(report["verdict"], "FAIL")
                self.assertIn(field, {item["field"] for item in report["failures"]})
                self.assertEqual(report["failures"][0]["a"], "select/SELECT700")

    def test_same_kind_overlap_is_reported_not_failed(self) -> None:
        rows = [make_row("t1", "noul", ["false", "true"], 0)]
        report = freeze.isolation_check({"train/A0": rows, "train/A0p": rows})
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(len(report["intersections"]), 3)
        self.assertTrue(all(item["allowed"] for item in report["intersections"]))
        with self.assertRaises(freeze.IsolationError):
            freeze.isolation_check({"TRAIN": rows, "CAL": rows})

    def test_recorded_input_hash_must_match(self) -> None:
        row = dict(make_row("t1", "noul", ["false", "true"], 0), input_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            freeze.isolation_check({"a": [row], "b": []})

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = write_jsonl(root / "train.jsonl", arm_rows())
            select = write_jsonl(
                root / "select.jsonl",
                [make_row("c1", "noul", ["false", "true"], 0, split="select")],
            )
            report = root / "isolation.json"
            with redirect_stdout(StringIO()):
                status = freeze.main(
                    [
                        "isolation",
                        "--partition",
                        f"train/A6={train}",
                        "--partition",
                        f"select/S={select}",
                        "--report",
                        str(report),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(
                json.loads(report.read_text(encoding="utf-8"))["verdict"], "FAIL"
            )
            self.assertEqual(stat.S_IMODE(os.stat(report).st_mode), 0o600)
            registry = root / "registry.json"
            registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
            manifest = root / "manifest.json"
            with redirect_stdout(StringIO()):
                status = freeze.main(
                    [
                        "freeze",
                        "--rows",
                        str(train),
                        "--arm-id",
                        "A6",
                        "--role",
                        "train",
                        "--license-registry",
                        str(registry),
                        "--out-manifest",
                        str(manifest),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["arm_id"], "A6"
            )


if __name__ == "__main__":
    unittest.main()
