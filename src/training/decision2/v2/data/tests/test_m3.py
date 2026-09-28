from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.build_a0_variants import native_prompt
from v2.data.m3 import guard, qualify, shards, teacher_targets
from v2.data.replay_targets import collector_digest
from v2.data.tests.test_build_a0_variants import _row

FROZEN = {"entries": 7, "sha256": "a" * 64}
PACKAGE = {"native_model_sha256": "n" * 64, "runtime_source_sha256": "s" * 64}


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _receipt(prompt: dict, answer: dict) -> dict:
    return {
        "id": prompt["id"],
        "answers": {"decision": answer},
        "source_input_sha256": collector_digest(prompt),
        "runtime_qualification": qualify.PENDING,
        **qualify.IDENTITY,
        **PACKAGE,
    }


def _answer(row: dict, p: float) -> dict:
    if row["task_type"] == "noul":
        return {"type": "noul", "noul": p}
    keys = [o["key"] for o in row["options"]]
    rest = (1.0 - p) / (len(keys) - 1)
    return {"type": row["task_type"], "probabilities": {k: (p if i == 0 else rest) for i, k in enumerate(keys)}}


class QualifyTest(unittest.TestCase):
    def test_draw_is_disjoint_and_stratified(self) -> None:
        kinds = {f"id{i:04d}": qualify.TYPES[i % 3] for i in range(3000)}
        tokens = {ident: (5000 if i % 7 == 0 else 100) for i, ident in enumerate(kinds)}
        repeat, warm = qualify.draw(kinds, tokens)
        self.assertEqual(len(repeat), 3 * qualify.REPEAT_PER_TYPE + qualify.REPEAT_LONG)
        self.assertEqual(len(warm), 3 * qualify.WARM_PER_TYPE + qualify.WARM_LONG)
        self.assertFalse(set(repeat) & set(warm))
        self.assertGreaterEqual(sum(tokens[i] >= qualify.LONG_TOKENS for i in repeat), qualify.REPEAT_LONG)
        self.assertEqual(qualify.draw(kinds, tokens), (repeat, warm))

    def test_compare_counts_drift_flips_and_validity(self) -> None:
        a = {
            "x": {"q": {"type": "choice", "probabilities": {"a": 0.6, "b": 0.4}}},
            "y": {"q": {"type": "noul", "noul": 0.2}},
            "z": {"q": {"type": "score", "error": "context_overflow"}},
        }
        same = qualify.compare(a, json.loads(json.dumps(a)), ["x", "y", "z"])
        self.assertEqual((same["identical_answers"], same["max_drift"], same["validity_mismatches"]), (3, 0.0, 0))
        b = json.loads(json.dumps(a))
        b["x"]["q"]["probabilities"] = {"a": 0.4, "b": 0.6}
        b["y"]["q"]["noul"] = 0.2005
        b["z"]["q"] = {"type": "score", "probabilities": {"0": 1.0, "1": 0.0}}
        diff = qualify.compare(a, b, ["x", "y", "z", "missing"])
        self.assertEqual(diff["argmax_mismatches"], 1)
        self.assertEqual(diff["validity_mismatches"], 1)
        self.assertEqual(diff["missing_prompts"], 1)
        self.assertAlmostEqual(diff["max_drift"], 0.2)
        self.assertEqual(diff["over_1e-3"], 1)

    def test_identity_check_requires_pins_and_fla(self) -> None:
        prompt = native_prompt(_row(1, "noul"))
        receipt = _receipt(prompt, {"type": "noul", "noul": 0.5})
        manifest = {
            "exit_code": 0,
            "image_id": qualify.IMAGE_ID,
            "fla_reference_fallback": False,
            "collector": {"loaded_parameters": qualify.LOADED_PARAMETERS},
        }
        self.assertTrue(qualify.identity_check([receipt], manifest, 1)["pass"])
        self.assertFalse(qualify.identity_check([receipt], dict(manifest, fla_reference_fallback=True), 1)["pass"])
        wrong = dict(receipt, model_revision="main")
        self.assertFalse(qualify.identity_check([wrong], manifest, 1)["pass"])


class ShardsTest(unittest.TestCase):
    def test_split_and_merge_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = _jsonl(root / "p.jsonl", [{"id": f"id{i}", "state": i, "questions": {}} for i in range(50)])
            parts = shards.split(prompts, 3, str(root / "w"))
            self.assertEqual(sum(v["rows"] for v in parts.values()), 50)
            outputs = []
            for k, name in enumerate(sorted(parts)):
                rows = [json.loads(line) for line in Path(name).read_text().splitlines()]
                self.assertTrue(all(shards.shard_of(r["id"], 3) == k for r in rows))
                outputs.append(_jsonl(root / f"o{k}.jsonl", [{"id": r["id"], "answers": {}} for r in rows]))
            merged = shards.merge(prompts, outputs, root / "m.jsonl")
            self.assertEqual(merged["rows"], 50)
            ids = [json.loads(line)["id"] for line in (root / "m.jsonl").read_text().splitlines()]
            self.assertEqual(ids, sorted(ids))
            with self.assertRaises(ValueError):
                shards.merge(prompts, outputs[:2], root / "m2.jsonl")


class GuardTest(unittest.TestCase):
    def test_guard_refuses_protected_and_non_train_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_row(i, "choice") for i in range(1, 5)]
            rows = _jsonl(root / "rows.jsonl", train)
            clean = _jsonl(root / "clean.jsonl", [native_prompt(r) for r in train])
            protected = _jsonl(root / "sel.jsonl", [_row(9, "noul")])
            ok = guard.guard(clean, rows, [protected], [])
            self.assertTrue(ok["pass"])
            leaked = dict(train[1], id="r0009x")
            panel = _jsonl(root / "panel.jsonl", [dict(native_prompt(leaked), id="panel-1")])
            hit = guard.guard(clean, rows, [protected], [panel])
            self.assertFalse(hit["pass"])
            self.assertEqual(hit["shared_prompt_digest"], 1)
            unknown = _jsonl(root / "unknown.jsonl", [native_prompt(_row(7, "score"))])
            self.assertEqual(guard.guard(unknown, rows, [], [])["not_a_row"], 1)


class TeacherTargetsTest(unittest.TestCase):
    def _fixture(self, root: Path, *, frozen_after: dict | None = None):
        train = [_row(1, "choice"), _row(2, "noul"), _row(3, "score")]
        rows = _jsonl(root / "rows.jsonl", train)
        prompts = [native_prompt(r) for r in train]
        prompt_file = _jsonl(root / "wave.prompts.jsonl", prompts)
        output = _jsonl(root / "s0.jsonl", [_receipt(p, _answer(r, 0.7)) for p, r in zip(prompts, train)])
        manifest = {
            "label": "aj-t-s0",
            "gpu": 2,
            "exit_code": 0,
            "image_id": qualify.IMAGE_ID,
            "fla_reference_fallback": False,
            "autotune_before": FROZEN,
            "autotune_after": frozen_after or FROZEN,
            "output_sha256": teacher_targets.file_sha256(output),
            "output_rows": 3,
            "input_sha256": "i" * 64,
            "mirror": {"commit": "c" * 40},
            "start_utc": "t0",
            "end_utc": "t1",
            "gpu_hours": 0.01,
            "collector": {"loaded_parameters": qualify.LOADED_PARAMETERS},
        }
        manifest_path = root / "s0.manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        qualification = root / "q.json"
        qualification.write_text(
            json.dumps({"pass": True, "autotune_frozen": FROZEN, "package_hashes": {k: [v] for k, v in PACKAGE.items()}})
        )
        guard_file = root / "guard.json"
        guard_file.write_text(json.dumps({"pass": True, "prompts_sha256": teacher_targets.file_sha256(prompt_file)}))
        argv = [
            "--wave", "aj-t", "--rows", str(rows), "--prompts", str(prompt_file),
            "--qualification", str(qualification), "--shard", f"{output}={manifest_path}",
            "--guard", str(guard_file), "--out", str(root / "t.jsonl"), "--report", str(root / "t.report.json"),
        ]
        return argv

    def test_converts_qualified_receipts_with_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(teacher_targets.main(self._fixture(root)), 0)
            targets = [json.loads(line) for line in (root / "t.jsonl").read_text().splitlines()]
            self.assertEqual([t["id"] for t in targets], ["r0001", "r0002", "r0003"])
            self.assertEqual(set(targets[0]), {"id", "input_sha256", "teacher_probs"})
            report = json.loads((root / "t.report.json").read_text())
            self.assertIn("closed OpenAI model", report["provenance_caveat"])
            self.assertEqual(report["rows"], 3)

    def test_rejects_changed_autotune_or_failed_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self._fixture(root, frozen_after={"entries": 8, "sha256": "b" * 64})
            with self.assertRaises(ValueError):
                teacher_targets.main(argv)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self._fixture(root)
            (root / "q.json").write_text(json.dumps({"pass": False}))
            with self.assertRaises(ValueError):
                teacher_targets.main(argv)


if __name__ == "__main__":
    unittest.main()
