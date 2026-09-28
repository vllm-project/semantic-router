from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.build_a0_variants import native_prompt
from v2.data.m2 import targets
from v2.data.replay_targets import collector_digest
from v2.data.tests.test_build_a0_variants import _row

MODEL, REV = "org/lux", "abc123"


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _receipt(prompt: dict, answer: dict | None, **extra) -> dict:
    return {
        "id": prompt["id"],
        "answers": {"decision": answer},
        "backend": "lux",
        "model_id": MODEL,
        "model_revision": REV,
        "revision_attested": True,
        "adapter_version": "native-v1",
        "model_config_sha256": "c" * 64,
        "runtime_matches_validated": True,
        "runtime_differences": {},
        "source_input_sha256": collector_digest(prompt),
        "native_error": None,
        **extra,
    }


class TargetsTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        rows = [_row(1, "choice"), _row(2, "noul"), _row(3, "score")]
        prompts = [native_prompt(r) for r in rows]
        over = {"kind": "native_input_over_budget", "question_id": "decision"}
        receipts = [
            _receipt(
                prompts[0],
                {
                    "type": "choice",
                    "probabilities": {"k0": 0.7, "k1": 0.1, "k2": 0.1, "k3": 0.1},
                },
            ),
            _receipt(prompts[1], {"type": "noul", "noul": 0.9}),
            _receipt(prompts[2], None, native_error=over),
        ]
        return {
            "rows": _jsonl(root / "rows.jsonl", rows + [_row(4, "noul")]),
            "prompts": _jsonl(root / "w.prompts.jsonl", prompts),
            "output": _jsonl(root / "w.jsonl", receipts),
            "receipts": receipts,
            "prompt_rows": prompts,
        }

    def _argv(self, root: Path, fx: dict, *extra: str) -> list[str]:
        return [
            "--rows",
            str(fx["rows"]),
            "--teacher-output",
            str(fx["output"]),
            "--teacher",
            "lux",
            "--model-id",
            MODEL,
            "--revision",
            REV,
            "--out",
            str(root / "t.jsonl"),
            "--report",
            str(root / "t.report.json"),
            *extra,
        ]

    def test_default_mode_keeps_the_m2_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            self.assertEqual(targets.main(self._argv(root, fx)), 0)
            out = _lines(root / "t.jsonl")
            self.assertEqual([t["id"] for t in out], ["r0001", "r0002"])
            self.assertEqual(set(out[0]), {"id", "input_sha256", "teacher_probs"})
            report = json.loads((root / "t.report.json").read_text())
            self.assertEqual((report["rows"], report["prompts"]), (2, 3))
            self.assertEqual(report["score"]["no_native_answer"], 1)
            self.assertNotIn("attestation_sha256", report)
            self.assertNotIn("no_target", report)

    def test_attestation_lists_every_prompt_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            prov = root / "prov.json"
            prov.write_text(
                json.dumps({"image_id": "img", "per_row": {"node": "node B", "gpu": 7}})
            )
            argv = self._argv(
                root,
                fx,
                "--prompts",
                str(fx["prompts"]),
                "--attestation",
                str(root / "t.attest.jsonl"),
                "--provenance",
                str(prov),
                "--wave",
                "w1",
            )
            self.assertEqual(targets.main(argv), 0)
            attest = _lines(root / "t.attest.jsonl")
            self.assertEqual([a["id"] for a in attest], ["r0001", "r0002", "r0003"])
            self.assertEqual([a["target"] for a in attest], [True, True, False])
            self.assertEqual(attest[2]["no_target_reason"], "native_input_over_budget")
            self.assertEqual(attest[2]["native_error"]["question_id"], "decision")
            self.assertEqual((attest[0]["node"], attest[0]["gpu"]), ("node B", 7))
            self.assertEqual(attest[0]["model_revision"], REV)
            self.assertEqual(
                attest[1]["source_input_sha256"],
                collector_digest(fx["prompt_rows"][1]),
            )
            report = json.loads((root / "t.report.json").read_text())
            self.assertEqual(report["no_target"], {"r0003": "native_input_over_budget"})
            self.assertEqual(report["wave"], "w1")
            self.assertEqual(report["provenance"]["image_id"], "img")
            self.assertEqual(
                report["attestation_sha256"],
                targets.file_sha256(root / "t.attest.jsonl"),
            )
            self.assertEqual(
                report["prompts_sha256"], targets.file_sha256(fx["prompts"])
            )

    def test_prompt_file_must_be_answered_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            _jsonl(fx["output"], fx["receipts"][:2])
            argv = self._argv(root, fx, "--prompts", str(fx["prompts"]))
            with self.assertRaises(ValueError):
                targets.main(argv)
            self.assertFalse((root / "t.jsonl").exists())
            extra = _receipt(native_prompt(_row(4, "noul")), None)
            _jsonl(fx["output"], fx["receipts"] + [extra])
            with self.assertRaises(ValueError):
                targets.main(argv)

    def test_rejects_a_sent_prompt_that_differs_from_its_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            tampered = [dict(p) for p in fx["prompt_rows"]]
            tampered[1]["state"] = "tampered"
            _jsonl(fx["prompts"], tampered)
            receipts = list(fx["receipts"])
            receipts[1] = _receipt(tampered[1], {"type": "noul", "noul": 0.9})
            _jsonl(fx["output"], receipts)
            with self.assertRaises(ValueError):
                targets.main(self._argv(root, fx, "--prompts", str(fx["prompts"])))

    def test_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            (root / "t.report.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                targets.main(self._argv(root, fx))
            self.assertFalse((root / "t.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
