from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.m2.targets import file_sha256
from v2.data.m3 import luxxl


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class ProvenanceTest(unittest.TestCase):
    def _argv(self, root: Path, *, summary: dict | None = None, rc: int = 0) -> list:
        prompts = _jsonl(
            root / "lux-xl-w1.prompts.jsonl",
            [{"id": f"p{i}", "state": "s", "questions": {}} for i in range(3)],
        )
        guard = root / "guard.json"
        guard.write_text(
            json.dumps(
                {
                    "pass": True,
                    "prompts": 3,
                    "prompts_sha256": file_sha256(prompts),
                    "protected_files": {"a.jsonl": 1, "b.jsonl": 2},
                }
            )
        )
        base = {
            "backend": "lux",
            "model_id": luxxl.LUX_ID,
            "revision": luxxl.LUX_REVISION,
            "adapter_version": luxxl.ADAPTER,
            "revision_attested": True,
            "runtime_matches_validated": True,
            "runtime_differences": {},
            "previously_completed": 0,
            "input_items": 3,
            "collected_now": 3,
            "over_budget_rows_now": 0,
            "model_config_sha256": "c" * 64,
            "output": "/data/somewhere/lux-xl-w1.jsonl",
        }
        log = root / "w1.log"
        log.write_text(
            f"warning\n[transformers] {luxxl.CONV1D_FALLBACK}.\n"
            + json.dumps(dict(base, **(summary or {})))
            + "\n"
        )
        teach = root / "teach.json"
        teach.write_text(
            json.dumps(
                {
                    "teacher": "lux",
                    "input": str(prompts),
                    "rc": rc,
                    "wall_s": 3600,
                    "end_utc": "t",
                }
            )
        )
        return [
            "provenance",
            "--wave",
            "lux-xl-w1",
            "--prompts",
            str(prompts),
            "--guard",
            str(guard),
            "--collector-log",
            str(log),
            "--teach-line",
            str(teach),
            "--image-id",
            luxxl.IMAGE_ID,
            "--launcher-sha256",
            "l" * 64,
            "--queue-sha256",
            "q" * 64,
            "--teach-mirror",
            "m" * 40,
            "--triton-cache-files",
            "5",
            "--triton-cache-sha256",
            "t" * 64,
            "--out",
            str(root / "prov.json"),
        ]

    def test_records_a_clean_run_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(luxxl.main(self._argv(root)), 0)
            text = (root / "prov.json").read_text()
            self.assertNotIn("/data/", text)
            prov = json.loads(text)
            run = prov["teacher_run"]
            self.assertFalse(run["causal_conv1d_kernel"])
            self.assertEqual(run["gpu_hours"], 1.0)
            self.assertEqual(prov["guard"]["protected_files"], 2)
            self.assertEqual(prov["per_row"]["node"], "node B")

    def test_rejects_partial_or_unattested_runs(self) -> None:
        for kwargs in (
            {"summary": {"collected_now": 2}},
            {"summary": {"previously_completed": 1}},
            {"summary": {"revision_attested": False}},
            {"summary": {"runtime_matches_validated": False}},
            {"rc": 1},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    luxxl.main(self._argv(Path(tmp), **kwargs))


class CoverageTest(unittest.TestCase):
    def test_counts_rows_gained_per_wave_and_rows_left(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _jsonl(root / "full.lux1.missing.jsonl", [{"id": i} for i in "abcd"])
            _jsonl(root / "ctrl.lux1.missing.jsonl", [{"id": i} for i in "cdxy"])
            w1 = _jsonl(root / "w1.jsonl", [{"id": "a"}, {"id": "c"}])
            w2 = _jsonl(root / "w2.jsonl", [{"id": "b"}, {"id": "d"}])

            def spec(rows: int, with_targets: int) -> dict:
                lux = {"rows_with_targets": with_targets, "rows_without": 4}
                return {"rows": rows, "sha256": "s", "coverage": {"lux1": lux}}

            manifest = {"recipes": {"full": spec(10, 6), "ctrl": spec(8, 4)}}
            out = luxxl.coverage(manifest, root, [("w1", w1), ("w2", w2)])
            full, ctrl = out["recipes"]["full"], out["recipes"]["ctrl"]
            self.assertEqual([s["gained"] for s in full["after_wave"]], [2, 2])
            self.assertEqual(full["after_wave"][-1]["rows_with_targets"], 10)
            self.assertEqual(full["left_without_target"], 0)
            self.assertEqual([s["gained"] for s in ctrl["after_wave"]], [1, 1])
            self.assertEqual(ctrl["left_without_target"], 2)
            self.assertEqual(ctrl["after_wave"][-1]["share"], 0.75)


if __name__ == "__main__":
    unittest.main()
