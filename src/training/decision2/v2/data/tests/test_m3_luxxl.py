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


def _summary(n: int) -> dict:
    return {
        "backend": "lux",
        "model_id": luxxl.LUX_ID,
        "revision": luxxl.LUX_REVISION,
        "adapter_version": luxxl.ADAPTER,
        "revision_attested": True,
        "runtime_matches_validated": True,
        "runtime_differences": {},
        "previously_completed": 0,
        "input_items": n,
        "collected_now": n,
        "over_budget_rows_now": 0,
        "model_config_sha256": "c" * 64,
    }


class ProvenanceTest(unittest.TestCase):
    def _argv(
        self,
        root: Path,
        *,
        summary: dict | None = None,
        rc: int = 0,
        repeat: dict | None = None,
        cache: list[dict] | None = None,
    ) -> list:
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
        argv = [
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
        if repeat is not None:
            (root / "repeat.json").write_text(json.dumps(repeat))
            argv += ["--repeat-check", str(root / "repeat.json")]
        if cache is not None:
            argv += ["--triton-cache-checks", str(_jsonl(root / "cache.jsonl", cache))]
        return argv

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

    def test_embeds_a_passed_repeat_and_an_unchanged_cache(self) -> None:
        frozen = {"files": 5, "tree_sha256": "t" * 64}
        cache = [
            dict(frozen, stage=s) for s in ("before", "after_wave", "after_repeat")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repeat = {"wave": "lux-xl-w1", "pass": True, "prompts": 2}
            argv = self._argv(root, repeat=repeat, cache=cache)
            self.assertEqual(luxxl.main(argv), 0)
            prov = json.loads((root / "prov.json").read_text())
            self.assertEqual(prov["repeat_check"], repeat)
            self.assertEqual(len(prov["teacher_run"]["triton_cache"]["checks"]), 3)

    def test_rejects_a_changed_cache_or_a_failed_repeat(self) -> None:
        frozen = {"files": 5, "tree_sha256": "t" * 64}
        passed = {"wave": "lux-xl-w1", "pass": True}
        for kwargs in (
            {"repeat": passed, "cache": [frozen, dict(frozen, files=6)]},
            {"repeat": passed, "cache": []},
            {"repeat": dict(passed, **{"pass": False}), "cache": [frozen]},
            {"repeat": dict(passed, wave="lux-xl-w2"), "cache": [frozen]},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    luxxl.main(self._argv(Path(tmp), **kwargs))


class RepeatTest(unittest.TestCase):
    def _files(self, root: Path, again: dict[str, dict] | None = None) -> list:
        ids = [f"p{i}" for i in range(10)]
        wave = _jsonl(
            root / "lux-xl-w1.prompts.jsonl",
            [{"id": i, "state": i, "questions": {}} for i in ids],
        )
        answers = {
            i: {"decision": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}}
            for i in ids
        }
        output = _jsonl(
            root / "lux-xl-w1.jsonl", [{"id": i, "answers": answers[i]} for i in ids]
        )
        prompts = root / "lux-xl-w1-r4.prompts.jsonl"
        self.assertEqual(
            luxxl.main(
                [
                    "repeat-prompts",
                    "--prompts",
                    str(wave),
                    "--n",
                    "4",
                    "--out",
                    str(prompts),
                ]
            ),
            0,
        )
        chosen = luxxl._ids(prompts)
        guard = root / "guard.json"
        guard.write_text(
            json.dumps(
                {"pass": True, "prompts": 4, "prompts_sha256": file_sha256(prompts)}
            )
        )
        log = root / "r4.log"
        log.write_text(json.dumps(_summary(4)) + "\n")
        teach = root / "teach.json"
        teach.write_text(
            json.dumps(
                {
                    "teacher": "lux",
                    "input": str(prompts),
                    "rc": 0,
                    "wall_s": 36,
                    "end_utc": "t",
                }
            )
        )
        repeated = dict(answers, **(again or {}))
        rerun = _jsonl(
            root / "lux-xl-w1-r4.jsonl",
            [{"id": i, "answers": repeated[i]} for i in chosen],
        )
        return [
            "repeat",
            "--wave",
            "lux-xl-w1",
            "--wave-prompts",
            str(wave),
            "--wave-output",
            str(output),
            "--prompts",
            str(prompts),
            "--guard",
            str(guard),
            "--output",
            str(rerun),
            "--collector-log",
            str(log),
            "--teach-line",
            str(teach),
            "--n",
            "4",
            "--out",
            str(root / "repeat.json"),
        ]

    def test_selection_is_hash_ordered_byte_identical_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root)
            wave = (root / "lux-xl-w1.prompts.jsonl").read_bytes().splitlines(True)
            chosen = (root / "lux-xl-w1-r4.prompts.jsonl").read_bytes().splitlines(True)
            ids = [json.loads(line)["id"] for line in chosen]
            self.assertEqual(ids, sorted(ids))
            self.assertTrue(set(chosen) <= set(wave))
            expected = sorted(
                luxxl.ranked([f"p{i}" for i in range(10)], luxxl.REPEAT_SALT)[:4]
            )
            self.assertEqual(ids, expected)

    def test_identical_rerun_passes_bitwise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(luxxl.main(self._files(root)), 0)
            text = (root / "repeat.json").read_text()
            self.assertNotIn(tmp, text)
            receipt = json.loads(text)
            self.assertTrue(receipt["pass"])
            self.assertTrue(receipt["bitwise_identical"])
            self.assertEqual(receipt["compare"]["questions"], 4)
            self.assertEqual(receipt["m2_repeat_max_abs_diff"]["max_abs_diff"], 0.0)

    def test_small_drift_passes_and_a_flip_fails(self) -> None:
        cases = (
            ({"a": 0.7005, "b": 0.2995}, True),
            ({"a": 0.4, "b": 0.6}, False),
            ({"a": 0.69, "b": 0.31}, False),
        )
        for probs, passed in cases:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = sorted(
                    luxxl.ranked([f"p{i}" for i in range(10)], luxxl.REPEAT_SALT)[:4]
                )[0]
                again = {
                    first: {"decision": {"type": "choice", "probabilities": probs}}
                }
                self.assertEqual(
                    luxxl.main(self._files(root, again)), 0 if passed else 3
                )
                receipt = json.loads((root / "repeat.json").read_text())
                self.assertEqual(receipt["pass"], passed)
                self.assertFalse(receipt["bitwise_identical"])

    def test_refuses_prompts_that_are_not_the_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self._files(root)
            prompts = root / "lux-xl-w1-r4.prompts.jsonl"
            prompts.write_bytes(prompts.read_bytes().splitlines(True)[0] * 4)
            with self.assertRaises(ValueError):
                luxxl.main(argv)


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
            luxxl.require_full(out, ["full"])
            for recipe in ("ctrl", "absent"):
                with self.assertRaises(ValueError):
                    luxxl.require_full(out, [recipe])


class ControlIdsTest(unittest.TestCase):
    def test_control_rows_outside_both_recipes_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lists = {
                "mx-xl-full": "ab",
                "mx-xl-short": "bc",
                "cx-xl-a7v1-full": "azd",
                "cx-xl-a7v1-short": "d",
                "cx-xl-v2v1-full": "cy",
                "cx-xl-v2v1-short": "",
            }
            for name, ids in lists.items():
                _jsonl(root / f"{name}.lux1.missing.jsonl", [{"id": i} for i in ids])
            self.assertEqual(luxxl.control_only_ids(root), ["d", "y", "z"])
            out = root / "c.jsonl"
            argv = ["control-ids", "--missing-dir", str(root), "--out", str(out)]
            self.assertEqual(luxxl.main(argv), 0)
            self.assertEqual(out.read_text().splitlines()[0], '{"id": "d"}')


if __name__ == "__main__":
    unittest.main()
