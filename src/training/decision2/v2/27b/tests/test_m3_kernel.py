import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

kernel = importlib.import_module("v2.27b.typed_collect_kernel")
readout = importlib.import_module("v2.27b.kernel_readout")
cache = importlib.import_module("v2.27b.triton_cache")
runtime_check = importlib.import_module("v2.dec.runtime_check")
adapters = importlib.import_module("v2.eval.adapters")
calibration = importlib.import_module("training.model.calibration")

HERE = Path(__file__).resolve().parents[1]
SHA = "a" * 64
BINDINGS = {
    "torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule",
    "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface.causal_conv1d_update",
}


def identity(cache_dir):
    return {
        "kernel_bindings": dict(BINDINGS),
        "fla_path": "/opt/decision-fla/fla",
        "flash_linear_attention": "0.5.2",
        "causal_conv1d": "1.7.0",
        "triton": "3.7.1",
        "triton_cache_dir": cache_dir,
        "triton_cache_autotuning": "1",
    }


class SiteTest(unittest.TestCase):
    def test_fla_overlay_is_kept_and_inserted_before_site_packages(self):
        path = [
            "",
            "/src",
            "/usr/lib/python3.12",
            "/usr/local/lib/python3.12/site-packages",
        ]
        self.assertTrue(kernel.ensure_site(path, "/opt/decision-fla"))
        self.assertEqual(path[3], "/opt/decision-fla")
        self.assertFalse(kernel.ensure_site(path, "/opt/decision-fla"))
        self.assertEqual(path.count("/opt/decision-fla"), 1)

    def test_import_does_not_strip_the_overlay(self):
        sys.path.append("/nonexistent/decision-fla")
        try:
            importlib.reload(kernel)
            self.assertIn("/nonexistent/decision-fla", sys.path)
        finally:
            sys.path.remove("/nonexistent/decision-fla")

    def test_fla_must_resolve_inside_the_overlay(self):
        spec = types.SimpleNamespace(origin="/opt/decision-fla/fla/__init__.py")
        with mock.patch("importlib.util.find_spec", return_value=spec):
            self.assertEqual(kernel.fla_origin("/opt/decision-fla"), spec.origin)
        shadow = types.SimpleNamespace(origin="/code/fla/__init__.py")
        with mock.patch("importlib.util.find_spec", return_value=shadow):
            with self.assertRaises(SystemExit):
                kernel.fla_origin("/opt/decision-fla")
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(SystemExit):
                kernel.fla_origin("/opt/decision-fla")


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cache = self.tmp / "triton-cache"
        self.cache.mkdir()
        for target, value in (
            ("ensure_site", False),
            ("fla_origin", "/opt/decision-fla/fla/__init__.py"),
        ):
            patcher = mock.patch.object(kernel, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_require_runtime_failure_aborts(self):
        with mock.patch.object(
            runtime_check, "require_runtime", side_effect=RuntimeError("not fla")
        ) as required:
            with self.assertRaises(SystemExit):
                kernel.kernel_runtime()
        required.assert_called_once()

    def test_cache_dir_must_exist(self):
        with mock.patch.object(
            runtime_check,
            "require_runtime",
            return_value=identity(str(self.tmp / "missing")),
        ):
            with self.assertRaises(SystemExit):
                kernel.kernel_runtime()

    def argv(self, output, *extra):
        prompts = self.tmp / "prompts.jsonl"
        prompts.write_text(
            "".join(json.dumps({"id": f"p{i}"}) + "\n" for i in range(3)),
            encoding="utf-8",
        )
        return [
            "--checkpoint", "ckpt", "--source-path", "base", "--model-id", "m",
            "--model-revision", "r", "--max-length", "32768", "--calibration", "c.json",
            "--input", str(prompts), "--output", str(output), *extra,
        ]  # fmt: skip

    def run_main(self, argv, after=None, require=None):
        calls = []

        def fake_infer():
            calls.append(list(sys.argv))
            out = Path(sys.argv[sys.argv.index("--output") + 1])
            out.write_text("{}\n", encoding="utf-8")

        ident = identity(str(self.cache))
        after = after or ident
        with (
            mock.patch.object(
                runtime_check, "require_runtime", return_value=ident, **(require or {})
            ),
            mock.patch.object(runtime_check, "runtime_identity", return_value=after),
            mock.patch.object(kernel.infer, "main", side_effect=fake_infer),
            mock.patch.dict(sys.modules, {"fla": types.ModuleType("fla")}),
            mock.patch.object(sys, "argv", ["x"]),
        ):
            kernel.main(argv)
        return calls

    def test_sidecar_records_the_kernel_runtime(self):
        output = self.tmp / "out" / "typed-final.predictions.jsonl"
        output.parent.mkdir()
        calls = self.run_main(self.argv(output, "--max-items", "2"))
        self.assertEqual(len(calls), 1)
        subset = Path(calls[0][calls[0].index("--input") + 1])
        self.assertEqual(len(subset.read_text().splitlines()), 2)
        self.assertNotIn("--max-items", calls[0])
        side = json.loads(
            output.with_name(output.name + ".runtime.json").read_text("utf-8")
        )
        self.assertTrue(side["fla_loaded"])
        self.assertEqual(side["kernel_bindings"], BINDINGS)
        self.assertEqual(side["kernel_bindings_after"], BINDINGS)
        self.assertEqual(side["triton_cache_dir"], str(self.cache))
        self.assertEqual(side["triton_cache_autotuning"], "1")
        self.assertEqual(side["max_length"], 32768)
        self.assertEqual(side["max_items"], 2)
        self.assertEqual(side["versions"]["flash_linear_attention"], "0.5.2")
        self.assertEqual(side["versions"]["causal_conv1d"], "1.7.0")
        self.assertIn("torch", side["versions"])

    def test_runtime_failure_stops_before_the_model(self):
        output = self.tmp / "o.jsonl"
        with self.assertRaises(SystemExit):
            self.run_main(
                self.argv(output), require={"side_effect": RuntimeError("reference")}
            )
        self.assertFalse(output.exists())

    def test_binding_drift_fails(self):
        output = self.tmp / "o.jsonl"
        drift = identity(str(self.cache))
        drift["kernel_bindings"] = {**BINDINGS, "causal_conv1d_fn": None}
        with self.assertRaises(SystemExit):
            self.run_main(self.argv(output), after=drift)
        self.assertFalse(output.with_name(output.name + ".runtime.json").exists())

    def test_exec_checks_runtime_before_the_module(self):
        args = types.SimpleNamespace(
            runtime=self.tmp / "rt.json", command=["--", "json.tool", "--help"]
        )
        with mock.patch.object(
            runtime_check, "require_runtime", side_effect=RuntimeError("reference")
        ), mock.patch("runpy.run_module") as run:
            with self.assertRaises(SystemExit):
                readout.run_module(args)
        run.assert_not_called()


class LongestTest(unittest.TestCase):
    def test_longest_admitted_question_wins(self):
        def item(i, *sizes):
            return {
                "id": f"p{i}",
                "state": "s",
                "questions": {
                    f"q{j}": {"type": "noul", "instructions": f"{size}"}
                    for j, size in enumerate(sizes)
                },
            }

        def length(row):
            size = int(row["instructions"])
            if size > 100:
                raise ValueError("exceeds max_length")
            return size

        rows = [item(0, 40), item(1, 90, 5), item(2, 150), item(3, 90)]
        best, info = kernel.select_longest(rows, length)
        self.assertEqual(best["id"], "p1")
        self.assertEqual(info["longest_question_tokens"], 90)
        self.assertEqual(info["questions_over_limit_in_input"], 1)
        with self.assertRaises(SystemExit):
            kernel.select_longest([item(0, 500)], length)


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.frozen = self.tmp / "frozen"
        (self.frozen / "AB").mkdir(parents=True)
        (self.frozen / "AB" / "k.autotune.json").write_text("{}")
        (self.frozen / "b.bin").write_bytes(b"\x00\x01")
        (self.frozen / "a-link").symlink_to("b.bin")

    def test_digest_matches_the_shell_recipe(self):
        digest = cache.tree_digest(cache.file_hashes(self.frozen))
        shell = subprocess.run(
            "find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum",
            shell=True, cwd=self.frozen, capture_output=True, text=True, check=True,
        ).stdout.split()[0]  # fmt: skip
        self.assertEqual(digest, shell)

    def test_copy_verifies_and_finish_lists_changes(self):
        expect = cache.tree_digest(cache.file_hashes(self.frozen))
        dest = self.tmp / "run" / "triton-cache"
        receipt = cache.copy(self.frozen, dest, expect)
        self.assertEqual(receipt["files"], 2)
        self.assertEqual(receipt["autotune_entries"], 1)
        with self.assertRaises(FileExistsError):
            cache.copy(self.frozen, dest, expect)
        (dest / "b.bin").write_bytes(b"changed")
        (dest / "CD").mkdir()
        (dest / "CD" / "new.autotune.json").write_text("{}")
        (dest / "AB" / "k.autotune.json").unlink()
        result = cache.finish(dest)
        self.assertEqual(result["pre_sha256"], expect)
        self.assertEqual(result["added"], ["CD/new.autotune.json"])
        self.assertEqual(result["changed"], ["b.bin"])
        self.assertEqual(result["removed"], ["AB/k.autotune.json"])
        self.assertFalse(result["unchanged"])
        self.assertTrue(dest.with_name("triton-cache.post.json").is_file())

    def test_copy_with_wrong_hash_refuses(self):
        dest = self.tmp / "bad"
        with self.assertRaises(SystemExit):
            cache.copy(self.frozen, dest, "0" * 64)
        self.assertFalse(dest.with_name("bad.copy.json").exists())


class ReadoutTest(unittest.TestCase):
    def test_trainer_select_rows_follow_the_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "BEST.json").write_text(
                json.dumps({"checkpoint": "checkpoint-0000012"})
            )
            (run / "COMPLETE.json").write_text(
                json.dumps({"status": "complete", "best": "checkpoint-0000012"})
            )
            predictions = [
                {"id": "n", "answer": {"type": "noul", "noul": 0.75}},
                {"id": "c", "answer": {"type": "choice", "choice": "b",
                                       "probabilities": {"a": 0.2, "b": 0.8}}},
                {"id": "s", "answer": {"type": "score", "score": 1.0,
                                       "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}}},
            ]  # fmt: skip
            (run / "select-step-0000012-predictions.jsonl").write_text(
                "".join(json.dumps(p) + "\n" for p in predictions)
            )
            rows = [
                {"id": "c", "options": [{"key": "b"}, {"key": "a"}]},
                {"id": "n", "options": [{"key": "true"}, {"key": "false"}]},
                {"id": "s", "options": [{"key": "0"}, {"key": "1"}, {"key": "2"}]},
            ]
            best, out = readout.trainer_select_rows(run, rows)
            self.assertEqual(best, "checkpoint-0000012")
            self.assertEqual(
                out,
                [
                    {"id": "c", "probabilities": [0.8, 0.2]},
                    {"id": "n", "probabilities": [0.75, 0.25]},
                    {"id": "s", "probabilities": [0.1, 0.8, 0.1]},
                ],
            )
            with self.assertRaises(SystemExit):
                readout.trainer_select_rows(run, rows[:2])

    def test_report_is_accepted_by_infer(self):
        records = [
            {"id": f"{kind}{i}", "task_type": kind, "label": i % 2,
             "logits": [float(i % 3), 1.0]}
            for kind in ("choice", "noul", "score")
            for i in range(4)
        ]  # fmt: skip
        ident = {
            "model_sha256": SHA,
            "files_sha256": {"checkpoint/adapter/a.safetensors": "b" * 64,
                             "source/model.safetensors": "c" * 64},
        }  # fmt: skip
        report = readout.build_report(records, ident, "d" * 64, {"max_length": 32768})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(report))
            temperatures, loaded = calibration.load_calibration(path, SHA)
        self.assertEqual(loaded["selection_policy"], "frozen_checkpoint")
        self.assertEqual(
            report["checkpoint_sha256"],
            hashlib.sha256(
                json.dumps(
                    {"adapter/a.safetensors": "b" * 64},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        calibrated = readout.probability_rows(records[:1], temperatures)
        raw = readout.probability_rows(records[:1], None)
        self.assertAlmostEqual(sum(calibrated[0]["probabilities"]), 1.0)
        self.assertEqual(
            raw[0]["probabilities"], calibration.probabilities([0.0, 1.0], 1.0)
        )


class SpecAndScriptTest(unittest.TestCase):
    def test_kernel_spec_matches_the_reference_spec(self):
        ref = adapters.load(None, HERE / "adapters" / "typed-lora.json")
        spec = adapters.load(None, HERE / "adapters" / "typed-lora-kernel.json")
        self.assertEqual(ref.module, "v2.27b.typed_collect")
        self.assertEqual(spec.module, "v2.27b.typed_collect_kernel")
        self.assertEqual(spec.args, ref.args)
        self.assertEqual(spec.requires, ref.requires)
        self.assertEqual(spec.image, ref.image)
        argv = spec.command(
            {"model": "m", "source": "b", "model_id": "i", "revision": "r",
             "max_length": "32768", "calibration": "c", "input": "in", "output": "out"}
        )  # fmt: skip
        self.assertEqual(argv[:3], ["python3", "-m", "v2.27b.typed_collect_kernel"])
        self.assertIn("32768", argv)
        self.assertTrue(adapters.module_path(HERE.parents[1], spec).is_file())

    def test_scripts_parse(self):
        for name in (
            "run_formal_kernel.sh",
            "run_dev_readout.sh",
            "warm_cache.sh",
            "kernel_common.sh",
            "run_lora_arm.sh",
            "run_formal_typed.sh",
        ):
            subprocess.run(["bash", "-n", str(HERE / name)], check=True)


if __name__ == "__main__":
    unittest.main()
