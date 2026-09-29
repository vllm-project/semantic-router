import contextlib
import hashlib
import importlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

readout = importlib.import_module("v2.27b.kernel_readout")
cache = importlib.import_module("v2.27b.triton_cache")
calibration = importlib.import_module("training.model.calibration")
infer = importlib.import_module("training.model.infer")

HERE = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class T1CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        records = [
            {"id": f"{kind}{i}", "task_type": kind, "label": i % 2,
             "logits": [float(i % 3) * 1.7, 0.4]}
            for kind in ("choice", "noul", "score")
            for i in range(6)
        ]  # fmt: skip
        ident = {"model_sha256": SHA, "files_sha256": {"checkpoint/x": "b" * 64}}
        report = readout.build_report(records, ident, "d" * 64, {"max_length": 32768})
        self.rejected = self.tmp / "cal698.json"
        self.rejected.write_text(json.dumps(report))
        self.receipt = self.tmp / "dev-calibration.json"
        self.write_receipt(adopt=False, candidate=sha(self.rejected))

    def write_receipt(self, adopt, candidate):
        self.receipt.write_text(
            json.dumps(
                {"adopt": adopt, "candidate_sha256": candidate, "worsened": ["x"]}
            )
        )

    def t1(self, name="t1.json"):
        out = self.tmp / name
        with contextlib.redirect_stdout(io.StringIO()):
            readout.main(
                ["t1-calibration", "--rejected", str(self.rejected),
                 "--adoption", str(self.receipt), "--output", str(out)]
            )  # fmt: skip
        return out

    def test_report_binds_the_rejected_fit_at_temperature_one(self):
        out = self.t1()
        temperatures, report = calibration.load_calibration(out, SHA)
        rejected = json.loads(self.rejected.read_text())
        self.assertEqual(temperatures, {"choice": 1.0, "noul": 1.0, "score": 1.0})
        for key in ("model_sha256", "checkpoint_sha256", "cal_sha256", "inference"):
            self.assertEqual(report[key], rejected[key])
        self.assertEqual(
            report["rejected_temperature_by_type"], rejected["temperature_by_type"]
        )
        self.assertEqual(report["rejected_calibration_sha256"], sha(self.rejected))
        self.assertEqual(report["adoption_receipt_sha256"], sha(self.receipt))
        with self.assertRaises(FileExistsError):
            self.t1()

    def test_refuses_an_adopted_or_different_fit(self):
        self.write_receipt(adopt=True, candidate=sha(self.rejected))
        with self.assertRaises(SystemExit):
            self.t1("a.json")
        self.write_receipt(adopt=False, candidate="0" * 64)
        with self.assertRaises(SystemExit):
            self.t1("b.json")

    def test_answers_equal_an_uncalibrated_package(self):
        temperatures, _ = calibration.load_calibration(self.t1(), SHA)
        item = {
            "id": "p",
            "state": "s",
            "questions": {
                "c": {"type": "choice", "instructions": "pick",
                      "criteria": {"a": "A", "b": "B", "c": "C"}},
                "n": {"type": "noul", "instructions": "yes?"},
                "s": {"type": "score", "instructions": "rate",
                      "criteria": ["low", "mid", "high"]},
            },
        }  # fmt: skip

        def run(temperature):
            predictions, _ = infer.run_prompts(
                [item],
                tokenizer=None,
                max_length=64,
                temperature=temperature,
                encode_fn=lambda row, tok, n: {"ids": [0], "k": len(row["options"])},
                predict_fn=lambda jobs: [
                    [0.1 + 0.73 * i * (-1) ** i for i in range(job["k"])]
                    for job in jobs
                ],
                model_sha256=SHA,
                adapter_sha256=SHA,
            )
            return predictions[0]["answers"]

        self.assertEqual(run(temperatures), run(1.0))


class CacheClassifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.frozen = self.tmp / "frozen"
        group = self.frozen / "G"
        group.mkdir(parents=True)
        (group / "k.hsaco").write_bytes(b"\x01")
        (group / "k.autotune.json").write_text("{}")
        self.children = {"k.hsaco": str(group / "k.hsaco")}
        (group / "__grp__k.json").write_text(json.dumps({"child_paths": self.children}))
        self.dest = self.tmp / "run" / "triton-cache"
        cache.copy(
            self.frozen, self.dest, cache.tree_digest(cache.file_hashes(self.frozen))
        )

    def rewrite(self, extra=None):
        value = {"child_paths": {"k.hsaco": str(self.dest / "G" / "k.hsaco")}}
        (self.dest / "G" / "__grp__k.json").write_text(
            json.dumps({**value, **(extra or {})})
        )

    def test_path_rewrites_and_new_kernels_pass(self):
        self.rewrite()
        (self.dest / "H").mkdir()
        (self.dest / "H" / "n.hsaco").write_bytes(b"\x02")
        result = cache.finish(self.dest)
        self.assertFalse(result["unchanged"])
        self.assertEqual(
            result["classified"]["group"]["path_rewrites"], ["G/__grp__k.json"]
        )
        self.assertEqual(result["classified"]["kernel"]["added"], ["H/n.hsaco"])
        self.assertTrue(result["frozen_check"]["passed"])

    def test_autotune_kernel_and_group_content_changes_fail(self):
        self.rewrite({"extra": 1})
        (self.dest / "G" / "k.hsaco").write_bytes(b"\x03")
        (self.dest / "H").mkdir()
        (self.dest / "H" / "n.autotune.json").write_text("{}")
        result = cache.finish(self.dest)
        self.assertFalse(result["frozen_check"]["passed"])
        self.assertEqual(
            sorted(result["frozen_check"]["problems"]),
            ["autotune added: H/n.autotune.json", "group changed: G/__grp__k.json",
             "kernel changed: G/k.hsaco"],
        )  # fmt: skip


class ArgcheckTest(unittest.TestCase):
    def check(self, command):
        with contextlib.redirect_stderr(io.StringIO()):
            return readout.check_command(command)

    def test_follows_exec_into_the_module(self):
        command = [
            "v2.27b.kernel_readout", "exec", "--runtime", "/o/rt.json", "--",
            "v2.27b.aho_eval", "--run-dir", "/r", "--source-path", "/b",
            "--slice", "A=/a", "--max-length", "32768", "--out-dir", "/o",
        ]  # fmt: skip
        self.assertEqual(
            self.check(command), ["v2.27b.kernel_readout", "v2.27b.aho_eval"]
        )
        with self.assertRaises(SystemExit) as caught:
            self.check(command[:-2] + ["--output", "/o"])
        self.assertEqual(caught.exception.code, 2)

    def test_follows_collect_into_the_adapter(self):
        command = [
            "v2.eval.same_panel", "collect", "--run-dir", "/r",
            "--adapter-spec", str(HERE / "adapters" / "typed-lora-kernel.json"),
            "--model-path", "/m", "--revision", "checkpoint-sha256:x",
            "--extra", "source=/b", "--extra", "calibration=/c.json",
            "--extra", "max_length=32768", "--panels", "typed-dev,css-pilot",
        ]  # fmt: skip
        self.assertEqual(
            self.check(command), ["v2.eval.same_panel", "training.model.infer"]
        )
        with self.assertRaises(ValueError):
            self.check(command[:-8] + command[-6:])


class ScriptTest(unittest.TestCase):
    def test_scripts_parse(self):
        for name in ("run_finalist.sh", "kernel_common.sh", "run_dev_readout.sh"):
            subprocess.run(["bash", "-n", str(HERE / name)], check=True)


if __name__ == "__main__":
    unittest.main()
