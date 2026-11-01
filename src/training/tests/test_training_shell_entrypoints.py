"""Exercise launcher dispatch without downloading models or training packages."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIERS = REPO_ROOT / "src/training/model_classifier"
TASKS = (
    ("classifier_model_fine_tuning_lora", "ft_linear_lora.py"),
    ("fact_check_fine_tuning_lora", "fact_check_bert_finetuning_lora.py"),
    ("pii_model_fine_tuning_lora", "pii_bert_finetuning_lora.py"),
    ("modality_routing_classifier", "modality_routing_bert_finetuning_lora.py"),
)
UV_STUB = """import json, os, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ['UV_CALL_LOG'], 'a') as log:
    log.write(json.dumps({'args': args, 'cwd': os.getcwd()}) + '\\n')
if '--mode' in args and args[args.index('--mode') + 1] == 'train':
    if os.environ.get('FAIL_TRAINING'):
        sys.exit(23)
    if args[5].endswith('modality_routing_bert_finetuning_lora.py'):
        model = args[args.index('--model') + 1]
        Path(f'lora_modality_router_{model}_r32_model').mkdir()
    if args[5].endswith('ft_linear_lora.py'):
        output = Path('lora_intent_classifier_mmbert-base_r32_model')
        output.mkdir()
        (output / 'adapter.txt').write_text('fixture')
"""


class TrainingShellEntrypointsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="training launchers ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # No ambient uv, pip, or Python executable can satisfy a bare invocation.
        for name in ("dirname", "date", "mkdir", "ls", "cp", "head", "mv"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (self.bin / name).symlink_to(executable)
        self.uv = self.bin / "uv"
        self.uv.write_text(f"#!{sys.executable}\n{UV_STUB}")
        self.uv.chmod(0o755)
        self.projects = self.root / "classifier projects"
        self.projects.mkdir()
        for area, trainer in TASKS:
            project = self.projects / area
            project.mkdir()
            (project / trainer).touch()
        self.wrapper = self.projects / "train_mmbert_gpu.sh"
        shutil.copyfile(CLASSIFIERS / self.wrapper.name, self.wrapper)
        self.modality = self.projects / TASKS[-1][0] / "run_training.sh"
        shutil.copyfile(CLASSIFIERS / TASKS[-1][0] / self.modality.name, self.modality)
        self.work = self.root / "work dir"
        self.log = self.root / "uv.jsonl"
        self.env = {
            "PATH": str(self.bin),
            "UV_CALL_LOG": str(self.log),
            "TRAINING_WORK_DIR": str(self.work),
        }
        self.bash = shutil.which("bash")
        self.make = shutil.which("make")

    def run_script(self, script, **environment):
        return subprocess.run(
            [self.bash, str(script)],
            cwd=self.root,
            env={**self.env, **environment},
            capture_output=True,
            text=True,
            check=False,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def assert_project(self, call, area):
        self.assertEqual(
            call["args"][:5],
            ["run", "--locked", "--project", str(self.projects / area), "python"],
        )

    def test_missing_uv_fails_before_cleanup_or_creating_outputs(self):
        self.uv.unlink()
        self.work.mkdir()
        sentinel = self.work / "lora_existing_mmbert_adapter"
        sentinel.mkdir()
        for script in (self.wrapper, self.modality):
            with self.subTest(script=script):
                result = self.run_script(script, TRAINING_CLEAN="1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("uv is required", result.stderr)
                self.assertIn("https://docs.astral.sh/uv/", result.stderr)
                self.assertTrue(sentinel.is_dir())
                self.assertFalse((self.work / "models").exists())

    def test_wrapper_dispatches_each_project_and_preserves_work_directory(self):
        result = self.run_script(
            self.wrapper, EPOCHS="1", BATCH_SIZE="2", MAX_SAMPLES="200"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        self.assert_project(calls[0], TASKS[0][0])
        self.assertEqual(calls[0]["args"][5], "-c")
        for call, (area, trainer) in zip(calls[1:], TASKS, strict=True):
            self.assert_project(call, area)
            args = call["args"]
            self.assertEqual(args[5], str(self.projects / area / trainer))
            self.assertEqual(call["cwd"], str(self.work))
            for flag, value in (
                ("--epochs", "1"),
                ("--batch-size", "2"),
                ("--max-samples", "200"),
            ):
                self.assertEqual(args[args.index(flag) + 1], value)
        copied = (
            self.work
            / "models/lora_intent_classifier_mmbert-base_r32_model/adapter.txt"
        )
        self.assertEqual(copied.read_text(), "fixture")

    def test_wrapper_preserves_default_training_parameters(self):
        result = self.run_script(self.wrapper)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[1]["args"]
        for flag, value in (
            ("--epochs", "10"),
            ("--batch-size", "64"),
            ("--max-samples", "10000"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_training_failure_stops_both_launchers(self):
        for script in (self.wrapper, self.modality):
            with self.subTest(script=script):
                self.log.write_text("")
                result = self.run_script(script, FAIL_TRAINING="1")
                self.assertEqual(result.returncode, 23, result.stderr)
                self.assertEqual(len(self.calls()), 2)

    def test_modality_diagnostic_training_and_inference_share_project(self):
        result = self.run_script(
            self.modality,
            VLLM_ENDPOINT="http://localhost:8000/v1",
            VLLM_MODEL="model name with spaces",
            LORA_RANK="16",
            LORA_ALPHA="32",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assert_project(call, TASKS[-1][0])
        args = calls[1]["args"]
        for flag, value in (
            ("--vllm-endpoint", "http://localhost:8000/v1"),
            ("--vllm-model", "model name with spaces"),
            ("--synthesize-both", "2000"),
            ("--lora-rank", "16"),
            ("--lora-alpha", "32"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(calls[2]["args"][6:8], ["--mode", "test"])

    def test_make_targets_dispatch_existing_projects(self):
        if not self.make:
            self.skipTest("make is unavailable")
        targets = {
            "intent": TASKS[0],
            "factcheck": TASKS[1],
            "pii": TASKS[2],
            "pii-quick": TASKS[2],
            "pii-presidio-only": TASKS[2],
        }
        for target, (area, trainer) in targets.items():
            with self.subTest(target=target):
                self.log.write_text("")
                result = subprocess.run(
                    [
                        self.make,
                        "-f",
                        str(REPO_ROOT / "tools/make/models.mk"),
                        f"train-mmbert32k-{target}",
                        f"LORA_DIR={self.projects}",
                    ],
                    cwd=self.root,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                (call,) = self.calls()
                self.assert_project(call, area)
                self.assertEqual(call["args"][5], str(self.projects / area / trainer))

    def test_make_target_reports_missing_uv_before_creating_outputs(self):
        if not self.make:
            self.skipTest("make is unavailable")
        self.uv.unlink()
        result = subprocess.run(
            [
                self.make,
                "-f",
                str(REPO_ROOT / "tools/make/models.mk"),
                "train-mmbert32k-intent",
            ],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uv is required", result.stderr)
        self.assertIn("https://docs.astral.sh/uv/", result.stderr)
        self.assertFalse((self.root / "models").exists())


if __name__ == "__main__":
    unittest.main()
