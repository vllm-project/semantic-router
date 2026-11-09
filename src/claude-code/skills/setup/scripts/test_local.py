import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import local


class Install(unittest.TestCase):
    def test_installs_the_runtime_from_a_pinned_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = local.install_steps(Path(tmp) / "venv", "cpu", sys.executable)
        self.assertRegex(
            steps[-1][-1],
            r"^vllm-srun @ git\+https://github\.com/vllm-project/semantic-router\.git@[0-9a-f]{40}#subdirectory=src/model-runtime$",
        )

    def test_constrains_the_dependencies_to_the_same_commits_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = local.install_steps(Path(tmp) / "venv", "cpu", sys.executable)
        locked = sys.version_info >= local.LOCKED_PYTHON
        self.assertEqual("-c" in steps[-1], locked)
        if locked:
            lock = steps[-1][steps[-1].index("-c") + 1]
            self.assertIn(
                f"/{local.COMMIT}/src/model-runtime/requirements-lock.txt", lock
            )
            self.assertIn(f"@{local.COMMIT}#", steps[-1][-1])


class Dtype(unittest.TestCase):
    def test_a_cpu_without_bf16_serves_in_float32(self):
        self.assertEqual(local.serve_dtype("cpu", "auto", False), "float32")

    def test_otherwise_the_runtime_keeps_its_own_default(self):
        for device, bf16 in (("cpu", True), ("cpu", None), ("cuda", False)):
            with self.subTest(device=device, bf16=bf16):
                self.assertIsNone(local.serve_dtype(device, "auto", bf16))

    def test_an_explicit_dtype_wins(self):
        self.assertEqual(local.serve_dtype("cpu", "bfloat16", False), "bfloat16")

    def test_the_serve_command_passes_the_dtype_only_when_set(self):
        venv = Path("/v")
        command = local.serve_command(venv, "m", "cpu", 8790, "float32")
        self.assertEqual(command[-2:], ["--dtype", "float32"])
        self.assertNotIn("--dtype", local.serve_command(venv, "m", "cpu", 8790))

    def test_a_full_disk_stops_the_install(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local.detect, "free_gib", return_value=0.5
        ), mock.patch.object(local, "run") as run, contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            code = local.main(["install", "--venv", str(Path(tmp) / "venv")])
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertIn("only 0.5 GiB disk free", err.getvalue())

    def test_a_gpu_install_checks_the_disk_too(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local.detect, "free_gib", return_value=3.0
        ), mock.patch.object(local, "run") as run, contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            code = local.main(
                ["install", "--device", "cuda", "--venv", str(Path(tmp) / "venv")]
            )
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertIn("only 3.0 GiB disk free", err.getvalue())

    def test_skip_disk_check_installs_anyway(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local.detect, "free_gib", return_value=0.5
        ), mock.patch.object(local, "run", return_value=0) as run:
            code = local.main(
                ["install", "--skip-disk-check", "--venv", str(Path(tmp) / "venv")]
            )
        self.assertEqual(code, 0)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
