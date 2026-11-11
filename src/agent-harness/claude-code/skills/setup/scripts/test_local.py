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

    def test_a_measured_bf16_decides_over_the_flags(self):
        fast = {"float32": 2.03, "bfloat16": 0.48}
        slow = {"float32": 1.0, "bfloat16": 9.0}
        self.assertIsNone(local.serve_dtype("cpu", "auto", False, fast))
        self.assertEqual(local.serve_dtype("cpu", "auto", True, slow), "float32")

    def test_bf16_is_kept_unless_clearly_slower(self):
        near = {"float32": 1.0, "bfloat16": 1.4}
        self.assertIsNone(local.serve_dtype("cpu", "auto", None, near))

    def test_measuring_needs_the_venv_and_a_readable_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "venv"
            self.assertIsNone(local.measure_dtypes(venv))
            (venv / "bin").mkdir(parents=True)
            (venv / "bin/python").touch()
            answer = mock.Mock(stdout='noise\n{"float32": 2.0, "bfloat16": 0.5}\n')
            with mock.patch.object(local.subprocess, "run", return_value=answer):
                self.assertEqual(
                    local.measure_dtypes(venv), {"float32": 2.0, "bfloat16": 0.5}
                )
            with mock.patch.object(
                local.subprocess, "run", return_value=mock.Mock(stdout="Traceback")
            ):
                self.assertIsNone(local.measure_dtypes(venv))
            with mock.patch.object(
                local.subprocess,
                "run",
                side_effect=local.subprocess.TimeoutExpired("python", 60),
            ):
                self.assertIsNone(local.measure_dtypes(venv))

    def test_start_measures_and_serves_in_the_faster_dtype(self):
        slow = {"float32": 1.0, "bfloat16": 9.0}
        with mock.patch.object(
            local, "measure_dtypes", return_value=slow
        ) as measured, mock.patch.object(
            local, "start", return_value=0
        ) as started, contextlib.redirect_stdout(
            io.StringIO()
        ) as out:
            local.main(["start", "--device", "cpu"])
        measured.assert_called_once()
        self.assertEqual(started.call_args.args[-1], "float32")
        self.assertIn(
            "bf16 9.00 ms, float32 1.00 ms per matmul, so float32", out.getvalue()
        )

    def test_a_dry_run_does_not_measure(self):
        with mock.patch.object(local, "measure_dtypes") as measured, mock.patch.object(
            local.detect, "cpu_bf16", return_value=True
        ), mock.patch.object(local, "start", return_value=0):
            local.main(["start", "--device", "cpu", "--dry-run"])
        measured.assert_not_called()


class Service(unittest.TestCase):
    def test_a_changed_systemd_unit_is_restarted_so_it_takes_its_new_flags(self):
        steps = local.enable_steps("systemd", Path("/u/vllm-sr.service"), changed=True)
        self.assertEqual(
            steps[-1], ["systemctl", "--user", "restart", f"{local.LABEL}.service"]
        )

    def test_an_unchanged_systemd_unit_is_only_started(self):
        steps = local.enable_steps("systemd", Path("/u/vllm-sr.service"), changed=False)
        self.assertEqual(
            steps[-1], ["systemctl", "--user", "start", f"{local.LABEL}.service"]
        )
        self.assertNotIn("restart", [word for step in steps for word in step])

    def test_rerunning_setup_with_a_new_port_restarts_the_running_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "vllm-sr.service"
            ran: list[list[str]] = []
            done = mock.Mock(returncode=0)
            with (
                mock.patch.object(
                    local,
                    "unit_text",
                    side_effect=lambda kind, command, log: (unit, " ".join(command)),
                ),
                mock.patch.object(
                    local.subprocess,
                    "run",
                    side_effect=lambda step, check: ran.append(step) or done,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                local.service(
                    "systemd",
                    Path(tmp),
                    "m",
                    "cpu",
                    8790,
                    Path(tmp) / "log",
                    True,
                    "float32",
                )
                ran.clear()
                local.service(
                    "systemd",
                    Path(tmp),
                    "m",
                    "cpu",
                    8791,
                    Path(tmp) / "log",
                    True,
                    "float32",
                )
            self.assertIn(
                ["systemctl", "--user", "restart", f"{local.LABEL}.service"], ran
            )


if __name__ == "__main__":
    unittest.main()
