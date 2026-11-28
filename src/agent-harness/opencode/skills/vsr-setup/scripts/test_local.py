import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import local


def venv_with(tmp: str, commit: str | None) -> Path:
    """A venv that has vllm-srun installed from `commit` (None: not installed)."""
    venv = Path(tmp) / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    if commit:
        (venv / "bin/vllm-srun").touch()
        info = venv / "lib/python3.12/site-packages/vllm_srun-0.2.0.dist-info"
        info.mkdir(parents=True)
        (info / "direct_url.json").write_text(
            json.dumps({"vcs_info": {"commit_id": commit}})
        )
    return venv


class Install(unittest.TestCase):
    def test_a_fresh_install_pins_the_commit_with_its_multimodal_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = local.install_steps(Path(tmp) / "venv", "cpu", sys.executable)
        self.assertEqual(steps[0][1:3], ["-m", "venv"])
        self.assertRegex(
            steps[-1][-1],
            r"^vllm-srun\[multimodal\] @ git\+https://github\.com/vllm-project/semantic-router\.git@"
            + local.COMMIT
            + r"#subdirectory=src/model-runtime$",
        )
        locked = sys.version_info >= local.LOCKED_PYTHON
        self.assertEqual("-c" in steps[-1], locked)
        if locked:
            lock = steps[-1][steps[-1].index("-c") + 1]
            self.assertTrue(
                lock.endswith(
                    f"/{local.COMMIT}/src/model-runtime/requirements-lock.txt"
                )
            )

    def test_an_older_commit_is_forced_out_then_installed_with_its_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "_has", return_value=True
        ):
            steps = local.install_steps(
                venv_with(tmp, next(iter(local.detect.OLDER))), "cuda", sys.executable
            )
        self.assertEqual(len(steps), 2)
        self.assertIn("--force-reinstall", steps[0])
        self.assertIn("--no-deps", steps[0])
        self.assertNotIn("-c", steps[0])
        self.assertNotIn("--force-reinstall", steps[1])

    def test_the_current_commit_without_pillow_installs_the_extra_again(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "_has", side_effect=lambda venv, module: module != "PIL"
        ):
            steps = local.install_steps(
                venv_with(tmp, local.COMMIT), "cuda", sys.executable
            )
        self.assertEqual(len(steps), 1)
        self.assertNotIn("--force-reinstall", steps[0])
        self.assertTrue(steps[0][-1].startswith("vllm-srun[multimodal] @"))

    def test_a_current_install_needs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "_has", return_value=True
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            venv = venv_with(tmp, local.COMMIT)
            self.assertEqual(local.install_steps(venv, "cpu", sys.executable), [])
            self.assertEqual(local.main(["install", "--venv", str(venv)]), 0)
        self.assertIn("nothing to install", out.getvalue())

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

    def test_skip_disk_check_installs_anyway(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local.detect, "free_gib", return_value=0.5
        ), mock.patch.object(local, "run", return_value=0) as run:
            code = local.main(
                ["install", "--skip-disk-check", "--venv", str(Path(tmp) / "venv")]
            )
        self.assertEqual(code, 0)
        run.assert_called_once()

    def test_the_serve_command_passes_no_dtype(self):
        command = local.serve_command(Path("/v"), "vllm-sr/vela3-lite", "cpu", 8790)
        self.assertNotIn("--dtype", command)
        self.assertEqual(command[-4:], ["--host", "127.0.0.1", "--port", "8790"])


class Service(unittest.TestCase):
    def test_the_names_are_the_claude_code_plugins(self):
        unit, _ = local.unit_text("systemd", ["x"], Path("/log"))
        plist, text = local.unit_text("launchd", ["x"], Path("/log"))
        self.assertEqual(unit.name, "vsr-runtime.service")
        self.assertEqual(plist.name, "local.vsr-runtime.plist")
        self.assertIn("<string>local.vsr-runtime</string>", text)

    def test_a_changed_systemd_unit_is_restarted_so_it_takes_its_new_flags(self):
        steps = local.enable_steps("systemd", Path("/u"), changed=True)
        self.assertEqual(
            steps[-1], ["systemctl", "--user", "restart", "vsr-runtime.service"]
        )
        steps = local.enable_steps("systemd", Path("/u"), changed=False)
        self.assertEqual(
            steps[-1], ["systemctl", "--user", "start", "vsr-runtime.service"]
        )


class Start(unittest.TestCase):
    def test_a_runtime_still_loading_is_not_started_twice(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local.runtime, "health", return_value={"ready": False}
        ), mock.patch.object(
            local, "port_open", return_value=False
        ), mock.patch.object(
            local.subprocess, "Popen"
        ) as popen, contextlib.redirect_stdout(
            io.StringIO()
        ) as out, mock.patch.object(
            local, "started_pid", return_value=4242
        ):
            code = local.start(Path(tmp), "m", "cpu", 8790, Path(tmp) / "log", False)
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertIn("already starting: pid 4242", out.getvalue())

    def test_a_dead_or_foreign_pid_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ):
            self.assertIsNone(local.started_pid(), "no pid file")
            (Path(tmp) / "runtime.pid").write_text("999999999")
            self.assertIsNone(local.started_pid(), "no such process")
            (Path(tmp) / "runtime.pid").write_text(str(local.os.getpid()))
            self.assertIsNone(
                local.started_pid(), "a live process that is not vllm-srun"
            )


class Restart(unittest.TestCase):
    def test_each_way_of_keeping_it_running_restarts_its_own_way(self):
        self.assertEqual(
            local.restart_plan("systemd"),
            [["systemctl", "--user", "restart", "vsr-runtime.service"]],
        )
        self.assertEqual(
            local.restart_plan("launchd")[0][:3], ["launchctl", "kickstart", "-k"]
        )
        self.assertEqual(local.restart_plan("pid"), "kill")
        self.assertEqual(local.restart_plan(None), "none")

    def test_a_live_runtime_is_stopped_and_started_again(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local.detect, "kept_by", return_value="pid"
        ), mock.patch.object(
            local, "started_pid", return_value=4242
        ), mock.patch.object(
            local.os, "kill"
        ) as kill, mock.patch.object(
            local, "alive", return_value=False
        ), mock.patch.object(
            local, "port_open", return_value=False
        ), mock.patch.object(
            local, "start", return_value=0
        ) as start, contextlib.redirect_stdout(
            io.StringIO()
        ):
            (Path(tmp) / "runtime.pid").write_text("4242")
            code = local.main(["restart", "--device", "cuda"])
            self.assertFalse((Path(tmp) / "runtime.pid").exists())
        self.assertEqual(code, 0)
        kill.assert_called_once_with(4242, local.signal.SIGTERM)
        self.assertEqual(start.call_args.args[2], "cuda")

    def test_a_stale_pid_is_never_signalled(self):
        # After a reboot the pid file may name another process; it is dropped, not killed.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local.detect, "kept_by", return_value="pid"
        ), mock.patch.object(
            local, "started_pid", return_value=None
        ), mock.patch.object(
            local.os, "kill"
        ) as kill, mock.patch.object(
            local, "port_open", return_value=False
        ), mock.patch.object(
            local, "start", return_value=0
        ) as start, contextlib.redirect_stdout(
            io.StringIO()
        ):
            (Path(tmp) / "runtime.pid").write_text("4242")
            code = local.main(["restart", "--device", "cpu"])
            self.assertFalse((Path(tmp) / "runtime.pid").exists())
        self.assertEqual(code, 0)
        kill.assert_not_called()
        start.assert_called_once()

    def test_a_runtime_that_will_not_stop_is_left_to_the_user(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local, "started_pid", return_value=4242
        ), mock.patch.object(
            local.os, "kill"
        ), mock.patch.object(
            local, "alive", return_value=True
        ), mock.patch.object(
            local.time, "sleep"
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            self.assertEqual(local.stop(False, timeout=0), 1)
        self.assertIn("still running", err.getvalue())

    def test_with_nothing_setup_started_and_nothing_answering_it_just_starts(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local.detect, "kept_by", return_value=None
        ), mock.patch.object(
            local.runtime, "health", return_value={"ready": False}
        ), mock.patch.object(
            local, "start", return_value=0
        ) as start:
            (Path(tmp) / "runtime.pid").write_text("4242")
            self.assertEqual(local.main(["restart", "--device", "cpu"]), 0)
            self.assertFalse((Path(tmp) / "runtime.pid").exists())
        start.assert_called_once()

    def test_a_runtime_someone_else_started_is_never_reported_as_restarted(self):
        # A stale pid file and a runtime the user started by hand on the port: restarting must not
        # end in `already running` as if the new code were loaded.
        with mock.patch.object(
            local.detect, "kept_by", return_value=None
        ), mock.patch.object(
            local.runtime, "health", return_value={"ready": True}
        ), mock.patch.object(
            local, "start"
        ) as start, contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            self.assertEqual(local.main(["restart", "--device", "cuda"]), 1)
        start.assert_not_called()
        self.assertIn("ask the user to restart theirs", err.getvalue())

    def test_a_port_still_answering_after_stop_is_left_to_the_user(self):
        with mock.patch.object(
            local.detect, "kept_by", return_value="pid"
        ), mock.patch.object(local, "stop", return_value=0), mock.patch.object(
            local, "port_open", return_value=True
        ), mock.patch.object(
            local, "start"
        ) as start, contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            self.assertEqual(local.main(["restart", "--device", "cuda"]), 1)
        start.assert_not_called()
        self.assertIn("another runtime serves it", err.getvalue())

    def test_start_restart_and_service_need_a_device(self):
        for command in ("start", "restart", "service"):
            with self.subTest(command=command), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                local.main([command])


class Pid(unittest.TestCase):
    def test_without_proc_the_command_comes_from_ps(self):
        with mock.patch.object(
            local.Path, "exists", return_value=False
        ), mock.patch.object(
            local.subprocess,
            "run",
            return_value=mock.Mock(stdout="/v/bin/python /v/bin/vllm-srun serve m\n"),
        ) as ps:
            self.assertIn("vllm-srun", local.command_of(4242))
        self.assertEqual(ps.call_args.args[0], ["ps", "-p", "4242", "-o", "command="])

    def test_a_live_pid_that_is_not_vllm_srun_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local, "command_of", return_value="/usr/bin/vim notes.txt"
        ):
            (Path(tmp) / "runtime.pid").write_text(str(local.os.getpid()))
            self.assertIsNone(local.started_pid())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(
            local, "command_of", return_value="python vllm-srun serve"
        ):
            (Path(tmp) / "runtime.pid").write_text(str(local.os.getpid()))
            self.assertEqual(local.started_pid(), local.os.getpid())


class Wait(unittest.TestCase):
    def test_a_runtime_that_exits_while_loading_is_said_at_once(self):
        # The pid file is read inside the loop, so a runtime that died before `wait` began counts too.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "HOME", Path(tmp)
        ), mock.patch.object(local, "alive", return_value=False), mock.patch.object(
            local, "port_open", return_value=False
        ), mock.patch.object(
            local.runtime, "health", return_value={"ready": False}
        ), contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            (Path(tmp) / "runtime.pid").write_text("4242")
            log = Path(tmp) / "log"
            log.write_text("RuntimeError: no CUDA device\n")
            self.assertEqual(local.wait(8790, 5, log), 1)
        self.assertIn("pid 4242) exited while loading", err.getvalue())
        self.assertIn("no CUDA device", err.getvalue())

    def test_a_ready_runtime_is_timed_and_a_slow_one_warned(self):
        with mock.patch.object(
            local.runtime, "health", return_value={"ready": True}
        ), mock.patch.object(
            local.runtime, "time_question", return_value=4100
        ), contextlib.redirect_stdout(
            io.StringIO()
        ) as out, contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            self.assertEqual(local.wait(8790, 5, Path("/nolog")), 0)
        self.assertIn(
            "ready: http://127.0.0.1:8790/v1/decisions · 4100 ms a question",
            out.getvalue(),
        )
        self.assertIn("warning: slow runtime", err.getvalue())


class Shared(unittest.TestCase):
    def test_a_commit_setup_does_not_know_is_left_as_it_is(self):
        # The Claude Code plugin installs into the same venv and may pin a newer commit.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "_has", return_value=True
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            venv = venv_with(tmp, "f" * 40)
            self.assertEqual(local.install_steps(venv, "cuda", sys.executable), [])
            self.assertEqual(local.main(["install", "--venv", str(venv)]), 0)
        self.assertIn("setup did not pin (fffffffff), left as it is", out.getvalue())


class LocalBuild(unittest.TestCase):
    def test_a_runtime_not_installed_from_git_is_left_as_it_is(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            local, "_has", return_value=True
        ):
            venv = venv_with(tmp, None)
            (venv / "bin/vllm-srun").touch()
            self.assertEqual(local.install_steps(venv, "cpu", sys.executable), [])

    def test_plist_paths_are_escaped(self):
        _, text = local.unit_text("launchd", ["/a&b/vllm-srun"], Path("/l<og"))
        self.assertIn("<string>/a&amp;b/vllm-srun</string>", text)
        self.assertIn("/l&lt;og", text)


if __name__ == "__main__":
    unittest.main()
