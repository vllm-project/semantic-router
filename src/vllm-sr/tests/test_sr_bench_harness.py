"""Harness integrity, bounded execution, and owned-resource recovery checks."""

import asyncio
import json
import os
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import pytest
from cli.sr_bench import harness_worker as worker
from cli.sr_bench.contracts import digest
from cli.sr_bench.sandbox_grade import _run_script


def test_tau_task_pin_and_auxiliary_roles(tmp_path, monkeypatch):
    task = {"id": "abc", "user_scenario": {"instructions": "task"}}
    directory = tmp_path / "data/tau2/domains/telecom"
    directory.mkdir(parents=True)
    path = directory / "tasks.json"
    path.write_text(json.dumps([task]))
    request = {
        "source_root": str(tmp_path),
        "case": {
            "metadata": {
                "domain": "telecom",
                "task_id": "abc",
                "source_task_sha256": digest(task),
            }
        },
    }
    monkeypatch.delenv("TAU2_DATA_DIR", raising=False)
    worker._verify_tau_task(request)
    assert os.environ["TAU2_DATA_DIR"] == str(tmp_path / "data")
    assert [
        worker._tau_role(name)
        for name in ("sr-bench-subject", "sr-bench-simulator", "upstream-grader")
    ] == ["subject", "simulator", "judge"]
    task["user_scenario"]["instructions"] = "changed"
    path.write_text(json.dumps([task]))
    with pytest.raises(ValueError, match="raw task differs"):
        worker._verify_tau_task(request)


def test_owned_cleanup_uses_only_exact_filters_and_retains_receipt(
    tmp_path, monkeypatch
):
    name = "sr-bench-" + "a" * 32
    worker._persist_resource(tmp_path, "containers", name)
    worker._persist_resource(tmp_path, "compose_projects", "task__unique__env")
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        selected = (
            "abcdef123456\n" if args[1] != "volume" else "task__unique__env_data\n"
        )
        return SimpleNamespace(stdout=selected, returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", run)
    assert worker.cleanup_owned_resources(tmp_path)["complete"] is True
    lists = [command for command in commands if "ls" in command]
    assert len(lists) == 4
    assert lists[0][-1] == "name=^/" + name + "$"
    assert all(
        command[-1] == "label=com.docker.compose.project=task__unique__env"
        for command in lists[1:]
    )
    assert not any("prune" in command for command in commands)
    assert (tmp_path / "owned-resources.json").is_file()
    receipt = json.loads((tmp_path / "owned-resources.json").read_text())
    receipt["containers"] = ["unrelated-container"]
    (tmp_path / "owned-resources.json").write_text(json.dumps(receipt))
    before = len(commands)
    with pytest.raises(ValueError, match="Invalid owned sandbox"):
        worker.cleanup_owned_resources(tmp_path)
    assert len(commands) == before


def test_sandbox_timeout_has_prelaunch_receipt_and_host_only_control(
    tmp_path, monkeypatch
):
    launches = []
    cleanup_calls = []

    def run(args, **kwargs):
        receipt = json.loads((tmp_path / "owned-resources.json").read_text())
        assert args[args.index("--name") + 1] in receipt["containers"]
        assert str(tmp_path) + ":/artifacts:rw" not in args
        assert str(tmp_path / "sandbox-lcb") + ":/artifacts:rw" in args
        launches.append(args)
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(worker.subprocess, "run", run)
    monkeypatch.setattr(
        worker,
        "cleanup_owned_resources",
        lambda path: cleanup_calls.append(path) or {"complete": True},
    )
    request = {
        "artifact_dir": str(tmp_path),
        "source_root": str(tmp_path),
        "case": {"benchmark": "livecodebench"},
        "config": {"sandbox_image": "sha256:" + "a" * 64},
        "limits": {"case_timeout_s": 2, "max_run_seconds": 100},
    }
    with pytest.raises(ValueError, match="deadline"):
        worker._sandbox(request, {"x": 1}, "lcb")
    assert len(launches) == 1 and cleanup_calls == [tmp_path]


def test_noisy_and_stalled_generated_code_are_bounded(tmp_path):
    code = tmp_path / "noisy.py"
    code.write_text("import os\nwhile True: os.write(1, b'x'*8192)\n")
    started = time.monotonic()
    correct, tail, reason = _run_script(
        code, dict(os.environ), timeout=3, max_log_bytes=32768
    )
    assert correct is False and reason == "output_limit"
    assert len(tail) <= 65536 and time.monotonic() - started < 2
    code.write_text("import time\ntime.sleep(10)\n")
    correct, _, reason = _run_script(code, dict(os.environ), timeout=0.1)
    assert correct is False and reason == "timeout"


def test_terminal_uses_task_image_and_rejects_mutable_sidecars(tmp_path):
    original = "example/task:tag"
    image = "example/task@sha256:" + "a" * 64
    environment = SimpleNamespace(
        task_env_config=SimpleNamespace(docker_image=original),
        _env_vars=SimpleNamespace(prebuilt_image_name=original),
        environment_dir=tmp_path,
    )
    request = {
        "config": {
            "task_images": {original: image},
            "sandbox_image": "wrong@sha256:" + "b" * 64,
        }
    }
    worker._freeze_terminal_image(environment, request)
    assert environment.task_env_config.docker_image == image
    assert environment._env_vars.prebuilt_image_name == image
    (tmp_path / "docker-compose.yaml").write_text(
        "services:\n  sidecar:\n    image: mutable:latest\n"
    )
    with pytest.raises(ValueError, match="Compose services"):
        worker._freeze_terminal_image(environment, request)


def test_harbor_cancel_records_before_environment_start_and_cleans(
    tmp_path, monkeypatch
):

    task = tmp_path / "task"
    task.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    started, cleanups, configs = [], [], []
    image = "task@sha256:" + "a" * 64

    class DockerEnvironment:
        session_id = "trial__unique__env"
        environment_dir = task
        task_env_config = SimpleNamespace(docker_image=image)
        _env_vars = SimpleNamespace(prebuilt_image_name=image)

        async def start(self, force_build):
            receipt = json.loads((artifact / "owned-resources.json").read_text())
            assert receipt["compose_projects"] == [self.session_id]
            assert force_build is False
            started.append(True)
            raise asyncio.CancelledError()

    class Job:
        @classmethod
        async def create(cls, config):
            configs.append(config)
            return cls()

        async def run(self):
            await DockerEnvironment().start(force_build=True)

    modules = {
        "harbor.job": {"Job": Job},
        "harbor.models.job.config": {
            "JobConfig": SimpleNamespace(model_validate=lambda value: value)
        },
        "harbor.environments.docker.docker": {
            "DockerEnvironment": DockerEnvironment,
            "_sanitize_docker_compose_project_name": lambda value: value,
        },
    }
    for name, values in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        worker,
        "cleanup_owned_resources",
        lambda directory: cleanups.append(directory) or {"complete": True},
    )
    request = {
        "case": {"metadata": {"task_path": str(task), "tree_sha256": digest({})}},
        "artifact_dir": str(artifact),
        "config": {},
        "limits": {
            "case_timeout_s": 12,
            "max_run_seconds": 1200,
            "max_log_bytes": 1024,
        },
    }
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker._terminal_job(request))
    assert started and cleanups == [str(artifact)]
    assert configs[0]["agents"][0]["override_timeout_sec"] == 12
