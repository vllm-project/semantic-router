import json
import subprocess

import pytest
from cli import container_mounts
from cli.container_mounts import (
    ContainerMountsUnavailableError,
    inspect_container_mounts,
)


def _runtime(monkeypatch: pytest.MonkeyPatch, name: str, fake_run) -> None:
    monkeypatch.setattr(container_mounts, "get_container_runtime", lambda: name)
    monkeypatch.setattr(container_mounts.subprocess, "run", fake_run)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_the_lookup_asks_for_a_container_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, runtime: str
):
    commands = []
    mounts = [{"Type": "volume", "Destination": "/data", "Name": "vllm-sr-redis-data"}]

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(mounts), stderr=""
        )

    _runtime(monkeypatch, runtime, fake_run)

    assert inspect_container_mounts("vllm-sr-redis") == mounts
    assert commands == [
        [
            runtime,
            "container",
            "inspect",
            "--format",
            "{{json .Mounts}}",
            "vllm-sr-redis",
        ]
    ]


@pytest.mark.parametrize(
    "runtime, stderr",
    [
        ("docker", "Error: No such container: vllm-sr-redis"),
        ("podman", "Error: no such container vllm-sr-redis"),
        (
            "podman",
            'Error: no container with name or ID "vllm-sr-redis" found: '
            "no such container",
        ),
    ],
)
def test_each_runtime_wording_of_a_missing_container_reads_as_absent(
    monkeypatch: pytest.MonkeyPatch, runtime: str, stderr: str
):
    def fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(125, command, stderr=stderr)

    _runtime(monkeypatch, runtime, fake_run)

    with pytest.raises(ContainerMountsUnavailableError) as raised:
        inspect_container_mounts("vllm-sr-redis")

    assert raised.value.container_absent


@pytest.mark.parametrize(
    "stderr",
    [
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
        "Is the docker daemon running?",
        'Error: template: inspect:1:18: executing "inspect" at <.Mounts>: '
        "can't evaluate field Mounts in type interface {}",
    ],
)
def test_a_reply_that_is_not_about_a_missing_container_is_not_absence(
    monkeypatch: pytest.MonkeyPatch, stderr: str
):
    def fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(125, command, stderr=stderr)

    _runtime(monkeypatch, "podman", fake_run)

    with pytest.raises(ContainerMountsUnavailableError) as raised:
        inspect_container_mounts("vllm-sr-redis")

    assert not raised.value.container_absent
