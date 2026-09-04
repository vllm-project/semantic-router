"""Which containers can reach this stack's data services over the network.

Publishing the storage ports on loopback only closes the north-south half of
the exposure. These tests cover the east-west half: the stores sit on a second
bridge network that Envoy, Dashboard, the observability
containers, and any user-selected OpenClaw workload never join, and Router is
the single container attached to both.

The ordering assertions are the point of several of these tests, not a
stylistic preference. Router dials Postgres as its process comes up, so
attaching the data network has to happen between creating the container and
starting it, and a test that only checked the end state would pass on an
implementation that races.
"""

import subprocess
from types import SimpleNamespace

import pytest
from cli import (
    container_services,
    container_start,
    container_start_runner,
    core,
    runtime_lifecycle,
    storage_backends,
    storage_secrets,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_backends import start_storage_backends


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _minimal_stack_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n"
        "    address: 0.0.0.0\n    port: 8899\n"
    )
    return config_path


def _stub_runtime_images(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *_a, **_k: None
    )


def _capture_run_commands(monkeypatch):
    """Record every runtime command, in the order the CLI issues them."""

    captured = []

    def fake_run(cmd, **_kwargs):
        captured.append(list(cmd))
        return SimpleNamespace(stdout="container-id\n", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def _network_operations(commands):
    return [
        (cmd[2], cmd[3], cmd[4])
        for cmd in commands
        if len(cmd) > 4 and cmd[1] == "network" and cmd[2] in {"connect", "disconnect"}
    ]


def _created_container(commands, container_name):
    for index, cmd in enumerate(commands):
        if "--name" in cmd and cmd[cmd.index("--name") + 1] == container_name:
            return index, cmd
    raise AssertionError(f"no creation command for {container_name}: {commands!r}")


def _network_argument(command):
    return command[command.index("--network") + 1]


def test_serve_creates_both_stack_networks_before_provisioning_storage(
    monkeypatch, tmp_path
):
    created_networks = []
    provisioned = []

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda _path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        core,
        "provision_storage_backends",
        lambda *_a, **_k: provisioned.append(tuple(created_networks)) or set(),
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_create_network",
        lambda name: created_networks.append(name) or (0, "", ""),
    )
    monkeypatch.setattr(runtime_lifecycle, "container_status", lambda _name: "running")
    monkeypatch.setattr(core, "container_start_vllm_sr", lambda **_k: (0, "", ""))
    monkeypatch.setattr(
        runtime_lifecycle, "container_network_connect", lambda *_a: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *_a, **_k: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *_a, **_k: (0, "ok", "")
    )
    monkeypatch.setattr(runtime_lifecycle, "load_openclaw_registry", lambda *_a: [])
    monkeypatch.setattr(core, "recover_openclaw_containers", lambda *_a, **_k: None)

    core.start_vllm_sr(
        str(_minimal_stack_config(tmp_path)), env_vars={}, enable_observability=False
    )

    stack_layout = resolve_runtime_stack()
    assert created_networks == [
        stack_layout.network_name,
        stack_layout.data_network_name,
    ]
    # The data network has to exist before anything is created on it.
    assert provisioned == [(stack_layout.network_name, stack_layout.data_network_name)]


def test_the_storage_backends_are_started_on_the_data_network(monkeypatch, tmp_path):
    stack_layout = resolve_runtime_stack()
    storage_secrets.ensure_storage_secrets(
        str(tmp_path),
        stack_layout=stack_layout,
        volumes=storage_secrets.StorageVolumes(postgres="pg-data", redis="redis-data"),
    )
    networks = {}

    for backend, attribute in (
        ("redis", "container_start_redis"),
        ("postgres", "container_start_postgres"),
        ("milvus", "container_start_milvus"),
    ):
        monkeypatch.setattr(
            storage_backends,
            attribute,
            # `backend=backend` binds the loop value into each stub rather than
            # letting all three close over the final one.
            lambda network, _layout, *, backend=backend, **_kwargs: networks.update(
                {backend: network}
            )
            or (0, "", ""),
        )

    started = start_storage_backends(
        {"redis", "postgres", "milvus"}, stack_layout, state_root_dir=str(tmp_path)
    )

    assert started == {"redis", "postgres", "milvus"}
    assert networks == {
        "redis": stack_layout.data_network_name,
        "postgres": stack_layout.data_network_name,
        # Milvus has no credentials of its own yet, but it is a data service
        # and network reachability is what this network split controls.
        "milvus": stack_layout.data_network_name,
    }


def test_router_joins_the_data_network_between_being_created_and_started(
    monkeypatch, tmp_path
):
    stack_layout = resolve_runtime_stack()
    _stub_runtime_images(monkeypatch, tmp_path)
    commands = _capture_run_commands(monkeypatch)

    return_code, _stdout, _stderr = container_start.container_start_vllm_sr(
        str(_minimal_stack_config(tmp_path)),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=False,
    )

    assert return_code == 0
    created_at, router_cmd = _created_container(
        commands, stack_layout.router_container_name
    )
    # Created, not run: a second network can only be attached to a container
    # that already exists.
    assert router_cmd[1] == "create"
    assert _network_argument(router_cmd) == stack_layout.network_name

    connect_cmd = [
        "docker",
        "network",
        "connect",
        stack_layout.data_network_name,
        stack_layout.router_container_name,
    ]
    start_cmd = ["docker", "start", stack_layout.router_container_name]
    assert created_at < commands.index(connect_cmd) < commands.index(start_cmd)


def test_setup_mode_attaches_the_data_network_to_a_router_it_leaves_created(
    monkeypatch, tmp_path
):
    stack_layout = resolve_runtime_stack()
    _stub_runtime_images(monkeypatch, tmp_path)
    commands = _capture_run_commands(monkeypatch)

    container_start.container_start_vllm_sr(
        str(_minimal_stack_config(tmp_path)),
        {"VLLM_SR_SETUP_MODE": "true"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=False,
    )

    # Setup mode leaves Router created for the activation reconciler to start
    # later; the network is attached now so that later start finds both.
    assert ["docker", "start", stack_layout.router_container_name] not in commands
    assert [
        "docker",
        "network",
        "connect",
        stack_layout.data_network_name,
        stack_layout.router_container_name,
    ] in commands


def test_envoy_and_dashboard_are_created_on_the_application_network_alone(
    monkeypatch, tmp_path
):
    stack_layout = resolve_runtime_stack()
    _stub_runtime_images(monkeypatch, tmp_path)
    commands = _capture_run_commands(monkeypatch)

    container_start.container_start_vllm_sr(
        str(_minimal_stack_config(tmp_path)),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=False,
    )

    for container_name in (
        stack_layout.envoy_container_name,
        stack_layout.dashboard_container_name,
    ):
        _index, cmd = _created_container(commands, container_name)
        assert _network_argument(cmd) == stack_layout.network_name
        assert stack_layout.data_network_name not in cmd
    # Router is the only container the data network is ever attached to.
    assert [name for _op, _net, name in _network_operations(commands)] == [
        stack_layout.router_container_name
    ]


def _reusable_running_storage(monkeypatch, commands):
    """Make every managed storage container look running and safely published."""

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")

    def fake_run(cmd, **_kwargs):
        commands.append(list(cmd))
        if "inspect" in cmd:
            return SimpleNamespace(
                stdout=(
                    '{"network_mode":"default","publish_all_ports":false,'
                    '"configured":{"6379/tcp":[{"HostIp":"127.0.0.1",'
                    '"HostPort":"6379"}]},"actual":{"6379/tcp":'
                    '[{"HostIp":"127.0.0.1","HostPort":"6379"}]}}'
                ),
                stderr="",
                returncode=0,
            )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(container_services.subprocess, "run", fake_run)


def test_re_serving_an_older_stack_moves_its_storage_off_the_application_network(
    monkeypatch, tmp_path
):
    """The migration path: containers provisioned before the split are running.

    Reuse is the only path such a stack takes -- its containers are not
    rebuilt, so nothing else would ever move them -- which makes this the step
    that actually applies the isolation to every stack already in the field.
    """

    stack_layout = resolve_runtime_stack()
    storage_secrets.ensure_storage_secrets(
        str(tmp_path),
        stack_layout=stack_layout,
        volumes=storage_secrets.StorageVolumes(postgres="pg-data", redis="redis-data"),
    )
    commands = []
    _reusable_running_storage(monkeypatch, commands)

    started = start_storage_backends(
        {"redis", "postgres"}, stack_layout, state_root_dir=str(tmp_path)
    )

    assert started == {"redis", "postgres"}
    assert _network_operations(commands) == [
        ("connect", stack_layout.data_network_name, stack_layout.redis_container_name),
        (
            "disconnect",
            stack_layout.network_name,
            stack_layout.redis_container_name,
        ),
        (
            "connect",
            stack_layout.data_network_name,
            stack_layout.postgres_container_name,
        ),
        (
            "disconnect",
            stack_layout.network_name,
            stack_layout.postgres_container_name,
        ),
    ]


@pytest.mark.parametrize(
    "message",
    [
        # One wording per marker the implementation matches on. Each is the
        # steady state on every `serve` after the first migrated one, so none
        # may read as a failure. Prefix and article vary between runtimes and
        # code paths, which is why the markers are substrings rather than
        # whole phrases -- that is a property of the marker, not five
        # behaviours worth asserting separately.
        "Error response from daemon: container abc is not connected to "
        "network vllm-sr-network",
        "Error response from daemon: network vllm-sr-network not found",
        "Error: unable to find network with name or ID vllm-sr-network",
    ],
)
def test_a_storage_container_already_off_the_application_network_is_not_a_failure(
    monkeypatch, message
):
    monkeypatch.setattr(
        container_services,
        "container_network_disconnect",
        lambda _network, _container: (1, "", message),
    )

    return_code, _stdout, _stderr = (
        container_services.container_network_disconnect_if_attached(
            "vllm-sr-network", "vllm-sr-redis"
        )
    )

    assert return_code == 0


def test_a_real_disconnect_failure_fails_the_reuse(monkeypatch):
    """Fail closed: a reported migration that did not happen is worse than
    a failed `serve`, because it claims an isolation the stack does not have."""

    stack_layout = resolve_runtime_stack()
    commands = []
    _reusable_running_storage(monkeypatch, commands)
    monkeypatch.setattr(
        container_services,
        "container_network_disconnect",
        lambda _network, _container: (1, "", "permission denied"),
    )

    return_code, _stdout, stderr = container_services.container_start_redis(
        stack_layout.data_network_name,
        stack_layout,
        redis_conf_file="/unused/redis.conf",
    )

    assert return_code != 0
    assert "permission denied" in stderr


def _stop_environment(monkeypatch, stack_layout, statuses, stopped, removed):
    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core, "_managed_container_statuses", lambda _stack_layout: statuses
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(
        core, "container_stop_container", lambda name: stopped.append(name) or True
    )
    monkeypatch.setattr(
        core, "container_remove_container", lambda name: removed.append(name) or True
    )


def _all_managed_names(stack_layout):
    return (
        *stack_layout.runtime_container_names,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )


def test_stop_removes_both_stack_networks(monkeypatch):
    stack_layout = resolve_runtime_stack()
    statuses = dict.fromkeys(_all_managed_names(stack_layout), "not found")
    statuses[stack_layout.router_container_name] = "running"
    removed_networks = []
    _stop_environment(monkeypatch, stack_layout, statuses, [], [])
    monkeypatch.setattr(
        core,
        "container_remove_network",
        lambda name: removed_networks.append(name) or (0, "", ""),
    )

    core.stop_vllm_sr()

    assert removed_networks == [
        stack_layout.network_name,
        stack_layout.data_network_name,
    ]


def test_detaching_a_preserved_container_rejects_a_single_network_name():
    """The parameter is a sequence, and a string must not pass as one.

    A string is iterable, so the previous single-name signature would still
    "work" here: the helper would disconnect the container from one network
    per character, every call would fail harmlessly, and `stop` would report a
    detach that never happened while Podman kept refusing to remove the
    network.
    """

    with pytest.raises(TypeError, match="sequence of network names"):
        storage_backends.detach_preserved_storage_container(
            "vllm-sr-network", "vllm-sr-redis"
        )


def test_a_failed_creation_does_not_remove_a_container_of_the_same_name(monkeypatch):
    """Rollback owns only what this run created.

    `create` fails when the name is already taken, and the container holding
    it belongs to whoever made it. Unwinding a name this run never created
    would delete somebody else's container on the way out.
    """

    removed = []

    def fake_run(cmd, **_kwargs):
        raise subprocess.CalledProcessError(125, cmd, stderr="name already in use")

    monkeypatch.setattr(container_start_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        container_start_runner, "container_status", lambda _n: "running"
    )
    monkeypatch.setattr(
        container_start_runner, "container_stop_container", lambda _n: True
    )
    monkeypatch.setattr(
        container_start_runner, "container_remove_container", removed.append
    )

    return_code, _stdout, stderr = container_start_runner.run_container_specs(
        [("router", "vllm-sr-router-container", (["docker", "create"],))],
        storage_secret_values={},
    )

    assert return_code == 125
    assert "name already in use" in stderr
    assert removed == []


def test_a_failure_after_creation_unwinds_the_container_it_created(monkeypatch):
    """The other half of the rollback contract.

    Router now takes three commands, so a failure can land after the container
    exists -- a `network connect` against a data network that was removed out
    from under the stack, say. Nothing else would clean that container up.
    """

    removed = []
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[1] == "network":
            raise subprocess.CalledProcessError(1, cmd, stderr="network not found")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(container_start_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        container_start_runner, "container_status", lambda _n: "created"
    )
    monkeypatch.setattr(
        container_start_runner, "container_stop_container", lambda _n: True
    )
    monkeypatch.setattr(
        container_start_runner, "container_remove_container", removed.append
    )

    return_code, _stdout, _stderr = container_start_runner.run_container_specs(
        [
            (
                "router",
                "vllm-sr-router-container",
                (
                    ["docker", "create", "vllm-sr-router-container"],
                    ["docker", "network", "connect", "d", "vllm-sr-router-container"],
                    ["docker", "start", "vllm-sr-router-container"],
                ),
            )
        ],
        storage_secret_values={},
    )

    assert return_code == 1
    # The create ran, the connect failed, and the start never happened.
    assert [cmd[1] for cmd in calls] == ["create", "network"]
    assert removed == ["vllm-sr-router-container"]
