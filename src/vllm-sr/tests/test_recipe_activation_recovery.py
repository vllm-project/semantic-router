import hashlib
import json
from pathlib import Path

import pytest
from cli import core
from cli import recipe_activation_recovery as recovery
from cli import recipe_activation_transaction as activation_transaction
from cli import recipe_topology_reconcile as topology
from cli.recipe_package import RECIPE_FILES, recipe_digest

TRANSACTION_ID = "0123456789abcdef0123456789abcdef"
TARGET_RECIPE_DIGEST = "sha256:" + "a" * 64
PREVIOUS_RECIPE_DIGEST = "sha256:" + "b" * 64
TARGET_CONFIG_DIGEST = "sha256:" + "c" * 64
PREVIOUS_CONFIG_DIGEST = "sha256:" + "d" * 64


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _previous_pointer(*, realized_digest: str) -> dict[str, str]:
    return {
        "schema_version": "vllm-sr/recipe-active/v1",
        "recipe_digest": PREVIOUS_RECIPE_DIGEST,
        "config_digest": PREVIOUS_CONFIG_DIGEST,
        "realized_config_digest": realized_digest,
        "activated_at": "2026-08-12T04:00:00Z",
    }


def _active_target_pointer() -> dict[str, str]:
    return {
        "schema_version": "vllm-sr/recipe-active/v1",
        "recipe_digest": TARGET_RECIPE_DIGEST,
        "config_digest": TARGET_CONFIG_DIGEST,
        "realized_config_digest": TARGET_CONFIG_DIGEST,
        "activated_at": "2026-08-12T04:01:00Z",
    }


def _write_valid_active_package(
    tmp_path: Path, *, stack_name: str = "vllm-sr"
) -> dict[str, object]:
    files = {
        "metadata.yaml": b"schema_version: vllm-sr/recipe-metadata/v1\nid: test\n",
        "config.yaml": b"version: v0.3\nlisteners:\n  - port: 8899\n",
        "probes.yaml": b"schema_version: vllm-sr/recipe-probes/v1\nname: test\n",
        "recipe.dsl": b'recipe "test" {}\n',
        "README.md": b"# Test Recipe\n",
    }
    raw_digest = _digest(files["config.yaml"])
    digest = recipe_digest(files)
    realized = b"version: v0.3\nlisteners:\n  - port: 8899\nsetup:\n  mode: false\n"

    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir(exist_ok=True)
    runtime_config = runtime_dir / recovery._runtime_config_filename(stack_name)
    runtime_config.write_bytes(realized)
    store_dir = runtime_dir / "recipe-store" / stack_name
    object_dir = store_dir / "objects" / "sha256" / digest.removeprefix("sha256:")
    object_dir.mkdir(parents=True)
    for filename in RECIPE_FILES:
        (object_dir / filename).write_bytes(files[filename])
    pointer = {
        "schema_version": "vllm-sr/recipe-active/v1",
        "recipe_digest": digest,
        "config_digest": raw_digest,
        "realized_config_digest": _digest(realized),
        "activated_at": "2026-08-12T04:01:00Z",
    }
    (store_dir / "active.json").write_text(json.dumps(pointer), encoding="utf-8")
    return {
        "files": files,
        "runtime_config": runtime_config,
        "store_dir": store_dir,
        "object_dir": object_dir,
        "pointer": pointer,
    }


def _write_crashed_activation(
    tmp_path: Path, *, previous_pointer: dict[str, str] | None
) -> tuple[Path, Path, Path, bytes, bytes]:
    previous_config = b"version: v0.3\nlisteners:\n  - port: 8899\n"
    unverified_target = b"version: v0.3\nlisteners:\n  - port: 9999\n"
    runtime_dir = tmp_path / ".vllm-sr"
    runtime_dir.mkdir()
    runtime_config = runtime_dir / "runtime-config.yaml"
    runtime_config.write_bytes(unverified_target)

    store_dir = runtime_dir / "recipe-store" / "vllm-sr"
    transaction_dir = store_dir / "transactions" / TRANSACTION_ID
    transaction_dir.mkdir(parents=True)
    backup_path = transaction_dir / "previous-config.yaml"
    backup_path.write_bytes(previous_config)
    (store_dir / "active.json").write_text(
        json.dumps(_active_target_pointer()), encoding="utf-8"
    )
    transaction: dict[str, object] = {
        "schema_version": "vllm-sr/recipe-activation-transaction/v1",
        "state": "pending",
        "operation": "activate",
        "topology_mode": "none",
        "id": TRANSACTION_ID,
        "target_recipe_digest": TARGET_RECIPE_DIGEST,
        "previous_config_digest": _digest(previous_config),
        "previous_config_backup": (
            f"transactions/{TRANSACTION_ID}/previous-config.yaml"
        ),
        "started_at": "2026-08-12T04:02:00Z",
    }
    if previous_pointer is not None:
        transaction["previous_recipe_digest"] = PREVIOUS_RECIPE_DIGEST
        transaction["previous_pointer"] = previous_pointer
    (store_dir / "activation-pending.json").write_text(
        json.dumps(transaction), encoding="utf-8"
    )
    return (
        runtime_config,
        store_dir,
        transaction_dir,
        previous_config,
        unverified_target,
    )


def _recover(runtime_config: Path, store_dir: Path, statuses=None) -> bool:
    statuses = statuses or {
        "router": "not found",
        "envoy": "exited",
        "dashboard": "created",
    }
    return recovery.recover_pending_recipe_activation(
        runtime_config_path=runtime_config,
        expected_runtime_config_path=runtime_config,
        recipe_store_dir=store_dir,
        managed_container_names=("router", "envoy", "dashboard"),
        status_provider=lambda name: statuses.get(name, "not found"),
    )


def test_recovers_crashed_activation_before_runtime_start(tmp_path: Path):
    # A pre-existing drift is valid transaction input. Its pointer digest need
    # not equal the exact runtime bytes backed up for rollback.
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, previous, _target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )

    assert _recover(runtime_config, store_dir) is True

    assert runtime_config.read_bytes() == previous
    assert (
        json.loads((store_dir / "active.json").read_text(encoding="utf-8")) == pointer
    )
    assert not (store_dir / "activation-pending.json").exists()
    assert not transaction_dir.exists()


def test_pending_recovery_rolls_back_journaled_storage_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, previous, _target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    topology_path = transaction_dir / "topology.json"
    topology_path.write_text("{}", encoding="utf-8")
    repair = {"containers": [], "service": "redis", "action": "repair"}
    observed: list[tuple[dict[str, object], bool]] = []
    monkeypatch.setattr(recovery, "load_recipe_topology", lambda _path: repair)
    monkeypatch.setattr(
        recovery,
        "rollback_recipe_topology",
        lambda state, restore_running: observed.append((state, restore_running)),
    )

    assert _recover(runtime_config, store_dir) is True

    assert runtime_config.read_bytes() == previous
    assert observed == [(repair, False)]


def test_managed_recovery_rejects_missing_topology_journal(tmp_path: Path):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, _previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    journal_path = store_dir / "activation-pending.json"
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["topology_mode"] = "managed"
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")

    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="topology does not match"
    ):
        _recover(runtime_config, store_dir)

    assert runtime_config.read_bytes() == target
    assert journal_path.is_file()
    assert transaction_dir.is_dir()


def test_recovery_without_previous_package_removes_active_pointer(tmp_path: Path):
    runtime_config, store_dir, _transaction_dir, previous, _target = (
        _write_crashed_activation(tmp_path, previous_pointer=None)
    )

    assert _recover(runtime_config, store_dir) is True

    assert runtime_config.read_bytes() == previous
    assert not (store_dir / "active.json").exists()


def test_active_recipe_package_for_stack_validates_full_state(tmp_path: Path):
    assert (
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )
        is False
    )

    active = _write_valid_active_package(tmp_path)
    assert (
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )
        is True
    )

    (active["store_dir"] / "active.json").write_text("{}", encoding="utf-8")
    with pytest.raises(recovery.RecipeActivationRecoveryError, match="prior pointer"):
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


def test_active_recipe_package_config_path_resolves_the_authored_config(
    tmp_path: Path,
):
    assert (
        recovery.active_recipe_package_config_path(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )
        is None
    )

    active = _write_valid_active_package(tmp_path)
    config_path = recovery.active_recipe_package_config_path(
        state_root_dir=tmp_path, stack_name="vllm-sr"
    )

    # The authored file, not the realized runtime config the CLI materialized.
    assert config_path == active["object_dir"] / "config.yaml"
    assert config_path.read_bytes() == active["files"]["config.yaml"]
    assert config_path.read_bytes() != active["runtime_config"].read_bytes()


def test_active_recipe_package_config_path_rejects_an_invalid_pointer(tmp_path: Path):
    active = _write_valid_active_package(tmp_path)
    (active["store_dir"] / "active.json").write_text("{}", encoding="utf-8")

    with pytest.raises(recovery.RecipeActivationRecoveryError, match="prior pointer"):
        recovery.active_recipe_package_config_path(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


def test_active_recipe_restart_rejects_runtime_drift(tmp_path: Path):
    active = _write_valid_active_package(tmp_path)
    active["runtime_config"].write_bytes(b"tampered runtime config\n")

    with pytest.raises(recovery.RecipeActivationRecoveryError, match="drifted"):
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


@pytest.mark.parametrize("missing", ["object", "file"])
def test_active_recipe_restart_rejects_missing_object_content(
    tmp_path: Path, missing: str
):
    active = _write_valid_active_package(tmp_path)
    if missing == "object":
        object_dir = active["object_dir"]
        for filename in RECIPE_FILES:
            (object_dir / filename).unlink()
        object_dir.rmdir()
    else:
        (active["object_dir"] / "README.md").unlink()

    with pytest.raises(recovery.RecipeActivationRecoveryError):
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


def test_active_recipe_restart_rejects_extra_object_entry(tmp_path: Path):
    active = _write_valid_active_package(tmp_path)
    (active["object_dir"] / "unexpected.json").write_text("{}", encoding="utf-8")

    with pytest.raises(recovery.RecipeActivationRecoveryError, match="exactly five"):
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


def test_active_recipe_restart_rejects_symlinked_object_file(tmp_path: Path):
    active = _write_valid_active_package(tmp_path)
    target = active["object_dir"] / "README.md"
    target.unlink()
    target.symlink_to(tmp_path / "outside.md")

    with pytest.raises(recovery.RecipeActivationRecoveryError, match="regular file"):
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


@pytest.mark.parametrize(
    ("filename", "message"),
    [("config.yaml", "raw config"), ("metadata.yaml", "object failed")],
)
def test_active_recipe_restart_rejects_tampered_object_bytes(
    tmp_path: Path, filename: str, message: str
):
    active = _write_valid_active_package(tmp_path)
    (active["object_dir"] / filename).write_bytes(b"tampered\n")

    with pytest.raises(recovery.RecipeActivationRecoveryError, match=message):
        recovery.active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="vllm-sr"
        )


def test_core_recovers_then_validates_before_cleanup_or_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    config = tmp_path / "config.yaml"
    config.write_text("version: v0.3\n", encoding="utf-8")
    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(
        core,
        "recover_pending_recipe_activation_for_stack",
        lambda **_kwargs: events.append("recover"),
    )

    def reject_active(**_kwargs):
        events.append("validate")
        raise recovery.RecipeActivationRecoveryError("invalid active state")

    monkeypatch.setattr(core, "active_recipe_package_for_stack", reject_active)
    monkeypatch.setattr(
        core,
        "ensure_clean_runtime_container",
        lambda _name: events.append("cleanup"),
    )
    monkeypatch.setattr(
        core,
        "provision_storage_backends",
        lambda *_args, **_kwargs: events.append("provision"),
    )

    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="invalid active state"
    ):
        core.start_vllm_sr(str(config), env_vars={})

    assert events == ["recover", "validate"]


def test_running_stack_blocks_recovery_without_mutation(tmp_path: Path):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, _transaction_dir, _previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )

    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="managed containers"
    ):
        _recover(
            runtime_config,
            store_dir,
            statuses={"router": "running", "envoy": "exited", "dashboard": "exited"},
        )

    assert runtime_config.read_bytes() == target
    assert (store_dir / "activation-pending.json").is_file()


def test_tampered_backup_aborts_without_mutation(tmp_path: Path):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, _previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    (transaction_dir / "previous-config.yaml").write_bytes(b"tampered\n")

    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="integrity validation"
    ):
        _recover(runtime_config, store_dir)

    assert runtime_config.read_bytes() == target
    assert (store_dir / "activation-pending.json").is_file()


def test_symlinked_backup_aborts_without_reading_outside_store(tmp_path: Path):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, _previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside sentinel\n", encoding="utf-8")
    backup = transaction_dir / "previous-config.yaml"
    backup.unlink()
    backup.symlink_to(outside)

    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="bounded regular file"
    ):
        _recover(runtime_config, store_dir)

    assert runtime_config.read_bytes() == target
    assert outside.read_text(encoding="utf-8") == "outside sentinel\n"
    assert (store_dir / "activation-pending.json").is_file()


def test_recovery_is_retryable_after_second_phase_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, previous, _target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    real_remove = recovery._remove_recovered_transaction

    def crash_before_journal_removal(**_kwargs):
        raise RuntimeError("simulated process crash")

    monkeypatch.setattr(
        recovery, "_remove_recovered_transaction", crash_before_journal_removal
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        _recover(runtime_config, store_dir)

    assert runtime_config.read_bytes() == previous
    assert (
        json.loads((store_dir / "active.json").read_text(encoding="utf-8")) == pointer
    )
    assert (store_dir / "activation-pending.json").is_file()
    assert transaction_dir.is_dir()

    monkeypatch.setattr(recovery, "_remove_recovered_transaction", real_remove)
    assert _recover(runtime_config, store_dir) is True
    assert not (store_dir / "activation-pending.json").exists()
    assert not transaction_dir.exists()


def test_recovery_cleanup_failure_retains_finalizing_journal_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, previous, _target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    backup = transaction_dir / "previous-config.yaml"
    real_unlink = Path.unlink

    def fail_backup_cleanup(path: Path, *args, **kwargs):
        if path == backup:
            raise OSError("simulated cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)

    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="could not be cleaned"
    ):
        _recover(runtime_config, store_dir)

    assert runtime_config.read_bytes() == previous
    journal_path = store_dir / "activation-pending.json"
    assert (
        json.loads(journal_path.read_text(encoding="utf-8"))["state"]
        == "rollback_finalizing"
    )
    assert backup.is_file()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert _recover(runtime_config, store_dir) is True
    assert not journal_path.exists()
    assert not transaction_dir.exists()


def test_committing_cleanup_failure_retries_without_replaying_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, _previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    journal_path = store_dir / "activation-pending.json"
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction.update(
        state="committing",
        operation="deactivate",
        target_recipe_digest=PREVIOUS_RECIPE_DIGEST,
        commit_config_digest=_digest(target),
    )
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    (store_dir / "active.json").unlink()
    backup = transaction_dir / "previous-config.yaml"
    real_unlink = Path.unlink

    def fail_backup_cleanup(path: Path, *args, **kwargs):
        if path == backup:
            raise OSError("simulated commit cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    with pytest.raises(
        recovery.RecipeActivationRecoveryError, match="could not be cleaned"
    ):
        _recover(runtime_config, store_dir)

    assert json.loads(journal_path.read_text(encoding="utf-8"))["state"] == "finalizing"
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert _recover(runtime_config, store_dir) is True
    assert not journal_path.exists()
    assert not transaction_dir.exists()


def test_recovery_finishes_committing_deactivation_without_rolling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    journal_path = store_dir / "activation-pending.json"
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["state"] = "committing"
    transaction["operation"] = "deactivate"
    transaction["topology_mode"] = "managed"
    transaction["target_recipe_digest"] = PREVIOUS_RECIPE_DIGEST
    transaction["commit_config_digest"] = _digest(target)
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    (store_dir / "active.json").unlink()
    topology_path = transaction_dir / "topology.json"
    names = topology._expected_container_names()
    suffix = "-recipe-backup-" + TRANSACTION_ID[:12]
    transitions = [
        {
            "service": service,
            "name": names[service],
            "backup_name": names[service] + suffix,
            "action": "replace",
            "was_running": True,
        }
        for service in ("router", "envoy")
    ]
    topology_path.write_text(
        json.dumps(
            {
                "schema_version": topology.TOPOLOGY_SCHEMA,
                "transaction_id": TRANSACTION_ID,
                "plan_digest": "sha256:" + "f" * 64,
                "listeners": [
                    {
                        "name": "public",
                        "address": "0.0.0.0",
                        "port": 8899,
                        "host_port": 8899,
                    }
                ],
                "storage_before": [],
                "storage_after": [],
                "containers": transitions,
                "credential_env": "",
            }
        ),
        encoding="utf-8",
    )
    statuses = {transition["name"]: "exited" for transition in transitions} | {
        transition["backup_name"]: "exited" for transition in transitions
    }
    removed: list[str] = []
    monkeypatch.setattr(topology, "_container_status", statuses.__getitem__)
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)

    assert _recover(runtime_config, store_dir) is True

    assert runtime_config.read_bytes() == target
    assert runtime_config.read_bytes() != previous
    assert removed == [transition["backup_name"] for transition in transitions]
    assert not journal_path.exists()
    assert not transaction_dir.exists()


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("config.yaml", "raw config"),
        ("README.md", "object failed"),
    ],
)
def test_committing_activation_rejects_tampered_object_before_topology_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    message: str,
):
    active = _write_valid_active_package(tmp_path)
    runtime_config = active["runtime_config"]
    store_dir = active["store_dir"]
    pointer = active["pointer"]
    transaction_dir = store_dir / "transactions" / TRANSACTION_ID
    transaction_dir.mkdir(parents=True)
    previous = b"version: v0.3\nlisteners:\n  - port: 7777\n"
    backup_path = transaction_dir / "previous-config.yaml"
    backup_path.write_bytes(previous)
    journal_path = store_dir / "activation-pending.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": activation_transaction.ACTIVATION_TRANSACTION_SCHEMA,
                "state": "committing",
                "operation": "activate",
                "topology_mode": "managed",
                "id": TRANSACTION_ID,
                "target_recipe_digest": pointer["recipe_digest"],
                "previous_config_digest": _digest(previous),
                "previous_config_backup": (
                    f"transactions/{TRANSACTION_ID}/previous-config.yaml"
                ),
                "commit_config_digest": pointer["realized_config_digest"],
                "started_at": "2026-08-12T04:02:00Z",
            }
        ),
        encoding="utf-8",
    )
    names = topology._expected_container_names()
    suffix = "-recipe-backup-" + TRANSACTION_ID[:12]
    transitions = [
        {
            "service": service,
            "name": names[service],
            "backup_name": names[service] + suffix,
            "action": "replace",
            "was_running": True,
        }
        for service in ("router", "envoy")
    ]
    topology_path = transaction_dir / "topology.json"
    topology_path.write_text(
        json.dumps(
            {
                "schema_version": topology.TOPOLOGY_SCHEMA,
                "transaction_id": TRANSACTION_ID,
                "plan_digest": "sha256:" + "f" * 64,
                "listeners": [
                    {
                        "name": "public",
                        "address": "0.0.0.0",
                        "port": 8899,
                        "host_port": 8899,
                    }
                ],
                "storage_before": [],
                "storage_after": [],
                "containers": transitions,
                "credential_env": "",
            }
        ),
        encoding="utf-8",
    )
    (active["object_dir"] / filename).write_bytes(b"tampered\n")
    removed: list[str] = []
    monkeypatch.setattr(topology, "_remove_if_present", removed.append)

    with pytest.raises(recovery.RecipeActivationRecoveryError, match=message):
        _recover(runtime_config, store_dir)

    assert removed == []
    assert journal_path.is_file()
    assert backup_path.is_file()
    assert topology_path.is_file()
    assert transaction_dir.is_dir()


def test_recovery_finishes_hot_switch_commit_without_topology_journal(tmp_path: Path):
    pointer = _previous_pointer(realized_digest="sha256:" + "e" * 64)
    runtime_config, store_dir, transaction_dir, previous, target = (
        _write_crashed_activation(tmp_path, previous_pointer=pointer)
    )
    journal_path = store_dir / "activation-pending.json"
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["state"] = "committing"
    transaction["operation"] = "deactivate"
    transaction["target_recipe_digest"] = PREVIOUS_RECIPE_DIGEST
    transaction["commit_config_digest"] = _digest(target)
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    (store_dir / "active.json").unlink()

    assert _recover(runtime_config, store_dir) is True

    assert runtime_config.read_bytes() == target
    assert runtime_config.read_bytes() != previous
    assert not journal_path.exists()
    assert not transaction_dir.exists()


def test_core_recovers_before_loading_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "config.yaml"
    source.write_text("version: v0.3\n", encoding="utf-8")
    runtime_config, _store_dir, _transaction_dir, previous, _target = (
        _write_crashed_activation(tmp_path, previous_pointer=None)
    )
    observed: list[bytes] = []

    class RuntimeLoadedError(Exception):
        pass

    def load_runtime(path):
        observed.append(Path(path).read_bytes())
        raise RuntimeLoadedError

    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(core, "container_status", lambda _name: "not found")
    monkeypatch.setattr(core, "container_status_strict", lambda _name: "not found")
    monkeypatch.setattr(core, "_load_runtime_config", load_runtime)

    with pytest.raises(RuntimeLoadedError):
        core.start_vllm_sr(
            str(runtime_config),
            env_vars={},
            source_config_file=str(source),
            runtime_config_file=str(runtime_config),
        )

    assert observed == [previous]
