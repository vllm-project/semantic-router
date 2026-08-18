"""Fail-closed recovery for interrupted local Recipe activations."""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path

from cli.commands.runtime_paths import (
    _atomic_write_private_bytes,
    _fsync_directory,
    _runtime_config_filename,
    write_runtime_config_bytes,
)
from cli.recipe_activation_recovery_io import (
    RecipeActivationRecoveryError,
    _digest_bytes,
    _is_digest,
    _is_rfc3339,
    _pending_journal_exists,
    _read_bounded_regular_file,
    _require_bounded_regular_file,
    _require_real_directory,
)
from cli.recipe_package import FILE_SIZE_LIMITS, RECIPE_FILES, recipe_digest
from cli.recipe_topology_reconcile import (
    TopologyReconcileError,
)
from cli.recipe_topology_reconcile import (
    _commit_offline as commit_recipe_topology_offline,
)
from cli.recipe_topology_reconcile import (
    _load_state as load_recipe_topology,
)
from cli.recipe_topology_reconcile import (
    _rollback as rollback_recipe_topology,
)
from cli.utils import get_logger

log = get_logger(__name__)

ACTIVATION_TRANSACTION_SCHEMA = "vllm-sr/recipe-activation-transaction/v1"
ACTIVE_POINTER_SCHEMA = "vllm-sr/recipe-active/v1"
MAX_ACTIVATION_TRANSACTION_BYTES = 128 * 1024
MAX_ACTIVE_POINTER_BYTES = 64 * 1024
MAX_RUNTIME_CONFIG_BYTES = 16 * 1024 * 1024

_TRANSACTION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_TRANSACTION_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "operation",
        "topology_mode",
        "id",
        "target_recipe_digest",
        "previous_recipe_digest",
        "previous_pointer",
        "previous_config_digest",
        "previous_config_backup",
        "commit_config_digest",
        "started_at",
    }
)
_TRANSACTION_REQUIRED_FIELDS = _TRANSACTION_FIELDS - {
    "previous_recipe_digest",
    "previous_pointer",
    "commit_config_digest",
}
_ACTIVE_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "recipe_digest",
        "config_digest",
        "realized_config_digest",
        "activated_at",
    }
)
_KNOWN_INACTIVE_CONTAINER_STATES = frozenset(
    {"not found", "created", "exited", "paused", "dead"}
)


def active_recipe_package_for_stack(
    *, state_root_dir: str | Path, stack_name: str
) -> bool:
    """Return whether a stack has a fully valid managed-package active state.

    The active pointer is a restart trust boundary. Validate its immutable
    object and the exact realized runtime bytes before any existing containers
    are stopped or support services are provisioned.
    """

    runtime_dir = Path(state_root_dir).expanduser().absolute() / ".vllm-sr"
    store_dir = runtime_dir / "recipe-store" / stack_name
    pointer_path = store_dir / "active.json"
    try:
        store_info = store_dir.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RecipeActivationRecoveryError(
            "The local Recipe package store could not be inspected safely."
        ) from error
    if stat.S_ISLNK(store_info.st_mode) or not stat.S_ISDIR(store_info.st_mode):
        raise RecipeActivationRecoveryError(
            "The local Recipe package store is not a real directory."
        )
    _require_real_directory(runtime_dir, "runtime state")
    _require_real_directory(store_dir.parent, "Recipe package store parent")
    try:
        encoded = _read_bounded_regular_file(
            pointer_path, MAX_ACTIVE_POINTER_BYTES, "active Recipe pointer"
        )
    except FileNotFoundError:
        return False
    try:
        pointer = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecipeActivationRecoveryError(
            "The active Recipe pointer is invalid."
        ) from error
    _validate_active_pointer(pointer)
    _validate_active_recipe_object(store_dir, pointer)
    _validate_active_runtime_config(runtime_dir, stack_name, pointer)
    return True


def _validate_active_recipe_object(store_dir: Path, pointer: dict[str, object]) -> None:
    digest_hex = str(pointer["recipe_digest"]).removeprefix("sha256:")
    objects_dir = store_dir / "objects"
    sha256_dir = objects_dir / "sha256"
    object_dir = sha256_dir / digest_hex
    _require_real_directory(objects_dir, "Recipe object store")
    _require_real_directory(sha256_dir, "Recipe SHA-256 object store")
    _require_real_directory(object_dir, "active Recipe object")

    try:
        entries = {entry.name: entry for entry in object_dir.iterdir()}
    except OSError as error:
        raise RecipeActivationRecoveryError(
            "The active Recipe object could not be inspected safely."
        ) from error
    if set(entries) != set(RECIPE_FILES):
        raise RecipeActivationRecoveryError(
            "The active Recipe object does not contain exactly five files."
        )

    files: dict[str, bytes] = {}
    for filename in RECIPE_FILES:
        try:
            data = _read_bounded_regular_file(
                entries[filename],
                FILE_SIZE_LIMITS[filename],
                f"active Recipe {filename}",
            )
        except FileNotFoundError as error:
            raise RecipeActivationRecoveryError(
                "The active Recipe object changed while it was being validated."
            ) from error
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RecipeActivationRecoveryError(
                f"The active Recipe {filename} is not valid UTF-8."
            ) from error
        files[filename] = data
    if _digest_bytes(files["config.yaml"]) != pointer["config_digest"]:
        raise RecipeActivationRecoveryError(
            "The active Recipe raw config failed integrity validation."
        )
    if recipe_digest(files) != pointer["recipe_digest"]:
        raise RecipeActivationRecoveryError(
            "The active Recipe object failed integrity validation."
        )


def _validate_active_runtime_config(
    runtime_dir: Path, stack_name: str, pointer: dict[str, object]
) -> None:
    runtime_path = runtime_dir / _runtime_config_filename(stack_name)
    try:
        runtime_config = _read_bounded_regular_file(
            runtime_path, MAX_RUNTIME_CONFIG_BYTES, "active runtime config"
        )
    except FileNotFoundError as error:
        raise RecipeActivationRecoveryError(
            "The active runtime config is missing."
        ) from error
    if _digest_bytes(runtime_config) != pointer["realized_config_digest"]:
        raise RecipeActivationRecoveryError(
            "The active runtime config drifted from the activated Recipe package."
        )


def recover_pending_recipe_activation_for_stack(
    *,
    runtime_config_path: str | Path,
    state_root_dir: str | Path,
    stack_name: str,
    managed_container_names: Iterable[str],
    status_provider: Callable[[str], str],
) -> bool:
    """Recover one stack's journal from its canonical host runtime workspace."""

    runtime_dir = Path(state_root_dir).expanduser().absolute() / ".vllm-sr"
    expected_runtime_config = runtime_dir / _runtime_config_filename(stack_name)
    return recover_pending_recipe_activation(
        runtime_config_path=runtime_config_path,
        expected_runtime_config_path=expected_runtime_config,
        recipe_store_dir=runtime_dir / "recipe-store" / stack_name,
        managed_container_names=managed_container_names,
        status_provider=status_provider,
    )


def recover_pending_recipe_activation(
    *,
    runtime_config_path: str | Path,
    expected_runtime_config_path: str | Path,
    recipe_store_dir: str | Path,
    managed_container_names: Iterable[str],
    status_provider: Callable[[str], str],
) -> bool:
    """Restore a journaled pre-activation state before local startup.

    Recovery is deliberately offline-only. The transaction journal remains in
    place until both the runtime config and prior active pointer are durable,
    making the operation safe to retry after another process crash.
    """

    store_dir = Path(recipe_store_dir).expanduser().absolute()
    journal_path = store_dir / "activation-pending.json"
    if not _pending_journal_exists(store_dir, journal_path):
        return False

    managed_container_names = tuple(managed_container_names)
    _require_inactive_containers(managed_container_names, status_provider)
    runtime_path = Path(runtime_config_path).expanduser().absolute()
    expected_runtime_path = Path(expected_runtime_config_path).expanduser().absolute()
    _validate_runtime_target(runtime_path, expected_runtime_path, store_dir)

    transaction, transaction_dir, backup_path, topology_path, topology_state = (
        _load_pending_transaction(store_dir, journal_path)
    )
    if _finish_finalizing_recovery(
        runtime_path=runtime_path,
        store_dir=store_dir,
        journal_path=journal_path,
        transaction_dir=transaction_dir,
        backup_path=backup_path,
        topology_path=topology_path,
        transaction=transaction,
    ):
        return True

    previous_config = _read_bounded_regular_file(
        backup_path, MAX_RUNTIME_CONFIG_BYTES, "activation backup"
    )
    if _digest_bytes(previous_config) != transaction["previous_config_digest"]:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation backup failed integrity validation."
        )

    _require_bounded_regular_file(
        runtime_path, MAX_RUNTIME_CONFIG_BYTES, "runtime config"
    )
    # Narrow the race with an external restart. The journal is only owned by
    # this offline recovery while every managed runtime container stays down.
    _require_inactive_containers(
        _recovery_container_names(managed_container_names, topology_state),
        status_provider,
    )
    _restore_or_commit_pending_activation(
        runtime_path=runtime_path,
        store_dir=store_dir,
        topology_path=topology_path,
        transaction=transaction,
        topology_state=topology_state,
        previous_config=previous_config,
    )

    _remove_recovered_transaction(
        store_dir=store_dir,
        journal_path=journal_path,
        transaction_dir=transaction_dir,
        backup_path=backup_path,
        topology_path=topology_path,
        transaction=transaction,
    )
    log.warning(
        "Recovered an interrupted Recipe activation before local runtime startup."
    )
    return True


def _load_pending_transaction(
    store_dir: Path, journal_path: Path
) -> tuple[dict[str, object], Path, Path, Path, dict | None]:
    transaction = _load_transaction(journal_path)
    transaction_dir = store_dir / "transactions" / str(transaction["id"])
    backup_path = transaction_dir / "previous-config.yaml"
    topology_path = transaction_dir / "topology.json"
    _validate_transaction_paths(
        store_dir, transaction_dir, backup_path, topology_path, transaction
    )
    topology_state = _load_pending_topology(topology_path)
    return transaction, transaction_dir, backup_path, topology_path, topology_state


def _load_pending_topology(topology_path: Path) -> dict | None:
    if not topology_path.exists():
        return None
    try:
        return load_recipe_topology(topology_path)
    except (OSError, ValueError, TopologyReconcileError) as error:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation topology is invalid."
        ) from error


def _finish_finalizing_recovery(
    *,
    runtime_path: Path,
    store_dir: Path,
    journal_path: Path,
    transaction_dir: Path,
    backup_path: Path,
    topology_path: Path,
    transaction: dict[str, object],
) -> bool:
    state = transaction["state"]
    if state not in {"finalizing", "rollback_finalizing"}:
        return False
    if state == "finalizing":
        _finish_committing_transaction(
            runtime_path=runtime_path,
            store_dir=store_dir,
            topology_path=topology_path,
            transaction=transaction,
            commit_topology=False,
        )
    else:
        _validate_rollback_finalizing_runtime(runtime_path, transaction)
        _restore_previous_pointer(store_dir, transaction.get("previous_pointer"))
    _remove_recovered_transaction(
        store_dir=store_dir,
        journal_path=journal_path,
        transaction_dir=transaction_dir,
        backup_path=backup_path,
        topology_path=topology_path,
        transaction=transaction,
    )
    log.warning("Finished interrupted Recipe activation cleanup.")
    return True


def _validate_rollback_finalizing_runtime(
    runtime_path: Path, transaction: dict[str, object]
) -> None:
    runtime = _read_bounded_regular_file(
        runtime_path, MAX_RUNTIME_CONFIG_BYTES, "rollback runtime config"
    )
    if _digest_bytes(runtime) != transaction["previous_config_digest"]:
        raise RecipeActivationRecoveryError(
            "The rollback-finalizing Recipe runtime config failed integrity validation."
        )


def _recovery_container_names(
    managed_container_names: Iterable[str], topology_state: dict | None
) -> tuple[str, ...]:
    names = list(managed_container_names)
    if topology_state is not None:
        for transition in topology_state["containers"]:
            names.append(transition["name"])
            if transition["backup_name"]:
                names.append(transition["backup_name"])
    return tuple(dict.fromkeys(names))


def _restore_or_commit_pending_activation(
    *,
    runtime_path: Path,
    store_dir: Path,
    topology_path: Path,
    transaction: dict[str, object],
    topology_state: dict | None,
    previous_config: bytes,
) -> None:
    try:
        if transaction["state"] == "committing":
            _finish_committing_transaction(
                runtime_path=runtime_path,
                store_dir=store_dir,
                topology_path=topology_path,
                transaction=transaction,
            )
            return
        if topology_state is not None:
            rollback_recipe_topology(topology_state, restore_running=False)
        write_runtime_config_bytes(runtime_path, previous_config)
        _restore_previous_pointer(store_dir, transaction.get("previous_pointer"))
    except RecipeActivationRecoveryError:
        raise
    except (OSError, ValueError, TopologyReconcileError) as error:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation could not restore its prior state safely."
        ) from error


def _require_inactive_containers(
    container_names: Iterable[str], status_provider: Callable[[str], str]
) -> None:
    try:
        statuses = [status_provider(name) for name in container_names]
    except Exception as error:
        raise RecipeActivationRecoveryError(
            "Managed container state could not be verified for Recipe recovery."
        ) from error
    if any(status not in _KNOWN_INACTIVE_CONTAINER_STATES for status in statuses):
        raise RecipeActivationRecoveryError(
            "A pending Recipe activation cannot be recovered while managed "
            "containers are active or their state is unknown."
        )


def _validate_runtime_target(
    runtime_path: Path, expected_runtime_path: Path, store_dir: Path
) -> None:
    if runtime_path != expected_runtime_path:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation does not match the selected runtime workspace."
        )
    runtime_dir = runtime_path.parent
    if (
        store_dir.parent.name != "recipe-store"
        or store_dir.parent.parent != runtime_dir
    ):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation store is outside the runtime workspace."
        )
    _require_real_directory(runtime_dir, "runtime state")
    _require_real_directory(store_dir.parent, "Recipe package store parent")


def _load_transaction(journal_path: Path) -> dict[str, object]:
    encoded = _read_bounded_regular_file(
        journal_path, MAX_ACTIVATION_TRANSACTION_BYTES, "activation journal"
    )
    try:
        transaction = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal is invalid."
        ) from error
    if not isinstance(transaction, dict):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal is invalid."
        )
    _normalize_legacy_transaction(transaction)
    _validate_transaction_fields(transaction)
    _validate_transaction_previous_state(transaction)
    return transaction


def _normalize_legacy_transaction(transaction: dict[str, object]) -> None:
    fields = frozenset(transaction)
    legacy_required = _TRANSACTION_REQUIRED_FIELDS - {"operation", "topology_mode"}
    legacy_allowed = _TRANSACTION_FIELDS - {
        "operation",
        "topology_mode",
        "commit_config_digest",
    }
    if legacy_required <= fields <= legacy_allowed and transaction.get("state") in {
        "pending",
        "inconsistent",
    }:
        transaction["operation"] = "activate"
        transaction["topology_mode"] = "none"


def _validate_transaction_fields(transaction: dict[str, object]) -> None:
    fields = frozenset(transaction)
    if not _TRANSACTION_REQUIRED_FIELDS <= fields <= _TRANSACTION_FIELDS:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid field set."
        )
    if transaction.get("schema_version") != ACTIVATION_TRANSACTION_SCHEMA:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal uses an unsupported schema."
        )
    _validate_transaction_mode(transaction)
    _validate_transaction_identity(transaction)
    _validate_transaction_digests(transaction)
    if not _is_rfc3339(transaction.get("started_at")):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid timestamp."
        )


def _validate_transaction_mode(transaction: dict[str, object]) -> None:
    if transaction.get("state") not in {
        "pending",
        "committing",
        "finalizing",
        "rollback_finalizing",
        "inconsistent",
    }:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid state."
        )
    if transaction.get("operation") not in {"activate", "deactivate"}:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid operation."
        )
    if transaction.get("topology_mode") not in {"none", "managed"}:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid topology mode."
        )


def _validate_transaction_identity(transaction: dict[str, object]) -> None:
    transaction_id = transaction.get("id")
    if not isinstance(transaction_id, str) or not _TRANSACTION_ID_PATTERN.fullmatch(
        transaction_id
    ):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid identity."
        )


def _validate_transaction_digests(transaction: dict[str, object]) -> None:
    for field in ("target_recipe_digest", "previous_config_digest"):
        if not _is_digest(transaction.get(field)):
            raise RecipeActivationRecoveryError(
                "The pending Recipe activation journal has an invalid digest."
            )
    commit_digest = transaction.get("commit_config_digest", "")
    if transaction.get("state") == "committing":
        if not _is_digest(commit_digest):
            raise RecipeActivationRecoveryError(
                "The pending Recipe activation has an invalid commit digest."
            )
    elif transaction.get("state") == "finalizing":
        if not _is_digest(commit_digest):
            raise RecipeActivationRecoveryError(
                "The finalizing Recipe activation has an invalid commit digest."
            )
    elif commit_digest:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation has an unexpected commit digest."
        )


def _validate_transaction_previous_state(transaction: dict[str, object]) -> None:
    previous_digest = transaction.get("previous_recipe_digest", "")
    if not isinstance(previous_digest, str) or (
        previous_digest and not _is_digest(previous_digest)
    ):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal has an invalid previous digest."
        )
    pointer = transaction.get("previous_pointer")
    if pointer is None:
        if previous_digest:
            raise RecipeActivationRecoveryError(
                "The pending Recipe activation journal has inconsistent prior state."
            )
    else:
        _validate_active_pointer(pointer)
        if pointer["recipe_digest"] != previous_digest:
            raise RecipeActivationRecoveryError(
                "The pending Recipe activation journal has inconsistent prior state."
            )


def _validate_active_pointer(pointer: object) -> None:
    if not isinstance(pointer, dict) or frozenset(pointer) != _ACTIVE_POINTER_FIELDS:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation contains an invalid prior pointer."
        )
    if pointer.get("schema_version") != ACTIVE_POINTER_SCHEMA:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation contains an invalid prior pointer."
        )
    for field in ("recipe_digest", "config_digest", "realized_config_digest"):
        if not _is_digest(pointer.get(field)):
            raise RecipeActivationRecoveryError(
                "The pending Recipe activation contains an invalid prior pointer."
            )
    if not _is_rfc3339(pointer.get("activated_at")):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation contains an invalid prior pointer."
        )


def _validate_transaction_paths(
    store_dir: Path,
    transaction_dir: Path,
    backup_path: Path,
    topology_path: Path,
    transaction: dict[str, object],
) -> None:
    expected_relative = f"transactions/{transaction['id']}/previous-config.yaml"
    if transaction.get("previous_config_backup") != expected_relative:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation has an invalid backup reference."
        )
    transactions_dir = store_dir / "transactions"
    _require_real_directory(transactions_dir, "activation transaction store")
    allowed = {backup_path, topology_path}
    if transaction.get("state") in {"finalizing", "rollback_finalizing"}:
        try:
            transaction_info = transaction_dir.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise RecipeActivationRecoveryError(
                "The activation transaction could not be inspected safely."
            ) from error
        if stat.S_ISLNK(transaction_info.st_mode) or not stat.S_ISDIR(
            transaction_info.st_mode
        ):
            raise RecipeActivationRecoveryError(
                "The activation transaction is not a real directory."
            )
        if not set(transaction_dir.iterdir()) <= allowed:
            raise RecipeActivationRecoveryError(
                "The finalizing Recipe activation has unexpected entries."
            )
        return
    _require_real_directory(transaction_dir, "activation transaction")
    entries = set(transaction_dir.iterdir())
    if backup_path not in entries or not entries <= allowed:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation transaction has unexpected entries."
        )
    has_topology = topology_path in entries
    topology_mode = transaction.get("topology_mode")
    transaction_state = transaction.get("state")
    # The topology writer durably creates topology.json before it upgrades the
    # outer transaction journal to managed mode. A crash between those writes
    # leaves a valid pending/inconsistent rollback state. The inverse cannot be
    # produced safely: managed mode without its topology journal is incomplete.
    missing_managed_topology = topology_mode == "managed" and not has_topology
    unexpected_hot_switch_topology = (
        has_topology
        and topology_mode != "managed"
        and transaction_state not in {"pending", "inconsistent"}
    )
    if missing_managed_topology or unexpected_hot_switch_topology:
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation topology does not match its journal."
        )


def _finish_committing_transaction(
    *,
    runtime_path: Path,
    store_dir: Path,
    topology_path: Path,
    transaction: dict[str, object],
    commit_topology: bool = True,
) -> None:
    runtime = _read_bounded_regular_file(
        runtime_path, MAX_RUNTIME_CONFIG_BYTES, "committing runtime config"
    )
    if _digest_bytes(runtime) != transaction["commit_config_digest"]:
        raise RecipeActivationRecoveryError(
            "The committing Recipe runtime config failed integrity validation."
        )
    is_deactivation = transaction["operation"] == "deactivate"
    pointer_path = store_dir / "active.json"
    if is_deactivation:
        try:
            _require_bounded_regular_file(
                pointer_path, MAX_ACTIVE_POINTER_BYTES, "active Recipe pointer"
            )
        except FileNotFoundError:
            pass
        else:
            raise RecipeActivationRecoveryError(
                "The committing Recipe deactivation retained an active pointer."
            )
    else:
        pointer = json.loads(
            _read_bounded_regular_file(
                pointer_path, MAX_ACTIVE_POINTER_BYTES, "active Recipe pointer"
            )
        )
        _validate_active_pointer(pointer)
        if (
            pointer["recipe_digest"] != transaction["target_recipe_digest"]
            or pointer["realized_config_digest"] != transaction["commit_config_digest"]
        ):
            raise RecipeActivationRecoveryError(
                "The committing Recipe pointer does not match its journal."
            )
        # Validate the immutable package before topology commit removes the
        # rollback containers. Startup performs the same check later, but that
        # is too late for this irreversible cleanup boundary.
        _validate_active_recipe_object(store_dir, pointer)
    if commit_topology and transaction["topology_mode"] == "managed":
        commit_recipe_topology_offline(load_recipe_topology(topology_path))


def _restore_previous_pointer(
    store_dir: Path, pointer: dict[str, object] | None
) -> None:
    pointer_path = store_dir / "active.json"
    if pointer is None:
        try:
            _require_bounded_regular_file(
                pointer_path, MAX_ACTIVE_POINTER_BYTES, "active Recipe pointer"
            )
        except FileNotFoundError:
            return
        pointer_path.unlink()
        _fsync_directory(store_dir)
        return

    encoded = (json.dumps(pointer, indent=2) + "\n").encode("utf-8")
    _atomic_write_private_bytes(pointer_path, encoded)


def _remove_recovered_transaction(
    *,
    store_dir: Path,
    journal_path: Path,
    transaction_dir: Path,
    backup_path: Path,
    topology_path: Path,
    transaction: dict[str, object],
) -> None:
    if transaction.get("state") not in {"finalizing", "rollback_finalizing"}:
        transaction = dict(transaction)
        transaction["state"] = (
            "finalizing"
            if transaction.get("state") == "committing"
            else "rollback_finalizing"
        )
        _atomic_write_private_bytes(
            journal_path, (json.dumps(transaction, indent=2) + "\n").encode("utf-8")
        )
    try:
        with suppress(FileNotFoundError):
            topology_path.unlink()
        with suppress(FileNotFoundError):
            backup_path.unlink()
        with suppress(FileNotFoundError):
            transaction_dir.rmdir()
        _fsync_directory(transaction_dir.parent)
    except RecipeActivationRecoveryError:
        raise
    except OSError as error:
        raise RecipeActivationRecoveryError(
            "The recovered Recipe activation artifacts could not be cleaned safely."
        ) from error
    try:
        _require_bounded_regular_file(
            journal_path, MAX_ACTIVATION_TRANSACTION_BYTES, "activation journal"
        )
        journal_path.unlink()
        _fsync_directory(store_dir)
    except RecipeActivationRecoveryError:
        raise
    except OSError as error:
        raise RecipeActivationRecoveryError(
            "The recovered Recipe activation journal could not be committed safely."
        ) from error
