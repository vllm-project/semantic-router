"""What a Recipe activation journal must look like before it can be trusted.

An activation writes a journal, mutates the store, then clears the journal, so a
crash leaves a document that recovery has to read before it does anything
destructive. Every rule that decides whether that document is coherent lives
here -- the schema, the field set per state, the digests, and the on-disk
entries the transaction directory is allowed to contain -- separate from
`recipe_activation_recovery`, which decides what to *do* about a journal that
passes.

Every rule fails closed: an unrecognized shape raises rather than being repaired
or ignored, because the alternative is acting on a half-written transaction.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

from cli.recipe_activation_recovery_io import (
    RecipeActivationRecoveryError,
    _is_digest,
    _is_rfc3339,
    _read_bounded_regular_file,
    _require_real_directory,
)

ACTIVATION_TRANSACTION_SCHEMA = "vllm-sr/recipe-activation-transaction/v1"
ACTIVE_POINTER_SCHEMA = "vllm-sr/recipe-active/v1"
MAX_ACTIVATION_TRANSACTION_BYTES = 128 * 1024

_TRANSACTION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
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
