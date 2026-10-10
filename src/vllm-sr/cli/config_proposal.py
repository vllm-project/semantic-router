"""Build a reviewable Router configuration proposal.

A bounded intent becomes candidate YAML, a redacted diff, provenance, and a
validation receipt. This module does not call a model, invent fields, write
the source config, or apply anything to a running Router.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

from cli.bootstrap import SETUP_MODE_KEY, validate_setup_metadata
from cli.catalog_provider_projection import provider_projection_errors
from cli.config_schema.validation import validate_config_structure
from cli.config_yaml import safe_load_router_config
from cli.model_catalog_validation import (
    redact_embedded_secret_literals,
)
from cli.parser import ConfigParseError, parse_user_config
from cli.terminal import echo
from cli.url_display import redact_url
from cli.validator import validate_user_config

PROPOSAL_SCHEMA = "vllm-sr/config-proposal/v1"
CANONICAL_VERSION = "v0.3"
GENERATOR = "vllm-sr-config-proposal"
# Textual until the canonical side-effect-free config diff API replaces it.
DIFF_SOURCE = "textual-unified-until-3477"

# Closed catalog. Each intent points at one maintained fragment. Adding an
# intent does not invent Router fields.
_INTENTS: dict[str, dict[str, str]] = {
    "selection.latency-aware": {
        "kind": "fragment",
        "fragment": "fragments/algorithm/selection/latency-aware.yaml",
        "source": "config/fragments/algorithm/selection/latency-aware.yaml",
        "summary": (
            "Attach the maintained latency-aware selection algorithm to one "
            "existing decision."
        ),
    },
}

# The privacy package keeps model cards and provider models on the document
# surfaces that own them. recipes[].routing cannot take that block directly.
_WITHHELD_INTENTS = {
    "recipe.privacy": (
        "unsupported: intent 'recipe.privacy' is withheld until a mapper "
        "places the privacy package's model cards and provider models on "
        "the document surfaces that own them. recipes[].routing cannot "
        "take that routing block directly."
    ),
}

# Same credential key names as cli.recipe_package. Callers lowercase first.
_CREDENTIAL_KEY_PATTERN = re.compile(
    r"(?:api_?key|access_?key|access_?token|authorization|proxy_?authorization|"
    r"cookie|client_?secret|password|secret|token|private_?key|x_?api_?key)$"
)
_CREDENTIAL_COLLECTION_KEYS = frozenset({"api_keys"})
# Lines shorter than this are ordinary words. The same floor is used for
# embedded api_key literals in catalog validation.
_MIN_SECRET_LINE_LENGTH = 8
_URL_IN_TEXT = re.compile(r"[a-z][a-z0-9+.-]*://[^\s\"']+", re.IGNORECASE)
_PRIVATE_PATH = re.compile(r"(?:~|/(?:home|Users|root|private)/)[^\s\"']+")


class ProposalUnsupportedError(ValueError):
    """The request is outside the maintained proposal contract."""


def supported_intents() -> tuple[str, ...]:
    """Return the intent ids this slice can propose."""

    return tuple(sorted(_INTENTS))


def propose(
    config_path: str | Path,
    intent: str,
    decision: str | None = None,
    *,
    recipe: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build one proposal document without writing the source config.

    The returned candidate YAML, diff, and diagnostics are redacted.
    Validation runs on the unredacted candidate in a temporary file that is
    deleted before return.
    """

    root = _repo_root(repo_root)
    source = Path(config_path)
    if not source.is_file():
        raise ProposalUnsupportedError(
            f"unsupported: configuration file {source.name!r} was not found"
        )

    raw = source.read_bytes()
    loaded = _load_yaml(raw, label=source.name)
    if not isinstance(loaded, dict):
        raise ProposalUnsupportedError(
            "unsupported: configuration file is not a canonical mapping"
        )
    version = loaded.get("version")
    if version != CANONICAL_VERSION:
        raise ProposalUnsupportedError(
            f"unsupported: config version {version!r} is not canonical "
            f"{CANONICAL_VERSION}"
        )

    withheld = _WITHHELD_INTENTS.get(intent)
    if withheld is not None:
        raise ProposalUnsupportedError(withheld)

    spec = _INTENTS.get(intent)
    if spec is None:
        supported = ", ".join(supported_intents())
        raise ProposalUnsupportedError(
            f"unsupported: intent {intent!r} is not a maintained proposal. "
            f"Supported intents: {supported}"
        )

    candidate = copy.deepcopy(loaded)
    with _intent_assets(root) as assets:
        if spec["kind"] == "fragment":
            if recipe is not None or not decision:
                raise ProposalUnsupportedError(
                    f"unsupported: intent {intent!r} changes one existing "
                    "routing decision"
                )
            fragment_rel = spec["fragment"]
            fragment_path = _maintained_fragment(assets, fragment_rel)
            source_raw = fragment_path.read_bytes()
            fragment = _load_yaml(source_raw, label=spec["source"])
            _attach_algorithm(
                candidate, decision, _fragment_algorithm(fragment, spec["source"])
            )
            source_record = {
                "kind": "fragment",
                "path": spec["source"],
                "sha256": _sha256(source_raw),
            }
            intent_record = {
                "id": intent,
                "summary": spec["summary"],
                "decision": decision,
            }
        else:
            raise ProposalUnsupportedError(
                f"unsupported: intent {intent!r} has no maintained applier"
            )

    secrets = _secret_literals(loaded) | _secret_literals(candidate)
    # Mask copied values before pattern redaction. A URL or PEM pattern would
    # otherwise rewrite part of the credential and the exact value would miss.
    # Multiline secrets are not replaced again after dumping: their trailing
    # newline is also the YAML line break, so that second pass splices lines.
    emitted = _single_line_secrets(secrets)
    before_yaml = _mask_known_literals(
        _dump_yaml(_redact(_mask_known_values(loaded, secrets))), emitted
    )
    after_yaml = _mask_known_literals(
        _dump_yaml(_redact(_mask_known_values(candidate, secrets))), emitted
    )
    diff = _mask_known_literals(_unified_diff(before_yaml, after_yaml), emitted)
    validation = _validation_receipt(candidate, secrets)

    return {
        "schema": PROPOSAL_SCHEMA,
        "applied": False,
        "intent": intent_record,
        "provenance": {
            "input": {
                "name": _public_path(source, root),
                "sha256": _sha256(raw),
                "version": CANONICAL_VERSION,
            },
            "sources": [source_record],
            "generator": GENERATOR,
            "diff_source": DIFF_SOURCE,
            "candidate_redacted": True,
        },
        "candidate_yaml": after_yaml,
        "diff": diff,
        "validation": validation,
    }


def propose_config_command(
    config_path: str,
    intent: str,
    decision: str | None = None,
    recipe: str | None = None,
) -> bool:
    """Print one proposal as JSON. Return whether validation passed."""

    document = propose(config_path, intent, decision, recipe=recipe)
    echo(json.dumps(document, indent=2, sort_keys=True) + "\n", nl=False)
    return document["validation"]["status"] == "valid"


def _repo_root(repo_root: Path | None) -> Path | None:
    """Return a source checkout when this module lives in one."""

    if repo_root is not None:
        return repo_root
    module_directory = Path(__file__).resolve().parent
    for repository_root in module_directory.parents:
        source_package = repository_root / "src" / "vllm-sr" / "cli"
        try:
            if source_package.resolve() == module_directory:
                return repository_root
        except OSError:
            continue
    return None


@contextmanager
def _intent_assets(repo_root: Path | None) -> Iterator[Path]:
    """Yield the directory that contains the latency-aware fragment.

    A source checkout reads ``config/`` in the repository. An installed CLI
    reads the same tree staged into ``cli.proposal_assets``.
    """

    if repo_root is not None:
        yield repo_root / "config"
        return
    packaged = files("cli.proposal_assets")
    marker = packaged.joinpath(
        "fragments", "algorithm", "selection", "latency-aware.yaml"
    )
    try:
        packaged_ready = marker.is_file()
    except OSError:
        packaged_ready = False
    if not packaged_ready:
        raise ProposalUnsupportedError(
            "unsupported: proposal assets are not installed with the CLI"
        )
    with as_file(packaged) as path:
        yield Path(path)


def _public_path(path: Path, repo_root: Path | None) -> str:
    """Return a repo-relative path, or only the file name outside the repo."""

    if repo_root is None:
        return path.name
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_yaml(raw: bytes, *, label: str) -> Any:
    try:
        return safe_load_router_config(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProposalUnsupportedError(
            f"unsupported: {label} is not UTF-8 YAML"
        ) from exc
    except yaml.YAMLError as exc:
        raise ProposalUnsupportedError(
            f"unsupported: {label} is not valid YAML"
        ) from exc


def _maintained_fragment(assets_root: Path, relative: str) -> Path:
    fragments_root = (assets_root / "fragments").resolve()
    path = (assets_root / relative).resolve()
    if not path.is_relative_to(fragments_root) or not path.is_file():
        raise ProposalUnsupportedError(
            f"unsupported: fragment {relative!r} is not a maintained config fragment"
        )
    return path


def _fragment_algorithm(fragment: Any, relative: str) -> dict[str, Any]:
    if not isinstance(fragment, dict) or set(fragment) != {"algorithm"}:
        raise ProposalUnsupportedError(
            f"unsupported: maintained fragment {relative!r} is not a single "
            "algorithm block"
        )
    algorithm = fragment["algorithm"]
    if not isinstance(algorithm, dict):
        raise ProposalUnsupportedError(
            f"unsupported: maintained fragment {relative!r} has no algorithm mapping"
        )
    return algorithm


def _attach_algorithm(
    candidate: dict[str, Any], decision_name: str, algorithm: dict[str, Any]
) -> None:
    routing = candidate.get("routing")
    decisions = routing.get("decisions") if isinstance(routing, dict) else None
    if not isinstance(decisions, list):
        raise ProposalUnsupportedError(
            "unsupported: canonical config has no routing.decisions list, "
            "and proposals do not invent decisions"
        )
    matches = [
        item
        for item in decisions
        if isinstance(item, dict) and item.get("name") == decision_name
    ]
    if len(matches) != 1:
        raise ProposalUnsupportedError(
            f"unsupported: decision {decision_name!r} is not in the canonical "
            "config, and proposals do not invent decisions"
        )
    matches[0]["algorithm"] = copy.deepcopy(algorithm)


def _mask_known_literals(text: str, secrets: set[str]) -> str:
    """Replace credential values copied out of secret fields."""

    for secret in _ordered_secrets(secrets):
        text = text.replace(secret, "***")
    return text


def _mask_known_values(value: Any, secrets: set[str]) -> Any:
    """Mask known credentials inside string values before YAML serialization."""

    if isinstance(value, dict):
        return {key: _mask_known_values(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_known_values(item, secrets) for item in value]
    if isinstance(value, str):
        return _mask_known_literals(value, secrets)
    return value


def _ordered_secrets(secrets: set[str]) -> list[str]:
    return sorted((item for item in secrets if item), key=len, reverse=True)


def _single_line_secrets(secrets: set[str]) -> set[str]:
    return {item for item in secrets if item and "\n" not in item and "\r" not in item}


def _dump_yaml(data: Any) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=4096,
    )


def _redact(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, list):
        if _is_secret_key(parent_key):
            return [_redact_secret_item(item) for item in value]
        return [_redact(item, parent_key) for item in value]
    if isinstance(value, dict):
        if _is_secret_key(parent_key):
            return "***"
        return {key: _redact(item, key) for key, item in value.items()}
    if _is_secret_key(parent_key) and value not in (None, ""):
        return "***"
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_secret_item(item: Any) -> Any:
    if isinstance(item, str) and item:
        return "***"
    if item in (None, ""):
        return item
    return _redact(item)


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    if normalized in _CREDENTIAL_COLLECTION_KEYS:
        return True
    return _CREDENTIAL_KEY_PATTERN.search(normalized) is not None


def _redact_text(text: str) -> str:
    text = _URL_IN_TEXT.sub(lambda match: redact_url(match.group(0)), text)
    text = _PRIVATE_PATH.sub("***", text)
    return redact_embedded_secret_literals(text)


def _secret_literals(value: Any, parent_key: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if _is_secret_key(parent_key):
            _remember_secret_text(found, value)
            return found
        for key, item in value.items():
            found.update(_secret_literals(item, key))
        return found
    if isinstance(value, list):
        for item in value:
            if _is_secret_key(parent_key):
                _remember_secret_text(found, item)
            else:
                found.update(_secret_literals(item, parent_key))
        return found
    if isinstance(value, str) and value and _is_secret_key(parent_key):
        _remember_secret_text(found, value)
    return found


def _remember_secret_text(found: set[str], value: Any) -> None:
    """Record a credential and each substantial line of a multiline one."""

    if isinstance(value, str) and value:
        found.add(value)
        if "\n" in value or "\r" in value:
            normalized = value.replace("\r\n", "\n").replace("\r", "\n")
            for raw_line in normalized.split("\n"):
                stripped = raw_line.strip()
                if len(stripped) >= _MIN_SECRET_LINE_LENGTH:
                    found.add(stripped)
        return
    if isinstance(value, dict):
        for item in value.values():
            _remember_secret_text(found, item)
        return
    if isinstance(value, list):
        for item in value:
            _remember_secret_text(found, item)


def _redact_diagnostics(
    diagnostics: list[str], secrets: set[str], temp_path: Path | None
) -> list[str]:
    cleaned: list[str] = []
    ordered_secrets = sorted((item for item in secrets if item), key=len, reverse=True)
    for item in diagnostics:
        text = item
        if temp_path is not None:
            text = text.replace(str(temp_path), "candidate.yaml")
        for secret in ordered_secrets:
            text = text.replace(secret, "***")
        cleaned.append(_redact_text(text))
    return cleaned


def _unified_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )


def _validation_receipt(candidate: dict[str, Any], secrets: set[str]) -> dict[str, Any]:
    schema_errors = validate_setup_metadata(candidate.get(SETUP_MODE_KEY))
    router_payload = {
        key: value for key, value in candidate.items() if key != SETUP_MODE_KEY
    }
    schema_errors.extend(validate_config_structure(router_payload))
    checks: list[dict[str, Any]] = [
        {
            "name": "schema",
            "ok": not schema_errors,
            "diagnostics": _redact_diagnostics(list(schema_errors), secrets, None),
        }
    ]
    if schema_errors:
        checks.append(
            _skipped_check("canonical_parse", "schema validation failed", secrets)
        )
        checks.append(_skipped_check("policy", "schema validation failed", secrets))
        return {"status": "invalid", "checks": checks}

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            delete=False,
        ) as handle:
            handle.write(_dump_yaml(candidate))
            temp_path = Path(handle.name)
        try:
            parsed = parse_user_config(str(temp_path), log_summary=False)
        except ConfigParseError as exc:
            checks.append(
                {
                    "name": "canonical_parse",
                    "ok": False,
                    "diagnostics": _redact_diagnostics([str(exc)], secrets, temp_path),
                }
            )
            checks.append(_skipped_check("policy", "canonical parsing failed", secrets))
            return {"status": "invalid", "checks": checks}
        checks.append({"name": "canonical_parse", "ok": True, "diagnostics": []})
        policy_errors = list(validate_user_config(parsed, log_summary=False))
        if not policy_errors:
            policy_errors.extend(provider_projection_errors(parsed))
        checks.append(
            {
                "name": "policy",
                "ok": not policy_errors,
                "diagnostics": _redact_diagnostics(
                    [str(error) for error in policy_errors], secrets, temp_path
                ),
            }
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    status = "valid" if all(check["ok"] for check in checks) else "invalid"
    return {"status": status, "checks": checks}


def _skipped_check(name: str, reason: str, secrets: set[str]) -> dict[str, Any]:
    return {
        "name": name,
        "ok": False,
        "diagnostics": _redact_diagnostics(
            [f"skipped because {reason}"], secrets, None
        ),
    }
