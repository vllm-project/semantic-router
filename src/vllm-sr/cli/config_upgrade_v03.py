"""Offline rewrite from the previous v0.3 contract to strict current v0.3."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from cli.models import UserConfig
from cli.validator import validate_user_config

CONTRACT_VERSION = "v0.3"
_OLD_PRICING_FIELDS = {
    "prompt_per_1m": "input_cost_per_million_tokens",
    "completion_per_1m": "output_cost_per_million_tokens",
    "cached_input_per_1m": "cache_read_cost_per_million_tokens",
    "cache_write_per_1m": "cache_write_cost_per_million_tokens",
}
_NEW_PRICING_FIELDS = frozenset(_OLD_PRICING_FIELDS.values())
_OLD_RELIABILITY_FIELDS = frozenset(
    {
        "lb_policy",
        "retry_count",
        "retry_on",
        "consecutive_5xx",
        "base_ejection_time",
        "max_ejection_percent",
        "health_check_path",
        "health_check_interval",
        "health_check_timeout",
    }
)
_RETRY_EVIDENCE = frozenset({"unavailable", "timeout"})
_CURRENCY_CODE_LENGTH = 3
_PLAINTEXT_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "client_secret",
        "encryption_key",
        "hmac_key",
        "password",
        "private_key",
        "reveal_key",
        "secret",
        "signing_key",
        "token",
        "tokens",
    }
)
_SECRET_HEADER_NAMES = frozenset({"authorization", "proxy-authorization", "x-api-key"})
_PUBLIC_PROVIDER_API_KEY_PATH = re.compile(
    r"^config\.providers\.models\[\d+\]\.backend_refs\[\d+\]\.api_key$"
)


@dataclass(frozen=True)
class MigrationIssue:
    """One actionable reason a rewrite cannot be proven safe."""

    code: str
    path: str
    message: str
    resolution: str


class ConfigMigrationError(RuntimeError):
    """Raised when an offline v0.3 rewrite cannot preserve behavior safely."""

    def __init__(self, issues: list[MigrationIssue]):
        self.issues = issues
        rendered = "; ".join(
            f"{issue.path}: {issue.message} ({issue.resolution})" for issue in issues
        )
        super().__init__(rendered)


@dataclass(frozen=True)
class MigrationSummary:
    """Bounded success metadata for one completed v0.3 rewrite."""

    source_version: str
    target_version: str
    models: int
    pricing_blocks: int
    control_blocks: int
    removed_noop_fields: int


@dataclass(frozen=True)
class MigrationResult:
    """Validated strict-v0.3 document and rewrite summary."""

    document: dict[str, Any]
    summary: MigrationSummary


def migrate_v03_config_data(source: dict[str, Any]) -> MigrationResult:
    """Rewrite exactly one previous-release v0.3 mapping into current v0.3."""

    if not isinstance(source, dict):
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "invalid_root",
                    "config",
                    "configuration root must be a mapping",
                    "provide one canonical v0.3 YAML document",
                )
            ]
        )
    if source.get("version") != CONTRACT_VERSION:
        raise ConfigMigrationError(
            [
                MigrationIssue(
                    "unsupported_source_version",
                    "version",
                    f"expected exactly {CONTRACT_VERSION}, found {source.get('version')!r}",
                    "upgrade to the previous canonical v0.3 release first",
                )
            ]
        )

    document = deepcopy(source)
    issues: list[MigrationIssue] = []
    _reject_removed_static_access(document, issues)
    removed_noop_fields = _rewrite_removed_router_fields(document, issues)
    _scan_plaintext_secrets(document, issues)
    models = _provider_models(document, issues)
    (
        currencies,
        pricing_blocks,
        control_blocks,
        removed_model_fields,
    ) = _rewrite_provider_models(models, issues)
    removed_noop_fields += removed_model_fields

    _merge_billing_currency(document, currencies, issues)
    if issues:
        raise ConfigMigrationError(issues)
    _validate_current_document(document)
    return MigrationResult(
        document=document,
        summary=MigrationSummary(
            source_version=CONTRACT_VERSION,
            target_version=CONTRACT_VERSION,
            models=len(models),
            pricing_blocks=pricing_blocks,
            control_blocks=control_blocks,
            removed_noop_fields=removed_noop_fields,
        ),
    )


def _provider_models(
    document: dict[str, Any], issues: list[MigrationIssue]
) -> list[Any]:
    providers = document.get("providers")
    if providers is None:
        providers = {}
        document["providers"] = providers
    if not isinstance(providers, dict):
        issues.append(
            MigrationIssue(
                "invalid_providers",
                "providers",
                "providers must be a mapping",
                "repair the previous v0.3 document before migration",
            )
        )
        return []
    models = providers.get("models") or []
    if not isinstance(models, list):
        issues.append(
            MigrationIssue(
                "invalid_models",
                "providers.models",
                "providers.models must be a list",
                "repair the previous v0.3 document before migration",
            )
        )
        return []
    return models


def _rewrite_provider_models(
    models: list[Any], issues: list[MigrationIssue]
) -> tuple[set[str], int, int, int]:
    currencies: set[str] = set()
    pricing_blocks = 0
    control_blocks = 0
    removed_noop_fields = 0
    for index, model in enumerate(models):
        path = f"providers.models[{index}]"
        if not isinstance(model, dict):
            issues.append(
                MigrationIssue(
                    "invalid_model",
                    path,
                    "Provider Model must be a mapping",
                    "repair the previous v0.3 document before migration",
                )
            )
            continue
        if "reliability" in model:
            control, removed = _rewrite_reliability(
                model.pop("reliability"), path, issues
            )
            removed_noop_fields += removed
            if control:
                if "control" in model and model["control"] != control:
                    issues.append(
                        MigrationIssue(
                            "control_collision",
                            f"{path}.control",
                            "existing control conflicts with migrated reliability",
                            "keep one unambiguous invocation-control policy",
                        )
                    )
                else:
                    model["control"] = control
                    control_blocks += 1
        pricing = model.get("pricing")
        if pricing is not None:
            rewritten, currency = _rewrite_pricing(pricing, path, issues)
            model["pricing"] = rewritten
            pricing_blocks += 1
            if currency:
                currencies.add(currency)
    return currencies, pricing_blocks, control_blocks, removed_noop_fields


def _reject_removed_static_access(
    document: dict[str, Any], issues: list[MigrationIssue]
) -> None:
    for index, listener in enumerate(document.get("listeners") or []):
        if isinstance(listener, dict) and "api_keys" in listener:
            if listener.get("api_keys") not in (None, []):
                issues.append(
                    MigrationIssue(
                        "removed_static_api_keys",
                        f"listeners[{index}].api_keys",
                        "static inference API keys cannot be rewritten safely",
                        "provision keys and grants through the Management API, then remove this field",
                    )
                )
            else:
                listener.pop("api_keys")
    global_config = document.get("global")
    if not isinstance(global_config, dict):
        return
    for field_name in ("authz", "ratelimit", "rate_limit", "rate_limits"):
        if field_name not in global_config:
            continue
        if global_config.get(field_name) not in (None, {}, []):
            issues.append(
                MigrationIssue(
                    "removed_static_access_policy",
                    f"global.{field_name}",
                    "static inference authorization and rate limits were removed",
                    "create equivalent dynamic access and quota resources through the Management API",
                )
            )
        else:
            global_config.pop(field_name)


def _rewrite_removed_router_fields(
    document: dict[str, Any], issues: list[MigrationIssue]
) -> int:
    """Remove inert legacy fields and fail closed for behavior-bearing values."""

    global_config = document.get("global")
    if not isinstance(global_config, dict):
        return 0
    router = global_config.get("router")
    if not isinstance(router, dict):
        return 0

    return _rewrite_config_source(router, issues) + _rewrite_skip_processing(
        router, issues
    )


def _rewrite_config_source(router: dict[str, Any], issues: list[MigrationIssue]) -> int:
    if "config_source" not in router:
        return 0
    config_source = router.get("config_source")
    if config_source in (None, "", "file"):
        router.pop("config_source")
        return 1
    if config_source == "kubernetes":
        issues.append(
            MigrationIssue(
                "removed_kubernetes_config_source",
                "global.router.config_source",
                "Kubernetes config_source cannot be preserved by the current single-authority contract",
                "use a file bootstrap, configure global.stores.management, and apply later changes through the versioned Management API",
            )
        )
    else:
        issues.append(
            MigrationIssue(
                "invalid_config_source",
                "global.router.config_source",
                f"unsupported previous config_source {config_source!r}",
                "use file for an offline rewrite, or migrate Kubernetes state explicitly through the Management API",
            )
        )
    return 0


def _rewrite_skip_processing(
    router: dict[str, Any], issues: list[MigrationIssue]
) -> int:
    if "skip_processing" not in router:
        return 0
    skip_processing = router.get("skip_processing")
    if skip_processing in (None, {}):
        router.pop("skip_processing")
        return 1
    if not isinstance(skip_processing, dict):
        issues.append(
            MigrationIssue(
                "invalid_skip_processing",
                "global.router.skip_processing",
                "skip_processing must be a mapping with only an enabled boolean",
                "remove the field after confirming no caller depends on a processing bypass",
            )
        )
        return 0

    unknown = set(skip_processing) - {"enabled"}
    if unknown:
        issues.append(
            MigrationIssue(
                "invalid_skip_processing",
                "global.router.skip_processing",
                "unsupported fields: " + ", ".join(sorted(unknown)),
                "remove the field after confirming no caller depends on a processing bypass",
            )
        )
        return 0
    enabled = skip_processing.get("enabled", False)
    if enabled is False or enabled is None:
        router.pop("skip_processing")
        return 1
    if enabled is True:
        issues.append(
            MigrationIssue(
                "removed_enabled_skip_processing",
                "global.router.skip_processing.enabled",
                "request-controlled processing bypass would evade Router authentication, authorization, and quota enforcement",
                "remove the bypass and use an authenticated health or management endpoint for operational traffic",
            )
        )
        return 0
    issues.append(
        MigrationIssue(
            "invalid_skip_processing",
            "global.router.skip_processing.enabled",
            "enabled must be a boolean",
            "set enabled to false before migration, then rerun",
        )
    )
    return 0


def _scan_plaintext_secrets(
    value: Any,
    issues: list[MigrationIssue],
    path: str = "config",
    *,
    header_map: bool = False,
) -> None:
    """Reject secret values that cannot be copied safely into a new file."""

    if isinstance(value, dict):
        for raw_name, child in value.items():
            name = str(raw_name)
            normalized = name.strip().lower().replace("-", "_")
            child_path = f"{path}.{name}"
            supported_provider_api_key = (
                normalized == "api_key"
                and _PUBLIC_PROVIDER_API_KEY_PATH.fullmatch(child_path) is not None
            )
            secret_field = (
                normalized in _PLAINTEXT_SECRET_FIELDS
                and not (
                    normalized in {"token", "tokens"} and _is_secret_reference(child)
                )
                and not supported_provider_api_key
            )
            secret_header = header_map and name.strip().lower() in _SECRET_HEADER_NAMES
            if (secret_field or secret_header) and child not in (None, "", [], {}):
                issues.append(
                    MigrationIssue(
                        "plaintext_secret",
                        child_path,
                        "plaintext secret cannot be copied into migrated output",
                        "move the value to an environment or file reference, then rerun",
                    )
                )
                continue
            _scan_plaintext_secrets(
                child,
                issues,
                child_path,
                header_map=normalized in {"headers", "extra_headers"},
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_plaintext_secrets(child, issues, f"{path}[{index}]")


def _is_secret_reference(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_is_secret_reference(item) for item in value)
    if not isinstance(value, dict):
        return False
    keys = {str(key).strip().lower() for key in value}
    return bool(keys & {"env", "file", "path"}) and keys <= {
        "env",
        "file",
        "name",
        "path",
        "role",
    }


def _rewrite_reliability(
    value: Any,
    model_path: str,
    issues: list[MigrationIssue],
) -> tuple[dict[str, Any], int]:
    path = f"{model_path}.reliability"
    if not isinstance(value, dict):
        issues.append(
            MigrationIssue(
                "invalid_reliability",
                path,
                "reliability must be a mapping",
                "repair the previous v0.3 document before migration",
            )
        )
        return {}, 0
    unknown = set(value) - _OLD_RELIABILITY_FIELDS
    if unknown:
        issues.append(
            MigrationIssue(
                "unknown_reliability_fields",
                path,
                "unsupported fields: " + ", ".join(sorted(unknown)),
                "remove fields that were never part of the previous v0.3 contract",
            )
        )

    parse_only_fields = _OLD_RELIABILITY_FIELDS - {"retry_count", "retry_on"}
    removed = sum(value.get(field) not in (None, "", 0) for field in parse_only_fields)
    control: dict[str, Any] = {}

    retry_count = value.get("retry_count", 0)
    if retry_count:
        triggers = _rewrite_retry_on(value.get("retry_on"), f"{path}.retry_on", issues)
        control["retry"] = {"count": retry_count, "on": triggers or ["unavailable"]}

    return control, removed


def _rewrite_retry_on(
    value: Any,
    path: str,
    issues: list[MigrationIssue],
) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw = []
        issues.append(
            MigrationIssue(
                "invalid_retry_on",
                path,
                "retry_on must be a comma-separated string or list",
                "use unavailable and/or timeout",
            )
        )
    mapped: list[str] = []
    for trigger in raw:
        normalized = str(trigger).strip().lower().replace("-", "_")
        if normalized in _RETRY_EVIDENCE:
            evidence = normalized
        elif normalized in {
            "connect_failure",
            "connection_failure",
            "refused_stream",
            "reset",
        }:
            evidence = "unavailable"
        elif normalized in {"5xx", "retriable_status_codes", "resource_exhausted"}:
            issues.append(
                MigrationIssue(
                    "unsafe_retry_trigger",
                    path,
                    f"transport retry trigger {trigger!r} does not prove zero billable work",
                    "remove it or choose unavailable/timeout only when the failure "
                    "has explicit pre-inference evidence",
                )
            )
            continue
        elif normalized in {"request_timeout", "stream_timeout"}:
            evidence = "timeout"
        else:
            issues.append(
                MigrationIssue(
                    "ambiguous_retry_trigger",
                    path,
                    f"cannot map transport retry trigger {trigger!r} to Router evidence",
                    "choose unavailable and/or timeout explicitly",
                )
            )
            continue
        if evidence not in mapped:
            mapped.append(evidence)
    return mapped


def _rewrite_pricing(
    value: Any,
    model_path: str,
    issues: list[MigrationIssue],
) -> tuple[dict[str, str], str | None]:
    path = f"{model_path}.pricing"
    if not isinstance(value, dict):
        issues.append(
            MigrationIssue(
                "invalid_pricing",
                path,
                "pricing must be a mapping",
                "repair the previous v0.3 document before migration",
            )
        )
        return {}, None
    allowed = set(_OLD_PRICING_FIELDS) | set(_NEW_PRICING_FIELDS) | {"currency"}
    unknown = set(value) - allowed
    if unknown:
        issues.append(
            MigrationIssue(
                "unknown_pricing_fields",
                path,
                "unsupported fields: " + ", ".join(sorted(unknown)),
                "remove fields that were never part of the previous v0.3 contract",
            )
        )
    result: dict[str, str] = {}
    for old_name, new_name in _OLD_PRICING_FIELDS.items():
        if old_name not in value or value[old_name] is None:
            continue
        if new_name in value and value[new_name] is not None:
            issues.append(
                MigrationIssue(
                    "pricing_collision",
                    f"{path}.{new_name}",
                    f"both {old_name} and {new_name} are configured",
                    "keep only one exact price",
                )
            )
            continue
        result[new_name] = _decimal_string(
            value[old_name], f"{path}.{old_name}", issues
        )
    for field_name in _NEW_PRICING_FIELDS:
        if field_name in value and value[field_name] is not None:
            result[field_name] = _decimal_string(
                value[field_name], f"{path}.{field_name}", issues
            )
    currency = value.get("currency")
    if currency is None and any(name in value for name in _OLD_PRICING_FIELDS):
        currency = "USD"
    if currency is not None and (
        not isinstance(currency, str)
        or not currency.isupper()
        or len(currency) != _CURRENCY_CODE_LENGTH
    ):
        issues.append(
            MigrationIssue(
                "invalid_currency",
                f"{path}.currency",
                "currency must be an uppercase three-letter code",
                "use one ISO-4217 currency such as USD",
            )
        )
        return result, None
    return result, currency


def _decimal_string(
    value: Any,
    path: str,
    issues: list[MigrationIssue],
) -> str:
    if isinstance(value, bool):
        parsed = None
    else:
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, ValueError):
            parsed = None
    if parsed is None or not parsed.is_finite() or parsed < 0:
        issues.append(
            MigrationIssue(
                "invalid_price",
                path,
                "price must be a finite non-negative decimal",
                "use an exact decimal value without exponent notation",
            )
        )
        return "0"
    rendered = format(parsed, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _merge_billing_currency(
    document: dict[str, Any],
    currencies: set[str],
    issues: list[MigrationIssue],
) -> None:
    if len(currencies) > 1:
        issues.append(
            MigrationIssue(
                "mixed_currencies",
                "providers.models[].pricing.currency",
                "Provider Models use more than one currency",
                "normalize prices into one global billing currency",
            )
        )
        return
    existing_global = document.get("global")
    if not currencies and existing_global is None:
        return
    global_config = document.setdefault("global", {})
    if not isinstance(global_config, dict):
        issues.append(
            MigrationIssue(
                "invalid_global",
                "global",
                "global must be a mapping",
                "repair the previous v0.3 document before migration",
            )
        )
        return
    existing = global_config.get("billing")
    existing_currency = existing.get("currency") if isinstance(existing, dict) else None
    migrated_currency = next(iter(currencies), None)
    if (
        existing_currency
        and migrated_currency
        and existing_currency != migrated_currency
    ):
        issues.append(
            MigrationIssue(
                "billing_currency_collision",
                "global.billing.currency",
                f"configured {existing_currency!r} conflicts with migrated {migrated_currency!r}",
                "choose one global billing currency",
            )
        )
        return
    if migrated_currency and not existing_currency:
        global_config["billing"] = {"currency": migrated_currency}


def _validate_current_document(document: dict[str, Any]) -> None:
    issues: list[MigrationIssue] = []
    try:
        config = UserConfig.model_validate(document)
    except PydanticValidationError as error:
        for item in error.errors():
            issues.append(
                MigrationIssue(
                    "invalid_current_schema",
                    ".".join(str(part) for part in item["loc"]) or "config",
                    item["msg"],
                    "repair the named v0.3 field and rerun",
                )
            )
        raise ConfigMigrationError(issues) from error
    for error in validate_user_config(config, log_summary=False):
        issues.append(
            MigrationIssue(
                "invalid_current_semantics",
                error.field or "config",
                error.message,
                "repair the referenced v0.3 relationship and rerun",
            )
        )
    if issues:
        raise ConfigMigrationError(issues)
