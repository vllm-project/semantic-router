from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .canonical import digest
from .dataset_io import MAX_ROWS, read_small
from .legacy_reasoning import (
    CONFIGURATIONS,
    ENVIRONMENT,
    MAX_EXPORT_BYTES,
    REQUIREMENTS,
    SOURCE_REVISION,
    configuration,
    inspect_runtime,
    worker_request,
)

FIELD: str = "legacy_reasoning_conditions"
PROTOCOL: str = "bench-reasoning-conditions-v1"
MODES: frozenset[str] = frozenset({"NR", "NR_REASONING", "AR", "XC", "Router_NR"})
MODE_ALIASES: dict[str, str] = {"standard": "NR", "reasoning": "NR_REASONING"}
TEMPLATE_CONTROLS: frozenset[str] = frozenset(
    {"thinking", "enable_thinking", "reasoning_effort"}
)
XC_SKIP_REASON: str = "native_first_ten_have_no_cot"


def canonical_mode(value: str) -> str:
    if not isinstance(value, str) or MODE_ALIASES.get(value, value) not in MODES:
        raise ValueError("Unknown reasoning mode")
    return MODE_ALIASES.get(value, value)


def native_dataset(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Unknown native reasoning dataset")
    return "mmlu-pro" if value.lower() in {"mmlu", "mmlu-pro"} else configuration(value)


def reasoning_controls(params: dict[str, Any]) -> dict[str, Any]:
    from .contracts import (  # noqa: PLC0415 - frozen-plan registry cycle
        DEFAULT_LIMITS,
        GENERATION_FIELDS,
        validate_request_params,
    )

    if not isinstance(params, dict) or set(params) - GENERATION_FIELDS:
        raise ValueError("Unsupported reasoning profile fields")
    kwargs = params.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    nested = {key: value for key, value in kwargs.items() if key in TEMPLATE_CONTROLS}
    controls: dict[str, Any] = {}
    if "reasoning_effort" in params:
        controls["reasoning_effort"] = params["reasoning_effort"]
    for key, value in nested.items():
        if key != "reasoning_effort" and type(value) is not bool:
            raise ValueError("Native thinking controls must be boolean")
    for value in [
        *([params["reasoning_effort"]] if "reasoning_effort" in params else []),
        *([nested["reasoning_effort"]] if "reasoning_effort" in nested else []),
    ]:
        if not isinstance(value, str):
            raise ValueError("reasoning_effort must be a supported string")
        validate_request_params({"reasoning_effort": value}, DEFAULT_LIMITS)
    if len(nested) + len(controls) > 1:
        raise ValueError("Conflicting reasoning controls")
    if nested:
        controls["chat_template_kwargs"] = nested
    return controls


def _without_controls(params: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(params)
    result.pop("reasoning_effort", None)
    if "chat_template_kwargs" in result:
        kwargs = result["chat_template_kwargs"]
        result["chat_template_kwargs"] = {
            key: value for key, value in kwargs.items() if key not in TEMPLATE_CONTROLS
        }
        if not result["chat_template_kwargs"]:
            result.pop("chat_template_kwargs")
    return result


def _group(target: dict[str, Any]) -> str:
    return digest(
        {key: target.get(key) for key in ("kind", "model", "base_url", "config_hash")}
    )


def _template(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or any(
        key in manifest
        for key in (
            "cases",
            "dataset",
            "execution_cells",
            "recovery",
            "baseline_run_id",
        )
    ):
        raise ValueError(
            "The condition template must not contain cases or derived work"
        )
    if manifest.get("profile", "quick") not in {"smoke", "quick"}:
        raise ValueError("Reasoning conditions do not establish a holdout")
    if manifest.get("mode", "live") != "live":
        raise ValueError("Reasoning conditions require live runs")
    if reasoning_controls(manifest.get("sampling", {})):
        raise ValueError("Reasoning controls belong on each target, not sampling")
    section = manifest.get(FIELD)
    if (
        not isinstance(section, dict)
        or set(section) != {"version", "conditions"}
        or type(section["version"]) is not int
        or section["version"] != 1
        or not isinstance(section["conditions"], list)
        or not section["conditions"]
    ):
        raise ValueError("The condition template requires version 1 and conditions")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("The condition template requires targets")
    by_id: dict[str, dict[str, Any]] = {}
    for target in targets:
        if (
            not isinstance(target, dict)
            or not isinstance(target.get("id"), str)
            or not target["id"]
            or target["id"] in by_id
        ):
            raise ValueError("Condition targets require unique IDs")
        reasoning_controls(target.get("request_params", {}))
        by_id[target["id"]] = target
    requested: list[dict[str, Any]] = []
    seen: set[str] = set()
    groups: set[tuple[str, str]] = set()
    for item in section["conditions"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"target_id", "mode"}
            or not isinstance(item["target_id"], str)
            or item["target_id"] not in by_id
        ):
            raise ValueError("Each condition requires a known target_id and mode")
        mode = canonical_mode(item["mode"])
        target = by_id[item["target_id"]]
        group = _group(target)
        if target["id"] in seen or (group, mode) in groups:
            raise ValueError("A condition template contains duplicate modes or targets")
        if mode == "Router_NR" and target.get("kind") != "mom":
            raise ValueError("Router_NR requires a mom target")
        seen.add(target["id"])
        groups.add((group, mode))
        requested.append(
            {
                "requested_mode": item["mode"],
                "mode": mode,
                "target": copy.deepcopy(target),
                "group": group,
            }
        )
    if seen != set(by_id):
        raise ValueError("Every target must have an explicit condition")
    return requested


def _controls_request(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "conditions": [
            {
                "target_id": item["target"]["id"],
                "model": item["target"]["model"],
                "mode": item["mode"],
            }
            for item in conditions
        ]
    }


def _validate_targets(manifest: dict[str, Any], targets: list[dict[str, Any]]) -> None:
    from .contracts import plan  # noqa: PLC0415 - frozen-plan registry cycle

    plan(
        {
            **{
                key: manifest[key]
                for key in (
                    "version",
                    "profile",
                    "cost_policy",
                    "sampling",
                    "limits",
                    "output_policy",
                )
                if key in manifest
            },
            "targets": targets,
            "cases": [
                {
                    "id": "condition-target-validation",
                    "benchmark": "mmlu-pro",
                    "messages": [{"role": "user", "content": "Target validation only"}],
                    "answer": "A",
                }
            ],
        }
    )


def _native_request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .setup import harness_paths  # noqa: PLC0415 - optional-harness registry cycle

    root, interpreter = harness_paths(ENVIRONMENT)
    return worker_request(
        operation=operation, payload=payload, root=root, interpreter=interpreter
    )


def _bind_controls(
    conditions: list[dict[str, Any]],
    native_controls: dict[str, Any],
    sampling: dict[str, Any],
    *,
    frozen: bool,
) -> None:
    nonreasoning: dict[str, dict[str, Any]] = {}
    for item in conditions:
        target, mode = item["target"], item["mode"]
        params = copy.deepcopy(target.get("request_params", {}))
        actual = reasoning_controls(params)
        expected = native_controls[target["id"]]
        if mode in {"NR", "NR_REASONING"}:
            if not expected:
                raise ValueError(
                    "Unknown model cannot substantiate a native reasoning toggle"
                )
            reasoning_controls(expected)
            if (actual and actual != expected) or (frozen and actual != expected):
                raise ValueError(
                    "Frozen target differs from its native reasoning control"
                )
            if not frozen:
                params.update(
                    {
                        key: value
                        for key, value in expected.items()
                        if key != "chat_template_kwargs"
                    }
                )
                if "chat_template_kwargs" in expected:
                    params["chat_template_kwargs"] = {
                        **sampling.get("chat_template_kwargs", {}),
                        **params.get("chat_template_kwargs", {}),
                        **expected["chat_template_kwargs"],
                    }
        elif mode == "AR":
            if not actual:
                raise ValueError("AR requires an explicit supported reasoning profile")
        elif actual:
            raise ValueError(f"{mode} must not carry a reasoning override")
        effective = {**sampling, **params}
        group = item["group"]
        other = _without_controls(effective)
        if group in nonreasoning and nonreasoning[group] != other:
            raise ValueError("Conditions changed non-reasoning request parameters")
        nonreasoning[group] = other
        if not frozen:
            target["request_params"] = params
        if frozen and item.get("controls") != reasoning_controls(effective):
            raise ValueError("Condition controls differ from the frozen request")
        item["controls"] = reasoning_controls(effective)


def selection_identity(section: dict[str, Any]) -> str:
    return digest(
        [
            section["source"],
            section["selection"],
            [
                [member["native_id"], member["native_sha256"]]
                for member in section["members"]
            ],
        ]
    )


def _validate_source(section: dict[str, Any], seed: int) -> str:
    source, selection = section["source"], section["selection"]
    if (
        not isinstance(source, dict)
        or set(source) != {"kind", "revision_verification", "files"}
        or not isinstance(source["kind"], str)
        or source["kind"] not in {"fixture", "local-import"}
        or source["revision_verification"] != "declared-revision-with-content-digest"
        or not isinstance(source["files"], list)
        or not source["files"]
        or not isinstance(selection, dict)
        or set(selection)
        != {"method", "dataset", "seed", "categories", "samples_per_category"}
        or selection["method"] != "native-loader"
        or type(selection["seed"]) is not int
        or not 0 <= selection["seed"] < 2**32
        or selection["seed"] != seed
    ):
        raise ValueError("Invalid frozen native source or selection")
    for entry in source["files"]:
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {"dataset", "configuration", "split", "revision", "sha256", "records"}
            or not all(
                isinstance(entry[key], str) and entry[key]
                for key in ("dataset", "split", "revision")
            )
            or (
                entry["configuration"] is not None
                and not isinstance(entry["configuration"], str)
            )
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or type(entry["records"]) is not int
            or entry["records"] <= 0
        ):
            raise ValueError("Invalid native source partition identity")
    categories, samples = selection["categories"], selection["samples_per_category"]
    if (
        categories is not None
        and (
            not isinstance(categories, list)
            or not categories
            or not all(
                isinstance(category, str) and category for category in categories
            )
        )
    ) or (samples is not None and (type(samples) is not int or samples <= 0)):
        raise ValueError("Invalid native selection parameters")
    name = native_dataset(selection["dataset"])
    if name != selection["dataset"]:
        raise ValueError("Frozen native selection requires its canonical dataset")
    return name


def _validate_member(
    *,
    case: dict[str, Any],
    member: dict[str, Any],
    section: dict[str, Any],
    options: dict[str, Any],
    explicit: bool,
) -> None:
    name = section["selection"]["dataset"]
    expected = "legacy-" + name + ("-explicit-cot" if explicit else "")
    if case["benchmark"] != expected:
        raise ValueError("Condition benchmark differs from its native selection")
    config = options.get(expected, {})
    if not isinstance(config, dict) or any(
        config.get(key) != section[key]
        for key in ("source", "selection", "source_revision", "requirements_sha256")
    ):
        raise ValueError("Condition source differs from its frozen adapter source")
    if explicit:
        payload = case.get("answer")
        if not isinstance(payload, dict):
            raise ValueError("Invalid explicit-CoT native payload")
        question = payload.get("question")
        plain = payload.get("plain_case")
        if (
            not isinstance(plain, dict)
            or digest(plain) != member["plain_sha256"]
            or plain.get("id") != member["plain_id"]
        ):
            raise ValueError("Explicit-CoT differs from its paired plain case")
    elif name != "mmlu-pro":
        question = case.get("answer")
    else:
        if case["id"] != "legacy-mmlu-pro/" + member["native_id"]:
            raise ValueError("Native MMLU question identity changed")
        return
    if (
        not isinstance(question, dict)
        or question.get("question_id") != member["native_id"]
        or digest(question) != member["native_sha256"]
        or not (
            question.get("cot_content") is None
            or isinstance(question["cot_content"], str)
        )
        or bool(question.get("cot_content") and question["cot_content"].strip())
        != member["has_cot"]
    ):
        raise ValueError("Native question or CoT eligibility changed")


def export_manifest(
    *,
    manifest: dict[str, Any],
    dataset: str,
    source_bundle: str | Path,
    seed: int = 42,
    categories: list[str] | None = None,
    samples_per_category: int | None = None,
) -> dict[str, Any]:
    from .contracts import plan  # noqa: PLC0415 - frozen-plan registry cycle

    conditions = _template(manifest)
    name = native_dataset(dataset)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Native selection seed must be an unsigned 32-bit integer")
    if samples_per_category is not None and (
        type(samples_per_category) is not int or samples_per_category <= 0
    ):
        raise ValueError("samples_per_category must be a positive integer")
    if categories is not None and (
        not categories
        or not all(isinstance(value, str) and value for value in categories)
    ):
        raise ValueError("categories must contain nonempty native category names")
    _validate_targets(manifest, [item["target"] for item in conditions])
    inspect_runtime()
    exported = _native_request(
        "export-modes",
        {
            "dataset": name,
            "source_bundle": str(Path(source_bundle).expanduser().resolve()),
            "seed": seed,
            "categories": categories,
            "samples_per_category": samples_per_category,
            **_controls_request(conditions),
        },
    )
    _bind_controls(
        conditions, exported["controls"], manifest.get("sampling", {}), frozen=False
    )
    eligible = any(member["has_cot"] for member in exported["members"][:10])
    for item in conditions:
        skipped = item["mode"] == "XC" and not eligible
        item.update(
            status="skipped" if skipped else "planned",
            reason=XC_SKIP_REASON if skipped else None,
        )
    active = [item for item in conditions if item["status"] == "planned"]
    if not active:
        raise ValueError("No executable conditions: native_first_ten_have_no_cot")
    selected: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, str]] = []
    for item in active:
        cases = (
            exported["explicit_cases"]
            if item["mode"] == "XC"
            else exported["plain_cases"]
        )
        for case in cases:
            selected[case["id"]] = case
            cells.append({"case_id": case["id"], "target_id": item["target"]["id"]})
    section = {
        "version": 1,
        "protocol": PROTOCOL,
        "source_revision": SOURCE_REVISION,
        "requirements_sha256": hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
        "source": exported["source"],
        "selection": exported["selection"],
        "members": exported["members"],
        "conditions": conditions,
    }
    section["selection_sha256"] = selection_identity(section)
    benchmarks = {case["benchmark"] for case in selected.values()}
    return plan(
        {
            **copy.deepcopy(manifest),
            "seed": seed,
            "cases": list(selected.values()),
            "targets": [item["target"] for item in active],
            "execution_cells": cells,
            "benchmark_options": {
                key: value
                for key, value in exported["benchmark_options"].items()
                if key in benchmarks
            },
            FIELD: section,
        }
    )


def validate_conditions(manifest: dict[str, Any], cache: dict[str, Any]) -> None:
    section = manifest.get(FIELD)
    if (
        not isinstance(section, dict)
        or set(section)
        != {
            "version",
            "protocol",
            "source_revision",
            "requirements_sha256",
            "source",
            "selection",
            "members",
            "conditions",
            "selection_sha256",
        }
        or type(section["version"]) is not int
        or section["version"] != 1
        or section["protocol"] != PROTOCOL
        or section["source_revision"] != SOURCE_REVISION
        or section["requirements_sha256"]
        != hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    ):
        raise ValueError("Invalid frozen reasoning condition contract")
    if manifest["mode"] != "live" or manifest["profile"] not in {"smoke", "quick"}:
        raise ValueError(
            "Reasoning conditions require live smoke/quick diagnostics, not a holdout"
        )
    if manifest.get("recovery") or manifest.get("baseline_run_id"):
        raise ValueError("Derived runs do not support reasoning condition matrices")
    _validate_source(section, manifest["seed"])
    members = section["members"]
    if not isinstance(members, list) or not 1 <= len(members) <= MAX_ROWS:
        raise ValueError("Invalid native selection members")
    member_keys = {
        "native_id",
        "native_sha256",
        "stratum",
        "has_cot",
        "plain_id",
        "plain_sha256",
        "xc_id",
        "xc_sha256",
    }
    seen: set[str] = set()
    for member in members:
        if (
            not isinstance(member, dict)
            or set(member) != member_keys
            or type(member["has_cot"]) is not bool
            or not all(
                isinstance(member[key], str) and member[key]
                for key in member_keys - {"has_cot"}
            )
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", member[key])
                for key in ("native_sha256", "plain_sha256", "xc_sha256")
            )
            or member["native_id"] in seen
        ):
            raise ValueError("Invalid or duplicate native selection identity")
        seen.add(member["native_id"])
    if section["selection_sha256"] != selection_identity(section):
        raise ValueError("Native selection identity changed")
    conditions = section["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Missing frozen reasoning conditions")
    for item in conditions:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "requested_mode",
                "mode",
                "target",
                "group",
                "controls",
                "status",
                "reason",
            }
            or not isinstance(item["target"], dict)
        ):
            raise ValueError("Invalid frozen reasoning condition")
    validated = _template(
        {
            **{
                key: value
                for key, value in manifest.items()
                if key not in {"cases", "dataset", "execution_cells", FIELD}
            },
            "targets": [item["target"] for item in conditions],
            FIELD: {
                "version": 1,
                "conditions": [
                    {
                        "target_id": item["target"].get("id"),
                        "mode": item["requested_mode"],
                    }
                    for item in conditions
                ],
            },
        }
    )
    if any(
        item["mode"] != expected["mode"] or item["group"] != expected["group"]
        for item, expected in zip(conditions, validated, strict=True)
    ):
        raise ValueError("Frozen condition binding changed")
    cases = {case["id"]: case for case in manifest["cases"]}
    expected_cells: set[tuple[str, str]] = set()
    expected_targets: list[dict[str, Any]] = []
    eligible = any(member["has_cot"] for member in members[:10])
    skipped_targets: list[dict[str, Any]] = []
    for item in conditions:
        skipped = item["mode"] == "XC" and not eligible
        if item["status"] != ("skipped" if skipped else "planned") or item[
            "reason"
        ] != (XC_SKIP_REASON if skipped else None):
            raise ValueError("Condition differs from native XC eligibility")
        if skipped:
            skipped_targets.append(item["target"])
            continue
        target = item["target"]
        expected_targets.append(target)
        prefix = "xc" if item["mode"] == "XC" else "plain"
        for member in members:
            identity = member[prefix + "_id"]
            case = cases.get(identity)
            if case is None or digest(case) != member[prefix + "_sha256"]:
                raise ValueError("Frozen condition case or prompt changed")
            if (
                not isinstance(case.get("metadata"), dict)
                or case["metadata"].get("stratum") != member["stratum"]
            ):
                raise ValueError("Native category identity changed")
            _validate_member(
                case=case,
                member=member,
                section=section,
                options=manifest.get("benchmark_options", {}),
                explicit=prefix == "xc",
            )
            expected_cells.add((identity, target["id"]))
    if manifest["targets"] != expected_targets or set(cases) != {
        case for case, _ in expected_cells
    }:
        raise ValueError("Condition targets or selected cases changed")
    cells = manifest.get("execution_cells", [])
    if {(cell["case_id"], cell["target_id"]) for cell in cells} != expected_cells:
        raise ValueError("Condition execution cells changed")
    if skipped_targets:
        _validate_targets(manifest, skipped_targets)
    inspect_runtime(cache=cache)
    controls = _native_request("controls", _controls_request(conditions))
    _bind_controls(
        copy.deepcopy(conditions), controls, manifest["sampling"], frozen=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze native reasoning conditions without inference."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(set(CONFIGURATIONS) | {"gpqa", "mmlu", "mmlu-pro"}),
        required=True,
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--samples-per-category", type=int)
    args = parser.parse_args()
    result = export_manifest(
        manifest=json.loads(read_small(args.manifest, MAX_EXPORT_BYTES)),
        dataset=args.dataset,
        source_bundle=args.source_bundle,
        seed=args.seed,
        categories=args.categories,
        samples_per_category=args.samples_per_category,
    )
    content = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    if len(content) > MAX_EXPORT_BYTES:
        raise ValueError("Reasoning condition export exceeds the supported byte limit")
    with os.fdopen(
        os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as output:
        output.write(content)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "plan_sha256": result["plan_sha256"],
                "model_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
