"""Verification contracts and explicit supported CPU qualification inventory."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "tools/ci/verification_catalog.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    return yaml.safe_load(CATALOG_PATH.read_text())


def profile_image_dependencies() -> dict[str, list[str]]:
    """Read the framework's declared fixture capabilities, not a second profile list.

    This deliberately accepts only the maintained registration shape. Adding a
    new capability expression requires teaching the planner its image identity;
    it cannot silently fall back to an undeclared per-test image build.
    """
    source = (ROOT / "e2e/profiles/all/imports.go").read_text()
    fixtures = {
        "providerMockerLocalImages": ["provider-mocker"],
        "dashboardLocalImages": ["dashboard"],
    }
    result = {}
    for match in re.finditer(
        r'register\(\s*"([^"\n]+)".*?framework\.ProfileCapabilities\{([^}]*)\}',
        source,
        re.S,
    ):
        name, capabilities = match.groups()
        images = ["extproc"]
        local = re.search(r"LocalImages:\s*([^,}]+)", capabilities)
        if local:
            expression = local.group(1).strip()
            if expression not in fixtures:
                raise ValueError(
                    f"profile {name}: undeclared image capability {expression}"
                )
            images.extend(fixtures[expression])
        result[name] = images
    return result


def verification_records(registry: dict) -> dict[str, dict]:
    catalog = load_catalog()
    defaults = catalog["defaults"]
    records = {
        name: {**defaults, **record}
        for name, record in catalog["verifications"].items()
    }
    dependencies = profile_image_dependencies()
    for name, profile in registry["profiles"].items():
        if name not in dependencies:
            raise ValueError(
                f"profile {name}: missing framework capability declaration"
            )
        images = dependencies[name]
        records[f"e2e.{name}"] = {
            **defaults,
            "activity": "test",
            "display_name": profile.get("display_name", ""),
            "boundary": ["e2e"],
            "executor": "e2e",
            "workflow": ".github/workflows/integration-test-k8s.yml",
            "profile": name,
            "images": images,
            "services": ["kind", "gateway", "controlled-backend"],
            "runtime": "candle",
            "device": "cpu",
            "inventory": f"e2e-profile:{name}",
            "contract": profile["coverage_role"],
        }
    return records


def full_cpu_ids() -> tuple[str, ...]:
    return tuple(load_catalog()["full_cpu"]["required"])


def catalog_errors(registry: dict) -> list[str]:
    records = verification_records(registry)
    catalog = load_catalog()
    errors = []
    required = full_cpu_ids()
    if len(required) != len(set(required)):
        errors.append("full CPU inventory has duplicate verification IDs")
    for name in required:
        if name not in records:
            errors.append(f"full CPU inventory references missing verification {name}")
    display_names = set()
    for name, record in records.items():
        label = record.get("display_name")
        if not isinstance(label, str) or not label.strip() or label == name:
            errors.append(f"verification {name} must declare a readable display_name")
        elif label in display_names:
            errors.append(f"verification {name} has a duplicate display_name: {label}")
        else:
            display_names.add(label)
        if record["executor"] == "tools":
            worker = catalog["component_workers"].get(record.get("worker"))
            if not worker or not worker.get("display_name"):
                errors.append(f"verification {name} lacks a declared component worker")
            if record["native"] or record["images"] or record["runtime"] != "none":
                errors.append(
                    f"verification {name} is incompatible with a lightweight worker"
                )
        elif record.get("worker"):
            errors.append(f"verification {name} cannot use a component worker")
        if not record.get("workflow") or not record.get("inventory"):
            errors.append(
                f"verification {name} must declare its workflow and inventory"
            )
        if set(record.get("boundary", [])) - {"unit", "integration", "e2e"}:
            errors.append(f"verification {name} has an invalid test boundary")
        if record.get("activity") == "test" and not record.get("boundary"):
            errors.append(f"test verification {name} has no boundary")
        if record.get("device") not in {"cpu", "none"}:
            errors.append(f"verification {name} has no declared hardware runner")
        if (execution := record.get("execution")) and (
            execution != {"mode": "qemu-user", "host_platform": "linux/amd64"}
            or record["platform"] != "linux/riscv64"
            or record["runtime"] != "candle"
            or record["device"] != "cpu"
            or record["native"]
        ):
            errors.append(
                f"verification {name} has an invalid emulated target contract"
            )
    # Only recurring public CI promises are required. Manual/experimental support
    # must not accidentally be upgraded by the CPU planner.
    public = ROOT / "website/docs/installation/support-matrix.md"
    for line in public.read_text().splitlines():
        if "**PR CI" not in line or not line.startswith("|"):
            continue
        cell = line.split("|")[1].strip()
        label = cell.split("]", 1)[0].removeprefix("[")
        ids = catalog["full_cpu"]["support_contracts"].get(label)
        if not ids or not set(ids) <= set(required):
            errors.append(
                f"public PR CI contract {label!r} lacks required CPU coverage"
            )
    return errors
