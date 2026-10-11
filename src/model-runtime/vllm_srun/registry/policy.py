"""Licence and access policy for packages the runtime is asked to serve."""

from __future__ import annotations

from typing import Any

from ..errors import PackageError

# Licences that restrict use beyond attribution; serving them needs an explicit
# --accept-licence with the same identifier.
RESTRICTED_MARKERS = ("nc", "non-commercial", "noncommercial", "research", "openrail")


def restricted(spdx: str) -> bool:
    lowered = spdx.lower()
    return any(marker in lowered for marker in RESTRICTED_MARKERS)


def licences(manifest: dict[str, Any]) -> list[str]:
    """Every licence identifier a package manifest declares, package first."""
    licence = manifest.get("licence") if isinstance(manifest, dict) else None
    found: list[str] = []
    if isinstance(licence, dict):
        if isinstance(licence.get("spdx"), str):
            found.append(licence["spdx"])
        for component in licence.get("components") or []:
            value = component.get("licence") if isinstance(component, dict) else None
            if isinstance(value, str) and value not in found:
                found.append(value)
    elif isinstance(licence, str):
        found.append(licence)
    return found


def check(manifest: dict[str, Any], accepted: tuple[str, ...] = ()) -> str | None:
    """The package licence; refuses restricted components that were not accepted."""
    declared = licences(manifest)
    accepted_lower = {value.lower() for value in accepted}
    for value in declared:
        if restricted(value) and value.lower() not in accepted_lower:
            raise PackageError(
                f"the package declares the restricted licence {value!r}; "
                f"serve it only with --accept-licence {value}"
            )
    return declared[0] if declared else None
