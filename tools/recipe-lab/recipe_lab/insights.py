"""Insight references: provenance links carried on ledger records (G2).

A candidate's provenance can cite the evidence it was derived from. Four
accepted forms:

  rec-<id>                            an existing ledger record_id
  bench:<run_id>                      a benchmark run id (format-checked only)
  attr:<path>@sha256:<hex>            an attribution artifact; the hash must
                                      match the file's raw bytes (A4 identity)
  diag:<path>@sha256:<hex>#<anchor>   a diagnostic artifact pinned to an anchor
                                      (line, section or entry id in the file)

`--insight-refs` is comma-separated. Unverifiable references raise
ContractError so the CLI exits 2: a ledger must not silently carry evidence
links that point at nothing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .contract import ContractError, dataset_content_hash

if TYPE_CHECKING:
    from .ledger import Ledger

_REC_RE = re.compile(r"^rec-[0-9a-f]{12}$")
_BENCH_RE = re.compile(r"^bench:[A-Za-z0-9][A-Za-z0-9._-]*$")
_ATTR_RE = re.compile(r"^(?P<path>.+)@sha256:(?P<hex>[0-9a-f]{64})$")
_DIAG_RE = re.compile(
    r"^(?P<path>.+)@sha256:(?P<hex>[0-9a-f]{64})#(?P<anchor>[A-Za-z0-9._/-]+)$"
)


def parse_insight_refs(raw: str | None) -> list[str]:
    """Split the comma-separated --insight-refs value."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_insight_refs(refs: list[str], ledger: Ledger | None = None) -> list[str]:
    """Validate every reference and dedupe preserving order."""
    resolved: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        _check_ref(ref, ledger)
        seen.add(ref)
        resolved.append(ref)
    return resolved


def merge_insight_refs(provenance: dict, refs: list[str]) -> None:
    """Merge validated refs into a provenance dict's insight_refs in place."""
    existing = provenance.setdefault("insight_refs", [])
    if not isinstance(existing, list):
        raise ContractError("candidate provenance.insight_refs must be a list")
    for ref in refs:
        if ref not in existing:
            existing.append(ref)


def _check_ref(ref: str, ledger: Ledger | None) -> None:
    if ref.startswith("rec-"):
        if not _REC_RE.match(ref):
            raise ContractError(
                f"malformed insight ref {ref!r}: record ids look like rec-<12 hex>"
            )
        if ledger is None:
            raise ContractError(
                f"insight ref {ref!r} needs a ledger to resolve against"
            )
        ids = {row.get("record_id") for row in ledger.read()}
        if ref not in ids:
            raise ContractError(f"insight ref {ref!r} not found in ledger")
        return
    if ref.startswith("bench:"):
        if not _BENCH_RE.match(ref):
            raise ContractError(
                f"malformed insight ref {ref!r}: expected bench:<run_id>"
            )
        return
    if ref.startswith("attr:"):
        match = _ATTR_RE.match(ref[len("attr:") :])
        if not match:
            raise ContractError(
                f"malformed insight ref {ref!r}: expected attr:<path>@sha256:<64 hex>"
            )
    elif ref.startswith("diag:"):
        match = _DIAG_RE.match(ref[len("diag:") :])
        if not match:
            raise ContractError(
                f"malformed insight ref {ref!r}: expected diag:<path>@sha256:<64 hex>#<anchor>"
            )
    else:
        raise ContractError(
            f"unsupported insight ref {ref!r}: use rec-<id>, bench:<run_id>, "
            "attr:<path>@sha256:<hex> or diag:<path>@sha256:<hex>#<anchor>"
        )
    actual = _file_hash(match.group("path"))
    expected = f"sha256:{match.group('hex')}"
    if actual != expected:
        raise ContractError(
            f"insight ref hash mismatch for {match.group('path')!r}: ref={expected} file={actual}"
        )


def _file_hash(path: str) -> str:
    try:
        return dataset_content_hash(path)
    except OSError as exc:
        raise ContractError(f"cannot read referenced file {path!r}: {exc}") from exc
