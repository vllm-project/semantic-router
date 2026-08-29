"""Read-only exact-pin verification for ignored external benchmark sources."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from cli.evaluation.benchmark_registry import (
    BenchmarkAdapterDescriptor,
    get_benchmark_adapter,
)
from cli.evaluation.suite_contract import BenchmarkSourceReceipt


class SourceVerificationError(ValueError):
    """An external source cannot be used as reproducible evaluation input."""


def _source_path(source_root: Path, cache_key: str) -> Path:
    root = source_root.expanduser().resolve()
    candidate = root / cache_key
    if candidate.is_symlink():
        raise SourceVerificationError(
            f"benchmark source {cache_key!r} must not be a symlink"
        )
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise SourceVerificationError(
            f"benchmark source {cache_key!r} is missing or outside the source root"
        ) from exc
    if not candidate.is_dir():
        raise SourceVerificationError(
            f"benchmark source {cache_key!r} is not a directory"
        )
    return candidate


def _git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(path), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceVerificationError("git source verification could not run") from exc
    if result.returncode != 0:
        raise SourceVerificationError("benchmark source is not a readable Git checkout")
    return result.stdout.strip()


def _verify_checkout(
    source_root: Path, cache_key: str, expected_revision: str
) -> tuple[str, bool, bool]:
    path = _source_path(source_root, cache_key)
    observed = _git(path, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", observed):
        raise SourceVerificationError(
            "benchmark source returned an invalid Git revision"
        )
    changes = _git(path, "status", "--porcelain", "--untracked-files=all")
    clean = changes == ""
    return observed, clean, observed == expected_revision and clean


def verify_benchmark_source(
    adapter_id: str, source_root: str | Path
) -> BenchmarkSourceReceipt:
    """Verify code and optional dataset checkouts without fetching or executing them."""

    descriptor = get_benchmark_adapter(adapter_id)
    root = Path(source_root)
    observed_source, source_clean, source_verified = _verify_checkout(
        root, descriptor.source_cache_key, descriptor.source_revision
    )
    observed_dataset: str | None = None
    dataset_clean: bool | None = None
    dataset_verified = descriptor.dataset_revision is None
    if descriptor.dataset_revision is not None:
        if descriptor.dataset_cache_key is None:
            raise SourceVerificationError("dataset pin has no source cache key")
        observed_dataset, dataset_clean, dataset_verified = _verify_checkout(
            root, descriptor.dataset_cache_key, descriptor.dataset_revision
        )
    return BenchmarkSourceReceipt(
        adapter_id=descriptor.id,
        expected_source_revision=descriptor.source_revision,
        observed_source_revision=observed_source,
        expected_dataset_revision=descriptor.dataset_revision,
        observed_dataset_revision=observed_dataset,
        source_clean=source_clean,
        dataset_clean=dataset_clean,
        verified=source_verified and dataset_verified,
    )


def require_verified_benchmark_source(
    descriptor: BenchmarkAdapterDescriptor, source_root: str | Path
) -> BenchmarkSourceReceipt:
    receipt = verify_benchmark_source(descriptor.id, source_root)
    if not receipt.verified:
        raise SourceVerificationError(
            f"benchmark source {descriptor.id!r} is dirty or does not match its exact pin"
        )
    return receipt
