from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verification_catalog as vc
from domain_registry import load_domain_registry
from executor_contract import MATRIX_EXECUTORS


def _baseline_registry() -> dict:
    return copy.deepcopy(load_domain_registry())


@contextmanager
def _catalog_with_extra(record_id: str, source_name: str):
    """Yield (registry, injected_clone) with an extra verification in the catalog.

    Uses mock.patch.object so the @lru_cache'd load_catalog() is never mutated:
    no cache_clear(), no cross-test leakage.
    """
    catalog = copy.deepcopy(vc.load_catalog())
    clone = copy.deepcopy(catalog["verifications"][source_name])
    clone["id"] = record_id
    catalog["verifications"][record_id] = clone
    with mock.patch.object(vc, "load_catalog", return_value=catalog):
        yield _baseline_registry(), clone


# --- 1. Baseline -------------------------------------------------------------


def test_baseline_catalog_is_valid():
    assert vc.catalog_errors(_baseline_registry()) == []


# --- 2. Non-matrix executor rejects a second verification --------------------


def test_non_matrix_executor_rejects_second_verification():
    records = vc.verification_records(_baseline_registry())
    target = next(
        name for name, rec in records.items() if rec["executor"] not in MATRIX_EXECUTORS
    )
    clone_id = f"{target}.secondary"

    with _catalog_with_extra(clone_id, target) as (registry, clone):
        errors = vc.catalog_errors(registry)

    assert any(
        f"executor {clone['executor']} supports exactly one verification" in e
        for e in errors
    ), (
        f"expected non-matrix multiplicity error for {clone['executor']!r}, "
        f"got: {errors}"
    )
    assert any(
        target in e and clone_id in e for e in errors
    ), f"error should name both ids, got: {errors}"


# --- 3. Matrix executor allows multiple verifications ------------------------


def test_matrix_executor_allows_multiple_verifications():
    records = vc.verification_records(_baseline_registry())
    # `records` may include profile-derived verifications (e2e.*) absent from
    # the catalog YAML; pick a catalog-backed one so we can inject a clone.
    target = next(
        name
        for name, rec in records.items()
        if rec["executor"] in MATRIX_EXECUTORS
        and name in vc.load_catalog()["verifications"]
    )
    clone_id = f"{target}.secondary"

    with _catalog_with_extra(clone_id, target) as (registry, clone):
        errors = vc.catalog_errors(registry)

    assert not any(
        "supports exactly one verification" in e and clone["executor"] in e
        for e in errors
    ), f"matrix executor {clone['executor']!r} must allow multiple, got: {errors}"
