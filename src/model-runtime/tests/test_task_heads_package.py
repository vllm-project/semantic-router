"""Task-head package label validation: ``id2label`` must be exactly 0..n-1."""

from __future__ import annotations

import pytest
from vllm_srun.families.task_heads.package import PackageError, labels_of


def test_labels_of_reads_indices_in_order():
    assert labels_of({"id2label": {"1": "unsafe", "0": "safe"}}) == ("safe", "unsafe")


def test_labels_of_rejects_duplicate_numeric_indices():
    with pytest.raises(PackageError, match="duplicate numeric indices"):
        labels_of({"id2label": {"0": "benign", "00": "jailbreak", "1": "other"}})


def test_labels_of_rejects_non_integer_keys():
    with pytest.raises(PackageError, match="keys must be integers"):
        labels_of({"id2label": {"0": "safe", "one": "unsafe"}})
