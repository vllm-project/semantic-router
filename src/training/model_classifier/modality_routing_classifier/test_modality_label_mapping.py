"""Tests for the checkpoint label mapping helpers in modality_label_mapping.py."""

import numpy as np
import pytest
from modality_label_mapping import (
    build_label_remap,
    check_output_size,
    logits_to_canonical_order,
)


def test_canonical_order_maps_to_itself():
    assert build_label_remap({0: "AR", 1: "DIFFUSION", 2: "BOTH"}) == {0: 0, 1: 1, 2: 2}


def test_permuted_order_is_remapped_by_name():
    # checkpoint class 0 is BOTH, 1 is AR, 2 is DIFFUSION
    assert build_label_remap({0: "BOTH", 1: "AR", 2: "DIFFUSION"}) == {0: 2, 1: 0, 2: 1}


@pytest.mark.parametrize(
    "id2label",
    [
        {},  # nothing to go on
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},  # generic labels look valid by id
        {0: "AR", 1: "DIFFUSION"},  # missing a class
        {0: "AR", 1: "DIFFUSION", 2: "BOTH", 3: "OTHER"},  # extra class
        {0: "AR", 1: "AR", 2: "BOTH"},  # duplicate label
        {0: "AR", 1: "DIFFUSION", 2: "both"},  # near miss on the name
    ],
)
def test_incompatible_id2label_fails_closed(id2label):
    with pytest.raises(ValueError, match="refusing to guess"):
        build_label_remap(id2label, "some/checkpoint")


def test_output_size_must_match_mapping():
    check_output_size({0: 0, 1: 1, 2: 2}, 3)
    with pytest.raises(ValueError, match="outputs"):
        check_output_size({0: 0, 1: 1, 2: 2}, 4)
    with pytest.raises(ValueError, match="outputs"):
        check_output_size({0: 0, 1: 1, 3: 2}, 3)


def test_teacher_logits_are_reordered_to_canonical_columns():
    # teacher columns are (BOTH, AR, DIFFUSION); canonical is (AR, DIFFUSION, BOTH)
    remap = build_label_remap({0: "BOTH", 1: "AR", 2: "DIFFUSION"})
    logits = np.array([[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]], dtype=np.float32)
    expected = np.array([[0.7, 0.2, 0.1], [0.05, 0.05, 0.9]], dtype=np.float32)
    np.testing.assert_array_equal(logits_to_canonical_order(logits, remap), expected)
