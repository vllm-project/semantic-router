"""Tests for the checkpoint label remap in evaluate_modality_candidate.py."""
import pytest

from evaluate_modality_candidate import build_label_remap


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
