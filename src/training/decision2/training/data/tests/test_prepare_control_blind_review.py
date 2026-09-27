"""Privacy and determinism guards for the ConTRoL blind-review packet."""

from training.data.prepare_control_blind_review import (
    make_packets,
    sample_pairs,
)


def test_blind_packet_excludes_labels_and_is_group_disjoint() -> None:
    pairs = [
        {
            "position": position,
            "premise": f"Unique context {position} " + "detail " * (position % 7 + 1),
            "hypothesis": f"Claim {position}",
            "label": ("contradiction", "neutral", "entailment")[position % 3],
            "group": f"group-{position}",
        }
        for position in range(180)
    ]
    sample = sample_pairs(pairs)
    blind, reveal = make_packets(sample)
    assert len({pair["group"] for pair in sample}) == 30
    assert len(blind) == len(reveal) == 30
    assert {row["task"] for row in blind} == {
        "choice",
        "support_noul",
        "contradiction_noul",
    }
    assert all("label" not in row and "source_position" not in row for row in blind)
    assert all("source_label" in row for row in reveal)
    assert [row["sample_id"] for row in blind] == [row["sample_id"] for row in reveal]
    assert sample_pairs(pairs) == sample
