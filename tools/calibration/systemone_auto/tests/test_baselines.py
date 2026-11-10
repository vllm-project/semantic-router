"""Random controls preserve action volume without multiplying source samples."""

from __future__ import annotations

from collections import Counter

from systemone_auto.baselines import RANDOM_SEEDS, random_assignment_control


def test_random_control_keeps_action_mix_and_source_sample_size():
    rows = [{"id": str(i), "group_id": str(i)} for i in range(4)]
    selected = [{"action": name} for name in ("kai", "kai", "eos", "sol")]
    visited = []

    def run(records, setting):
        assert Counter(setting["actions"].values()) == Counter(
            row["action"] for row in selected
        )
        assert setting["delivery_kind"] == "cascade"
        visited.append(setting["actions"])
        return [
            {
                "group_id": row["group_id"],
                "result": {"correct": setting["actions"][row["id"]] == "kai"},
            }
            for row in records
        ]

    def aggregate(samples):
        assert len(samples) == len(rows)
        return {
            "bundle_accuracy": 0.5,
            "bundle_error": 0.5,
            "mean_typed_loss": 0.5,
            "mean_policy_cost_ms": 2,
            "mean_accumulated_call_ms": 3,
            "mean_calls": 1.5,
            "escalation_fraction": 0.5,
        }

    result = random_assignment_control(rows, selected, "cascade", run, aggregate)
    assert result["source_groups"] == len(rows)
    assert result["seeds"] == list(RANDOM_SEEDS)
    assert len(visited) == len(RANDOM_SEEDS)
    assert result["mean"]["mean_calls"] == 1.5
    low, high = result["bundle_accuracy_group_bootstrap_95pct"]
    assert 0 <= low <= 0.5 <= high <= 1
