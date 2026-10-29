"""The guard miss correlation tool (``tools/guard_miss_correlation.py``, issue #4669)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "guard_miss_correlation.py"


@pytest.fixture(scope="module")
def gm():
    spec = importlib.util.spec_from_file_location("guard_miss_correlation", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_arm(
    tmp_path: Path,
    name: str,
    jb_silent: np.ndarray,
    s_silent: np.ndarray,
    gold: list[tuple[bool, bool]],
) -> str:
    """A suite and one arm's predictions for texts with the given (jailbreak, unsafe) gold."""
    suite = tmp_path / "suite" / "toxicchat"
    for task, positive, negative, index, silent in (
        ("jailbreak", "jailbreak", "benign", 0, jb_silent),
        ("safety", "unsafe", "safe", 1, s_silent),
    ):
        (suite / task).mkdir(parents=True, exist_ok=True)
        pred_dir = tmp_path / name / task
        pred_dir.mkdir(parents=True, exist_ok=True)
        rows, preds = [], []
        for i, g in enumerate(gold):
            rid = f"{task}-{i}"
            rows.append(
                {
                    "id": rid,
                    "text": f"text {i}",
                    "label": positive if g[index] else negative,
                    "group": f"g{i}",
                }
            )
            preds.append({"id": rid, "scores": {positive: 0.1 if silent[i] else 0.99}})
        (suite / task / "hold-toxicchat.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (pred_dir / "hold-toxicchat.pred.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in preds)
        )
    return str(tmp_path / "suite")


def gold_cells(both: int, safety_only: int, neither: int) -> list[tuple[bool, bool]]:
    return (
        [(True, True)] * both
        + [(False, True)] * safety_only
        + [(False, False)] * neither
    )


def test_independent_misses_give_a_ratio_near_one_and_correlated_ones_do_not(
    gm, tmp_path
) -> None:
    gold = gold_cells(400, 100, 500)
    rng = np.random.default_rng(1)
    n = len(gold)
    # Each head is silent on a fifth of the texts, independently ...
    independent = (rng.random(n) < 0.2, rng.random(n) < 0.2)
    # ... or together, on the same fifth.
    shared = rng.random(n) < 0.2
    suite = write_arm(tmp_path, "ind", *independent, gold)
    write_arm(tmp_path, "cor", shared, shared, gold)
    ind = gm.summarize(
        {"a": gm.build_table(suite, str(tmp_path / "ind"), (0.5, 0.5))}, 300, 0
    )["arms"]["a"]
    cor = gm.summarize(
        {"a": gm.build_table(suite, str(tmp_path / "cor"), (0.5, 0.5))}, 300, 0
    )["arms"]["a"]
    assert ind["rule_positives"] == cor["rule_positives"] == 500
    assert ind["ratio_ci"][0] < 1 < ind["ratio_ci"][1]
    assert cor["ratio"] > 2 and cor["ratio_ci"][0] > 1
    assert cor["odds_ratio"] > 10 > ind["odds_ratio"]


def test_a_miss_needs_both_heads_silent_on_a_rule_positive_text(gm, tmp_path) -> None:
    gold = gold_cells(2, 1, 1)
    jb_silent = np.array([True, True, True, True])
    s_silent = np.array([True, False, True, True])
    suite = write_arm(tmp_path, "arm", jb_silent, s_silent, gold)
    counts = gm.miss_counts(gm.build_table(suite, str(tmp_path / "arm"), (0.5, 0.5)))
    # Text 0 and the safety-only text are missed; text 1 is caught by safety; the clean text is not a miss.
    assert (
        counts["both"][0] == 1
        and counts["safety_only"][0] == 1
        and "jailbreak_only" not in counts
    )
    assert gm.pooled_ratio(counts) == pytest.approx(2 / (2 * 1.0 * 0.5 + 1 * 1.0 * 1.0))


def test_the_smallest_detectable_ratio_falls_as_more_misses_are_expected(gm) -> None:
    assert (
        gm.smallest_detectable_ratio(2.0)
        > gm.smallest_detectable_ratio(20.0)
        > gm.smallest_detectable_ratio(200.0)
        > 1
    )
    assert math.isinf(gm.smallest_detectable_ratio(0.0))
    assert gm.poisson_upper_tail(0, 3.0) == 1.0
    assert gm.poisson_upper_tail(1, 1.0) == pytest.approx(1 - math.exp(-1))


def test_the_gate_reads_pass_fail_and_underpowered(gm) -> None:
    def report(ratio, ci, diff):
        return {
            "arms": {"a": {"ratio": ratio, "ratio_ci": ci}},
            "difference_ratio": {"ci": diff},
        }

    assert gm.verdict(report(2.0, [1.3, 3.1], [0.4, 1.9]), "a") == "pass"
    assert gm.verdict(report(1.0, [0.8, 1.2], [-0.5, 0.5]), "a") == "fail"
    assert gm.verdict(report(1.6, [0.9, 2.8], [-0.2, 1.5]), "a") == "underpowered"
    assert gm.verdict(report(2.0, [1.3, 3.1], [-0.1, 1.9]), "a") == "underpowered"
    assert (
        gm.verdict(report(math.nan, [None, None], [None, None]), "a") == "underpowered"
    )


def test_mantel_haenszel_adds_a_half_to_a_cell_with_an_empty_count(
    gm, tmp_path
) -> None:
    # Every text has the heads silent together or fired together: b and c are empty in each cell.
    gold = gold_cells(10, 0, 10)
    silent = np.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5)
    suite = write_arm(tmp_path, "arm", silent, silent, gold)
    odds = gm.mantel_haenszel(gm.build_table(suite, str(tmp_path / "arm"), (0.5, 0.5)))
    assert math.isfinite(odds) and odds > 1


def test_matched_thresholds_keep_the_false_positive_rate_of_the_reference(
    gm, tmp_path
) -> None:
    gold = gold_cells(2, 0, 8)
    jb_silent = np.array(
        [False, False] + [True] * 6 + [False] * 2
    )  # fires on 2 of 8 negatives
    ref = gm.build_table(
        write_arm(tmp_path, "ref", jb_silent, jb_silent, gold),
        str(tmp_path / "ref"),
        (0.5, 0.5),
    )
    # The other arm scores its negatives 0.1 ... 0.8 and its positives 0.95.
    suite = write_arm(
        tmp_path, "oth", np.zeros(10, dtype=bool), np.zeros(10, dtype=bool), gold
    )
    for task, positive in (("jailbreak", "jailbreak"), ("safety", "unsafe")):
        path = tmp_path / "oth" / task / "hold-toxicchat.pred.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for i, row in enumerate(rows):
            row["scores"] = {positive: 0.95 if i < 2 else 0.1 * (i - 1)}
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    other = gm.build_table(suite, str(tmp_path / "oth"), (0.5, 0.5))
    jb_threshold, _ = gm.matched_thresholds(ref, other)
    fired = gm.refire(other, (jb_threshold, 0.5)).jb_fires
    assert fired[~other.jb_gold].sum() == 2 and fired[other.jb_gold].all()


def test_the_command_writes_a_report_with_a_matched_comparison_arm(
    gm, tmp_path
) -> None:
    gold = gold_cells(60, 20, 120)
    rng = np.random.default_rng(3)
    shared = rng.random(len(gold)) < 0.3
    suite = write_arm(tmp_path, "a", shared, shared, gold)
    write_arm(
        tmp_path, "b", rng.random(len(gold)) < 0.3, rng.random(len(gold)) < 0.3, gold
    )
    out = tmp_path / "guard.json"
    code = gm.main(["--suite", suite, "--a", str(tmp_path / "a"), "--a-thresholds", "0.5,0.5",
                    "--b", str(tmp_path / "b"), "--b-match-fpr", "--rounds", "50", "--out", str(out)])  # fmt: skip
    report = json.loads(out.read_text())
    assert code == 0 and report["verdict"] in {"pass", "fail", "underpowered"}
    assert set(report["arms"]) == {"a", "b"} and "difference_ratio" in report
    assert report["arms"]["a"]["rule_positives"] == 80
    with pytest.raises(SystemExit):
        gm.main(["--suite", suite, "--a", str(tmp_path / "a"), "--a-thresholds", "0.5,0.5", "--b-match-fpr", "--out", str(out)])  # fmt: skip
