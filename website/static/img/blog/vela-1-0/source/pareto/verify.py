"""Audit the pinned observations and Pareto memberships using two algorithms."""

import hashlib
import json
import math
from pathlib import Path


def pairwise_frontier(points):
    return [
        index
        for index, point in enumerate(points)
        if not any(
            other["parameters_billion"] <= point["parameters_billion"]
            and other["score"] >= point["score"]
            and (
                other["parameters_billion"] < point["parameters_billion"]
                or other["score"] > point["score"]
            )
            for other in points
        )
    ]


def sweep_frontier(points):
    ordered = sorted(
        range(len(points)),
        key=lambda i: (points[i]["parameters_billion"], -points[i]["score"]),
    )
    best = float("-inf")
    previous_size = None
    kept = []
    for index in ordered:
        point = points[index]
        size, score = point["parameters_billion"], point["score"]
        if score > best or (score == best and size == previous_size):
            kept.append(index)
            best = score
            previous_size = size
    return sorted(kept)


def verify_panels(root):
    raw = (root / "general-rank-data.json").read_bytes()
    data = json.loads(raw)
    summaries = {}
    for name, panel in data["panels"].items():
        points = panel["complete_points"]
        known = [
            p
            for p in points
            if p["parameters_billion"] is not None and p["parameters_billion"] > 0
        ]
        assert {p["id"] for p in known} == set(panel["plotted_ids"])
        for point in points:
            scores = point["scores_by_task"]
            assert set(scores) == set(panel["task_types"])
            groups = {}
            for task, value in scores.items():
                groups.setdefault(panel["task_types"][task], []).append(value)
            means = [sum(values) / len(values) for values in groups.values()]
            assert math.isclose(
                sum(scores.values()) / len(scores), point["mean_task"], abs_tol=1e-12
            )
            assert math.isclose(
                sum(means) / len(means), point["mean_task_type"], abs_tol=1e-12
            )
        for metric in ("mean_task_type", "mean_task"):
            coordinates = [{**p, "score": p[metric]} for p in known]
            indexes = pairwise_frontier(coordinates)
            assert indexes == sweep_frontier(coordinates)
            frontier = {known[i]["id"] for i in indexes}
            assert frontier == set(panel["known_size_frontier"][metric])
            for point in points:
                record = point["rank"][metric]
                assert record["global_population"] == len(points)
                assert record["global_rank"] == 1 + sum(
                    other[metric] > point[metric] for other in points
                )
                assert math.isclose(record["score_100"], point[metric] * 100)
                if point in known:
                    eligible = [
                        p
                        for p in known
                        if p["parameters_billion"] <= point["parameters_billion"]
                    ]
                    size_rank = record["at_most_size"]
                    assert size_rank["population"] == len(eligible)
                    assert size_rank["rank"] == 1 + sum(
                        p[metric] > point[metric] for p in eligible
                    )
                    gap = (max(p[metric] for p in eligible) - point[metric]) * 100
                    assert math.isclose(gap, size_rank["gap_to_best_pp"], abs_tol=1e-10)
                    assert record["known_size_frontier"] == (point["id"] in frontier)
                    assert point["on_known_size_frontier"][metric] == (
                        point["id"] in frontier
                    )
        summaries[name] = {
            "complete_observations": len(points),
            "plotted_known_size_observations": len(known),
            "coverage": panel["coverage"],
            "comparison_modes": data["comparison_modes"][name],
            "vela": [
                {
                    "model": p["model"],
                    "parameters_billion": p["parameters_billion"],
                    "rank": p["rank"],
                }
                for p in points
                if p["id"].startswith("current:")
            ],
        }
    for variant in ("nano", "mini"):
        meta = json.loads((root / f"general-rank-figures-{variant}.json").read_text())
        assert hashlib.sha256(raw).hexdigest() == meta["data_sha256"]
        for key, figure in meta["figures"].items():
            name = key.rsplit("-", 1)[0]
            panel = data["panels"][name]
            current = next(
                p for p in panel["complete_points"] if p["id"] == f"current:{variant}"
            )
            assert figure["rank"] == current["rank"]["mean_task_type"]
            assert set(figure["plotted_ids"]) == set(panel["plotted_ids"])
            assert set(figure["known_size_frontier"]) == set(
                panel["known_size_frontier"]["mean_task_type"]
            )
            svg = root / f"omni-{variant}-general-{name.lower()}.svg"
            expected = figure["exports_sha256"][f"general-{name.lower()}-{variant}.svg"]
            assert hashlib.sha256(svg.read_bytes()).hexdigest() == expected
    return summaries


def main():
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / "provenance.json").read_text())
    for item in provenance["files"]:
        digest = hashlib.sha256((root / item["file"]).read_bytes()).hexdigest()
        assert digest == item["sha256"], item["file"]
    data = json.loads((root / "pareto-data.json").read_text())
    additional = json.loads((root / "pareto-new-frontiers.json").read_text())
    data["tasks"].update(additional["tasks"])
    audit = {
        "source_revision": provenance["revision"],
        "companion_mini_revision": provenance["companion_mini_revision"],
        "complete_panels": verify_panels(root),
        "tasks": {},
    }
    for key, task in data["tasks"].items():
        points = task["points"]
        pairwise = pairwise_frontier(points)
        assert pairwise == sweep_frontier(points), key
        assert pairwise == [
            i for i, point in enumerate(points) if point["on_displayed_frontier"]
        ], key
        vela = [point for point in points if point["kind"] == "vela"]
        # Keep this tolerant of the upstream kind naming; model ids are stable.
        if not vela:
            vela = [point for point in points if "/Vela-" in point["model"]]
        audit["tasks"][key] = {
            "task": task["task"],
            "metric": task["metric"],
            "observation_count": len(points),
            "frontier_count": len(pairwise),
            "vela": [
                {
                    name: point[name]
                    for name in (
                        "model",
                        "parameters_billion",
                        "score",
                        "on_displayed_frontier",
                        "evidence_type",
                    )
                }
                for point in vela
            ],
            "frontier": [
                {
                    name: points[i][name]
                    for name in ("model", "parameters_billion", "score")
                }
                for i in pairwise
            ],
        }
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    count = sum(task["observation_count"] for task in audit["tasks"].values())
    complete = sum(
        panel["complete_observations"] for panel in audit["complete_panels"].values()
    )
    print(
        f"Verified {count} task observations, {complete} complete-panel observations, "
        "both aggregations/ranks/frontiers and all source hashes."
    )


if __name__ == "__main__":
    main()
