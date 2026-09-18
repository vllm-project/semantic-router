"""Audit the pinned observations and Pareto memberships using two algorithms."""

import hashlib
import json
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


def main():
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / "provenance.json").read_text())
    for item in provenance["files"]:
        digest = hashlib.sha256((root / item["file"]).read_bytes()).hexdigest()
        assert digest == item["sha256"], item["file"]
    data = json.loads((root / "pareto-data.json").read_text())
    audit = {"source_revision": provenance["revision"], "tasks": {}}
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
    print("Verified all 420 observations, all four frontiers and source hashes.")


if __name__ == "__main__":
    main()
