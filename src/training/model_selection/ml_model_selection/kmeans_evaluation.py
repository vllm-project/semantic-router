#!/usr/bin/env python3
"""Held-out evaluation of the KMeans selector against the shared baselines.

Fits KMeans on the train split of a query-level split and scores it, the
baselines and the oracle on the test split with `evaluation`. A test query
whose candidates are all unknown to the trained set counts as an abstention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
from data_loader import RoutingRecord, create_feature_vector
from evaluation import PolicyReport, Selector, evaluate_against_baselines
from models import KMeansModel, NoEligibleCandidateError
from objective import SelectorObjective
from query_outcome_set import (
    QueryOutcomeSet,
    build_query_outcome_sets,
    split_by_query,
)

SELECTOR_NAME: Final = "kmeans"


def kmeans_selector(
    model: KMeansModel, features_by_query: Mapping[str, np.ndarray]
) -> Selector:
    """The trained model as a selector: nearest cluster, best eligible candidate."""

    def select(query: str, eligible: tuple[str, ...]) -> str | None:
        vector = features_by_query.get(query)
        if vector is None:
            raise ValueError(f"no feature vector for test query {query!r}")
        try:
            return model.predict(vector, eligible)
        except NoEligibleCandidateError:
            return None

    return select


def evaluate_kmeans(
    train: Sequence[QueryOutcomeSet],
    test: Sequence[QueryOutcomeSet],
    features: Mapping[str, np.ndarray],
    objective: SelectorObjective,
    *,
    n_clusters: int = 8,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    min_support: int = 1,
) -> dict[str, PolicyReport]:
    """Fit on `train`, then report KMeans, every baseline and the oracle on `test`.

    `features` maps query_id to the query's feature vector.
    """
    model = KMeansModel(
        n_clusters=n_clusters,
        objective=objective,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        min_support=min_support,
    )
    model.train_snapshots(train, features)
    missing = [s.query for s in test if s.query_id not in features]
    if missing:
        raise ValueError(f"no feature vector for test query {missing[0]!r}")
    if len({s.query for s in test}) != len(test):
        raise ValueError("test queries must be unique by text")
    by_query = {snapshot.query: features[snapshot.query_id] for snapshot in test}
    return evaluate_against_baselines(
        kmeans_selector(model, by_query),
        train,
        test,
        objective,
        selector_name=SELECTOR_NAME,
        seed=seed,
    )


def held_out_kmeans_report(
    records: Sequence[RoutingRecord],
    embeddings: Mapping[str, np.ndarray],
    objective: SelectorObjective,
    *,
    source: str,
    split_seed: int = 0,
    n_clusters: int = 8,
    seed: int = 42,
) -> dict[str, PolicyReport]:
    """From benchmark records and query embeddings to a held-out report.

    Queries are split before anything is derived from them, so no query is on
    both sides, and a query with a single candidate (no choice) is dropped.
    """
    snapshots = build_query_outcome_sets(records, source, drop_single_candidate=True)
    missing = [s.query for s in snapshots if s.query not in embeddings]
    if missing:
        raise ValueError(
            f"no embedding for {len(missing)} queries, e.g. {missing[0]!r}"
        )
    split = split_by_query(snapshots, seed=split_seed)
    features = {
        s.query_id: create_feature_vector(embeddings[s.query], s.category)
        for s in (*split.train, *split.test)
    }
    return evaluate_kmeans(
        split.train,
        split.test,
        features,
        objective,
        n_clusters=n_clusters,
        seed=seed,
    )


def report_to_dict(reports: Mapping[str, PolicyReport]) -> dict[str, object]:
    """A JSON-serializable form of the reports."""
    return {name: asdict(report) for name, report in reports.items()}


def summarize(reports: Mapping[str, PolicyReport]) -> str:
    """One line per policy, best mean utility first."""
    ordered = sorted(reports.values(), key=lambda r: -r.overall.mean_utility)
    lines = [
        f"{'policy':<14} {'coverage':>8} {'utility':>8} {'regret':>8} {'quality':>8}"
    ]
    lines += [
        f"{r.name:<14} {r.overall.coverage:>8.3f} {r.overall.mean_utility:>8.3f} "
        f"{r.overall.mean_regret:>8.3f} {r.overall.mean_quality:>8.3f}"
        for r in ordered
    ]
    return "\n".join(lines)


def write_held_out_report(
    reports: Mapping[str, PolicyReport], output_dir: Path
) -> Path:
    path = output_dir / "kmeans_evaluation.json"
    path.write_text(json.dumps(report_to_dict(reports), indent=2, sort_keys=True))
    return path
