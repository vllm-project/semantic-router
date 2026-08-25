"""Clean v0.3 Recipe-scoped ML selector contract tests."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.models import RecipeRouting  # noqa: E402


def ml_decision(name: str, algorithm_type: str, *, models_path: str, family: dict):
    return {
        "name": name,
        "rules": {},
        "algorithm": {
            "type": algorithm_type,
            "ml": {
                "models_path": models_path,
                "embedding_dim": 1024,
                algorithm_type: family,
            },
        },
    }


def test_recipe_document_merges_non_conflicting_ml_families():
    document = RecipeRouting(
        decisions=[
            ml_decision("nearest", "knn", models_path="/models/a", family={"k": 5}),
            ml_decision(
                "boundary",
                "svm",
                models_path="/models/a",
                family={"kernel": "rbf"},
            ),
        ]
    )

    assert document.decisions[0].algorithm.ml.knn.k == 5
    assert document.decisions[1].algorithm.ml.svm.kernel == "rbf"


def test_recipe_document_rejects_conflicting_shared_ml_settings():
    with pytest.raises(ValidationError, match=r"conflicting algorithm\.ml shared"):
        RecipeRouting(
            decisions=[
                ml_decision(
                    "nearest",
                    "knn",
                    models_path="/models/a",
                    family={"k": 5},
                ),
                ml_decision(
                    "boundary",
                    "svm",
                    models_path="/models/b",
                    family={"kernel": "rbf"},
                ),
            ]
        )


def test_recipe_document_rejects_conflicting_same_family_settings():
    with pytest.raises(ValidationError, match=r"conflicting algorithm\.ml\.knn"):
        RecipeRouting(
            decisions=[
                ml_decision(
                    "nearest-a",
                    "knn",
                    models_path="/models/a",
                    family={"k": 5},
                ),
                ml_decision(
                    "nearest-b",
                    "knn",
                    models_path="/models/a",
                    family={"k": 9},
                ),
            ]
        )


def test_different_recipes_can_use_different_ml_settings():
    fast = RecipeRouting(
        decisions=[
            ml_decision(
                "choose",
                "knn",
                models_path="/models/fast",
                family={"k": 3},
            )
        ]
    )
    deep = RecipeRouting(
        decisions=[
            ml_decision(
                "choose",
                "knn",
                models_path="/models/deep",
                family={"k": 11},
            )
        ]
    )

    assert fast.decisions[0].algorithm.ml.knn.k == 3
    assert deep.decisions[0].algorithm.ml.knn.k == 11
