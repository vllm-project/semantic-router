"""Inputs for differential testing against the production Go sampling helpers."""

from __future__ import annotations

import random
from dataclasses import asdict

from cli.evaluation.router_learning_policy import ArmState

STATE_FIELDS = {
    "quality_seed": "QualitySeed",
    "seed_weight": "SeedWeight",
    "good_fit": "GoodFitCount",
    "underpowered": "UnderpoweredCount",
    "overprovisioned": "OverprovisionedCount",
    "failed": "FailedCount",
    "latency_ewma": "LatencyEWMA",
    "cache_hit_ewma": "CacheHitEWMA",
    "cache_write_ewma": "CacheWriteEWMA",
    "input_cost_multiplier_ewma": "InputCostMultiplierEWMA",
}


def go_state(state: ArmState) -> dict:
    data = asdict(state)
    return {
        **{target: data[source] for source, target in STATE_FIELDS.items()},
        "LastUpdated": (
            "1970-01-01T00:00:01Z" if state.updated else "0001-01-01T00:00:00Z"
        ),
    }


def python_state(data: dict) -> ArmState:
    return ArmState(
        **{key: data[value] for key, value in STATE_FIELDS.items()},
        updated=data["LastUpdated"] != "0001-01-01T00:00:00Z",
    )


def parity_cases() -> list[dict]:
    rng = random.Random(2346)
    cases = []
    # Every score term is isolated, including clamps and no catalog pricing.
    states = [ArmState(), ArmState(updated=True)] + [
        ArmState(**{field: value}, updated=True)
        for field, values in {
            "good_fit": [1, 8],
            "underpowered": [1, 8],
            "overprovisioned": [1, 8],
            "failed": [1, 8],
            "latency_ewma": [-0.2, 0.6, 2.0],
            "cache_hit_ewma": [-0.2, 0.8, 2.0],
            "input_cost_multiplier_ewma": [-0.2, 0.5, 2.0],
        }.items()
        for value in values
    ]
    for scope in ("decision", "tier", "global"):
        for sampling in (False, True):
            for index, state in enumerate(states):
                cases.append(
                    {
                        "Base": "base",
                        "Scope": scope,
                        "Sampling": sampling,
                        "Arms": [
                            {
                                "Model": "base",
                                "State": go_state(ArmState(updated=True)),
                                "Cost": 0.0 if index % 3 == 0 else 1.0,
                                "Sample": 0.50,
                            },
                            {
                                "Model": "other",
                                "State": go_state(state),
                                "Cost": 0.0 if index % 3 == 0 else 3.0,
                                "Sample": 0.53,
                            },
                        ],
                    }
                )
    # Combined states, seed overrides, absent bases, ordering, and near ties.
    for index in range(160):
        arms = []
        for model in ("z-model", "a-model", "m-model"):
            state = ArmState(
                quality_seed=rng.random(),
                seed_weight=rng.choice([0, 2, 7]),
                good_fit=rng.randrange(5),
                underpowered=rng.randrange(5),
                overprovisioned=rng.randrange(5),
                failed=rng.randrange(5),
                latency_ewma=rng.random() * 2,
                cache_hit_ewma=rng.random(),
                input_cost_multiplier_ewma=rng.random(),
                updated=rng.choice([True, False]),
            )
            arms.append(
                {
                    "Model": model,
                    "State": go_state(state),
                    "Cost": rng.choice([0.0, 1.0, 4.0]),
                    "Sample": rng.random(),
                }
            )
        cases.append(
            {
                "Base": rng.choice(["z-model", "absent", ""]),
                "Scope": rng.choice(["decision", "tier", "global"]),
                "Sampling": index % 2 == 0,
                "Arms": arms,
            }
        )
    # Exactly equal scores must sort by ascending model ID regardless of input order.
    cases.append(
        {
            "Base": "absent",
            "Scope": "decision",
            "Sampling": True,
            "Arms": [
                {
                    "Model": model,
                    "State": go_state(ArmState(updated=True)),
                    "Cost": 0.0,
                    "Sample": 0.5,
                }
                for model in ("z-model", "a-model")
            ],
        }
    )
    return cases


def parity_events() -> list[dict]:
    events = [
        {"Verdict": verdict, "Weight": weight}
        for verdict in ("good_fit", "underpowered", "overprovisioned", "failed")
        for weight in (0.0, 0.4, 1.0, 2.8)
    ]
    for observed in (0.8, 0.2, 0.0, -1.0, 0.6):
        events.append(
            {
                "Telemetry": {
                    "LatencyObserved": True,
                    "LatencySeconds": observed,
                    "CacheObserved": True,
                    "CacheHitRatio": observed,
                    "CacheWritePressure": observed,
                    "InputCostObserved": True,
                    "InputCostMultiplier": observed,
                    "ProviderFailureObserved": observed == 0.2,
                }
            }
        )
    events.append(
        {
            "Telemetry": {
                "LatencyObserved": False,
                "LatencySeconds": 9.0,
                "CacheObserved": False,
                "CacheHitRatio": 9.0,
                "InputCostObserved": False,
                "InputCostMultiplier": 9.0,
            }
        }
    )
    return events
