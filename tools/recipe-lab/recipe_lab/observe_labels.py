"""Cold-start C2: observe the router's own domain labels for a case split.

A seed config gives every request a deterministic route to observe: 14 domain
decisions (one `type: domain` condition each, named after the MMLU-Pro
categories the router's domain classifier emits) plus a catch-all decision
named `unmatched`. Decision names ARE the labels: a case whose record names a
domain decision carries that label, and the catch-all (no domain matched)
records `other` - no domain decision fired.

Verified against the router source (C2 prerequisites):

- The 14 supported routing domain names are fixed by
  pkg/config/domain_contract.go (`supportedRoutingDomains`); mirrored here.
- One request produces exactly one replay record: startRouterReplay is
  guarded by `ctx.RouterReplayID == ""` (pkg/extproc/recorder.go), so the
  request-id join sees at most one record per case.
- The record carries the matched decision name (`decision`), the domain
  classifier's own top category (`category`, an independent cross-check
  source), and the decision confidence (`confidence_score` +
  `confidence_score_available`; pkg/routerreplay/store/store.go).
- Decisions referencing a domain auto-generate the domain signal
  (pkg/config/canonical_domain_autogen_test.go), so the seed config needs no
  explicit domains section; a domain condition is
  `{type: domain, name: <label>}`.
- The seed config mirrors the proven examples/smoke shape: top-level
  `routing` profile (the default recipe), entrypoint `vllm-sr/auto` ->
  `recipe: default`, `global.model_catalog.system.decision_model` explicitly
  pinned to a declared deployment (#5620: never rely on the default), and the
  response cache disabled so every request really reaches the decision path.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .contract import ContractError, dataset_content_hash
from .runner import RunnerEnv, probe_records

# Mirror of pkg/config/domain_contract.go `supportedRoutingDomains`: the exact
# label set the repository's domain classifier emits (and the validator
# enforces on domain conditions).
SUPPORTED_ROUTING_DOMAINS = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)

# The catch-all decision of the seed config. It is deliberately NOT a domain
# name (the validator would reject a duplicate decision name), and its records
# are labeled `other`: no domain decision fired.
FALLBACK_DECISION_NAME = "unmatched"

OTHER_LABEL = "other"

# The request model name routes through the built-in entrypoint into the
# default recipe (the seed config's top-level routing profile).
ROUTER_MODEL = "vllm-sr/auto"

# x-request-id tag for the observe probe; the record keeps it verbatim, so
# labels join cases by request id regardless of prompt truncation.
REQUEST_ID_PREFIX = "rl-observe-"

# Replay capture bound: MCQ prompts (question + 10 options) can exceed the
# 4096 default, and a truncated prompt breaks any prompt-join debugging. The
# request-id join does not depend on it; this only keeps the stored prompt
# readable.
SEED_MAX_BODY_BYTES = 16384

REPLAY_PLUGIN = {
    "type": "router_replay",
    "configuration": {
        "enabled": True,
        "max_records": 5000,
        "max_body_bytes": SEED_MAX_BODY_BYTES,
    },
}


_HTTP_OK = 200


def label_for_decision(decision: str) -> str:
    """The router label of one record: the matched decision name when it is a
    routing domain, `other` for the catch-all (or anything else)."""
    decision = (decision or "").strip()
    return decision if decision in SUPPORTED_ROUTING_DOMAINS else OTHER_LABEL


def build_seed_config(
    *,
    backend_endpoint: str = "127.0.0.1:8000",
    model_name: str = "local/seed-backend",
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The cold-start seed config: every decision observes one domain.

    All decisions point at the same backend model - the seed observes LABELS,
    not answers - and every decision mounts the router_replay plugin so the
    matched decision name surfaces in the record.

    ``classification`` (Y2) carries the shared classification sections read
    from the classification source config (``load_classification_source``);
    they are injected verbatim into ``global.model_catalog`` so the seed
    classifies exactly like the bootstrap candidates do. Without it the
    router's defaults apply - which the candidates (scaffold copies of the
    maintained config) do not, and the observed labels would describe a
    classifier the final config never runs.
    """
    domain_decisions = [
        {
            "name": domain,
            "description": f"Observe the {domain} domain label.",
            "priority": 200,
            "tier": 1,
            "rules": {
                "operator": "AND",
                "conditions": [{"type": "domain", "name": domain}],
            },
            "modelRefs": [{"model": model_name, "use_reasoning": False}],
            "plugins": [dict(REPLAY_PLUGIN)],
        }
        for domain in SUPPORTED_ROUTING_DOMAINS
    ]
    model_catalog: dict[str, Any] = {
        "deployments": {
            "primary": {"provider": "model_runtime", "artifact": "vllm-sr/d3-lite"}
        },
        "system": {"decision_model": {"deployment": "primary"}},
    }
    if classification:
        model_catalog.update(classification)
    return {
        "version": "v0.3",
        "listeners": [
            {"name": "http-seed", "address": "0.0.0.0", "port": 8899, "timeout": "300s"}
        ],
        "global": {
            "stores": {"response_cache": {"enabled": False, "backend_type": "memory"}},
            "model_catalog": model_catalog,
        },
        "providers": {
            "defaults": {"model": model_name},
            "models": [
                {
                    "name": model_name,
                    "provider_model_id": model_name,
                    "api_format": "openai",
                    "backend_refs": [
                        {
                            "name": "seed-backend",
                            "endpoint": backend_endpoint,
                            "protocol": "http",
                            "provider": "vllm",
                            "weight": 1,
                        }
                    ],
                }
            ],
        },
        "routing": {
            "modelCards": [
                {
                    "name": model_name,
                    "context_window_size": 131072,
                    "description": "Seed observation backend; answers are irrelevant.",
                    "capabilities": ["chat", "text"],
                    "modality": "text",
                }
            ],
            "decisions": [
                *domain_decisions,
                {
                    "name": FALLBACK_DECISION_NAME,
                    "description": "Catch-all: no domain decision matched.",
                    "priority": 100,
                    "tier": 2,
                    "modelRefs": [{"model": model_name, "use_reasoning": False}],
                    "plugins": [dict(REPLAY_PLUGIN)],
                },
            ],
        },
        "entrypoints": [{"model_names": [ROUTER_MODEL], "recipe": "default"}],
    }


def seed_config_yaml(config: dict[str, Any]) -> str:
    """Deterministic YAML text of the seed config (same dict => same bytes)."""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def write_seed_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(seed_config_yaml(config), encoding="utf-8")
    return path


def extract_confidence(record: dict[str, Any]) -> float | None:
    """The record's decision confidence, or None when not scoreable.

    `confidence_score_available` gates the value: a catch-all match (or an
    unavailable classifier) must record null, never a stale or zero score.
    """
    available = record.get("confidence_score_available")
    if available is None:
        available = record.get("ConfidenceScoreAvailable")
    score = record.get("confidence_score")
    if score is None:
        score = record.get("ConfidenceScore")
    if not available or not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    score = float(score)
    return score if math.isfinite(score) else None


def request_id_for_case(case_id: str) -> str:
    """The x-request-id sent for one observed case."""
    return f"{REQUEST_ID_PREFIX}{case_id}"


def labels_from_records(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    label_of: Any = None,
    request_id_prefix: str = REQUEST_ID_PREFIX,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join replay records to cases by request id and extract label rows.

    One output row per case that produced exactly one record:
    `{"case_id", "router_label", "confidence"}` (sorted by case_id, so the
    file is byte-stable across runs that observe the same routing). Cases
    without a record are reported in stats["missing"], never silently folded
    into `other` - a lost record is not a routing observation.

    ``label_of`` maps the record's matched decision name to the reported
    label; the default reads the record's ``category`` (the classifier's
    selected domain) and falls back to the decision name only when the
    category is absent or not a routing domain. The lane observer
    (observe_lanes) passes the identity mapping: in a template config every
    decision IS a lane.
    """
    # label_of stays None for the default projection. Reassigning it to
    # label_for_decision here would make the in-loop category branch dead
    # code: every label would come from the decision name, which folds
    # bootstrap candidates' route-<model> decisions into `other` and breaks
    # the Y2 seed-vs-candidate agreement the bootstrap gate checks.
    by_request_id: dict[str, dict[str, Any]] = {}
    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen_case_ids:
            raise ContractError(f"duplicate case_id in the observed split: {case_id}")
        seen_case_ids.add(case_id)
        by_request_id[f"{request_id_prefix}{case_id}"] = case

    rows: list[dict[str, Any]] = []
    stats = {
        "matched": 0,
        "no_request_id": 0,
        "unknown_request_id": 0,
        "duplicate_record": 0,
    }
    seen_request_ids: set[str] = set()
    for record in records:
        request_id = str(record.get("request_id") or record.get("RequestID") or "")
        if not request_id:
            stats["no_request_id"] += 1
            continue
        case = by_request_id.get(request_id)
        if case is None:
            stats["unknown_request_id"] += 1
            continue
        if request_id in seen_request_ids:
            stats["duplicate_record"] += 1
            continue
        seen_request_ids.add(request_id)
        category = str(record.get("category") or record.get("Category") or "").strip()
        decision = str(record.get("decision") or record.get("Decision") or "")
        if label_of is None:
            # The record's `category` is the classifier's selected domain -
            # the router fills it on every request (extproc recorder), and it
            # is the observation target of C2 and of the Y2 seed-vs-candidate
            # agreement. Older records may only carry the matched decision
            # name, whose spelling differs per config (seed decisions are
            # domain-named while bootstrap candidates name decisions
            # `route-<model>`), so fall back to the decision spelling.
            label = (
                category
                if category in SUPPORTED_ROUTING_DOMAINS
                else label_for_decision(decision)
            )
        else:
            # a custom mapper owns the whole projection (the lane observer
            # passes the identity mapping: in a template config every
            # decision IS a lane)
            label = label_of(decision)
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "router_label": label,
                "confidence": extract_confidence(record),
            }
        )
        stats["matched"] += 1
    stats["missing"] = sum(1 for rid in by_request_id if rid not in seen_request_ids)
    rows.sort(key=lambda row: row["case_id"])
    return rows, stats


def observe_split(
    *,
    env: RunnerEnv,
    config: str | Path,
    cases: list[dict[str, Any]],
    router_model: str = ROUTER_MODEL,
    request_timeout: float = 180.0,
    trial_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Observe one case split through the seed config (C2).

    Boots the stack once via probe_records with tagged request ids, joins the
    records to cases, and summarizes the label distribution. The label
    coverage is computed against the FULL case count (a lost record is never
    covered), which is the number the provenance records (C2 completion
    criterion).
    """
    records, statuses, timings = probe_records(
        env,
        config,
        cases,
        router_model=router_model,
        request_timeout=request_timeout,
        trial_timeout_s=trial_timeout_s,
        request_id_prefix=REQUEST_ID_PREFIX,
    )
    rows, stats = labels_from_records(records, cases)
    label_counts = dict(sorted(Counter(r["router_label"] for r in rows).items()))
    labeled = sum(
        count for label, count in label_counts.items() if label != OTHER_LABEL
    )
    summary = {
        "cases": len(cases),
        "observed": len(rows),
        "missing": stats["missing"],
        "labeled": labeled,
        "other": label_counts.get(OTHER_LABEL, 0),
        # C2 completion criterion: fraction of the split the router labeled
        # with a real domain (not other, not lost).
        "label_coverage": round(labeled / len(cases), 6) if cases else 0.0,
        "label_counts": label_counts,
        "non_200_requests": sum(
            1 for status in statuses.values() if status != _HTTP_OK
        ),
    }
    return {
        "rows": rows,
        "stats": stats,
        "statuses": statuses,
        "timings": timings,
        "summary": summary,
    }


def write_labels(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Byte-stable labels JSONL (sorted rows, sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["case_id"])
    with path.open("w", encoding="utf-8") as fh:
        for row in ordered:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_labels_summary(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def seed_config_fingerprint(path: str | Path) -> str:
    """The seed config identity for provenance (sha256 of the file bytes)."""
    return dataset_content_hash(path)
