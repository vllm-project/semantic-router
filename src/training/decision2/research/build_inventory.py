"""Rebuild the screened Jev catalog from the locally pinned source snapshot.

The large upstream JSONL snapshot stays local. The derived inventory contains
attribution, identities, screening decisions and limitations, not its prose.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" / "all-methods-2026-09-26.jsonl"
EXPECTED_SHA256 = "612332c1ac1209554c054afb60b2bdfb4854064dcb927a470a1096c9304ba2c5"
OUTPUT = HERE / "jev-catalog-inventory-2026-09-26.jsonl"
SOURCE_URL = "https://hanxiao.io/all-about-jev/all-methods.jsonl"

# Indexes refer only to the SHA-256 pinned snapshot. P0 means worth an exact
# unified comparison, not that a source's performance claim was reproduced.
MODEL_P0 = {7, 14, 23, 79, 87, 105, 109, 120, 163, 164, 167, 169, 181, 187, 195}
MODEL_P1 = {
    12,
    32,
    34,
    35,
    46,
    64,
    67,
    76,
    77,
    78,
    81,
    95,
    97,
    127,
    131,
    142,
    160,
    183,
    192,
    194,
    196,
}
MODEL_INDEX_ROWS = {0, 1, 2, 3, 9, 133}
BENCH_P0 = {16, 51, 66, 129, 131}
BENCH_P1 = {2, 4, 15, 53, 57, 59, 81, 86, 87, 95, 100, 106, 130, 132, 139, 150, 165}
BENCH_INDEX_ROWS = {0, 33, 90, 96, 109, 110, 116, 117, 118, 140, 144, 147}
PAPER_P0 = {4, 6, 9, 10, 14, 15, 17, 19, 20, 24, 26, 34, 35, 41, 44, 51, 53}
PAPER_P1 = {0, 1, 2, 5, 7, 8, 12, 13, 16, 18, 22, 31, 32, 36, 39, 43, 48}

MODEL_GROUPS = {
    0: "catalog-search-jev-27b",
    1: "catalog-search-typed-decisions",
    2: "catalog-search-kev",
    3: "catalog-search-laya",
    9: "catalog-search-hf",
    7: "flymy-decision",
    14: "jevk5",
    15: "jevk5",
    23: "lev",
    77: "flymy-decision",
    78: "flymy-decision",
    79: "eikos",
    87: "jevk5",
    105: "apus-openjev",
    109: "decision-1.0",
    120: "jevk5",
    121: "kev",
    133: "openjev-27b",
    163: "zefancai-open-jev",
    164: "openjev-27b",
    167: "this-that",
    181: "laya",
    187: "kev",
    195: "decider",
}
MODEL_CAUTION = {
    7: "author metrics use public JevBench items; sealed judge tier unavailable",
    14: "public JevBench is report-only; 64-item hard set and development proxy are small; Decision Index train-split overlap disclosed",
    23: "some source data have non-commercial terms; S1Bench is public and self-reported",
    79: "author's JevBench numbers use public items, not the official sealed leaderboard; calibration fit sets were too easy",
    87: "public JevBench only; sibling card discloses MMLU-Pro test-split training",
    105: "catalog points to family bundle; pin separate Apache-2.0 4B/9B/35B BF16 releases; Frozen80 reused development panel and merged probabilities differ from adapter",
    109: "one catalog row groups six separately released Decision 1.0 models",
    120: "public JevBench and index-proxy scores are not independent blind evidence; sealed drop disclosed for earlier version",
    133: "exact duplicate weights of openjev/openjev at this snapshot; count once; non-commercial license",
    163: "reported test/OOD split is synthetic mixture; no full-data baseline; some training rows omitted from redistributable dataset",
    164: "non-commercial license; large 27B text model; same weights as AlquimiaAi/openjev",
    167: "Score/Noul need declared-option projections in our shared protocol; author cohort is small",
    169: "adapter requires pinned base; no separate temperature fit for current checkpoint",
    181: "short-context encoder; representational capacity and admitted-input limits differ from LLM backbones",
    187: "catalog groups Kev sizes; pin exact 4B/0.8B native runtime before comparison",
    195: "catalog URL is decider-2b; decider-4b and 35B are distinct checkpoints; do not merge versions",
}
BENCH_CAUTION = {
    4: "paired dialect-bias probe; fairness slice, not overall accuracy benchmark",
    16: "public candidate-set stress data; possible source overlap; use as diagnostic, not final blind test",
    51: "public mixed gold/silver typed suite; keep reliability strata separate",
    66: "frozen public 1,071-row corpus; pretraining leakage possible; use development/regression only",
    129: "temperature refit is post-hoc diagnostic; model version hidden behind gateway; synthetic rules",
    131: "public questions and sealed aggregate differ; report Intelligence/calibration/speed/cost separately, not composite as accuracy",
}
PAPER_CAUTION = {
    4: "judging study, not a universal model accuracy ordering",
    6: "adaptive context attack uses target feedback; measure fixed-budget and clean accuracy separately",
    9: "prompt-injection setting and target success must not be conflated with ordinary error rate",
    10: "small repeated scientific Choice study; downstream correctness distinct from answer accuracy",
    14: "ContractNLI request-format robustness; repeated agreement can be consistently wrong",
    15: "judge cascade uses chosen threshold; compare fixed-model, fallback, and oracle costs",
    17: "option-name/rubric binding counterfactual; type validity stays high despite semantic failure",
    19: "real human audit, but published recalibration uses same labels; hold out calibration for transport claims",
    20: "strongest human-label transfer evidence; per-task macro and uncertainty matter more than single pooled score",
    24: "external answer-correctness calibrator is a different task from typed question answering",
    26: "author 68-question cohort and public benchmark cannot establish unseen-family superiority",
    34: "temperature theory does not imply cross-domain transfer of one fitted temperature",
    35: "judge shortcut bias is a design warning rather than direct Jev benchmark",
    41: "general calibration survey; report NLL/Brier and group conditional calibration",
    44: "confidence elicitation for RLHF LMs; direct applicability to native decision heads unproven",
    51: "BERT/RoBERTa calibration under shift; not a direct Jev comparison",
    53: "classic temperature scaling; independent calibration partition needed",
}


def main() -> None:
    raw = SOURCE.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit(f"source SHA-256 changed: {actual}")
    source_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    expected = {"model": 200, "benchmark": 169, "paper": 54}
    counts = Counter(row["category"] for row in source_rows)
    if {k: counts[k] for k in expected} != expected:
        raise SystemExit(f"category counts changed: {counts}")

    primary_models = {
        r["url"]: r
        for r in json.loads(
            (HERE / "source" / "model-primary-check-2026-09-26.json").read_text()
        )
        if "revision" in r
    }
    primary_bench = {
        r["url"].lower(): r
        for r in json.loads(
            (HERE / "source" / "benchmark-primary-check-2026-09-26.json").read_text()
        )
        if "revision" in r
    }
    primary_papers = {
        r["url"]: r
        for r in json.loads(
            (HERE / "source" / "paper-primary-check-2026-09-26.json").read_text()
        )
        if r["status"] == 200
    }

    rows = []
    category_index = Counter()
    for item in source_rows:
        category = item["category"]
        if category not in expected:
            continue
        idx = category_index[category]
        category_index[category] += 1
        if category == "model":
            tier = (
                "P0"
                if idx in MODEL_P0
                else (
                    "P1"
                    if idx in MODEL_P1
                    else "P3" if idx in MODEL_INDEX_ROWS else "P2"
                )
            )
            role = (
                "exact_duplicate_do_not_run"
                if idx == 133
                else (
                    "exact_native_comparison_candidate"
                    if tier == "P0"
                    else (
                        "followup_candidate"
                        if tier == "P1"
                        else (
                            "catalog_index_only"
                            if tier == "P3"
                            else "context_or_specialist"
                        )
                    )
                )
            )
            primary = primary_models.get(item["url"], {})
            caution = MODEL_CAUTION.get(
                idx,
                (
                    "catalog claim only; native interface and data overlap need verification"
                    if not primary
                    else "published card is author-reported; no score reused as our result"
                ),
            )
            group = MODEL_GROUPS.get(idx, item["url"].rstrip("/").lower())
        elif category == "benchmark":
            tier = (
                "P0"
                if idx in BENCH_P0
                else (
                    "P1"
                    if idx in BENCH_P1
                    else "P3" if idx in BENCH_INDEX_ROWS else "P2"
                )
            )
            role = (
                "development_suite_or_protocol"
                if tier == "P0"
                else (
                    "targeted_diagnostic_candidate"
                    if tier == "P1"
                    else (
                        "index_or_commentary"
                        if tier == "P3"
                        else "domain_or_unverified_suite"
                    )
                )
            )
            primary = primary_bench.get(item["url"].rstrip("/").lower(), {})
            caution = BENCH_CAUTION.get(
                idx,
                "public benchmark; inspect gold provenance, license, train overlap and harness before use",
            )
            group = (
                "jevbench-fstandhartinger"
                if idx in {33, 131}
                else item["url"].rstrip("/").lower()
            )
        else:
            tier = "P0" if idx in PAPER_P0 else "P1" if idx in PAPER_P1 else "P2"
            role = (
                "direct_method_or_failure_evidence"
                if tier == "P0"
                else (
                    "adjacent_method_or_application"
                    if tier == "P1"
                    else "background_or_low_transfer"
                )
            )
            primary = primary_papers.get(item["url"], {})
            caution = PAPER_CAUTION.get(
                idx,
                "paper result is specific to its protocol; transfer to Decision 2.0 is a hypothesis",
            )
            group = item["url"].rstrip("/").lower()
        record = {
            "category": category,
            "source_index": idx,
            "source_id": item["id"],
            "name": item["name"],
            "url": item["url"],
            "entity_group": group,
            "priority": tier,
            "evaluation_role": role,
            "evidence_status": (
                (
                    "pinned_primary_card"
                    if category == "model"
                    else (
                        "pinned_primary_repo"
                        if category == "benchmark"
                        else "primary_abstract_verified"
                    )
                )
                if primary
                else "catalog_only"
            ),
            "primary_revision": primary.get("revision", ""),
            "catalog_date": item.get("date", ""),
            "catalog_updated": item.get("updated", ""),
            "catalog_license_unverified": item.get("license", ""),
            "catalog_score_claim_unverified": item.get("score", ""),
            "caution": caution,
            "source_snapshot_sha256": EXPECTED_SHA256,
        }
        if category == "paper":
            record["primary_title"] = primary.get("primary_title", "")
            record["primary_abstract_sha256"] = primary.get("abstract_sha256", "")
        if category == "model" and primary:
            record["primary_license"] = primary.get("license", "")
            record["primary_card_sha256"] = primary.get("card_sha256", "")
        if category == "benchmark" and primary:
            record["primary_readme_sha256"] = primary.get("readme_sha256", "")
        if category == "model" and idx == 133:
            record["duplicate_of"] = "https://huggingface.co/openjev/openjev"
            record["duplicate_proof"] = (
                "14/14 config/index/shard blobs identical via HF API blobs=true on 2026-09-26"
            )
        rows.append(record)

    OUTPUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    )
    print(
        json.dumps(
            {
                "source_sha256": actual,
                "records": len(rows),
                "by_category": dict(Counter(r["category"] for r in rows)),
                "by_priority": dict(Counter(r["priority"] for r in rows)),
                "inventory_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
