"""Build live-tier sr-bench manifests from the local MMLU-Pro snapshot.

Outputs three files:
  cases-mcq.jsonl      - labeled cases for the capability-aware mock backend
                         (prompt / truth / expected_contains = answer letter)
  baseline-manifest.json - single target straight at the mock backend
  candidate-manifest.json - single target + mom target (the router)

The capability model: the local lane fails deep-reasoning questions, the
router's candidate therefore moves accuracy by routing them to the frontier
lane. The mom target needs the router's frozen config hash - pass it with
--config-hash (the live-smoke script grabs it from the router's management
surface before writing the final candidate manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from build_mmlu_cases import DEEP_CATEGORIES, load_rows


def letter(index: int) -> str:
    return chr(ord("A") + index)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="build live-tier manifests from MMLU-Pro"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mock-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--router-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument(
        "--router-preview-url",
        default="http://127.0.0.1:8080/api/v1/routing/preview",
        help="router management preview endpoint (required by sr-bench preview)",
    )
    parser.add_argument("--router-model", default="vllm-sr/auto")
    parser.add_argument(
        "--config-hash", required=True, help="router's frozen config hash (64 hex)"
    )
    parser.add_argument("--per-deep", type=int, default=6)
    parser.add_argument("--per-standard", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    data_sub = data_dir / "data"
    rows = load_rows(data_sub if data_sub.is_dir() else data_dir)

    deep = [r for r in rows if str(r.get("category", "")).lower() in DEEP_CATEGORIES]
    standard = [
        r for r in rows if str(r.get("category", "")).lower() not in DEEP_CATEGORIES
    ]
    rng = random.Random(args.seed)
    rng.shuffle(deep)
    rng.shuffle(standard)

    cases: list[dict] = []
    index = 0
    for pool, label, limit in (
        (deep, "cloud_frontier_reasoning", args.per_deep),
        (standard, "local_standard", args.per_standard),
    ):
        taken = 0
        for row in pool:
            if taken >= limit:
                break
            options = row.get("options")
            answer = row.get("answer")
            question = str(row.get("question", "")).strip()
            if not isinstance(options, list) or not isinstance(answer, str):
                continue
            idx = ord(answer.strip().upper()) - ord("A")
            if not question or idx < 0 or idx >= len(options):
                continue
            index += 1
            prompt = f"{question}\n" + "\n".join(
                f"{letter(i)}. {o}" for i, o in enumerate(options) if o
            )
            cases.append(
                {
                    "case_id": f"mmlu-live-{index:03d}",
                    "prompt": prompt,
                    "truth": label,
                    "expected_contains": letter(idx),
                    "answer": letter(idx),
                    "options_text": [str(o) for o in options if o],
                }
            )
            taken += 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_path = out_dir / "cases-mcq.jsonl"
    with cases_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    manifest_cases = [
        {
            "id": c["case_id"],
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": c["prompt"]}],
            "answer": c["answer"],
        }
        for c in cases
    ]
    limits = {"total_timeout_s": 1200, "idle_timeout_s": 300, "max_run_seconds": 3600}
    # capability_only: the smoke validates the evaluation pipeline, not cost
    # accounting - require_priced would demand the full router usage-receipt
    # chain (SKILL: "capability_only permits unpriced evaluation")
    prices = {
        "local/private-qwen": {
            "input": 0,
            "cached_input": 0,
            "cache_write": 0,
            "output": 0,
        },
        "cloud/frontier-reasoning": {
            "input": 0,
            "cached_input": 0,
            "cache_write": 0,
            "output": 0,
        },
        "vllm-sr/auto": {"input": 0, "cached_input": 0, "cache_write": 0, "output": 0},
    }

    single_target = {
        "id": "single",
        "kind": "single",
        # the baseline is "no routing": everything hits the local low-cost
        # lane, so the capability model fails the deep questions and the
        # router candidate has real headroom to show
        "model": "local/private-qwen",
        "base_url": args.mock_url,
        "prices": prices,
    }
    baseline_manifest = {
        "version": "sr-bench-1.0",
        "name": "recipe-lab-live-baseline",
        "cost_policy": "capability_only",
        "targets": [single_target],
        "cases": manifest_cases,
        "limits": limits,
    }
    candidate_manifest = {
        "version": "sr-bench-1.0",
        "name": "recipe-lab-live-candidate",
        "cost_policy": "capability_only",
        "targets": [
            single_target,
            {
                "id": "mom",
                "kind": "mom",
                "model": args.router_model,
                "base_url": args.router_url,
                "config_hash": args.config_hash,
                # the router's benchmark binding only supports a direct-call
                # limit of 1 (pkg/extproc/benchmark_config_binding.go)
                "max_inference_calls": 1,
                "prices": prices,
            },
        ],
        "cases": manifest_cases,
        "limits": limits,
    }
    (out_dir / "baseline-manifest.json").write_text(
        json.dumps(baseline_manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "candidate-manifest.json").write_text(
        json.dumps(candidate_manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # routing-distribution preview takes a mom-only manifest (sr-bench:
    # "preview only supports mom targets") and the mom target must declare
    # the router's management preview_url
    mom_target = dict(candidate_manifest["targets"][1])
    mom_target["preview_url"] = args.router_preview_url
    preview_manifest = {
        "version": "sr-bench-1.0",
        "name": "recipe-lab-live-preview",
        "cost_policy": "capability_only",
        "targets": [mom_target],
        "cases": manifest_cases,
        "limits": limits,
    }
    (out_dir / "preview-manifest.json").write_text(
        json.dumps(preview_manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "cases": len(cases),
                "deep": sum(
                    1 for c in cases if c["truth"] == "cloud_frontier_reasoning"
                ),
                "standard": sum(1 for c in cases if c["truth"] == "local_standard"),
                "cases_hash": f"sha256:{digest}",
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
