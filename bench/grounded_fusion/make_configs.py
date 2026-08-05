"""Generate the two bench router configs (grounding on / off) from config/config.yaml.

Both configs are identical except ``algorithm.fusion.grounding.enabled``. They:
  - add a deterministic ``deliberate_sentinel`` regex keyword rule + a top-priority
    fusion decision keyed to it (the harness prepends the sentinel to every prompt),
  - bind the fusion panel/judge to a local Ollama proxy via provider backend_refs,
  - wire the NLI model (models/mom-halugate-explainer) so PANEL-mode grounding
    actually fires instead of silently falling back to plain fusion.

Usage:
    .venv-bench/bin/python -m bench.grounded_fusion.make_configs \
        --base config/config.yaml --out-dir bench/grounded_fusion
"""

from __future__ import annotations

import argparse
import copy

import yaml

SENTINEL = "deliberate-eval"
PANEL = ["qwen3:8b", "llama3.1:8b", "gemma3:12b"]
JUDGE = "qwen3:14b"
OLLAMA_BACKEND = {
    "base_url": "http://localhost:11435/v1",
    "provider": "openai",
    "chat_path": "/chat/completions",
}


def _ollama_provider_model(name: str) -> dict:
    return {
        "name": name,
        "provider_model_id": name,
        "api_format": "openai",
        "backend_refs": [
            {
                "name": f"ollama-{name.replace(':', '-')}",
                "weight": 100,
                "type": "chat",
                **OLLAMA_BACKEND,
            }
        ],
    }


def _model_card(name: str) -> dict:
    return {
        "name": name,
        "param_size": "8B",
        "context_window_size": 32768,
        "description": f"Local Ollama model {name} for the grounded-fusion benchmark.",
        "capabilities": ["chat", "reasoning"],
        "quality_score": 0.7,
        "modality": "ar",
        "tags": ["bench", "ollama"],
    }


def _sentinel_rule() -> dict:
    return {
        "name": "deliberate_sentinel",
        "operator": "OR",
        "method": "regex",
        "keywords": [SENTINEL],
        "case_sensitive": False,
    }


def _fusion_decision(grounding_on: bool, policy: str = "weight") -> dict:
    return {
        "name": "grounded-fusion-bench",
        "description": "Deterministic fusion route for the grounded-fusion DRACO benchmark.",
        "priority": 100000,
        "rules": {
            "operator": "AND",
            "conditions": [{"type": "keyword", "name": "deliberate_sentinel"}],
        },
        "modelRefs": [
            {"model": m, "use_reasoning": False, "weight": 1.0} for m in PANEL
        ],
        "algorithm": {
            "type": "fusion",
            "fusion": {
                "model": JUDGE,
                "analysis_models": PANEL,
                # Serialize panel calls: loading 3 local models concurrently
                # thrashes Ollama under memory pressure and 502s some panel
                # responses (thinning the panel). One at a time is reliable.
                "max_concurrent": 1,
                "max_completion_tokens": 1024,
                "temperature": 0.0,
                "include_analysis": True,
                "include_intermediate_responses": True,
                "on_error": "skip",
                "judge_prompt_version": "fusion-v1",
                "grounding": {
                    "enabled": grounding_on,
                    "reference": "panel",  # DRACO ships no context -> cross-model NLI
                    # policy controls how the score is used. weight (default, no
                    # drop) is the production default; filter (hard-drop below
                    # min_score) is known to hurt on contested factual items and is
                    # kept only so later CRs can A/B annotate vs weight vs filter.
                    # min_score/min_keep apply to the filter policy only.
                    "policy": policy,
                    "min_score": 0.34,
                    "min_keep": 1,
                    "nli_contradiction_penalty": 1.0,
                    "on_error": "fail" if grounding_on else "skip",
                },
            },
        },
    }


def build(base: dict, grounding_on: bool, policy: str = "weight") -> dict:
    c = copy.deepcopy(base)

    # provider models + matching modelCards for the local Ollama panel/judge
    # (every providers.models entry must have a routing.modelCards entry)
    existing = {m["name"] for m in c["providers"]["models"]}
    cards = {m["name"] for m in c["routing"]["modelCards"]}
    for name in [*PANEL, JUDGE]:
        if name not in existing:
            c["providers"]["models"].append(_ollama_provider_model(name))
        if name not in cards:
            c["routing"]["modelCards"].append(_model_card(name))

    # Reduce routing to ONLY what the benchmark needs: a single regex keyword
    # signal + a single fusion decision keyed to it. This avoids building the
    # embedding-backed domain/complexity classifiers (which need a working
    # embedding model and otherwise fatal at startup) -- the benchmark routes
    # purely on the sentinel the harness prepends to every prompt.
    c["routing"]["signals"] = {"keywords": [_sentinel_rule()]}
    c["routing"].pop("projections", None)
    c["routing"]["decisions"] = [_fusion_decision(grounding_on, policy)]

    # disable external-dependency stores (milvus/redis/llama_stack) for a
    # self-contained local run -- otherwise startup fatals on missing services.
    for store in ("semantic_cache", "memory", "vector_store"):
        if store in c["global"].get("stores", {}):
            c["global"]["stores"][store]["enabled"] = False

    # Ensure hallucination_mitigation is enabled so the detector + NLI model init
    # (initializeHallucinationDetector -> wireFusionGroundingBackends). The NLI
    # model for PANEL-mode grounding comes from the `explainer` block
    # (models/mom-halugate-explainer), which the reference config already sets.
    hm = c["global"]["model_catalog"]["modules"]["hallucination_mitigation"]
    hm["enabled"] = True
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config/config.yaml")
    ap.add_argument("--out-dir", default="bench/grounded_fusion")
    ap.add_argument(
        "--policy",
        default="weight",
        choices=["weight", "annotate", "filter"],
        help="grounding policy for the 'on' arm: weight/annotate keep all panel "
        "responses; filter hard-drops below min_score (known to hurt). Use this to "
        "A/B the policies in follow-up experiments.",
    )
    args = ap.parse_args()
    with open(args.base) as f:
        base = yaml.safe_load(f)
    for on in (True, False):
        cfg = build(base, on, args.policy)
        path = f"{args.out_dir}/config-fusion-{'on' if on else 'off'}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, width=120)
        print(f"wrote {path} (grounding={'on' if on else 'off'})")


if __name__ == "__main__":
    main()
