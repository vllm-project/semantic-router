"""Generate the two bench router configs (grounding on / off) from config/config.yaml.

Both configs are identical except ``algorithm.fusion.grounding.enabled``. They:
  - replace the recipes with one: a deterministic ``deliberate_sentinel`` regex
    keyword rule and a top-priority fusion decision keyed to it (the harness
    prepends the sentinel to every prompt), served under ``vllm-sr/fusion``,
  - add the local Ollama panel/judge as provider models, with their model facts
    and an operator rating.
Grounding asks the decision model's hallucination preset about each answer,
against the context, or against every peer answer in ``panel`` mode; the Router
starts that model for a fusion decision that grounds.

Usage:
    .venv-bench/bin/python -m bench.grounded_fusion.make_configs \
        --base config/config.yaml --out-dir bench/grounded_fusion
"""

from __future__ import annotations

import argparse
import copy

import yaml

SENTINEL = "deliberate-eval"
FUSION_MODEL = "vllm-sr/fusion"
PANEL = ["qwen3:8b", "llama3.1:8b", "gemma3:12b"]
JUDGE = "qwen3:14b"
OLLAMA_BACKEND = {
    "endpoint": "http://localhost:11435/v1",
    "provider": "ollama",
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
                **OLLAMA_BACKEND,
            }
        ],
        "param_size": "8B",
        "context_window_size": 32768,
        "description": f"Local Ollama model {name} for the grounded-fusion benchmark.",
        "capabilities": ["chat", "reasoning"],
        "tags": ["bench", "ollama"],
    }


def _rating(name: str) -> dict:
    return {
        "model": name,
        "benchmark": "vllm-sr/operator-rating@1.0.0",
        "metrics": {"score": 0.7},
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
            "on_error": "skip",
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
                "judge_prompt_version": "fusion-v1",
                "grounding": {
                    "enabled": grounding_on,
                    "reference": "panel",  # DRACO ships no context -> answers read against peers
                    # policy controls how the score is used. weight (default, no
                    # drop) is the production default; filter (hard-drop below
                    # min_score) is known to hurt on contested factual items and is
                    # kept only so later CRs can A/B annotate vs weight vs filter.
                    # min_score/min_keep apply to the filter policy only.
                    "policy": policy,
                    "min_score": 0.34,
                    "min_keep": 1,
                    "contradiction_penalty": 1.0,
                    "on_error": "fail" if grounding_on else "skip",
                },
            },
        },
    }


def _bench_recipe(grounding_on: bool, policy: str) -> dict:
    # Only what the benchmark needs: the sentinel keyword rule and the fusion
    # decision keyed to it.
    return {
        "name": "default",
        "signals": {"keywords": [_sentinel_rule()]},
        "decisions": [_fusion_decision(grounding_on, policy)],
    }


def build(base: dict, grounding_on: bool, policy: str = "weight") -> dict:
    c = copy.deepcopy(base)

    existing = {m["name"] for m in c["providers"]["models"]}
    rated = {r["model"] for r in c.setdefault("evaluation", {}).get("records", [])}
    for name in [*PANEL, JUDGE]:
        if name not in existing:
            c["providers"]["models"].append(_ollama_provider_model(name))
        if name not in rated:
            c["evaluation"].setdefault("records", []).append(_rating(name))

    c["recipes"] = [_bench_recipe(grounding_on, policy)]
    c["entrypoints"] = [
        {"model_names": [FUSION_MODEL], "recipe": "default", "default": True}
    ]
    # A listener's vSR rules name rules of the recipes replaced above.
    for listener in c.get("listeners", []):
        systemone = (listener.get("apis") or {}).get("systemone") or {}
        (systemone.get("vsr") or {}).pop("rules", None)

    # Drop the request plugins (the context plugin recalls from the memory
    # store) and the external-dependency stores (llama_stack, redis) for a
    # self-contained local run; startup otherwise fails on missing services.
    c["global"]["plugins"] = []
    stores = c["global"].get("stores", {})
    if "vector_store" in stores:
        stores["vector_store"]["enabled"] = False
    stores.pop("tool_sessions", None)

    # No authenticator fronts the local run, and the bench recipe asks no
    # delegated deployment; one that nothing asks fails to load.
    deployments = c["global"].get("model_catalog", {}).get("deployments", {})
    for name, deployment in list(deployments.items()):
        if (deployment.get("auth") or {}).get("credential") == "delegated":
            del deployments[name]
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
