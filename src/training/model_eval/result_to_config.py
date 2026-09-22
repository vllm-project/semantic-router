"""Analyze MMLU-Pro results and generate a canonical v0.3 config scaffold.

Two output modes (selectable via --mode):

  --mode full   (default, upstream behavior)
      Generate a complete v0.3 config scaffold: providers, routing.modelCards,
      routing.signals.domains, routing.decisions, global.*. Use this when you
      are bootstrapping a new deployment from evaluation results.

  --mode patch
      Generate only the routing.signals.domains block plus a
      merge_instructions note, as a YAML patch. Use this when you already
      have a config.yaml and want to update only the per-domain model_scores
      ranking + use_reasoning flags based on new evaluation results. Similar
      to bench/reasoning/canonical_patch.py's patch mode.

Optional cost-aware extensions (additive, off by default):
  --cost-lambda LAMBDA           Enable cost-aware scoring: score = acc - λ * norm_cost
  --token-costs-dir DIR          Per-model per-category avg tokens (required for λ>0)
  --nothink-results-dir DIR      Think-vs-nothink gap -> use_reasoning (replaces heuristic)

When --cost-lambda > 0 is given without --token-costs-dir, a warning is
printed and λ is ignored (graceful degradation to pure-accuracy ranking).

How to read the output
----------------------

  - Full mode: the output YAML is a complete config — load it with
    `vllm-sr serve` or merge into your deployment config.
  - Patch mode: the output YAML contains routing.signals.domains and
    merge_instructions. Open it, copy the domains block, and paste it
    under routing.signals in your existing config.yaml.

The terminal output also prints a "Config saved to <path>" line and,
when --cost-lambda > 0, a "Warning" line if --token-costs-dir was missing.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import yaml

try:
    from .constants import MODEL_REGISTRY
except ImportError:
    from constants import MODEL_REGISTRY

DEFAULT_OUTPUT_FILE = "config/config.eval.yaml"


def served_model_path(role):
    """Use the same native classifier identity as the served evaluator."""
    return "models/" + MODEL_REGISTRY[role]["id"].split("/")[-1]


DEFAULT_EMBEDDINGS = {
    "semantic": {
        "mmbert_model_path": "models/Vela-1.0-Encoder-307M-Embedding",
        "use_cpu": True,
        "embedding_config": {
            "model_type": "mmbert",
            "preload_embeddings": True,
            "target_dimension": 768,
            "target_layer": 22,
            "min_score_threshold": 0.5,
        },
    }
}

DEFAULT_RESPONSE_CACHE = {
    "enabled": True,
    "embedding_model": "mmbert",
    "max_entries": 1000,
    "ttl_seconds": 3600,
}

DEFAULT_TOOLS = {
    "enabled": True,
    "top_k": 3,
    "similarity_threshold": 0.2,
    "tools_db_path": "config/runtime/tools/tools_db.json",
    "fallback_to_empty": True,
}

DEFAULT_PROMPT_GUARD = {
    "enabled": True,
    "model_id": served_model_path("jailbreak"),
    "threshold": 0.5,
    "use_cpu": True,
    "variant": "mmbert32k",
    "jailbreak_mapping_path": served_model_path("jailbreak")
    + "/jailbreak_type_mapping.json",
    "positive_labels": ["jailbreak"],
}

DEFAULT_DOMAIN_CLASSIFIER = {
    "model_id": served_model_path("intent"),
    "threshold": 0.5,
    "use_cpu": True,
    "variant": "mmbert32k",
    "category_mapping_path": served_model_path("intent") + "/category_mapping.json",
    "fallback_category": "other",
}

DEFAULT_PII_CLASSIFIER = {
    "model_id": served_model_path("pii"),
    "threshold": 0.9,
    "use_cpu": True,
    "use_mmbert_32k": True,
    "pii_mapping_path": served_model_path("pii") + "/pii_mapping.json",
}

CATEGORY_REASONING = {
    "math": True,
    "physics": True,
    "chemistry": True,
    "computer science": True,
    "engineering": True,
    "biology": True,
    "business": False,
    "law": False,
    "psychology": False,
    "history": False,
    "economics": False,
    "philosophy": False,
    "health": False,
    "other": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MMLU-Pro results and generate a canonical v0.3 config scaffold"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing MMLU-Pro results",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output file for the generated canonical config scaffold",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "patch"),
        default="full",
        help="Output mode: 'full' generates a complete v0.3 config scaffold "
        "(default, upstream behavior). 'patch' generates only the "
        "routing.signals.domains section as a merge patch — useful when you "
        "already have a config and want to update just the routing policy "
        "based on new eval results (similar to bench/reasoning/canonical_patch.py).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Similarity threshold for the generated semantic cache override",
    )
    parser.add_argument(
        "--backend-endpoint",
        type=str,
        default="127.0.0.1:8000",
        help="Endpoint to bind generated providers.models[].backend_refs[] entries to",
    )
    parser.add_argument(
        "--backend-protocol",
        type=str,
        default="http",
        help="Protocol for generated backend_refs entries",
    )
    parser.add_argument(
        "--provider-id",
        type=str,
        default="vllm",
        help="Catalog Provider ID for generated backend_refs entries",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default="openai",
        help="API format for generated providers.models entries",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default="openai",
        help="Provider name used in generated external_model_ids",
    )
    # ---- Cost-aware extensions (optional, additive) ----
    # When --cost-lambda > 0, score = acc - lambda * norm_cost is used
    # instead of pure accuracy for per-category model ranking. Requires
    # --token-costs-dir. When omitted, behavior is identical to upstream.
    parser.add_argument(
        "--cost-lambda",
        type=float,
        default=0.0,
        help="Cost-aware scoring weight: score = acc - lambda * norm_cost. "
        "0.0 = pure accuracy (default, upstream behavior). "
        "When > 0, --token-costs-dir must also be provided "
        "(otherwise a warning is printed and lambda is ignored).",
    )
    parser.add_argument(
        "--token-costs-dir",
        type=str,
        default=None,
        help="Directory with per-model token cost JSON files. "
        "Each file is named {model_name}.json and contains a "
        '{"category": avg_tokens} mapping. '
        "Two ways to produce these files: "
        "(1) Run cost_aware_calib.py --token-costs-dir <dir> "
        '--model "name,path/to/judged.jsonl" [...]; '
        "(2) Write them manually following the format above.",
    )
    parser.add_argument(
        "--nothink-results-dir",
        type=str,
        default=None,
        help="Directory with nothink-form results (same layout as "
        "--results-dir: subdirs with analysis.json). When provided, "
        "use_reasoning is derived from measured think-vs-nothink gap "
        "per category instead of the static CATEGORY_REASONING heuristic.",
    )
    return parser.parse_args()


def collect_model_accuracies(results_dir):
    """Collect all model accuracies by category from result files."""
    category_accuracies = defaultdict(lambda: defaultdict(float))
    analysis_files = glob.glob(
        os.path.join(results_dir, "**/analysis.json"), recursive=True
    )

    for file_path in analysis_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        if "_cot" in dir_name:
            model_name = dir_name.replace("_cot", "")
        else:
            model_name = dir_name.replace("_direct", "")

        model_name = (
            model_name.replace("_", "/", 1) if "_" in model_name else model_name
        )

        with open(file_path, encoding="utf-8") as handle:
            analysis = json.load(handle)

        for category, accuracy in analysis.get("category_accuracy", {}).items():
            category_accuracies[category][model_name] = max(
                category_accuracies[category][model_name],
                float(accuracy),
            )

    return category_accuracies


def calculate_average_accuracies(category_accuracies):
    """Compute average per-model accuracy across categories after variant collapse."""
    averages = defaultdict(list)
    for models in category_accuracies.values():
        for model_name, accuracy in models.items():
            if model_name == "auto":
                continue
            averages[model_name].append(float(accuracy))

    return {
        model_name: sum(scores) / len(scores)
        for model_name, scores in averages.items()
        if scores
    }


def build_provider_models(
    ranked_models,
    backend_endpoint,
    backend_protocol,
    provider_id,
    api_format,
    provider_name,
):
    provider_models = []
    for model_name, _average_accuracy in ranked_models:
        provider_models.append(
            {
                "name": model_name,
                "provider_model_id": model_name,
                "api_format": api_format,
                "external_model_ids": {provider_name: model_name},
                "backend_refs": [
                    {
                        "name": f"{model_name}-backend",
                        "endpoint": backend_endpoint,
                        "protocol": backend_protocol,
                        "provider": provider_id,
                        "weight": 1,
                    }
                ],
            }
        )
    return provider_models


def build_routing_model_cards(ranked_models):
    model_cards = []
    for model_name, _average_accuracy in ranked_models:
        model_cards.append(
            {
                "name": model_name,
                "description": (
                    "Generated from MMLU-Pro evaluation results for category-aware routing."
                ),
                "capabilities": ["chat"],
                "tags": ["generated", "mmlu-pro"],
                "modality": "ar",
            }
        )
    return model_cards


def build_domain_signals(category_accuracies):
    domains = []
    for category_name, models in sorted(category_accuracies.items()):
        ranked_models = sorted(
            (
                (model_name, float(accuracy))
                for model_name, accuracy in models.items()
                if model_name != "auto"
            ),
            key=lambda item: (-item[1], item[0]),
        )
        domains.append(
            {
                "name": category_name,
                "description": (
                    f"MMLU-Pro category generated from evaluation results: {category_name}."
                ),
                "mmlu_categories": [category_name],
                "model_scores": [
                    {
                        "model": model_name,
                        "score": round(accuracy, 6),
                        "use_reasoning": CATEGORY_REASONING.get(
                            category_name.lower(), False
                        ),
                    }
                    for model_name, accuracy in ranked_models
                ],
            }
        )
    return domains


def generate_config_yaml(
    category_accuracies,
    similarity_threshold,
    backend_endpoint,
    backend_protocol,
    provider_id,
    api_format,
    provider_name,
):
    """Generate a canonical v0.3 config scaffold from MMLU-Pro results."""
    average_accuracies = calculate_average_accuracies(category_accuracies)
    if not average_accuracies:
        raise ValueError("No non-auto model results were found in the input directory")

    ranked_models = sorted(
        average_accuracies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    default_model = ranked_models[0][0]

    return {
        "version": "v0.3",
        "listeners": [],
        "providers": {
            "defaults": {
                "model": default_model,
                "reasoning_effort": "medium",
            },
            "models": build_provider_models(
                ranked_models,
                backend_endpoint,
                backend_protocol,
                provider_id,
                api_format,
                provider_name,
            ),
        },
        "routing": {
            "modelCards": build_routing_model_cards(ranked_models),
            "signals": {
                "domains": build_domain_signals(category_accuracies),
            },
            "decisions": [],
        },
        "global": {
            "stores": {
                "response_cache": {
                    **DEFAULT_RESPONSE_CACHE,
                    "similarity_threshold": similarity_threshold,
                }
            },
            "integrations": {
                "tools": DEFAULT_TOOLS,
            },
            "model_catalog": {
                "embeddings": DEFAULT_EMBEDDINGS,
                "modules": {
                    "prompt_guard": DEFAULT_PROMPT_GUARD,
                    "classifier": {
                        "domain": DEFAULT_DOMAIN_CLASSIFIER,
                        "pii": DEFAULT_PII_CLASSIFIER,
                    },
                },
            },
        },
    }


def save_config(config, output_file):
    """Save the config dictionary as a YAML file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)

    print(f"Config saved to {output_file}")


# ============================================================================
# Cost-aware extensions (additive, optional).
# When --cost-lambda > 0 and --token-costs-dir provided, the per-category model
# ranking uses score = acc - lambda * norm_cost instead of pure accuracy.
# When --nothink-results-dir provided, use_reasoning is derived from the
# measured think-vs-nothink accuracy gap per category instead of the static
# CATEGORY_REASONING heuristic. All existing behavior is preserved when these
# optional flags are omitted.
# ============================================================================


def load_token_costs(token_costs_dir):
    """Load per-model token cost JSONs from a directory.

    Returns {model_name: {category: avg_tokens}}. Missing dir or files -> {}.
    Model ids are decoded with the same reversible encoding used by
    cost_aware_calib.write_token_costs, so ``org/model`` round-trips.
    """
    if not token_costs_dir or not os.path.isdir(token_costs_dir):
        return {}
    out = {}
    for fname in os.listdir(token_costs_dir):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        model_name = stem.replace("__slash__", "/")
        with open(os.path.join(token_costs_dir, fname), encoding="utf-8") as f:
            out[model_name] = {k: float(v) for k, v in json.load(f).items()}
    return out


def _norm_cost(token_costs, model_name, category_name):
    """Normalize a model's token cost for a category to [0, 1].

    Uses the same per-category, cross-model normalization as
    cost_aware_calib.py: norm_cost = cost_m(b) / max_over_models(cost(b)).
    This keeps the calibration scan and the generated config in lock-step,
    so the recommended lambda reproduces the measured policy and savings.
    Returns 0.0 when no token cost data is available for the model/category.
    """
    if not token_costs or model_name not in token_costs:
        return 0.0
    if category_name not in token_costs[model_name]:
        return 0.0
    # Collect this category's cost across all models that have data for it,
    # then normalize by the max — matches cost_aware_calib.py scan().
    costs_in_cat = {
        m: float(costs[category_name])
        for m, costs in token_costs.items()
        if category_name in costs
    }
    if model_name not in costs_in_cat or not costs_in_cat:
        return 0.0
    cmax = max(costs_in_cat.values()) or 1.0
    return costs_in_cat[model_name] / max(cmax, 1.0)


def collect_nothink_accuracies(nothink_results_dir):
    """Collect per-category nothink-form accuracies, mirroring
    collect_model_accuracies but against the --nothink-results-dir argument.

    Returns {category: {model_name: accuracy}}. Missing dir -> {}.
    """
    if not nothink_results_dir or not os.path.isdir(nothink_results_dir):
        return {}
    return collect_model_accuracies(nothink_results_dir)


def _measured_use_reasoning(
    category_name,
    think_acc,
    nothink_acc,
    gap_min_pp=0.02,
):
    """Decide use_reasoning for one model from its measured think-vs-nothink gap.

    A model is routed to think mode when think_acc - nothink_acc >= gap_min_pp
    (in accuracy points; default 0.02 = 2pp, i.e., ~2 questions on a
    100-question MMLU-Pro category). Falls back to the static
    CATEGORY_REASONING heuristic when either side is missing.
    """
    if think_acc is None or nothink_acc is None:
        return CATEGORY_REASONING.get(category_name.lower(), False)
    gap = float(think_acc) - float(nothink_acc)
    return gap >= gap_min_pp


def build_cost_aware_domain_signals(
    category_accuracies,
    token_costs,
    cost_lambda,
    nothink_accuracies,
):
    """Like build_domain_signals but uses cost-aware score and measured
    use_reasoning. Falls back to pure-accuracy ranking when token_costs is
    empty, and to CATEGORY_REASONING when nothink_accuracies is empty.
    """
    domains = []
    for category_name, models in sorted(category_accuracies.items()):
        ranked_models = sorted(
            (
                (model_name, float(accuracy))
                for model_name, accuracy in models.items()
                if model_name != "auto"
            ),
            key=lambda item: (-item[1], item[0]),
        )

        nothink_accs_cat = nothink_accuracies.get(category_name, {})

        model_scores = []
        for model_name, accuracy in ranked_models:
            if cost_lambda > 0 and token_costs:
                nc = _norm_cost(token_costs, model_name, category_name)
                score = float(accuracy) - cost_lambda * nc
            else:
                score = float(accuracy)

            if nothink_accuracies:
                # Compare this model's matched think-vs-nothink accuracy,
                # not all models' — so reasoning is only enabled when it
                # helps *this* model (e.g., A think>nothink → true,
                # B think<nothink → false).
                t_acc = float(accuracy)
                n_acc = nothink_accs_cat.get(model_name)
                use_reasoning = _measured_use_reasoning(
                    category_name,
                    t_acc,
                    n_acc,
                )
            else:
                use_reasoning = CATEGORY_REASONING.get(category_name.lower(), False)

            model_scores.append(
                {
                    "model": model_name,
                    "score": round(score, 6),
                    "use_reasoning": use_reasoning,
                }
            )

        # When cost_lambda > 0, re-rank model_scores by the new score
        if cost_lambda > 0 and token_costs:
            model_scores.sort(key=lambda m: (-m["score"], m["model"]))

        domains.append(
            {
                "name": category_name,
                "description": (
                    f"MMLU-Pro category generated from evaluation results: {category_name}."
                ),
                "mmlu_categories": [category_name],
                "model_scores": model_scores,
            }
        )
    return domains


def main():
    args = parse_args()

    print(f"Analyzing MMLU-Pro results in {args.results_dir}...")
    category_accuracies = collect_model_accuracies(args.results_dir)

    # Cost-aware post-processing: compute the (possibly cost-aware) signals
    # block. When --cost-lambda or --nothink-results-dir is set, replace the
    # pure-accuracy ranking with cost-aware ranking + measured use_reasoning.
    use_cost_aware = args.cost_lambda > 0 or args.nothink_results_dir
    token_costs = {}
    nothink_accuracies = {}
    if use_cost_aware:
        token_costs = (
            load_token_costs(args.token_costs_dir) if args.token_costs_dir else {}
        )
        nothink_accuracies = collect_nothink_accuracies(args.nothink_results_dir)

        if args.cost_lambda > 0 and not token_costs:
            # Graceful degradation: no token costs -> λ has no effect,
            # fall back to pure-accuracy ranking (same as λ=0).
            print(
                f"Warning: --cost-lambda={args.cost_lambda} ignored "
                f"(no --token-costs-dir provided); using pure-accuracy ranking."
            )
            args.cost_lambda = 0.0

    if use_cost_aware and (args.cost_lambda > 0 or nothink_accuracies):
        signals_domains = build_cost_aware_domain_signals(
            category_accuracies,
            token_costs,
            args.cost_lambda,
            nothink_accuracies,
        )
    else:
        signals_domains = build_domain_signals(category_accuracies)

    # ---- Mode: patch ----
    # Generate only the routing.signals.domains block as a merge patch.
    # The user merges this into an existing config — similar to
    # bench/reasoning/canonical_patch.py's patch mode.
    if args.mode == "patch":
        patch = {
            "routing": {
                "signals": {
                    "domains": signals_domains,
                },
            },
            "merge_instructions": (
                "Merge the 'routing.signals.domains' block into your "
                "existing config.yaml under routing.signals. This patch "
                "only updates per-domain model_scores ranking + "
                "use_reasoning. All other config sections (providers, "
                "modelCards, decisions, global.*) are left untouched."
            ),
        }
        if args.cost_lambda > 0:
            patch["lambda_used"] = args.cost_lambda
        if nothink_accuracies:
            patch["use_reasoning_source"] = "measured_gap"

        print(f"Saving patch to {args.output_file}...")
        save_config(patch, args.output_file)
        print("Done! Merge the patch into your existing config.yaml.")
        return

    # ---- Mode: full (default, upstream behavior) ----
    print("Generating canonical v0.3 config scaffold...")
    config = generate_config_yaml(
        category_accuracies,
        args.similarity_threshold,
        args.backend_endpoint,
        args.backend_protocol,
        args.provider_id,
        args.api_format,
        args.provider_name,
    )
    # Replace the pure-accuracy signals.domains with the cost-aware version
    # when cost-aware post-processing was requested.
    config["routing"]["signals"]["domains"] = signals_domains

    print(f"Saving config to {args.output_file}...")
    save_config(config, args.output_file)

    print("Done!")


if __name__ == "__main__":
    main()
