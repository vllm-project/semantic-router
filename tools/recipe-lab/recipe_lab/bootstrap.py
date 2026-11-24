"""Cold-start C4+C5: candidate configs from the C3 profile.

C4 (scaffold): for every ``--lambdas`` value the real
``result_to_config.py`` runs as a subprocess over the calib split only
(``profile/calib/results`` + ``profile/calib/token_costs``), producing
``scaffold-lambda-<lambda>.yaml``. The script is invoked, never copied -
provenance records its path, content sha256, git blob hash and the full
command line (same pattern as C3). §3.2-5: with lambda > 0 the script
silently degrades to pure accuracy when token costs are missing and only
prints ``Warning: --cost-lambda ...``; this module rejects that warning
with exit 2 so a lambda scan can never silently become a no-op. The
profile's every calib label must appear in the scaffold's
``routing.signals.domains`` (the labels drive the decisions).

C5 (completion): the scaffold is deliberately incomplete (§3.2-2/3: empty
decisions, one shared backend endpoint, cache and tools enabled) and is
turned into a runnable canonical config:

- listeners: contract default port 8899; entrypoints mirror the C2 seed
  (``vllm-sr/auto`` -> the default recipe).
- providers.models: every alias's ``backend_refs`` points at the spec's
  own direct endpoint (the scaffold shares one), and ``pricing`` comes
  from the contract's assumed pricing when declared (never a fake zero).
- decisions: labels sharing the same ``model_scores`` first place merge
  into one ``route-<alias>`` decision whose ``type: domain`` conditions
  are OR-joined; ``modelRefs`` list every model in that domain's score
  order (the static selector picks by the matched domain's scores,
  pkg/selection/static.go, so only exact ties fall back to ref order);
  priorities follow calib traffic share, deterministically; a catch-all
  decision sits below every domain decision. Every decision mounts the
  ``router_replay`` plugin so replay-tier evaluation (``run``/``verify``)
  can observe the routed model - operators may drop it in production.
- providers.defaults.model: the best mean over the lambda-adjusted
  domain scores (the scaffold keeps the pure-accuracy default).
- global: the scaffold's sparse global merged with the C2 seed model
  catalog (deployments + system decision-model pinning, #5620); the
  response cache and tools are DISABLED - a cache would poison replay
  and live evidence with repeat hits, and the reasons ship in the
  sidecar for the bundle write-up.
- classification (Y2 same-source): the candidate's classification sections
  (modules/embeddings) come from the shared classification source config -
  the same sections the observe seed was built with - never from the
  scaffold's own copies, and the scaffold can never override the seed's
  deployments/system. The sidecar records the source's hash and the
  candidate's classification projection hash (bundle verifies it).
- label agreement (Y2): after every candidate is emitted, the dev split is
  observed once through the seed config and once through each candidate;
  a candidate whose dev labels disagree with the seed's in more than 1% of
  the cases fails the command (the profile's grouping premise does not
  hold for that candidate).
- validation: each completed candidate must pass
  ``<validate-bin> config validate --offline --config`` (the
  ``run --validate`` path) before it is emitted; a rejected candidate
  fails the command.

Anti-leak: only calib outputs feed bootstrap; ``dev`` belongs to the
answer bank (C6) and ``frozen_test`` is quarantined.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .classification import classification_projection_hash
from .contract import (
    ContractError,
    ExperimentSpec,
    dataset_content_hash,
    lambda_tag,
)
from .observe_labels import (
    REPLAY_PLUGIN,
    ROUTER_MODEL,
    build_seed_config,
    observe_split,
    write_seed_config,
)
from .profile import script_provenance
from .runner import RunnerEnv, RunnerError, RunnerTimeoutError

BOOTSTRAP_SCHEMA = "recipe-lab-bootstrap-v1"
RESULT_TO_CONFIG_SCRIPT = "result_to_config.py"
FALLBACK_DECISION_NAME = "route-fallback"
DEFAULT_LISTENER_PORT = 8899
DEFAULT_LAMBDAS = "0,0.1,0.2,0.3"
WARNING_MARKER = "Warning: --cost-lambda"
# Y2: a candidate whose dev labels disagree with the observe seed's on more
# than this fraction of the dev split is rejected (exit 2).
LABEL_AGREEMENT_MIN = 0.99
# The mismatched case ids listed in the rejection message (the full list
# ships in the summary).
MISMATCH_PREVIEW = 20

CACHE_DISABLED_REASON = (
    "disabled for evaluation: a warm cache would let repeated cases hit the "
    "cache instead of the routed model and poison replay/live evidence. "
    "Re-enable for production if wanted."
)
TOOLS_DISABLED_REASON = (
    "disabled for evaluation: tool integrations are not part of the MCQ "
    "cold-start contract. Re-enable for production if wanted."
)


def parse_lambdas(raw: str) -> list[float]:
    """Parse the comma-separated lambda list (deduped, order kept)."""
    values: list[float] = []
    seen: set[str] = set()
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        try:
            lam = float(chunk)
        except ValueError as exc:
            raise ContractError(f"--lambdas entry {chunk!r} is not a number") from exc
        if lam < 0:
            raise ContractError(f"--lambdas entry {chunk!r} must be >= 0")
        tag = lambda_tag(lam)
        if tag not in seen:
            seen.add(tag)
            values.append(lam)
    if not values:
        raise ContractError("--lambdas is empty")
    return values


def load_profile(profile_path: str | Path) -> dict[str, Any]:
    profile_path = Path(profile_path)
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read profile: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"{profile_path} is not valid JSON: {exc}") from exc
    if (
        not isinstance(profile, dict)
        or profile.get("schema") != "recipe-lab-profile-v1"
    ):
        raise ContractError(
            f"{profile_path} is not a recipe-lab-profile-v1 document: "
            "bootstrap consumes the C3 profile directory"
        )
    if "calib" not in (profile.get("splits") or []):
        raise ContractError(
            "profile has no calib split: bootstrap consumes the calib split "
            "only (anti-leak); rerun profile with --splits calib"
        )
    return profile


def calib_label_counts(profile: dict[str, Any]) -> dict[str, int]:
    """calib label -> case count (union over models; all models share tasks)."""
    stats = (profile.get("model_split_stats") or {}).get("calib") or {}
    counts: dict[str, int] = {}
    for alias, block in stats.items():
        labels = (block or {}).get("labels") or {}
        if not isinstance(labels, dict):
            raise ContractError(
                f"profile calib stats for {alias!r} have no labels block"
            )
        for label, entry in labels.items():
            n = entry.get("n") if isinstance(entry, dict) else None
            if not isinstance(n, int) or n < 0:
                raise ContractError(
                    f"profile calib stats for {alias!r} label {label!r} has no case count"
                )
            counts[label] = max(counts.get(label, 0), n)
    if not counts:
        raise ContractError("profile has no calib labels: nothing to bootstrap from")
    return counts


def run_result_to_config(
    *,
    script: Path,
    results_dir: Path,
    token_costs_dir: Path,
    output_file: Path,
    lam: float,
    python_bin: str,
    timeout_s: float,
) -> list[str]:
    """C4: run the real result_to_config.py for one lambda (§3.2-5 guard)."""
    argv = [
        str(python_bin),
        str(script),
        "--results-dir",
        str(results_dir),
        "--mode",
        "full",
        "--cost-lambda",
        repr(lam),
        "--token-costs-dir",
        str(token_costs_dir),
        "--output-file",
        str(output_file),
    ]
    try:
        proc = subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContractError(
            f"result_to_config for lambda {lam} exceeded the trial budget "
            f"({timeout_s:.0f}s): {exc}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-4000:]
        raise ContractError(
            f"result_to_config for lambda {lam} failed (exit {proc.returncode}):\n{tail}"
        )
    # §3.2-5: missing token costs degrade lambda to 0 with only a Warning.
    if WARNING_MARKER in (proc.stdout or ""):
        raise ContractError(
            f"result_to_config ignored --cost-lambda {lam} (token costs missing "
            f"or empty under {token_costs_dir}): rerun profile so calib/token_costs "
            "exists, or drop the lambda from --lambdas"
        )
    return argv


def _scaffold_domains(scaffold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # The current result_to_config.py emits the canonical layout
    # (recipes[].signals.domains); the donor-era script and the handcrafted
    # fixtures emit routing.signals.domains. Read either.
    domains = [
        domain
        for recipe in scaffold.get("recipes") or []
        for domain in ((recipe.get("signals") or {}).get("domains") or [])
    ]
    if not domains:
        domains = ((scaffold.get("routing") or {}).get("signals") or {}).get(
            "domains"
        ) or []
    return {domain.get("name"): domain for domain in domains}


def verify_scaffold(
    scaffold: dict[str, Any],
    *,
    spec: ExperimentSpec,
    labels: set[str],
) -> None:
    """The scaffold must cover every profile label and every spec alias."""
    domains = _scaffold_domains(scaffold)
    missing = sorted(labels - set(domains))
    if missing:
        raise ContractError(
            "scaffold is missing domain signals for profile labels: "
            + ", ".join(missing)
        )
    for name, domain in domains.items():
        scores = domain.get("model_scores") or []
        if not scores:
            raise ContractError(f"scaffold domain {name!r} has no model_scores")
    scaffold_aliases = {
        entry.get("name")
        for entry in (scaffold.get("providers") or {}).get("models") or []
    }
    spec_aliases = {model.name for model in spec.models}
    unmatched = sorted(spec_aliases - scaffold_aliases)
    if unmatched:
        raise ContractError(
            "scaffold is missing evaluated models (no results?): "
            + ", ".join(unmatched)
        )
    unknown = sorted(scaffold_aliases - spec_aliases)
    if unknown:
        raise ContractError(
            "scaffold declares models outside the contract: " + ", ".join(unknown)
        )


def _endpoint_parts(url: str, alias: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if not parsed.netloc or parsed.scheme not in ("http", "https"):
        raise ContractError(
            f"model {alias!r} endpoint {url!r} has no http(s) host:port"
        )
    return parsed.netloc, parsed.scheme


def _provider_models(
    scaffold: dict[str, Any], spec: ExperimentSpec
) -> list[dict[str, Any]]:
    """Rewrite each alias's backend_refs to the contract endpoint (+ pricing)."""
    spec_models = {model.name: model for model in spec.models}
    assumed = (spec.cost_basis.assumed_pricing or {}) if spec.cost_basis else {}
    out: list[dict[str, Any]] = []
    for entry in (scaffold.get("providers") or {}).get("models") or []:
        alias = entry["name"]
        model = spec_models.get(alias)
        if model is None:
            raise ContractError(
                f"scaffold declares model {alias!r} outside the contract's models section"
            )
        endpoint, protocol = _endpoint_parts(model.endpoint, alias)
        rewritten = dict(entry)
        rewritten["backend_refs"] = [
            {
                "name": f"{alias}-backend",
                "endpoint": endpoint,
                "protocol": protocol,
                "provider": "vllm",
                "weight": 1,
            }
        ]
        pricing = assumed.get(alias)
        if pricing:
            rewritten["pricing"] = {
                "prompt_per_1m": pricing["prompt_per_1m"],
                "completion_per_1m": pricing["completion_per_1m"],
            }
        out.append(rewritten)
    return out


def _lambda_default_model(domains: dict[str, dict[str, Any]]) -> str:
    """Best mean over the lambda-adjusted domain scores (C5 step 4)."""
    totals: dict[str, list[float]] = {}
    for domain in domains.values():
        for score in domain["model_scores"]:
            totals.setdefault(score["model"], []).append(float(score["score"]))
    averaged = {
        model: sum(scores) / len(scores) for model, scores in totals.items() if scores
    }
    return min(averaged.items(), key=lambda item: (-item[1], item[0]))[0]


def _route_decision_name(alias: str, taken: set[str]) -> str:
    name = f"route-{alias.replace('/', '-')}"
    if name in taken:
        raise ContractError(
            f"decision name collision for model {alias!r}: {name!r} is taken"
        )
    return name


def complete_candidate(
    *,
    scaffold: dict[str, Any],
    spec: ExperimentSpec,
    label_counts: dict[str, int],
    lam: float,
    classification: dict[str, Any],
    listener_port: int = DEFAULT_LISTENER_PORT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """C5: complete the scaffold into a runnable canonical config.

    Returns ``(config, selection)`` - the config to emit and the selection
    table for the sidecar (default model, per-label pick, decision layout).

    ``classification`` (Y2) is the shared classification source sections
    (``load_classification_source``): they replace the scaffold's own
    modules/embeddings copies so the candidate classifies exactly like the
    observe seed did, and the scaffold can never override the seed's
    deployments/system pinning.
    """
    domains = _scaffold_domains(scaffold)
    # result_to_config scores cost-aware candidates as acc - λ*norm_cost,
    # which goes negative for large lambdas; the router only accepts domain
    # scores in [0, 1]. Clamp the emitted values in place - the score ORDER
    # (what the static selector consumes) is untouched.
    for domain in domains.values():
        for score in domain["model_scores"]:
            try:
                score["score"] = min(1.0, max(0.0, float(score.get("score", 0.0))))
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"scaffold domain {domain.get('name')!r} has a non-numeric "
                    f"model score: {score.get('score')!r}"
                ) from exc
    default_model = _lambda_default_model(domains)

    # Group labels by their cost-aware first place; order groups (and
    # priorities) by calib traffic share, deterministically.
    groups: dict[str, list[str]] = {}
    for label, domain in domains.items():
        top = domain["model_scores"][0]["model"]
        groups.setdefault(top, []).append(label)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -sum(label_counts.get(label, 0) for label in item[1]),
            item[0],
        ),
    )

    decisions: list[dict[str, Any]] = []
    decision_table: list[dict[str, Any]] = []
    taken_names: set[str] = set()
    for rank, (alias, labels) in enumerate(ordered):
        # modelRefs follow the alphabetically-first label's domain order;
        # with several labels the matched domain's own scores decide at
        # runtime (static.go), so this only fixes the tie order.
        first_domain = domains[min(labels)]
        model_refs = [
            {
                "model": score["model"],
                "use_reasoning": bool(score.get("use_reasoning", False)),
            }
            for score in first_domain["model_scores"]
        ]
        name = _route_decision_name(alias, taken_names)
        taken_names.add(name)
        decisions.append(
            {
                "name": name,
                "description": (
                    f"Bootstrap (lambda={lambda_tag(lam)}): labels {', '.join(labels)} "
                    f"prefer {alias}."
                ),
                "priority": len(ordered) - rank,
                "tier": 1,
                "rules": {
                    "operator": "OR",
                    "conditions": [
                        {"type": "domain", "name": label} for label in labels
                    ],
                },
                "modelRefs": model_refs,
                "plugins": [dict(REPLAY_PLUGIN)],
            }
        )
        decision_table.append(
            {
                "name": name,
                "labels": labels,
                "priority": len(ordered) - rank,
                "model_refs": [ref["model"] for ref in model_refs],
            }
        )
    # The catch-all sits below every domain decision.
    decisions.append(
        {
            "name": FALLBACK_DECISION_NAME,
            "description": "Catch-all: no bootstrapped domain decision matched.",
            "priority": 0,
            "tier": 2,
            "modelRefs": [{"model": default_model, "use_reasoning": False}],
            "plugins": [dict(REPLAY_PLUGIN)],
        }
    )

    seed_catalog = build_seed_config(classification=classification)["global"][
        "model_catalog"
    ]
    scaffold_global = copy.deepcopy(scaffold.get("global") or {})
    scaffold_catalog = scaffold_global.get("model_catalog") or {}
    # Y2 same-source: the classification sections come from the shared source
    # (via the seed catalog), never from the scaffold's copies; deployments
    # and system stay the seed's own pinning.
    scaffold_catalog.pop("modules", None)
    scaffold_catalog.pop("embeddings", None)
    scaffold_catalog.pop("deployments", None)
    scaffold_catalog.pop("system", None)
    model_catalog = {**seed_catalog, **scaffold_catalog}
    config: dict[str, Any] = {
        # The emitted candidate keeps the legacy routing form the loader
        # accepts (the landed observe_labels seed uses the same shape); the
        # version tag must name that form, not the scaffold's own.
        "version": "v0.3",
        "listeners": [
            {
                "name": "http-gateway",
                "address": "0.0.0.0",
                "port": listener_port,
                "timeout": "300s",
            }
        ],
        "providers": {
            "defaults": {
                **((scaffold.get("providers") or {}).get("defaults") or {}),
                "model": default_model,
            },
            "models": _provider_models(scaffold, spec),
        },
        "routing": {
            "modelCards": (scaffold.get("routing") or {}).get("modelCards") or [],
            "signals": {"domains": list(domains.values())},
            "decisions": decisions,
        },
        "entrypoints": [{"model_names": [ROUTER_MODEL], "recipe": "default"}],
        "global": {
            "stores": {"response_cache": {"enabled": False, "backend_type": "memory"}},
            "integrations": {"tools": {"enabled": False}},
            "model_catalog": model_catalog,
        },
    }
    selection = {
        "default_model": default_model,
        "decisions": decision_table,
        "disabled": {
            "response_cache": CACHE_DISABLED_REASON,
            "tools": TOOLS_DISABLED_REASON,
        },
    }
    return config, selection


def validate_config(path: str | Path, validate_bin: str) -> str | None:
    """None when the config validates, else the rejection reason.

    --offline pins the gate to the CLI's structural rules: the default
    Router verdict depends on whichever image happens to sit on the
    machine, and a stale one rejects canonical configs.
    """
    try:
        proc = subprocess.run(
            [validate_bin, "config", "validate", "--offline", "--config", str(path)],
            capture_output=True,
            text=True,
            timeout=300.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ContractError(f"validator binary not found: {validate_bin}") from exc
    except subprocess.TimeoutExpired:
        return "config validate timed out"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        return f"config validate exit {proc.returncode}: {tail}"
    return None


def _candidate_yaml(config: dict[str, Any]) -> str:
    """Deterministic YAML text (same dict => same bytes)."""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _label_table(
    profile: dict[str, Any],
    *,
    label_counts: dict[str, int],
    domains: dict[str, dict[str, Any]],
    spec: ExperimentSpec,
) -> dict[str, dict[str, Any]]:
    """Per-label pick with the profile's accuracy/cost for the picked model."""
    stats = (profile.get("model_split_stats") or {}).get("calib") or {}
    total = sum(label_counts.values()) or 1
    priced = bool((spec.cost_basis.assumed_pricing or {}) if spec.cost_basis else False)
    table: dict[str, dict[str, Any]] = {}
    for label in sorted(domains):
        top = domains[label]["model_scores"][0]["model"]
        entry: dict[str, Any] = {
            "n": label_counts.get(label, 0),
            "traffic_share": round(label_counts.get(label, 0) / total, 6),
            "selected_model": top,
        }
        block = ((stats.get(top) or {}).get("labels") or {}).get(label) or {}
        for key in ("accuracy", "accuracy_ci95", "cost_per_question"):
            if block.get(key) is not None:
                entry[key] = block[key]
        if priced and "cost_per_question" in entry:
            entry["cost_basis"] = "assumed_pricing"
        table[label] = entry
    return table


def _observe_labels(
    env: RunnerEnv,
    config: Path,
    cases: list[dict[str, Any]],
    request_timeout: float,
    trial_timeout_s: float,
) -> list[dict[str, Any]]:
    """One observe_split call with the bootstrap failure contract."""
    try:
        return observe_split(
            env=env,
            config=config,
            cases=cases,
            request_timeout=request_timeout,
            trial_timeout_s=trial_timeout_s,
        )["rows"]
    except RunnerTimeoutError as exc:
        raise ContractError(
            f"label-agreement observation exceeded the trial budget: {exc}"
        ) from exc
    except RunnerError as exc:
        raise ContractError(f"label-agreement observation failed: {exc}") from exc


def verify_label_agreement(
    *,
    env: RunnerEnv,
    seed_config: str | Path,
    candidates: list[tuple[str, str | Path]],
    cases: list[dict[str, Any]],
    request_timeout: float,
    trial_timeout_s: float,
) -> dict[str, dict[str, Any]]:
    """Y2: every candidate's dev labels must match the observe seed's.

    The seed config and each candidate are observed on the dev split (one
    stack lifecycle each, isolated work dirs so replay records cannot bleed
    across runs); the per-case labels must agree on at least
    ``LABEL_AGREEMENT_MIN`` of the split, else ContractError (exit 2) naming
    the mismatched cases. Returns the per-candidate agreement summary.
    """
    seed_rows = _observe_labels(
        dataclasses.replace(env, work_dir=env.work_dir / "seed"),
        Path(seed_config),
        cases,
        request_timeout,
        trial_timeout_s,
    )
    seed_labels = {row["case_id"]: row["router_label"] for row in seed_rows}
    results: dict[str, dict[str, Any]] = {}
    for tag, path in candidates:
        rows = _observe_labels(
            dataclasses.replace(env, work_dir=env.work_dir / f"lambda-{tag}"),
            Path(path),
            cases,
            request_timeout,
            trial_timeout_s,
        )
        labels = {row["case_id"]: row["router_label"] for row in rows}
        mismatched = sorted(
            case_id
            for case_id in set(seed_labels) | set(labels)
            if seed_labels.get(case_id) != labels.get(case_id)
        )
        agreement = (
            round((len(cases) - len(mismatched)) / len(cases), 6) if cases else 1.0
        )
        results[tag] = {
            "cases": len(cases),
            "mismatched_count": len(mismatched),
            "mismatched": mismatched,
            "agreement": agreement,
        }
        if agreement < LABEL_AGREEMENT_MIN:
            preview = ", ".join(mismatched[:MISMATCH_PREVIEW])
            if len(mismatched) > MISMATCH_PREVIEW:
                preview += ", …"
            raise ContractError(
                f"candidate lambda {tag} disagrees with the observe seed on "
                f"{len(mismatched)}/{len(cases)} dev labels (agreement "
                f"{agreement} < {LABEL_AGREEMENT_MIN}): {preview}"
            )
    return results


def run_bootstrap(
    *,
    spec: ExperimentSpec,
    profile_dir: str | Path,
    model_eval_dir: str | Path,
    lambdas: str,
    out_dir: str | Path,
    python_bin: str,
    validate_bin: str,
    classification: dict[str, Any],
    classification_path: str | Path,
    dev_cases: list[dict[str, Any]],
    env: RunnerEnv,
    request_timeout: float = 180.0,
    listener_port: int = DEFAULT_LISTENER_PORT,
) -> dict[str, Any]:
    """Run the C4+C5 chain for every lambda, then the Y2 label check.

    Returns the summary the CLI prints: candidate paths, lambdas, the
    default models, the per-candidate decision layout and the per-candidate
    dev label agreement against the observe seed.
    """
    profile_path = Path(profile_dir) / "profile.json"
    profile = load_profile(profile_path)
    label_counts = calib_label_counts(profile)
    if not spec.models:
        raise ContractError(
            "spec declares no models section: bootstrap needs the candidate "
            "models and their direct endpoints"
        )
    script = Path(model_eval_dir) / RESULT_TO_CONFIG_SCRIPT
    if not script.is_file():
        raise ContractError(f"model_eval script not found: {script}")
    results_dir = Path(profile_dir) / "calib" / "results"
    token_costs_dir = Path(profile_dir) / "calib" / "token_costs"
    if not results_dir.is_dir() or not token_costs_dir.is_dir():
        raise ContractError(
            f"profile is missing calib/results or calib/token_costs under {profile_dir}: "
            "rerun profile with the calib split"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timeout_s = spec.budget.max_minutes_per_trial * 60
    lam_values = parse_lambdas(lambdas)
    script_prov = script_provenance(script)
    classification_sha = dataset_content_hash(classification_path)

    prepared: list[dict[str, Any]] = []
    for lam in lam_values:
        tag = lambda_tag(lam)
        scaffold_path = out / f"scaffold-lambda-{tag}.yaml"
        command = run_result_to_config(
            script=script,
            results_dir=results_dir,
            token_costs_dir=token_costs_dir,
            output_file=scaffold_path,
            lam=lam,
            python_bin=python_bin,
            timeout_s=timeout_s,
        )
        scaffold = yaml.safe_load(scaffold_path.read_text(encoding="utf-8"))
        if not isinstance(scaffold, dict):
            raise ContractError(f"{scaffold_path} is not a YAML mapping")
        verify_scaffold(scaffold, spec=spec, labels=set(label_counts))

        config, selection = complete_candidate(
            scaffold=scaffold,
            spec=spec,
            label_counts=label_counts,
            lam=lam,
            classification=classification,
            listener_port=listener_port,
        )
        # Validate before emitting: a rejected candidate never lands in out/.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".yaml", dir=out, delete=False
        ) as handle:
            handle.write(_candidate_yaml(config))
            staged = Path(handle.name)
        try:
            reason = validate_config(staged, validate_bin)
            if reason is not None:
                raise ContractError(
                    f"candidate for lambda {tag} was rejected by config validate: {reason}"
                )
            candidate_path = out / f"candidate-lambda-{tag}.yaml"
            os.replace(staged, candidate_path)
        finally:
            if staged.exists():
                staged.unlink()

        prepared.append(
            {
                "lambda": lam,
                "tag": tag,
                "scaffold": str(scaffold_path),
                "candidate": str(candidate_path),
                "command": command,
                "selection": selection,
                "scaffold_obj": scaffold,
                "classification_projection": classification_projection_hash(config),
            }
        )

    # Y2 ②: the profile grouped calib by the seed config's labels; every
    # candidate must reproduce them on dev before it can be shipped.
    seed_config = build_seed_config(
        backend_endpoint=f"127.0.0.1:{env.backend_port}",
        classification=classification,
    )
    seed_path = write_seed_config(seed_config, out / "seed-config.yaml")
    agreement = verify_label_agreement(
        env=env,
        seed_config=seed_path,
        candidates=[(item["tag"], item["candidate"]) for item in prepared],
        cases=dev_cases,
        request_timeout=request_timeout,
        trial_timeout_s=timeout_s,
    )

    candidates: list[dict[str, Any]] = []
    for item in prepared:
        tag = item["tag"]
        sidecar = {
            "schema": BOOTSTRAP_SCHEMA,
            "proposer": "bootstrap",
            "lambda": item["lambda"],
            "profile": {
                "path": str(profile_path),
                "sha256": dataset_content_hash(profile_path),
            },
            "scaffold": {
                "path": item["scaffold"],
                "sha256": dataset_content_hash(item["scaffold"]),
                "command": item["command"],
                "script": script_prov,
            },
            "candidate": {
                "path": item["candidate"],
                "sha256": dataset_content_hash(item["candidate"]),
            },
            # Y2 ③: the shared classification source and the candidate's own
            # projection - bundle verifies the packaged config against it.
            "classification": {
                "path": str(classification_path),
                "sha256": classification_sha,
                "projection_sha256": item["classification_projection"],
            },
            "dev_label_agreement": agreement[tag],
            "default_model": item["selection"]["default_model"],
            "labels": _label_table(
                profile,
                label_counts=label_counts,
                domains=_scaffold_domains(item["scaffold_obj"]),
                spec=spec,
            ),
            "decisions": item["selection"]["decisions"],
            "disabled": item["selection"]["disabled"],
        }
        sidecar_path = out / f"candidate-lambda-{tag}.provenance.json"
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        candidates.append(
            {
                "lambda": item["lambda"],
                "scaffold": item["scaffold"],
                "candidate": item["candidate"],
                "provenance": str(sidecar_path),
                "default_model": item["selection"]["default_model"],
                "decisions": len(item["selection"]["decisions"]) + 1,
                "label_agreement": agreement[tag]["agreement"],
            }
        )

    return {
        "profile": str(profile_path),
        "lambdas": lam_values,
        "seed_config": str(seed_path),
        "classification_source": str(classification_path),
        "classification_source_sha256": classification_sha,
        "candidates": candidates,
    }
