"""Offline Router Learning recipe analysis command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import click
import requests
import yaml

from cli.commands.recipe_learning_artifacts import (
    build_candidate_recipes,
    build_experience_seed_pack,
    build_recipe_learning_experiments,
    build_recipe_patch_suggestions,
    empty_recipe_patch,
)
from cli.commands.recipe_learning_findings import build_recipe_learning_findings
from cli.commands.recipe_learning_metrics import (
    DecisionMetrics,
    EvalCase,
    optional_numeric,
    record_decision,
    record_decision_tier,
    tier_metrics_key,
    update_experience_counts,
    update_metrics,
)
from cli.consts import DEFAULT_API_PORT, DEFAULT_LISTENER_PORT

_DEFAULT_REPLAY_LIMIT = 100
_MAX_REPLAY_LIMIT = 500


def default_replay_endpoint() -> str:
    return f"http://localhost:{DEFAULT_LISTENER_PORT}/v1/router_replay"


def normalize_replay_endpoint(endpoint: str, limit: int) -> str:
    return candidate_replay_endpoints(endpoint, limit)[0]


def candidate_replay_endpoints(endpoint: str, limit: int) -> list[str]:
    endpoint = endpoint.strip()
    if not endpoint:
        endpoint = default_replay_endpoint()
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]
    if not endpoint.endswith("/v1/router_replay"):
        endpoint = urljoin(endpoint + "/", "v1/router_replay")

    candidates = [endpoint]
    listener_endpoint = listener_replay_endpoint(endpoint)
    if listener_endpoint and listener_endpoint not in candidates:
        candidates.append(listener_endpoint)

    return [
        normalize_replay_endpoint_query(candidate, limit) for candidate in candidates
    ]


def listener_replay_endpoint(endpoint: str) -> str | None:
    parts = urlsplit(endpoint)
    if parts.port != DEFAULT_API_PORT:
        return None
    if (
        parts.path
        and parts.path not in ("", "/")
        and not parts.path.rstrip("/").endswith("/v1/router_replay")
    ):
        return None
    host = parts.hostname
    if not host:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo += f":{parts.password}"
        userinfo += "@"
    netloc = f"{userinfo}{host}:{DEFAULT_LISTENER_PORT}"
    return urlunsplit((parts.scheme, netloc, "/v1/router_replay", "", ""))


def normalize_replay_endpoint_query(endpoint: str, limit: int) -> str:
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("showDetails", "true")
    query.setdefault("limit", str(max(1, min(limit, _MAX_REPLAY_LIMIT))))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def load_replay_records(
    replay_file: Path | None, endpoint: str | None, limit: int, timeout: int
) -> list[dict[str, Any]]:
    if replay_file is not None and endpoint:
        raise ValueError("Provide only one of --replay-file or --endpoint")
    if replay_file is not None:
        payload = json.loads(replay_file.read_text(encoding="utf-8"))
    else:
        payload = fetch_replay_payload(endpoint or "", limit, timeout)
    return normalize_replay_payload(payload)


def fetch_replay_payload(endpoint: str, limit: int, timeout: int) -> Any:
    errors: list[str] = []
    for url in candidate_replay_endpoints(endpoint, limit):
        try:
            response = requests.get(url, timeout=timeout)
        except requests.ConnectionError as exc:
            errors.append(f"{url}: not reachable ({exc})")
            continue
        except requests.Timeout as exc:
            errors.append(f"{url}: timed out after {timeout}s ({exc})")
            continue
        except requests.RequestException as exc:
            errors.append(f"{url}: request failed ({exc})")
            continue
        if response.status_code == requests.codes.ok:
            return response.json()
        errors.append(f"{url}: HTTP {response.status_code} - {response.text}")

    tried = "; ".join(errors)
    raise ValueError(
        "Router replay endpoint is not reachable. Start vllm-sr serve, pass "
        f"--replay-file, or pass the Envoy listener /v1/router_replay URL. Tried: {tried}"
    )


def normalize_replay_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            records = payload["data"]
        elif isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            records = [payload]
    else:
        raise ValueError(
            "Replay input must be a record, a record list, or a router_replay.list payload"
        )

    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Replay record {index} must be an object")
        normalized.append(record)
    return normalized


def load_eval_cases(cases_file: Path | None) -> dict[str, EvalCase]:
    if cases_file is None:
        return {}
    payload = json.loads(cases_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        payload = payload["cases"]
    if not isinstance(payload, list):
        raise ValueError(
            "--cases-file must contain a JSON array or an object with cases[]"
        )

    cases: dict[str, EvalCase] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Eval case {index} must be an object")
        case = EvalCase(
            replay_id=str(item.get("replay_id") or item.get("id") or "").strip(),
            request_id=str(item.get("request_id") or "").strip(),
            expected_decision=str(item.get("expected_decision") or "").strip(),
            expected_model=str(item.get("expected_model") or "").strip(),
            max_cost=optional_numeric(
                item.get("max_cost")
                if "max_cost" in item
                else item.get("expected_max_cost")
            ),
            max_latency_ms=optional_numeric(
                item.get("max_latency_ms")
                if "max_latency_ms" in item
                else item.get("expected_max_latency_ms")
            ),
        )
        for key in (case.replay_id, case.request_id):
            if key:
                cases[key] = case
    return cases


def build_recipe_learning_artifact(
    records: list[dict[str, Any]],
    cases: dict[str, EvalCase] | None = None,
    recipe: dict[str, Any] | None = None,
    *,
    generate_patches: bool = True,
) -> dict[str, Any]:
    metrics_payload, experience_counts = build_metrics_payload(records, cases or {})
    findings = build_recipe_learning_findings(metrics_payload)
    recipe_patch = (
        build_recipe_patch_suggestions(findings)
        if generate_patches
        else empty_recipe_patch("report_only")
    )
    candidate_recipes = build_candidate_recipes(recipe, recipe_patch)
    return {
        "object": "router_learning.recipe_learning",
        "metrics": metrics_payload,
        "findings": findings,
        "candidate_recipes": candidate_recipes,
        "experiment_results": build_recipe_learning_experiments(
            metrics_payload,
            candidate_recipes,
        ),
        "recipe_patch": recipe_patch,
        "experience_seed_pack": build_experience_seed_pack(experience_counts),
    }


def build_metrics_payload(
    records: list[dict[str, Any]], case_index: dict[str, EvalCase]
) -> tuple[dict[str, Any], dict[tuple[str, int, str], dict[str, int]]]:
    global_metrics = DecisionMetrics()
    per_decision: dict[str, DecisionMetrics] = {}
    per_tier: dict[str, DecisionMetrics] = {}
    experience_counts: dict[tuple[str, int, str], dict[str, int]] = {}

    for record in records:
        decision = record_decision(record)
        metrics = per_decision.setdefault(decision, DecisionMetrics())
        tier = tier_metrics_key(record_decision_tier(record))
        tier_metrics = per_tier.setdefault(tier, DecisionMetrics())
        update_metrics(global_metrics, record, case_index)
        update_metrics(metrics, record, case_index)
        update_metrics(tier_metrics, record, case_index)
        update_experience_counts(experience_counts, record)

    return (
        {
            "records": len(records),
            "overall": global_metrics.to_json(),
            "per_decision": {
                decision: metrics.to_json()
                for decision, metrics in sorted(per_decision.items())
            },
            "per_tier": {
                tier: metrics.to_json() for tier, metrics in sorted(per_tier.items())
            },
        },
        experience_counts,
    )


def write_recipe_learning_artifacts(artifact: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "summary.json": artifact,
        "metrics.json": artifact["metrics"],
        "findings.json": artifact["findings"],
        "recipe_patch.json": artifact["recipe_patch"],
        "experiment_results.json": artifact["experiment_results"],
        "experience_seed_pack.json": artifact["experience_seed_pack"],
    }
    for name, payload in files.items():
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    write_candidate_recipes(artifact, output_dir)


def write_candidate_recipes(artifact: dict[str, Any], output_dir: Path) -> None:
    for candidate in artifact.get("candidate_recipes", []):
        if not isinstance(candidate, dict) or candidate.get("recipe") is None:
            continue
        candidate_id = str(candidate.get("id") or "candidate")
        (output_dir / f"{candidate_id}.yaml").write_text(
            yaml.safe_dump(candidate["recipe"], sort_keys=False),
            encoding="utf-8",
        )


def summarize_recipe_learning_artifact(artifact: dict[str, Any]) -> str:
    metrics = artifact["metrics"]["overall"]
    findings = artifact["findings"]
    lines = [
        "Router Learning recipe analysis",
        f"records: {artifact['metrics']['records']}",
        f"learning coverage: {metrics['learning_coverage']:.2f}",
        f"outcome coverage: {metrics['outcome_coverage']:.2f}",
        f"switch rate: {metrics['switch_rate']:.2f}",
        f"cache preservation: {metrics['cache_preservation']:.2f}",
        f"cost savings: {metrics['cost_savings']:.6f}",
        f"findings: {len(findings)}",
        f"candidate recipes: {len(artifact['candidate_recipes'])}",
    ]
    for item in findings[:5]:
        lines.append(
            f"- [{item['severity']}] {item['decision']}: {item['type']} - {item['message']}"
        )
    return "\n".join(lines)


@click.command("recipe-learning")
@click.option(
    "--replay-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Router replay JSON file. Accepts a router_replay.list payload or a record array.",
)
@click.option(
    "--endpoint",
    default=None,
    help=(
        "Router base URL or /v1/router_replay endpoint. "
        f"Defaults to {default_replay_endpoint()} when --replay-file is omitted."
    ),
)
@click.option(
    "--cases-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Optional eval cases JSON with replay_id/request_id plus expected_decision or expected_model.",
)
@click.option(
    "--recipe-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Optional current recipe YAML used to materialize complete candidate recipe variants.",
)
@click.option(
    "--limit",
    default=_DEFAULT_REPLAY_LIMIT,
    show_default=True,
    help="Replay records to fetch from the endpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory for metrics/findings/patch/seed-pack artifacts.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Print the full recipe-learning artifact.",
)
@click.option(
    "--report-only",
    is_flag=True,
    help="Compute metrics and findings without generating recipe patches.",
)
@click.option(
    "--timeout", default=15, show_default=True, help="HTTP request timeout in seconds."
)
def recipe_learning(
    replay_file: Path | None,
    endpoint: str | None,
    cases_file: Path | None,
    recipe_file: Path | None,
    limit: int,
    output_dir: Path | None,
    output_json: bool,
    report_only: bool,
    timeout: int,
) -> None:
    """Analyze replay and outcomes to produce recipe-learning artifacts."""

    records = load_replay_records(replay_file, endpoint, limit, timeout)
    cases = load_eval_cases(cases_file)
    recipe = load_recipe_file(recipe_file)
    artifact = build_recipe_learning_artifact(
        records,
        cases,
        recipe,
        generate_patches=not report_only,
    )
    if output_dir is not None:
        write_recipe_learning_artifacts(artifact, output_dir)
    if output_json:
        click.echo(json.dumps(artifact, indent=2, ensure_ascii=False))
        return
    click.echo(summarize_recipe_learning_artifact(artifact))
    if output_dir is not None:
        click.echo(f"artifacts: {output_dir}")


def load_recipe_file(recipe_file: Path | None) -> dict[str, Any] | None:
    if recipe_file is None:
        return None
    payload = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--recipe-file must contain a YAML object")
    return payload
