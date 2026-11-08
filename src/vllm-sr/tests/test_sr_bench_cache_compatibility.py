from __future__ import annotations

import csv
import json
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench import cache_probe
from cli.sr_bench.contracts import SESSION_AWARE, plan
from cli.sr_bench.service import Server
from test_sr_bench_cache_probe import (
    ROOT,
    Provider,
    benchmark_cli,
    document,
    native_cli,
    run_cli,
    running_provider,
    usage,
    wait_run,
)
from test_sr_bench_cache_probe import (
    provider as provider,  # noqa: PLC0414 - pytest fixture re-export
)
from test_sr_bench_cache_probe import (
    service as service,  # noqa: PLC0414 - pytest fixture re-export
)

from bench import cache_token_probe as native

DECLARED_REF: str = "fixture-ref"
DECLARED_IMAGE: str = "fixture-image"


def paired_document(
    *, root: Path, router: Provider, backend: Provider, repeats: int = 3
) -> dict[str, Any]:
    value = document(root=root, provider=router, repeats=repeats)
    value["targets"][0].update(id="router", kind="mom", config_hash="fixture-config")
    value["targets"].append(
        {
            "id": "direct-backend",
            "kind": "single",
            "model": "requested-model",
            "base_url": backend.base_url,
            "session_mode": SESSION_AWARE,
        }
    )
    return value


def consume_cache_artifact(
    *, artifact: Path, output: Path, reporting: str = "reported_zero"
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = [
        "--cache-aggregate",
        str(artifact),
        "--min-cached-token-reporting",
        reporting,
        "--min-cached-token-field-rate",
        "1",
        "--min-cached-prompt-ratio",
        "0",
    ]
    branch = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/session_routing_branch_image_benchmark.py"),
            "--diagnostic-summary",
            str(output / "unprovided-diagnostic.json"),
            "--ref",
            DECLARED_REF,
            "--image-tag",
            DECLARED_IMAGE,
            "--output-dir",
            str(output / "branch"),
            *common,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    ga = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/session_routing_ga_report.py"),
            "--output-dir",
            str(output / "ga"),
            *common,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert branch.returncode == ga.returncode == 1, branch.stderr + ga.stderr
    branch_report = json.loads((output / "branch/summary.json").read_text())
    ga_report = json.loads((output / "ga/ga-readiness.json").read_text())
    assert branch_report["checks"]["branch_image_benchmark_ok"] is False
    assert ga_report["ga_ready"] is False
    cache_gate = next(
        row for row in ga_report["requirements"] if row["id"] == "cache_token_reporting"
    )
    return branch_report, cache_gate


@pytest.mark.parametrize("source", ["chat", "input", "deepseek"])
@pytest.mark.parametrize("cached", [0, 40])
def test_native_summary_turns_and_release_consumers_match_on_complete_evidence(
    tmp_path: Path,
    provider: Provider,
    service: Server,
    source: str,
    cached: int,
) -> None:
    with running_provider() as backend:
        provider.usage_events = backend.usage_events = [
            usage(source=source, cached=cached)
        ]
        native_directory = tmp_path / "native"
        original = native_cli(
            provider=provider,
            output=native_directory,
            repeats=3,
            extra_args=[
                "--baseline-base-url",
                backend.base_url,
                "--baseline-model",
                "requested-model",
                "--evidence-ref",
                DECLARED_REF,
                "--evidence-image-tag",
                DECLARED_IMAGE,
            ],
        )
        assert original.returncode == 0, original.stderr
        native_result = json.loads(original.stdout)["aggregate_summary"]
        provider.requests.clear()
        backend.requests.clear()
        value = paired_document(root=tmp_path, router=provider, backend=backend)
        value["benchmark_options"] = {
            cache_probe.BENCHMARK_ID: {
                "evidence_ref": DECLARED_REF,
                "evidence_image_tag": DECLARED_IMAGE,
            }
        }
        result = run_cli(service=service, value=value)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        export = report["cache_probe_compatibility"]
        assert export["version"] == "cache-probe-compatibility-v1"
        assert export["status"] == "complete"
        assert export["validation_failures"] == []
        assert export["evidence_identity_source"] == "declared"
        assert all(
            target["scored"] == 0
            and target["sr_bench_score"] is None
            and target["accuracy"] is None
            for target in report["summary"]["targets"]
        )
        for arm, directory in (
            ("router", native_directory),
            ("baseline", native_directory / "baseline"),
        ):
            summary = export[arm]
            expected = native_result[arm]
            for key, expected_value in expected.items():
                if key not in {"latency_ms", "requests_per_second"}:
                    assert summary[key] == expected_value, (arm, key)
            assert summary["stable_prefix_sha256"] == native.probe_prompt_hash()
            assert (
                summary["stable_prefix_full_sha256"] == cache_probe.STABLE_PREFIX_SHA256
            )
            assert summary["validation_failures"] == []
            with (directory / "turns.csv").open(newline="") as stream:
                native_rows = list(csv.DictReader(stream))
            for row, expected_row in zip(summary["turns"], native_rows, strict=True):
                for key in (
                    "repeat_index",
                    "status",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_tokens",
                ):
                    assert row[key] == int(expected_row[key]), (arm, key)
                for key in ("label", "cached_token_source", "response_model", "error"):
                    assert row[key] == expected_row[key], (arm, key)
                assert row["success"] is True
                assert row["cached_token_field_present"] is True
                assert row["transport_status"] == "completed"
                assert row["latency_ms"] >= 0
            assert summary["requests_per_second"] == round(
                summary["requests"] / summary["sequence_wall_time_s"], 3
            )
        snapshot = tmp_path / "shared.json"
        snapshot.write_text(json.dumps(export))
        shared_branch, shared_ga = consume_cache_artifact(
            artifact=snapshot, output=tmp_path / "shared-consumers"
        )
        native_branch, native_ga = consume_cache_artifact(
            artifact=native_directory / "aggregate-summary.json",
            output=tmp_path / "native-consumers",
        )
        assert shared_branch["checks"]["cache_token_probe_ok"] is True
        assert shared_ga["status"] == native_ga["status"] == "passed"
        assert shared_branch["evidence"]["cache_aggregates"][0]["paths"] == (
            native_branch["evidence"]["cache_aggregates"][0]["paths"]
        )
        assert shared_ga["metrics"] == native_ga["metrics"]
        reread = benchmark_cli(service=service, args=["report", report["run_id"]])
        assert reread.returncode == 0, reread.stderr
        assert json.loads(reread.stdout)["cache_probe_compatibility"] == export


@pytest.mark.parametrize(
    "problem",
    [
        "cache_missing",
        "root_unqualified",
        "usage_partial",
        "model_missing",
        "config_missing",
    ],
)
def test_unknown_evidence_cannot_pass_release_readers_via_native_defaults(
    tmp_path: Path, provider: Provider, service: Server, problem: str
) -> None:
    with running_provider() as backend:
        value = paired_document(root=tmp_path, router=provider, backend=backend)
        value["benchmark_options"] = {
            cache_probe.BENCHMARK_ID: {
                "evidence_ref": DECLARED_REF,
                "evidence_image_tag": DECLARED_IMAGE,
            }
        }
        if problem == "cache_missing":
            provider.usage_events = [usage(cached=None)]
        elif problem == "root_unqualified":
            provider.usage_events = [usage(source="root", cached=40)]
        elif problem == "usage_partial":
            raw = usage()
            raw.pop("completion_tokens")
            provider.usage_events = [raw]
        elif problem == "model_missing":
            provider.omit_model = True
        else:
            provider.config_hash = None
        result = run_cli(service=service, value=value)
        assert result.returncode == (
            2 if problem == "config_missing" else 0
        ), result.stderr
        export = json.loads(result.stdout)["cache_probe_compatibility"]
        assert export["status"] == "incomplete"
        assert export["validation_failures"]
        if problem in {"cache_missing", "root_unqualified", "usage_partial"}:
            assert export["router"]["cached_prompt_ratio"] is None
        if problem == "cache_missing":
            assert export["router"]["prompt_tokens"] == 300
            assert export["router"]["cached_tokens"] is None
        if problem == "usage_partial":
            assert export["router"]["prompt_tokens"] == 300
            assert export["router"]["completion_tokens"] is None
        if problem == "model_missing":
            assert all(
                row["response_model"] is None for row in export["router"]["turns"]
            )
        artifact = tmp_path / "incomplete.json"
        artifact.write_text(json.dumps(export))
        branch, ga = consume_cache_artifact(
            artifact=artifact, output=tmp_path / "consumer", reporting="missing"
        )
        assert branch["checks"]["cache_token_probe_ok"] is False
        assert ga["status"] == "blocked"
        assert ga["failures"]


@pytest.mark.parametrize("fault", ["http", "connection", "truncated", "length"])
def test_failed_calls_and_unstarted_baseline_remain_in_compatibility_output(
    tmp_path: Path, provider: Provider, service: Server, fault: str
) -> None:
    with running_provider() as backend:
        provider.fault = fault
        value = paired_document(root=tmp_path, router=provider, backend=backend)
        result = run_cli(service=service, value=value)
        assert result.returncode == 2, result.stderr
        report = json.loads(result.stdout)
        export = report["cache_probe_compatibility"]
        assert export["validation_failures"]
        assert export["router"]["requests"] == export["baseline"]["requests"] == 3
        assert len(export["router"]["turns"]) == len(export["baseline"]["turns"]) == 3
        assert export["router"]["requests_per_second"] is None
        assert export["baseline"]["requests_per_second"] is None
        for row in [export["router"]["turns"][-1], *export["baseline"]["turns"]]:
            assert row["transport_status"] == "undispatched"
            assert all(
                row[key] is None
                for key in (
                    "status",
                    "success",
                    "latency_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_tokens",
                    "cached_token_field_present",
                    "response_model",
                )
            )
        if fault == "connection":
            assert export["router"]["turns"][1]["status"] is None
            assert "0" not in export["router"]["status_counts"]
        assert backend.requests == []


def test_direct_only_and_mixed_runs_do_not_claim_a_native_router_pair(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    value = document(root=tmp_path, provider=provider, repeats=1)
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    export = json.loads(result.stdout)["cache_probe_compatibility"]
    assert export["status"] == "unsupported"
    assert export["validation_failures"]
    assert "router" not in export
    value = plan(value)
    for key in ("dataset", "adapter_versions", "plan_sha256"):
        value.pop(key)
    value["cases"].append(
        {
            "id": "binary",
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": "Return A"}],
            "answer": "A",
        }
    )
    path = service.store.root / "mixed.json"
    path.write_text(json.dumps(value))
    result = benchmark_cli(
        service=service,
        args=[
            "run",
            "--manifest",
            str(path),
            "--idempotency-key",
            "mixed-compatibility",
        ],
    )
    assert result.returncode == 0, result.stderr
    export = json.loads(result.stdout)["cache_probe_compatibility"]
    assert export["status"] == "unsupported"
    assert "router" not in export


@pytest.mark.parametrize(
    "problem",
    [
        "conflicting_cache",
        "malformed_cache",
        "null_cache",
        "zero_prompt",
        "prompt_aliases",
        "snapshot_drift",
    ],
)
def test_unsupported_usage_never_becomes_complete_compatibility(
    tmp_path: Path, provider: Provider, service: Server, problem: str
) -> None:
    with running_provider() as backend:
        raw = usage()
        if problem == "conflicting_cache":
            raw["prompt_cache_hit_tokens"] = 41
        elif problem == "malformed_cache":
            raw = usage(source="deepseek", cached="40")
        elif problem == "null_cache":
            raw["prompt_tokens_details"]["cached_tokens"] = None
        elif problem == "zero_prompt":
            raw.update(prompt_tokens=0, prompt_tokens_details={"cached_tokens": 0})
        elif problem == "prompt_aliases":
            raw["input_tokens"] = 101
        provider.usage_events = [raw]
        if problem == "snapshot_drift":
            provider.usage_events.append({"prompt_tokens": 110, "completion_tokens": 3})
        value = paired_document(root=tmp_path, router=provider, backend=backend)
        result = run_cli(service=service, value=value)
        assert result.returncode == (2 if problem == "null_cache" else 0), result.stderr
        report = json.loads(result.stdout)
        export = report["cache_probe_compatibility"]
        assert export["status"] == "incomplete"
        assert export["validation_failures"]
        if problem in {
            "conflicting_cache",
            "malformed_cache",
            "null_cache",
            "zero_prompt",
            "snapshot_drift",
        }:
            assert export["router"]["cached_prompt_ratio"] is None
        if problem == "prompt_aliases":
            assert export["router"]["prompt_tokens"] is None
        if problem == "snapshot_drift":
            assert export["router"]["prompt_tokens"] == 330
            assert any(
                "snapshots differ" in reason for reason in export["validation_failures"]
            )
        original_calls = service.store.calls(report["run_id"])
        rendered = benchmark_cli(service=service, args=["report", report["run_id"]])
        assert rendered.returncode == 0, rendered.stderr
        assert service.store.calls(report["run_id"]) == original_calls


def test_declared_evidence_is_frozen_and_not_runtime_provenance(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    with running_provider() as backend:
        value = paired_document(root=tmp_path, router=provider, backend=backend)
        original = plan(value)
        value["benchmark_options"] = {
            cache_probe.BENCHMARK_ID: {
                "evidence_ref": DECLARED_REF,
                "evidence_image_tag": DECLARED_IMAGE,
            }
        }
        frozen = plan(value)
        assert original["case_sha256"] == frozen["case_sha256"]
        assert original["plan_sha256"] != frozen["plan_sha256"]
        result = run_cli(service=service, value=frozen)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        export = report["cache_probe_compatibility"]
        assert export["evidence_identity_source"] == "declared"
        assert export["router"]["evidence_ref"] == DECLARED_REF
        assert export["router"]["evidence_image_tag"] == DECLARED_IMAGE
        assert "evidence_ref" not in export["baseline"]
        assert "evidence_image_tag" not in export["baseline"]
        assert all(
            row["config_hash"] == "fixture-config" for row in export["router"]["turns"]
        )
        assert report["provenance"]["runner"].get("evidence_image_tag") is None


@pytest.mark.parametrize("field", ["evidence_ref", "evidence_image_tag"])
@pytest.mark.parametrize("value", [None, "", 42, "a\nb", " leading", "a" * 257])
def test_invalid_declared_identity_is_rejected_before_dispatch(
    tmp_path: Path, provider: Provider, field: str, value: object
) -> None:
    document_value = document(root=tmp_path, provider=provider)
    document_value["benchmark_options"] = {cache_probe.BENCHMARK_ID: {field: value}}
    with pytest.raises(ValueError, match=field):
        plan(document_value)
    assert provider.requests == []


@pytest.mark.parametrize("problem", ["unpriced_usage", "cost_budget"])
def test_compatibility_does_not_override_accounting_or_budget_failures(
    tmp_path: Path, provider: Provider, service: Server, problem: str
) -> None:
    with running_provider() as backend:
        value = paired_document(root=tmp_path, router=provider, backend=backend)
        value["cost_policy"] = "require_priced"
        prices = {
            name: {"input": 1, "cached_input": 0.1, "cache_write": 2, "output": 3}
            for name in ("requested-model", "reported-model")
        }
        for target in value["targets"]:
            target["prices"] = prices
        value["targets"][0]["max_inference_calls"] = 1
        if problem == "unpriced_usage":
            raw = usage()
            raw.pop("completion_tokens")
            provider.usage_events = [raw]
        else:
            value["limits"]["max_cost_usd"] = 0.00000001
        result = run_cli(service=service, value=value)
        assert result.returncode == 2, result.stderr
        report = json.loads(result.stdout)
        assert report["status"] == "failed"
        export = report["cache_probe_compatibility"]
        assert export["status"] == "incomplete"
        assert export["validation_failures"]
        assert report["summary"]["targets"][0]["scored"] == 0
        assert report["summary"]["targets"][0]["sr_bench_score"] is None
        assert len(provider.requests) == (1 if problem == "unpriced_usage" else 0)
        assert backend.requests == []
        assert all(row["status"] is None for row in export["baseline"]["turns"])


def test_cancelled_sequence_keeps_every_native_slot_without_zero_defaults(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    with running_provider() as backend:
        provider.fault = "cancel"
        result = run_cli(
            service=service,
            value=paired_document(root=tmp_path, router=provider, backend=backend),
            detach=True,
        )
        assert result.returncode == 0, result.stderr
        run_id = json.loads(result.stdout)["id"]
        assert provider.entered.wait(timeout=5)
        response = benchmark_cli(service=service, args=["cancel", run_id])
        assert response.returncode == 0, response.stderr
        provider.release.set()
        assert wait_run(service.store, run_id)["status"] == "cancelled"
        rendered = benchmark_cli(service=service, args=["report", run_id])
        assert rendered.returncode == 0, rendered.stderr
        export = json.loads(rendered.stdout)["cache_probe_compatibility"]
        assert export["status"] == "incomplete"
        assert len(export["router"]["turns"]) == len(export["baseline"]["turns"]) == 3
        assert export["router"]["turns"][1]["transport_status"] == "cancelled"
        assert export["router"]["turns"][2]["status"] is None
        assert export["router"]["turns"][2]["response_model"] is None
        assert all(row["cached_tokens"] is None for row in export["baseline"]["turns"])
        assert backend.requests == []


def test_recovered_baseline_cannot_be_projected_as_a_complete_native_pair(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    with running_provider() as backend:
        provider.fault = "http"
        result = run_cli(
            service=service,
            value=paired_document(root=tmp_path, router=provider, backend=backend),
        )
        assert result.returncode == 2, result.stderr
        parent = json.loads(result.stdout)["run_id"]
        original_calls = service.store.calls(parent)
        path = tmp_path / "recovery.json"
        proposed = benchmark_cli(
            service=service, args=["recover-plan", parent, "--output", str(path)]
        )
        assert proposed.returncode == 0, proposed.stderr
        cells = json.loads(proposed.stdout)["eligible_cells"]
        assert cells == [
            {
                "case_id": cache_probe.BENCHMARK_ID + "/repeated-prefix",
                "target_id": "direct-backend",
            }
        ]
        recovered = benchmark_cli(
            service=service,
            args=[
                "recover",
                parent,
                "--plan",
                str(path),
                "--idempotency-key",
                "cache-baseline-recovery",
            ],
        )
        assert recovered.returncode == 0, recovered.stderr
        child = json.loads(recovered.stdout)["id"]
        assert wait_run(service.store, child)["status"] == "completed"
        rendered = benchmark_cli(service=service, args=["report", child])
        assert rendered.returncode == 0, rendered.stderr
        export = json.loads(rendered.stdout)["cache_probe_compatibility"]
        assert export["status"] == "unsupported"
        assert "router" not in export
        assert export["validation_failures"]
        assert service.store.calls(parent) == original_calls
        assert len(provider.requests) == 2
        assert len(backend.requests) == 3


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        server = self.server
        assert isinstance(server, Provider)
        server.requests.append(
            {
                "path": self.path,
                "body": json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                ),
            }
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {"config_hash": server.config_hash, "selected_model": "reported-model"}
            ).encode()
        )


def test_preview_does_not_invent_live_native_cache_observations(
    tmp_path: Path, provider: Provider, service: Server
) -> None:
    provider.RequestHandlerClass = PreviewHandler
    value = document(root=tmp_path, provider=provider)
    value["mode"] = "preview"
    value["targets"][0].update(
        kind="mom",
        config_hash="fixture-config",
        preview_url=provider.base_url + "/preview",
    )
    result = run_cli(service=service, value=value)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["cache_probe_compatibility"]["status"] == "unsupported"
    assert "router" not in report["cache_probe_compatibility"]
    assert report["cache_probe_compatibility"]["validation_failures"]
    assert service.store.calls(report["run_id"]) == []
    assert [row["path"] for row in provider.requests] == ["/v1/preview"]
