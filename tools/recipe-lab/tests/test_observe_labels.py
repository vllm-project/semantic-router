"""C2 observe-labels: seed config, label extraction, and the offline probe.

No GPU, no real router binary: the offline probe runs against a fake router
that records one replay record per request (mirroring the real recorder's
one-record-per-request guard) with the decision/confidence the state file
prescribes. The real-router label coverage is the C2 completion criterion and
is recorded by the command summary when run against the live stack.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

import pytest
import yaml
from recipe_lab import API_VERSION
from recipe_lab.classification import load_classification_source
from recipe_lab.cli import main
from recipe_lab.contract import ContractError
from recipe_lab.observe_labels import (
    FALLBACK_DECISION_NAME,
    OTHER_LABEL,
    REQUEST_ID_PREFIX,
    ROUTER_MODEL,
    SUPPORTED_ROUTING_DOMAINS,
    build_seed_config,
    extract_confidence,
    label_for_decision,
    labels_from_records,
    observe_split,
    seed_config_yaml,
    write_labels,
    write_seed_config,
)
from recipe_lab.runner import RunnerEnv

# The shared classification source (a minimal stand-in for the maintained
# config/config.yaml): both sections the Y2 same-source rule requires.
CLASSIFICATION_SOURCE_YAML = """\
global:
  model_catalog:
    embeddings:
      semantic:
        use_cpu: true
        embedding_config:
          model_type: mmbert
    modules:
      classifier:
        domain:
          enabled: true
          fallback_category: other
      prompt_guard:
        enabled: true
"""


def _write_classification_source(tmp_path: Path, *, drop: str = "") -> str:
    text = CLASSIFICATION_SOURCE_YAML
    if drop == "embeddings":
        text = text.replace(
            "    embeddings:\n      semantic:\n        use_cpu: true\n        embedding_config:\n          model_type: mmbert\n",
            "",
        )
    path = tmp_path / "classification-source.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# A fake router binary for the restart-mode observe probe: gateway + management
# on the config's own ports, records driven by a per-case route table in the
# shared state file (fake_state.json next to the config the runner boots from,
# i.e. inside the work directory). Each chat request looks up the route by the
# case id inside its rl-observe-<case_id> x-request-id and appends exactly one
# record.
OBSERVE_FAKE_ROUTER_SCRIPT = """#!/usr/bin/env python3
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

config_path = sys.argv[sys.argv.index("-config") + 1]
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
GATE = int(cfg["listeners"][0]["port"])
API = int(sys.argv[sys.argv.index("-api-port") + 1])
STATE = os.path.join(os.path.dirname(config_path), "fake_state.json")
lock = threading.Lock()


def _load():
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        # a boot without a pre-seeded state observes no routes: every case
        # records the catch-all label (used by the bootstrap label check)
        return {"records": [], "routes": {}}


def _save(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length) or b"{}")
        rid = self.headers.get("x-request-id", "")
        case_id = rid[len("rl-observe-"):] if rid.startswith("rl-observe-") else ""
        with lock:
            state = _load()
            route = state["routes"].get(case_id, {})
            record = {
                "request_id": rid,
                "decision": route.get("decision", ""),
                "selected_model": "local/seed-backend",
            }
            if "confidence_score" in route:
                record["confidence_score"] = route["confidence_score"]
                record["confidence_score_available"] = route.get(
                    "confidence_score_available", True
                )
            if route.get("category"):
                record["category"] = route["category"]
            state["records"].append(record)
            _save(state)
        self._send(200, {"choices": [{"message": {"content": "ok"}}]})

    def do_GET(self):
        if self.path.startswith("/api/v1/observability/replays"):
            with lock:
                records = list(_load()["records"])
            self._send(200, {"data": records, "has_more": False})
            return
        self._send(404, {"error": "not found"})


gateway = ThreadingHTTPServer(("127.0.0.1", GATE), Handler)
management = ThreadingHTTPServer(("127.0.0.1", API), Handler)
with open(
    os.path.join(os.path.dirname(config_path), "router-runtime.json"),
    "w",
    encoding="utf-8",
) as fh:
    json.dump({"ready": True}, fh)
threading.Thread(target=management.serve_forever, daemon=True).start()
gateway.serve_forever()
"""


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _write_state(path: Path, routes: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": [], "routes": routes}), encoding="utf-8")


def _observe_env(tmp_path: Path, work_dir: str) -> RunnerEnv:
    router = tmp_path / "fake_router.py"
    router.write_text(OBSERVE_FAKE_ROUTER_SCRIPT)
    router.chmod(0o755)
    mock = tmp_path / "mock_backend.py"
    mock.write_text("import time\ntime.sleep(300)\n")
    return RunnerEnv(
        router_bin=str(router),
        mock_backend_script=str(mock),
        python_bin=sys.executable,
        work_dir=tmp_path / work_dir,
        backend_port=_free_port(),
        ready_timeout_s=30,
    )


def test_seed_config_structure_pins_everything_the_observe_needs():
    config = build_seed_config()
    decisions = config["routing"]["decisions"]
    by_name = {decision["name"]: decision for decision in decisions}
    # 14 domain decisions + the catch-all, nothing else.
    assert set(by_name) == set(SUPPORTED_ROUTING_DOMAINS) | {FALLBACK_DECISION_NAME}
    for domain in SUPPORTED_ROUTING_DOMAINS:
        decision = by_name[domain]
        # exactly one domain condition, named after the decision itself
        assert decision["rules"]["conditions"] == [{"type": "domain", "name": domain}]
        assert decision["modelRefs"], "every decision routes to the seed backend"
        assert any(
            plugin["type"] == "router_replay" for plugin in decision["plugins"]
        ), "the matched decision name must surface in the replay records"
    fallback = by_name[FALLBACK_DECISION_NAME]
    assert "rules" not in fallback  # catch-all: no conditions of its own
    # the entrypoint routes the default request model into the default recipe
    entrypoint = config["entrypoints"][0]
    assert entrypoint["model_names"] == [ROUTER_MODEL]
    assert entrypoint["recipe"] == "default"
    # #5620: never rely on defaults - pin the decision model deployment
    assert config["global"]["model_catalog"]["system"]["decision_model"] == {
        "deployment": "primary"
    }
    # labels must come from live routing decisions, never from the cache
    assert config["global"]["stores"]["response_cache"]["enabled"] is False


def test_seed_config_is_byte_deterministic(tmp_path):
    first = seed_config_yaml(build_seed_config())
    assert seed_config_yaml(build_seed_config()) == first
    other = seed_config_yaml(build_seed_config(backend_endpoint="127.0.0.1:9999"))
    assert other != first  # the endpoint is part of the document
    assert "127.0.0.1:9999" in other
    one = write_seed_config(build_seed_config(), tmp_path / "one" / "seed-config.yaml")
    two = write_seed_config(build_seed_config(), tmp_path / "two" / "seed-config.yaml")
    assert one.read_bytes() == two.read_bytes()


def test_seed_config_injects_shared_classification_sections(tmp_path):
    # Y2: the seed classifies with the same modules/embeddings every bootstrap
    # candidate will carry; the seed's own deployments/system pinning survives.
    classification = load_classification_source(_write_classification_source(tmp_path))
    config = build_seed_config(classification=classification)
    catalog = config["global"]["model_catalog"]
    assert catalog["embeddings"] == classification["embeddings"]
    assert catalog["modules"] == classification["modules"]
    assert catalog["deployments"] == {
        "primary": {"provider": "model_runtime", "artifact": "vllm-sr/d3-lite"}
    }
    assert catalog["system"] == {"decision_model": {"deployment": "primary"}}
    # without a source the seed would classify with the router's defaults -
    # exactly the divergence Y2 closes against the scaffold-derived candidates
    bare = build_seed_config()
    assert "modules" not in bare["global"]["model_catalog"]
    assert "embeddings" not in bare["global"]["model_catalog"]
    # the injected document stays byte-deterministic
    assert seed_config_yaml(
        build_seed_config(classification=classification)
    ) == seed_config_yaml(config)


def test_load_classification_source_requires_both_sections(tmp_path):
    sections = load_classification_source(_write_classification_source(tmp_path))
    assert set(sections) == {"modules", "embeddings"}
    assert (
        sections["embeddings"]["semantic"]["embedding_config"]["model_type"] == "mmbert"
    )
    assert sections["modules"]["classifier"]["domain"]["fallback_category"] == "other"
    # an incomplete source would silently reintroduce the default-vs-scaffold
    # divergence Y2 closes
    broken = _write_classification_source(tmp_path, drop="embeddings")
    with pytest.raises(ContractError, match=r"no global\.model_catalog\.embeddings"):
        load_classification_source(broken)


def test_label_for_decision_maps_domains_and_catch_all():
    for domain in SUPPORTED_ROUTING_DOMAINS:
        assert label_for_decision(domain) == domain
    assert label_for_decision(FALLBACK_DECISION_NAME) == OTHER_LABEL
    assert label_for_decision("") == OTHER_LABEL
    assert label_for_decision("some_unknown_decision") == OTHER_LABEL
    assert label_for_decision(" math ") == "math"


def test_extract_confidence_gates_on_availability():
    assert (
        extract_confidence(
            {"confidence_score": 0.87, "confidence_score_available": True}
        )
        == 0.87
    )
    # unavailable score must never leak as a number
    assert (
        extract_confidence(
            {"confidence_score": 0.87, "confidence_score_available": False}
        )
        is None
    )
    assert extract_confidence({"confidence_score_available": True}) is None
    assert extract_confidence({"confidence_score": 0.87}) is None
    assert (
        extract_confidence(
            {"confidence_score": "0.87", "confidence_score_available": True}
        )
        is None
    )
    # camelCase wiring tolerated
    assert (
        extract_confidence({"ConfidenceScore": 0.5, "ConfidenceScoreAvailable": True})
        == 0.5
    )


def test_labels_from_records_joins_by_request_id_and_reports_gaps():
    cases = [{"case_id": f"c{i}", "prompt": f"p{i}", "truth": "x"} for i in range(4)]
    records = [
        {
            "request_id": REQUEST_ID_PREFIX + "c0",
            "decision": "math",
            "confidence_score": 0.9,
            "confidence_score_available": True,
        },
        {
            "request_id": REQUEST_ID_PREFIX + "c1",
            "decision": FALLBACK_DECISION_NAME,
            "confidence_score": 0.0,
            "confidence_score_available": False,
        },
        {"request_id": "rl-other-tag-c2", "decision": "law"},  # unknown request id
        {"request_id": "", "decision": "law"},  # no request id
        {
            "request_id": REQUEST_ID_PREFIX + "c3",
            "decision": "physics",
            "confidence_score": 0.42,
            "confidence_score_available": True,
        },
        # a repeated record for the same request id is counted, never joined twice
        {"request_id": REQUEST_ID_PREFIX + "c3", "decision": "physics"},
    ]
    rows, stats = labels_from_records(records, cases)
    assert [row["case_id"] for row in rows] == ["c0", "c1", "c3"]
    assert rows[0] == {"case_id": "c0", "router_label": "math", "confidence": 0.9}
    assert rows[1] == {"case_id": "c1", "router_label": OTHER_LABEL, "confidence": None}
    assert rows[2] == {"case_id": "c3", "router_label": "physics", "confidence": 0.42}
    assert stats["matched"] == 3
    assert stats["missing"] == 1  # c2 produced no record
    assert stats["unknown_request_id"] == 1
    assert stats["no_request_id"] == 1
    assert stats["duplicate_record"] == 1


def test_labels_from_records_rejects_duplicate_case_ids():
    cases = [
        {"case_id": "c0", "prompt": "p", "truth": "t"},
        {"case_id": "c0", "prompt": "q", "truth": "t"},
    ]
    with pytest.raises(ContractError):
        labels_from_records([], cases)


def test_write_labels_is_byte_stable(tmp_path):
    rows = [
        {"case_id": "b", "router_label": "math", "confidence": None},
        {"case_id": "a", "router_label": OTHER_LABEL, "confidence": 0.5},
    ]
    path = write_labels(rows, tmp_path / "labels" / "labels-dev.jsonl")
    expected = (
        '{"case_id": "a", "confidence": 0.5, "router_label": "other"}\n'
        '{"case_id": "b", "confidence": null, "router_label": "math"}\n'
    )
    assert path.read_text(encoding="utf-8") == expected
    path2 = write_labels(rows, tmp_path / "labels2" / "labels-dev.jsonl")
    assert path.read_bytes() == path2.read_bytes()


def test_observe_split_with_fake_router_labels_match_seed_decisions(tmp_path):
    # C2 test criterion: a mock fixture asserts the labels correspond to the
    # seed decisions one to one (decision name IS the label; the catch-all
    # records other).
    state_path = tmp_path / "work" / "fake_state.json"
    cases = [
        {"case_id": "math-1", "prompt": "What is 2+2?", "truth": "math"},
        {"case_id": "law-1", "prompt": "Which amendment governs?", "truth": "law"},
        {"case_id": "wild-1", "prompt": "Free-form chatter", "truth": "other"},
    ]
    _write_state(
        state_path,
        {
            "math-1": {
                "decision": "math",
                "confidence_score": 0.93,
                "category": "math",
            },
            "law-1": {
                "decision": "law",
                "confidence_score": 0.81,
                "category": "law",
            },
            "wild-1": {
                "decision": FALLBACK_DECISION_NAME,
                "confidence_score": 0.0,
                "confidence_score_available": False,
            },
        },
    )
    config_path = write_seed_config(build_seed_config(), tmp_path / "seed-config.yaml")
    result = observe_split(
        env=_observe_env(tmp_path, "work"),
        config=config_path,
        cases=cases,
        trial_timeout_s=120.0,
    )
    labels = {row["case_id"]: row["router_label"] for row in result["rows"]}
    assert labels == {"math-1": "math", "law-1": "law", "wild-1": OTHER_LABEL}
    confidences = {row["case_id"]: row["confidence"] for row in result["rows"]}
    assert confidences == {"math-1": 0.93, "law-1": 0.81, "wild-1": None}
    summary = result["summary"]
    assert summary["cases"] == 3
    assert summary["observed"] == 3
    assert summary["missing"] == 0
    assert summary["labeled"] == 2
    assert summary["other"] == 1
    assert summary["label_coverage"] == round(2 / 3, 6)
    assert summary["label_counts"] == {"law": 1, "math": 1, "other": 1}
    assert all(status == 200 for status in result["statuses"].values())
    assert set(result["timings"]) == {"switch_s", "request_s", "fetch_s"}
    # the observe probe tagged every request with the candidate-scoped prefix
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert {record["request_id"] for record in state["records"]} == {
        REQUEST_ID_PREFIX + case["case_id"] for case in cases
    }


def _write_observe_spec(tmp_path: Path, split_cases: dict[str, str]) -> str:
    """Spec with calib/dev/frozen_test manifests covering their own bytes."""
    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 2.0,
        },
        "workloads": [
            {
                "name": "mmlu-pro",
                "profile": "quick",
                "splits": {
                    name: {
                        "manifest": f"{name}.jsonl",
                        "content_hash": "sha256:"
                        + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    }
                    for name, payload in split_cases.items()
                },
            }
        ],
        "attribution": {"oracle_policy": {"math": "m"}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 10},
    }
    for name, payload in split_cases.items():
        (tmp_path / f"{name}.jsonl").write_text(payload, encoding="utf-8")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _write_cli_stack(tmp_path: Path, state_path: Path) -> tuple[str, str]:
    router = tmp_path / "fake_router.py"
    router.write_text(OBSERVE_FAKE_ROUTER_SCRIPT)
    router.chmod(0o755)
    mock = tmp_path / "mock_backend.py"
    mock.write_text("import time\ntime.sleep(300)\n")
    return str(router), str(mock)


def test_observe_labels_cli_writes_labels_and_summary(tmp_path, capsys):
    # CLI path: spec split resolution + seed config written next to the labels
    # + summary carrying the coverage the provenance records.
    calib_payload = (
        '{"case_id":"cm","prompt":"p1","truth":"math"}\n'
        '{"case_id":"cw","prompt":"p2","truth":"other"}\n'
    )
    spec_path = _write_observe_spec(
        tmp_path,
        {
            "calib": calib_payload,
            "dev": '{"case_id":"d1","prompt":"pd","truth":"math"}\n',
            "frozen_test": '{"case_id":"f1","prompt":"pf","truth":"math"}\n',
        },
    )
    out = tmp_path / "labels"
    state_path = out / "work" / "fake_state.json"
    _write_state(
        state_path,
        {
            "cm": {"decision": "math", "confidence_score": 0.77},
            "cw": {"decision": FALLBACK_DECISION_NAME},
        },
    )
    router, mock = _write_cli_stack(tmp_path, state_path)
    classification_source = _write_classification_source(tmp_path)
    exit_code = main(
        [
            "observe-labels",
            "--spec",
            spec_path,
            "--split",
            "calib",
            "--out",
            str(out),
            "--router-bin",
            router,
            "--mock-backend",
            mock,
            "--classification-config",
            classification_source,
            "--ready-timeout",
            "30",
        ]
    )
    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["cases"] == 2
    assert parsed["observed"] == 2
    assert parsed["label_counts"] == {"math": 1, "other": 1}
    labels_path = Path(parsed["labels"])
    assert labels_path.name == "labels-calib.jsonl"
    rows = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["case_id"]: row["router_label"] for row in rows} == {
        "cm": "math",
        "cw": OTHER_LABEL,
    }
    summary = json.loads(
        (out / "labels-calib-summary.json").read_text(encoding="utf-8")
    )
    assert summary["split"] == "calib"
    assert summary["workload"] == "mmlu-pro"
    assert summary["label_coverage"] == 0.5
    assert summary["seed_config_sha256"].startswith("sha256:")
    assert (out / "seed-config.yaml").is_file()
    assert summary["join_stats"]["missing"] == 0
    # Y2: the summary proves where the seed's classification came from ...
    assert summary["classification_source"] == classification_source
    assert summary["classification_source_sha256"].startswith("sha256:")
    # ... and the written seed config carries the injected sections
    seeded = yaml.safe_load((out / "seed-config.yaml").read_text(encoding="utf-8"))
    catalog = seeded["global"]["model_catalog"]
    assert (
        catalog["embeddings"]["semantic"]["embedding_config"]["model_type"] == "mmbert"
    )
    assert catalog["modules"]["classifier"]["domain"]["fallback_category"] == "other"


def test_observe_labels_cli_allows_frozen_test(tmp_path, capsys):
    """Y1: observe-labels may label frozen_test through the seed config.

    The model_eval verify driver rewrites the frozen split's task categories
    from the C2 labels file, so the cold-start chain observes frozen_test
    like any other split. This is labeling, not tuning: the profile only
    consumes the calib/dev labels, so the quarantine still holds.
    """
    spec_path = _write_observe_spec(
        tmp_path,
        {
            "calib": '{"case_id":"c1","prompt":"p","truth":"math"}\n',
            "dev": '{"case_id":"d1","prompt":"pd","truth":"math"}\n',
            "frozen_test": '{"case_id":"f1","prompt":"pf","truth":"math"}\n',
        },
    )
    out = tmp_path / "labels"
    state_path = out / "work" / "fake_state.json"
    _write_state(
        state_path,
        {"f1": {"decision": "history", "confidence_score": 0.66}},
    )
    router, mock = _write_cli_stack(tmp_path, state_path)
    exit_code = main(
        [
            "observe-labels",
            "--spec",
            spec_path,
            "--split",
            "frozen_test",
            "--out",
            str(out),
            "--router-bin",
            router,
            "--mock-backend",
            mock,
            "--classification-config",
            _write_classification_source(tmp_path),
            "--ready-timeout",
            "30",
        ]
    )
    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["observed"] == 1
    labels = Path(parsed["labels"]).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in labels.splitlines() if line.strip()]
    assert [row["case_id"] for row in rows] == ["f1"]
    assert rows[0]["router_label"] == "history"
    summary = json.loads(
        (tmp_path / "labels" / "labels-frozen_test-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["split"] == "frozen_test"
