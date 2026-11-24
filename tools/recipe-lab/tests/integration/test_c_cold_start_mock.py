"""Y1: the cold-start chain runs end to end on the mock stack (V1 items 6/7).

Criterion: from one labeled MCQ jsonl the cold-start script drives the full
generation flow - split -> observe-labels (now including the frozen split,
the Y1 gate) -> profile -> bootstrap -> bank import -> replay ranking ->
auto-picked live ``verify --final`` -> bundle - on the fully mocked stack
(real router with the d3-lite classifier, real model_eval scripts, mock
OpenAI backends), and the ledger, verify document and bundle all agree.

The fixture pins behavior with an answer plan: ``team/model-a`` answers
math/physics, ``team/model-b`` answers law/history, and the seed backend
answers like model-a. So the replay ranking, the live baseline accuracy and
the random mix R are all computable from the split composition alone.

Skips unless ``RECIPE_LAB_ROUTER_BIN`` (real router) and
``RECIPE_LAB_VLLM_SR_BIN`` (config validator) are set; the model_eval
driver additionally needs the optional ``openai`` package. The one
heavyweight test runs the whole chain in one pass - the phases share the
long-lived mock backend and the split, so splitting it across tests would
multiply the router startups without adding coverage.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.contract import API_VERSION
from recipe_lab.model_eval_tasks import render_mcq_prompt
from recipe_lab.observe_lanes import OBSERVE_MODEL_NAME

pytestmark = [pytest.mark.integration]

RECIPE_LAB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RECIPE_LAB_ROOT.parents[1]
SCRIPT = RECIPE_LAB_ROOT / "examples" / "cold-start" / "cold-start.sh"
MODEL_EVAL_DIR = REPO_ROOT / "src" / "training" / "model_eval"

CATEGORIES = ("math", "physics", "law", "history")
PER_CATEGORY = 10
ALIAS_A = "team/model-a"
ALIAS_B = "team/model-b"
SEED_SERVED = "local/seed-backend"
METRIC_KEY = "capability.accuracy"

STEM = {
    "math": "What is 7 times {n}?",
    "physics": "A force of {n} newtons acts on a 1 kg mass; what is the acceleration?",
    "law": "In legal case {n}, which branch of government interprets the statute?",
    "history": "In history module {n}, which document limited the monarchy?",
}
OPTIONS = {
    "math": ["7n", "7+n", "7-n", "7 divided by n"],
    "physics": ["{n} m/s^2", "{n} kg", "{n} joules", "{n} newtons"],
    "law": ["The judiciary", "The executive", "The legislature", "The police"],
    "history": ["Magna Carta", "Code of Hammurabi", "Rosetta Stone", "Domesday Book"],
}


def _fixture_cases() -> list[dict]:
    """Stereotypical MCQ rows: category + expected letter + rendered prompt."""
    rows: list[dict] = []
    for category in CATEGORIES:
        for n in range(1, PER_CATEGORY + 1):
            question = STEM[category].format(n=n)
            options = [option.format(n=n) for option in OPTIONS[category]]
            case_id = f"{category}-{n:02d}"
            rows.append(
                {
                    "case_id": case_id,
                    "prompt": render_mcq_prompt(question, options),
                    "truth": "A",
                    "source_id": case_id,
                    "id": case_id,
                    "source": "recipe-lab-fixture",
                    "category": category,
                    "question": question,
                    "options": options,
                    "answer": "A",
                    "expected": "A",
                }
            )
    return rows


def _answer_plan() -> dict:
    """served model -> category -> correct? (mock backend contract)."""
    plan_a = {"math": True, "physics": True, "law": False, "history": False}
    plan_b = {"math": False, "physics": False, "law": True, "history": True}
    return {ALIAS_A: plan_a, ALIAS_B: plan_b, SEED_SERVED: plan_a}


def _template_fixture_cases() -> list[dict]:
    """Exercise balance's deterministic long_context guard and short fast lane.

    The built-in balance probes use 960001 input bytes for the 240K boundary.
    Put the padding inside the question so the MCQ renderer preserves it.
    The response plan remains fixed by category before splitting.
    """
    cases = _fixture_cases()
    for case in cases:
        if case["category"] in {"law", "history"}:
            case["question"] = (
                "Archive follows.\n" + " " * 960001 + "\n" + case["question"]
            )
            case["prompt"] = render_mcq_prompt(case["question"], case["options"])
    return cases


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _openai_available() -> bool:
    return importlib.util.find_spec("openai") is not None


@pytest.fixture(scope="module")
def router_bin() -> str:
    value = os.environ.get("RECIPE_LAB_ROUTER_BIN", "")
    if not value:
        pytest.skip("RECIPE_LAB_ROUTER_BIN is not set; real router required")
    return value


@pytest.fixture(scope="module")
def validate_bin() -> str:
    value = os.environ.get("RECIPE_LAB_VLLM_SR_BIN", "")
    if not value:
        pytest.skip("RECIPE_LAB_VLLM_SR_BIN is not set; config validator required")
    return value


@pytest.fixture(scope="module")
def recipe_lab_bin(tmp_path_factory) -> Path:
    """A dir with the CLI wrapper and python the cold-start script expects."""
    bin_dir = tmp_path_factory.mktemp("recipe-lab-bin")
    wrapper = bin_dir / "recipe-lab"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec env PYTHONPATH="{RECIPE_LAB_ROOT}" "{sys.executable}" '
        '-m recipe_lab.cli "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    return bin_dir


def _run_recipe_lab(recipe_lab_bin: Path, argv: list[str]) -> None:
    proc = subprocess.run(
        [str(recipe_lab_bin / "recipe-lab"), *argv],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _category_share(split_rows: list[dict], categories: tuple[str, ...]) -> float:
    return sum(row["category"] in categories for row in split_rows) / len(split_rows)


def test_cold_start_model_eval_chain_end_to_end(
    router_bin: str,
    validate_bin: str,
    recipe_lab_bin: Path,
    hf_home: str | None,
    tmp_path: Path,
) -> None:
    """The whole cold-start flow lands a PASS bundle on the mock stack."""
    if not _openai_available():
        pytest.skip("openai is not installed; the model_eval driver needs it")

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _fixture_cases()),
        encoding="utf-8",
    )

    mock_port = _free_port()
    backend_port = _free_port()

    # Deterministic pre-split: the script reruns the identical split, and the
    # spec pins both renderings (cases manifest + model_eval tasks) per split.
    splits_dir = tmp_path / "splits"
    _run_recipe_lab(
        recipe_lab_bin,
        ["split", "--cases", str(cases_path), "--output-dir", str(splits_dir)],
    )
    for split in ("calib", "dev", "frozen_test"):
        _run_recipe_lab(
            recipe_lab_bin,
            [
                "cases",
                "to-model-eval",
                "--source",
                str(splits_dir / f"{split}.jsonl"),
                "--out",
                str(splits_dir / f"tasks-{split}.jsonl"),
            ],
        )

    def split_spec(name: str) -> dict:
        return {
            "manifest": f"splits/{name}.jsonl",
            "content_hash": _sha256(splits_dir / f"{name}.jsonl"),
            "model_eval_tasks": f"splits/tasks-{name}.jsonl",
            "model_eval_tasks_sha256": _sha256(splits_dir / f"tasks-{name}.jsonl"),
        }

    answer_plan_path = tmp_path / "answer-plan.json"
    answer_plan_path.write_text(json.dumps(_answer_plan()), encoding="utf-8")

    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": METRIC_KEY,
            "direction": "maximize",
            "min_improvement_pp": 1.0,
            "min_cases": 2,
            "replay_metric": "routed_accuracy",
        },
        "workloads": [
            {
                "name": "mcq-cold-start",
                "profile": "quick",
                "driver": "model_eval",
                "split_rule": "group by source_id; seed 42; 40/30/30",
                "splits": {
                    "calib": split_spec("calib"),
                    "dev": split_spec("dev"),
                    "frozen_test": split_spec("frozen_test"),
                },
            }
        ],
        # Attribution is not exercised without a baseline candidate; the
        # mapping keeps the spec honest about the intended route.
        "attribution": {
            "oracle_policy": {
                "math": ALIAS_A,
                "physics": ALIAS_A,
                "law": ALIAS_B,
                "history": ALIAS_B,
            }
        },
        "budget": {"rounds": 1, "max_minutes_per_trial": 10},
        "references": {"strong": ALIAS_B, "weak": ALIAS_A},
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                ALIAS_A: {"prompt_per_1m": 1.0, "completion_per_1m": 1.0},
                ALIAS_B: {"prompt_per_1m": 4.0, "completion_per_1m": 4.0},
            },
        },
        "models": [
            {
                "name": ALIAS_A,
                "endpoint": f"http://127.0.0.1:{mock_port}/v1",
                "served_name": ALIAS_A,
            },
            {
                "name": ALIAS_B,
                "endpoint": f"http://127.0.0.1:{mock_port}/v1",
                "served_name": ALIAS_B,
            },
        ],
        "generation": {"temperature": 0, "max_tokens": 64},
        "profile": {"min_label_cases": 1},
        "gates": [],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "RECIPE_LAB_BIN": str(recipe_lab_bin),
            "ROUTER_BIN": router_bin,
            "CASES": str(cases_path),
            "SPEC": str(spec_path),
            "MODEL_EVAL_DIR": str(MODEL_EVAL_DIR),
            # this test pins the LEGACY domain-scaffold chain (the
            # compatibility path); the template chain has its own test
            "MODE": "legacy",
            "LAMBDAS": "0,10",
            "ANSWER_PLAN": str(answer_plan_path),
            "MOCK_PORT": str(mock_port),
            "DRIVER": "model_eval",
            "VERIFY_REPETITIONS": "1",
            "WORK_ROOT": str(tmp_path / "work"),
            "VALIDATE_BIN": validate_bin,
            "BACKEND_PORT": str(backend_port),
        }
    )
    if hf_home:
        env["HF_HOME"] = hf_home

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=3600,
        check=False,
    )
    assert proc.returncode == 0, (
        f"cold-start.sh exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-4000:]}"
    )
    assert "COLD-START-DONE" in proc.stdout

    work_dir = tmp_path / "work"
    (work,) = list(work_dir.glob("cold-start-*"))

    ledger = [
        json.loads(line)
        for line in (work / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Replay ranking: bank-driven, so it is exact regardless of the live
    # mock's answers. lambda 0 routes every dev case to its plan-correct
    # model (requires the classifier to label all stereotypical prompts
    # correctly); lambda 10 collapses onto the cheap model and keeps only
    # the math/physics share.
    replay = [
        record
        for record in ledger
        if record["tier"] == "replay" and record["verdict"] == "RANKED"
    ]
    assert len(replay) == 2
    by_candidate = {Path(r["candidate"]["ref"]).name: r for r in replay}
    assert set(by_candidate) == {
        "candidate-lambda-0.yaml",
        "candidate-lambda-10.yaml",
    }
    lam0 = by_candidate["candidate-lambda-0.yaml"]
    lam10 = by_candidate["candidate-lambda-10.yaml"]
    dev_rows = _read_jsonl(splits_dir / "dev.jsonl")
    expected_lam10 = _category_share(dev_rows, ("math", "physics"))
    assert lam0["metrics"]["routed_accuracy"] == pytest.approx(1.0)
    assert lam10["metrics"]["routed_accuracy"] == pytest.approx(expected_lam10)
    assert lam0["metrics"]["routed_accuracy"] > lam10["metrics"]["routed_accuracy"]

    # The lambda-10 sidecar: one route-team-model-a decision covering every
    # observed label (cost dominates every per-label accuracy). The observed
    # labels are the scaffold's domains - the classifier's labels on calib,
    # which on this stack never surface a clean "history" domain - so
    # coverage is pinned against the scaffold, not the fixture categories.
    scaffold10 = yaml.safe_load(
        (work / "cands" / "scaffold-lambda-10.yaml").read_text(encoding="utf-8")
    )
    # The current result_to_config emits recipes[].signals.domains; the
    # donor-era script emitted routing.signals.domains.
    scaffold_domains = (
        [
            domain
            for recipe in scaffold10.get("recipes") or []
            for domain in ((recipe.get("signals") or {}).get("domains") or [])
        ]
        or ((scaffold10.get("routing") or {}).get("signals") or {}).get("domains")
        or []
    )
    observed_labels = {domain["name"] for domain in scaffold_domains}
    sidecar10 = json.loads(
        (work / "cands" / "candidate-lambda-10.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    decisions10 = sidecar10["decisions"]
    assert len(decisions10) == 1
    assert decisions10[0]["name"] == "route-team-model-a"
    assert decisions10[0]["model_refs"][0] == ALIAS_A
    assert set(decisions10[0]["labels"]) == observed_labels
    assert sidecar10["default_model"] == ALIAS_A

    # The lambda-0 sidecar: each plan-correct model leads at least one
    # label group (accuracy dominates; the exact grouping follows the
    # classifier's labels).
    sidecar0 = json.loads(
        (work / "cands" / "candidate-lambda-0.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    leaders = {decision["model_refs"][0] for decision in sidecar0["decisions"]}
    assert {ALIAS_A, ALIAS_B} <= leaders

    # Candidate hardening (C5): cache and tools off.
    candidate0 = yaml.safe_load(
        (work / "cands" / "candidate-lambda-0.yaml").read_text(encoding="utf-8")
    )
    assert candidate0["global"]["stores"]["response_cache"]["enabled"] is False
    assert candidate0["global"]["integrations"]["tools"]["enabled"] is False

    # Live final verdict on the frozen split: the seed baseline keeps only
    # the math/physics share, the lambda-0 candidate answers everything.
    live = [record for record in ledger if record["tier"] == "live"]
    assert len(live) == 1
    record = live[0]
    assert record["verdict"] == "PASS"
    assert record["final"] is True
    frozen_rows = _read_jsonl(splits_dir / "frozen_test.jsonl")
    expected_baseline = _category_share(frozen_rows, ("math", "physics"))
    assert record["metrics"][METRIC_KEY] == pytest.approx(1.0)
    assert record["notes"].startswith("baseline=")
    notes_baseline = float(record["notes"].split("baseline=", 1)[1].split(";", 1)[0])
    assert notes_baseline == pytest.approx(expected_baseline)

    bundle_dir = work / "bundle"
    for name in ("report.md", "config.yaml", "profile.json", "validate.txt"):
        assert (bundle_dir / name).is_file(), name
    # Y4: the graded config is the generated candidate the router served.
    assert record["candidate"]["config_sha256"] == _sha256(bundle_dir / "config.yaml")

    result = json.loads((work / "verify" / "result.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "PASS"
    assert result["metrics"]["baseline"] == pytest.approx(expected_baseline)
    assert result["metrics"]["candidate"] == pytest.approx(1.0)
    # C8: the random mix R is evaluable (candidate replay shares + both
    # reference arms) and never beats the candidate.
    assert result["random_mix"]["evaluable"] is True
    assert result["random_mix"]["expected_accuracy"] is not None
    assert (
        result["random_mix"]["expected_accuracy"]
        <= result["metrics"]["candidate"] + 1e-9
    )
    assert result["distribution"]
    assert set(result["distribution"]) == {ALIAS_A, ALIAS_B}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.integration
def test_cold_start_template_chain_end_to_end(
    router_bin: str,
    validate_bin: str,
    recipe_lab_bin: Path,
    hf_home: str | None,
    tmp_path: Path,
) -> None:
    """MODE=template (sub-issue-8 §2.2): observe-lanes -> profile --lanes ->
    bindings -> run -> verify --final --lanes -> bundle --bindings, with the
    REAL ``vllm-sr recipe builtin list/init`` (never a stand-in).

    Gated on ``RECIPE_LAB_TEMPLATE_STACK=1``: the observation config and the
    generated candidates serve the mom-v1 bundle's pinned decision model
    (Vela 2.0 0.8B on CPU through model-runtime), which the plain mock stack
    cannot answer for. Run this where that artifact is available; the
    baseline arm is an all-weak bindings derivative built here with the real
    init (the "route everything to the cheapest model" reference).
    """
    if not _openai_available():
        pytest.skip("openai is not installed; the model_eval driver needs it")
    if os.environ.get("RECIPE_LAB_TEMPLATE_STACK") != "1":
        pytest.skip(
            "RECIPE_LAB_TEMPLATE_STACK is not set; the template path needs "
            "the mom-v1 decision model (Vela 2.0 0.8B) servable by "
            "model-runtime"
        )

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in _template_fixture_cases()
        ),
        encoding="utf-8",
    )

    mock_port = _free_port()
    backend_port = _free_port()

    splits_dir = tmp_path / "splits"
    _run_recipe_lab(
        recipe_lab_bin,
        ["split", "--cases", str(cases_path), "--output-dir", str(splits_dir)],
    )
    for split in ("calib", "dev", "frozen_test"):
        _run_recipe_lab(
            recipe_lab_bin,
            [
                "cases",
                "to-model-eval",
                "--source",
                str(splits_dir / f"{split}.jsonl"),
                "--out",
                str(splits_dir / f"tasks-{split}.jsonl"),
            ],
        )

    def split_spec(name: str) -> dict:
        return {
            "manifest": f"splits/{name}.jsonl",
            "content_hash": _sha256(splits_dir / f"{name}.jsonl"),
            "model_eval_tasks": f"splits/tasks-{name}.jsonl",
            "model_eval_tasks_sha256": _sha256(splits_dir / f"tasks-{name}.jsonl"),
        }

    answer_plan_path = tmp_path / "answer-plan.json"
    answer_plan_path.write_text(json.dumps(_answer_plan()), encoding="utf-8")

    # The providers config is the template path's only endpoint/credential
    # source: both candidate models point at the long-lived mock backend and
    # carry the declared capabilities the recipe's candidate_requirements
    # demand. The bundle's decision model is intentionally NOT declared -
    # init copies the pinned deployment.
    providers = {
        "version": "v0.3",
        "listeners": [
            {
                "name": "gw",
                "address": "0.0.0.0",
                "port": _free_port(),
                "timeout": "300s",
            }
        ],
        "providers": {
            "defaults": {"model": ALIAS_A},
            "models": [
                {
                    "name": alias,
                    "provider_model_id": alias,
                    "api_format": "openai",
                    "backend_refs": [
                        {
                            "name": f"{alias}-backend",
                            "endpoint": f"127.0.0.1:{mock_port}",
                            "protocol": "http",
                            "provider": "vllm",
                            "weight": 1,
                        }
                    ],
                }
                for alias in (ALIAS_A, ALIAS_B)
            ],
        },
        "routing": {
            "modelCards": [
                {
                    "name": ALIAS_A,
                    "context_window_size": 262144,
                    "max_output_tokens": 8192,
                    "description": "cheap generalist",
                    "capabilities": ["chat", "text", "vision", "tools"],
                    "modality": "multimodal",
                },
                {
                    "name": ALIAS_B,
                    "context_window_size": 262144,
                    "max_output_tokens": 8192,
                    "description": "strong reasoning generalist",
                    "capabilities": ["chat", "text", "tools"],
                    "modality": "text",
                },
            ]
        },
        "global": {
            "stores": {"response_cache": {"enabled": False, "backend_type": "memory"}}
        },
    }
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(
        yaml.safe_dump(providers, sort_keys=False), encoding="utf-8"
    )

    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": METRIC_KEY,
            "direction": "maximize",
            "min_improvement_pp": 1.0,
            "min_cases": 2,
            "replay_metric": "routed_accuracy",
        },
        "workloads": [
            {
                "name": "mcq-cold-start-template",
                "profile": "quick",
                "driver": "model_eval",
                "split_rule": "group by source_id; seed 42; 40/30/30",
                "splits": {
                    "calib": split_spec("calib"),
                    "dev": split_spec("dev"),
                    "frozen_test": split_spec("frozen_test"),
                },
            }
        ],
        "attribution": {"oracle_policy": {"math": ALIAS_A}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 10},
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                ALIAS_A: {"prompt_per_1m": 1.0, "completion_per_1m": 1.0},
                ALIAS_B: {"prompt_per_1m": 4.0, "completion_per_1m": 4.0},
            },
        },
        "models": [
            {
                "name": ALIAS_A,
                "endpoint": f"http://127.0.0.1:{mock_port}/v1",
                "served_name": ALIAS_A,
                "capabilities": {
                    "vision": True,
                    "tools": True,
                    "context_window": 262144,
                },
            },
            {
                "name": ALIAS_B,
                "endpoint": f"http://127.0.0.1:{mock_port}/v1",
                "served_name": ALIAS_B,
                "capabilities": {"tools": True, "context_window": 262144},
            },
        ],
        "template": {"bundle": "mom-v1", "recipe": "balance"},
        "providers_config": "providers.yaml",
        "bootstrap": {"min_lane_cases": 1, "lambdas": [0]},
        "generation": {"temperature": 0, "max_tokens": 64},
        "profile": {"min_label_cases": 1},
        "gates": [],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    # The baseline arm: every lane bound to the cheap model via the REAL
    # `recipe builtin init` (the "all-weakest" reference config).
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    # This independent reference is initialized outside recipe-lab. Use a
    # local replay store for this ephemeral gateway on the integration host.
    baseline_providers = yaml.safe_load(providers_path.read_text(encoding="utf-8"))
    baseline_providers["global"]["services"] = {
        "router_replay": {"store_backend": "memory"}
    }
    baseline_providers_path = baseline_dir / "providers.yaml"
    baseline_providers_path.write_text(
        yaml.safe_dump(baseline_providers), encoding="utf-8"
    )
    all_weak_bindings = baseline_dir / "bindings.yaml"
    lanes = [
        "long_context",
        "vision",
        "recovery",
        "agentic",
        "facts",
        "frontier",
        "code",
        "hard",
        "standard",
        "fast",
    ]
    all_weak_bindings.write_text(
        yaml.safe_dump(
            {
                lane: (
                    [{"model": ALIAS_A}, {"model": ALIAS_B}]
                    if lane == "frontier"
                    else [{"model": ALIAS_A}]
                )
                for lane in lanes
            }
        ),
        encoding="utf-8",
    )
    baseline_config = baseline_dir / "baseline.yaml"
    init = subprocess.run(
        [
            validate_bin,
            "recipe",
            "builtin",
            "init",
            "balance",
            "--bundle",
            "mom-v1",
            "--config",
            str(baseline_providers_path),
            "--bindings",
            str(all_weak_bindings),
            "--model-name",
            OBSERVE_MODEL_NAME,
            "--output",
            str(baseline_config),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert init.returncode == 0, init.stdout + init.stderr

    env = dict(os.environ)
    env.update(
        {
            "RECIPE_LAB_BIN": str(recipe_lab_bin),
            "ROUTER_BIN": router_bin,
            "CASES": str(cases_path),
            "SPEC": str(spec_path),
            "MODEL_EVAL_DIR": str(MODEL_EVAL_DIR),
            "MODE": "template",
            "VLLM_SR_BIN": validate_bin,
            "LAMBDAS": "0",
            "ANSWER_PLAN": str(answer_plan_path),
            "MOCK_PORT": str(mock_port),
            "DRIVER": "model_eval",
            "BASELINE_CONFIG": str(baseline_config),
            "VERIFY_REPETITIONS": "1",
            "WORK_ROOT": str(tmp_path / "work"),
            "VALIDATE_BIN": validate_bin,
            "BACKEND_PORT": str(backend_port),
        }
    )
    if hf_home:
        env["HF_HOME"] = hf_home

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=3600,
        check=False,
    )
    assert proc.returncode == 0, (
        f"cold-start.sh (MODE=template) exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-4000:]}"
    )
    assert "COLD-START-DONE" in proc.stdout

    _assert_template_outputs(tmp_path, splits_dir)


def _assert_template_outputs(tmp_path: Path, splits_dir: Path) -> None:
    work_dir = tmp_path / "work"
    (work,) = list(work_dir.glob("cold-start-*"))

    # The template chain's artifacts: lanes observations, the bindings
    # sidecar and the bundle's bindings.yaml copy.
    calib_lanes = _read_jsonl(work / "observe-lanes" / "lanes-calib.jsonl")
    assert {"fast", "long_context"} <= {row["lane"] for row in calib_lanes}
    sidecars = list((work / "cands").glob("bindings-lambda-*.provenance.json"))
    assert sidecars, "bindings produced no provenance sidecar"
    sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert sidecar["schema"] == "recipe-lab-bindings-v1"
    assert sidecar["template"]["bundle"] == "mom-v1"
    assert sidecar["measured_lanes"] or sidecar["unmeasured_lanes"]

    bundle_dir = work / "bundle"
    for name in (
        "report.md",
        "config.yaml",
        "bindings.yaml",
        "profile.json",
        "validate.txt",
    ):
        assert (bundle_dir / name).is_file(), name
    provenance = json.loads(
        (bundle_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["template"]["bundle"] == "mom-v1"
    assert _sha256(bundle_dir / "config.yaml") == sidecar["candidate"]["sha256"]
    assert _sha256(Path(sidecar["candidate"]["path"])) == sidecar["candidate"]["sha256"]
    live = [
        record
        for record in _read_jsonl(work / "ledger.jsonl")
        if record["tier"] == "live"
    ]
    assert len(live) == 1
    assert live[0]["candidate"]["config_sha256"] == sidecar["candidate"]["sha256"]
    assert live[0]["final"] and live[0]["verdict"] == "PASS"
    frozen = _read_jsonl(splits_dir / "frozen_test.jsonl")
    result = json.loads((work / "verify" / "result.json").read_text(encoding="utf-8"))
    assert result["metrics"]["baseline"] == pytest.approx(
        _category_share(frozen, ("math", "physics"))
    )
    assert result["metrics"]["candidate"] == pytest.approx(1.0)
    expected_fast = sum(row["category"] in {"math", "physics"} for row in frozen)
    # The fixture has no references section, so the optional random-mix
    # distribution is unavailable. Check the gateway's actual frozen requests.
    for arm, fast_alias, long_alias in (
        ("baseline", ALIAS_A, ALIAS_A),
        ("candidate", ALIAS_A, ALIAS_B),
    ):
        log = work / "verify" / f"{arm}-router.log"
        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.startswith("{")
        ]
        completed = {
            event["request_id"]
            for event in events
            if event.get("msg") == "access"
            and event.get("path") == "/v1/chat/completions"
            and event.get("response_code") == 200
        }
        routed = [
            event for event in events if event.get("msg") == "router_replay_start"
        ]
        assert len(routed) == len(completed) == len(frozen)
        assert {event["request_id"] for event in routed} == completed
        actual = [(event["decision"], event["selected_model"]) for event in routed]
        assert actual.count(("fast", fast_alias)) == expected_fast
        assert actual.count(("long_context", long_alias)) == len(frozen) - expected_fast
    assert provenance["lanes"] is not None
    # The report states the lane coverage honestly.
    report = (bundle_dir / "report.md").read_text(encoding="utf-8")
    assert "Lane coverage" in report
    assert "unmeasured lanes" in report


def test_profile_rejects_truncating_generation(tmp_path: Path) -> None:
    """A generation cap under the mock's truncation floor is output-quality
    poison: the profile refuses instead of ranking on unreliable evidence."""
    if not _openai_available():
        pytest.skip("openai is not installed; the model_eval driver needs it")

    categories = ("math", "law")
    cases: dict[str, list[dict]] = {"calib": [], "dev": []}
    for split, rows in cases.items():
        for index, category in enumerate(categories, 1):
            question = STEM[category].format(n=index)
            options = [option.format(n=index) for option in OPTIONS[category]]
            rows.append(
                {
                    "case_id": f"{category}-{index}",
                    "prompt": render_mcq_prompt(question, options),
                    "truth": "A",
                }
            )
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    port = _free_port()
    mock_proc = subprocess.Popen(
        [
            sys.executable,
            str(RECIPE_LAB_ROOT / "examples" / "smoke" / "mock_backend.py"),
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        spec = {
            "api_version": API_VERSION,
            "objective": {
                "metric_key": METRIC_KEY,
                "direction": "maximize",
                "min_improvement_pct": 1.0,
            },
            "workloads": [
                {
                    "name": "truncation",
                    "profile": "quick",
                    "driver": "model_eval",
                    "split_rule": "hand-written two-split fixture",
                    "splits": {
                        "calib": {
                            "manifest": "calib.jsonl",
                            "content_hash": _sha256(tmp_path / "calib.jsonl"),
                        },
                        "dev": {
                            "manifest": "dev.jsonl",
                            "content_hash": _sha256(tmp_path / "dev.jsonl"),
                        },
                    },
                }
            ],
            "attribution": {"oracle_policy": {"math": "mock/small"}},
            "models": [
                {
                    "name": "mock/small",
                    "endpoint": f"http://127.0.0.1:{port}/v1",
                    "served_name": "mock/small",
                }
            ],
            # Under the mock's TRUNCATION_MAX_TOKENS=16: every completion
            # finishes with "length".
            "generation": {"temperature": 0, "max_tokens": 8},
            "profile": {"min_label_cases": 1},
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        for split in ("calib", "dev"):
            (labels_dir / f"labels-{split}.jsonl").write_text(
                "".join(
                    json.dumps({"case_id": row["case_id"], "router_label": "math"})
                    + "\n"
                    for row in cases[split]
                ),
                encoding="utf-8",
            )

        from tests.integration.conftest import wait_http_ready

        wait_http_ready(f"http://127.0.0.1:{port}/v1/models")

        rc = main(
            [
                "profile",
                "--spec",
                str(spec_path),
                "--labels",
                str(labels_dir),
                "--splits",
                "calib,dev",
                "--out",
                str(tmp_path / "profile"),
                "--model-eval-dir",
                str(MODEL_EVAL_DIR),
                "--python-bin",
                sys.executable,
                "--force",
            ]
        )
        assert rc == 2
    finally:
        mock_proc.terminate()
        try:
            mock_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock_proc.kill()
