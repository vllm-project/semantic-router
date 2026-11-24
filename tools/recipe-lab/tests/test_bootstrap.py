"""C4+C5 tests: bootstrap candidates from the C3 profile.

C4 drives the REAL result_to_config.py (PyYAML is its only dependency, and
the maintained config it reads lives in this checkout), so the scaffold
ranking assertions cover the actual upstream script - including the
§3.2-5 silent lambda-degradation warning. C5 completion is exercised both
through the real scaffolds and through a handcrafted golden scaffold with
2 models x 3 labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml
from recipe_lab.bootstrap import (
    FALLBACK_DECISION_NAME,
    complete_candidate,
    lambda_tag,
    parse_lambdas,
    run_bootstrap,
    validate_config,
)
from recipe_lab.cli import main
from recipe_lab.contract import API_VERSION, ContractError, ExperimentSpec
from recipe_lab.profile import PROFILE_SCHEMA, results_dirname
from recipe_lab.runner import RunnerEnv
from tests.test_observe_labels import (
    FAKE_ROUTER_TEXT,
    _free_port,
    _write_state,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MODEL_EVAL = REPO_ROOT / "src" / "training" / "model_eval"

SMALL = "mock/small"
LARGE = "mock/large"
SMALL_ENDPOINT = "http://127.0.0.1:8001/v1"
LARGE_ENDPOINT = "http://127.0.0.1:8002/v1"

# Pure accuracies (lambda 0). The history margin is 0.02, so lambda 0.1
# flips history's first place from LARGE to SMALL.
ACCURACY = {
    SMALL: {"math": 0.6, "business": 0.7, "history": 0.78},
    LARGE: {"math": 0.9, "business": 0.65, "history": 0.8},
}
TOKENS = {SMALL: 100.0, LARGE: 500.0}
LABEL_COUNTS = {"math": 40, "business": 30, "history": 20}

# Y2: the shared classification source sections. A minimal stand-in for the
# maintained config/config.yaml classification machinery (canonical v0.5
# catalogs carry no classifier modules; the embeddings block names the
# deployment that embeds for the shared machinery).
CLASSIFICATION = {"embeddings": {"default": {"deployment": "primary"}}}

# The dev split the Y2 label-agreement check observes.
DEV_PAYLOAD = (
    '{"case_id":"d1","prompt":"dev prompt one","truth":"math"}\n'
    '{"case_id":"d2","prompt":"dev prompt two","truth":"business"}\n'
)


def _write_classification_source(tmp_path: Path) -> str:
    path = tmp_path / "classification-source.yaml"
    path.write_text(
        yaml.safe_dump({"global": {"model_catalog": CLASSIFICATION}}),
        encoding="utf-8",
    )
    return str(path)


def _dev_cases() -> list[dict]:
    return [json.loads(line) for line in DEV_PAYLOAD.splitlines() if line]


def _write_fake_stack(tmp_path: Path) -> tuple[str, str]:
    router = tmp_path / "fake_router.py"
    router.write_text(FAKE_ROUTER_TEXT)
    router.chmod(0o755)
    mock = tmp_path / "mock_backend.py"
    mock.write_text("import time\ntime.sleep(300)\n")
    return str(router), str(mock)


def _label_check_env(tmp_path: Path, work_dir: str = "label-check") -> RunnerEnv:
    router, mock = _write_fake_stack(tmp_path)
    return RunnerEnv(
        router_bin=router,
        mock_backend_script=mock,
        python_bin=sys.executable,
        work_dir=tmp_path / work_dir,
        backend_port=_free_port(),
        ready_timeout_s=30,
    )


def _write_spec(
    tmp_path: Path,
    *,
    priced_models: tuple[str, ...] = (SMALL, LARGE),
    dev_split: bool = True,
) -> str:
    spec: dict = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 2.0,
        },
        "workloads": [
            {
                "name": "office",
                "profile": "quick",
                "splits": {
                    "calib": {
                        "manifest": "calib.jsonl",
                        "content_hash": "sha256:" + "0" * 64,
                    }
                },
            }
        ],
        "attribution": {"oracle_policy": {"math": LARGE}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 5},
        "models": [
            {"name": SMALL, "endpoint": SMALL_ENDPOINT},
            {"name": LARGE, "endpoint": LARGE_ENDPOINT},
        ],
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                name: {"prompt_per_1m": 0.1, "completion_per_1m": 0.4}
                for name in priced_models
            },
        },
    }
    if dev_split:
        # Y2: bootstrap observes the dev split for the label-agreement check
        # (the manifest's hash is verified like any other split's).
        (tmp_path / "dev.jsonl").write_text(DEV_PAYLOAD, encoding="utf-8")
        spec["workloads"][0]["splits"]["dev"] = {
            "manifest": "dev.jsonl",
            "content_hash": "sha256:"
            + hashlib.sha256(DEV_PAYLOAD.encode("utf-8")).hexdigest(),
        }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _load_spec(path: str) -> ExperimentSpec:
    return ExperimentSpec.from_yaml(path)


def _write_profile(tmp_path: Path, *, with_tokens: bool = True) -> Path:
    profile_dir = tmp_path / "profile"
    for alias in (SMALL, LARGE):
        rdir = profile_dir / "calib" / "results" / results_dirname(alias)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "analysis.json").write_text(
            json.dumps({"category_accuracy": ACCURACY[alias]}), encoding="utf-8"
        )
        if with_tokens:
            token_dir = profile_dir / "calib" / "token_costs"
            token_dir.mkdir(parents=True, exist_ok=True)
            (token_dir / f"{quote(alias, safe='')}.json").write_text(
                json.dumps(dict.fromkeys(ACCURACY[alias], TOKENS[alias])),
                encoding="utf-8",
            )
    stats = {
        alias: {
            "labels": {
                label: {"n": count, "accuracy": ACCURACY[alias][label]}
                for label, count in LABEL_COUNTS.items()
            },
            "total": {"n": sum(LABEL_COUNTS.values())},
        }
        for alias in (SMALL, LARGE)
    }
    profile = {
        "schema": PROFILE_SCHEMA,
        "splits": ["calib"],
        "models": [SMALL, LARGE],
        "label_merges": {},
        "uncovered_cases": {},
        "quality": {},
        "model_split_stats": {"calib": stats},
        "inputs": {},
    }
    (profile_dir / "profile.json").write_text(
        json.dumps(profile, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return profile_dir


def _write_validator(tmp_path: Path, *, ok: bool) -> str:
    body = "#!/bin/sh\nexit 0\n" if ok else "#!/bin/sh\necho 'bad config' >&2\nexit 1\n"
    validator = tmp_path / f"validator-{'ok' if ok else 'bad'}.sh"
    validator.write_text(body, encoding="utf-8")
    validator.chmod(0o755)
    return str(validator)


def _score(model: str, score: float, reasoning: bool = False) -> dict:
    return {"model": model, "score": score, "use_reasoning": reasoning}


def _golden_scaffold() -> dict:
    """2 models x 3 labels, lambda-adjusted scores, pure-accuracy default.

    First places: math -> LARGE (share 40), business+other -> SMALL
    (share 30+20). The lambda-adjusted mean flips the default model from
    the scaffold's pure-accuracy LARGE to SMALL.
    """
    return {
        "version": "v0.5",
        "listeners": [],
        "providers": {
            "defaults": {"model": LARGE, "reasoning_effort": "medium"},
            "models": [
                {
                    "name": SMALL,
                    "provider_model_id": SMALL,
                    "api_format": "openai",
                    "external_model_ids": {"openai": SMALL},
                    "capabilities": ["chat"],
                    "backend_refs": [
                        {
                            "name": f"{SMALL}-backend",
                            "endpoint": "127.0.0.1:8000",
                            "protocol": "http",
                            "provider": "vllm",
                            "weight": 1,
                        }
                    ],
                },
                {
                    "name": LARGE,
                    "provider_model_id": LARGE,
                    "api_format": "openai",
                    "external_model_ids": {"openai": LARGE},
                    "capabilities": ["chat"],
                    "backend_refs": [
                        {
                            "name": f"{LARGE}-backend",
                            "endpoint": "127.0.0.1:8000",
                            "protocol": "http",
                            "provider": "vllm",
                            "weight": 1,
                        }
                    ],
                },
            ],
        },
        "recipes": [
            {
                "name": "default",
                "signals": {
                    "domains": [
                        {
                            "name": "math",
                            "description": "d",
                            "mmlu_categories": ["math"],
                            "model_scores": [
                                _score(LARGE, 0.80, True),
                                _score(SMALL, 0.58),
                            ],
                        },
                        {
                            "name": "business",
                            "description": "d",
                            "mmlu_categories": ["business"],
                            "model_scores": [_score(SMALL, 0.68), _score(LARGE, 0.55)],
                        },
                        {
                            "name": "other",
                            "description": "d",
                            "mmlu_categories": ["other"],
                            "model_scores": [_score(SMALL, 0.67), _score(LARGE, 0.48)],
                        },
                    ]
                },
                "decisions": [],
            }
        ],
        "global": {
            "model_catalog": {
                "embeddings": {"semantic": {"mmbert_model_path": "x"}},
                "modules": {"prompt_guard": {"enabled": True}},
            },
        },
    }


def _spec(
    tmp_path: Path, *, priced_models: tuple[str, ...] = (SMALL, LARGE)
) -> ExperimentSpec:
    return _load_spec(_write_spec(tmp_path, priced_models=priced_models))


def test_lambda_tag_and_parse():
    assert lambda_tag(0.0) == "0"
    assert lambda_tag(0.1) == "0.1"
    assert lambda_tag(0.25) == "0.25"
    assert parse_lambdas("0,0.1,0.2,0.3") == [0.0, 0.1, 0.2, 0.3]
    # duplicate tags collapse, order preserved
    assert parse_lambdas("0,0.0,0.1") == [0.0, 0.1]
    with pytest.raises(ContractError):
        parse_lambdas("abc")
    with pytest.raises(ContractError):
        parse_lambdas("-1")
    with pytest.raises(ContractError):
        parse_lambdas("")


def test_complete_candidate_golden(tmp_path):
    # Only SMALL is priced: the unpriced model must never get a fake zero.
    spec = _spec(tmp_path, priced_models=(SMALL,))
    scaffold = _golden_scaffold()
    # The scaffold carries its own catalog copies (embeddings/modules above,
    # deployments/system added here): under Y2 they all lose to the shared
    # classification source and the seed's pinning.
    scaffold["global"]["model_catalog"]["deployments"] = {
        "sneaky": {"provider": "model_runtime", "artifact": "sneaky"}
    }
    scaffold["global"]["model_catalog"]["system"] = {
        "decision_model": {"deployment": "sneaky"}
    }
    config, selection = complete_candidate(
        scaffold=scaffold,
        spec=spec,
        label_counts={"math": 40, "business": 30, "other": 20},
        lam=0.1,
        classification=CLASSIFICATION,
    )

    # Default model: best mean over the lambda-adjusted domain scores -
    # NOT the scaffold's pure-accuracy default.
    assert config["providers"]["defaults"]["model"] == SMALL
    assert config["providers"]["defaults"]["reasoning_effort"] == "medium"
    assert selection["default_model"] == SMALL

    # Decisions grouped by first place, priorities by calib traffic share
    # (business+other=50 beats math=40), fallback last.
    decisions = config["recipes"][0]["decisions"]
    assert [d["name"] for d in decisions] == [
        "route-mock-small",
        "route-mock-large",
        FALLBACK_DECISION_NAME,
    ]
    merged = decisions[0]
    assert merged["priority"] == 2
    assert merged["rules"] == {
        "operator": "OR",
        "conditions": [
            {"type": "domain", "name": "business"},
            {"type": "domain", "name": "other"},
        ],
    }
    # modelRefs follow the first label's domain order (static selector
    # picks by the matched domain's scores; refs fix exact ties).
    assert merged["modelRefs"] == [
        {"model": SMALL, "use_reasoning": False},
        {"model": LARGE, "use_reasoning": False},
    ]
    solo = decisions[1]
    assert solo["priority"] == 1
    assert solo["rules"]["conditions"] == [{"type": "domain", "name": "math"}]
    assert solo["modelRefs"] == [
        {"model": LARGE, "use_reasoning": True},
        {"model": SMALL, "use_reasoning": False},
    ]
    fallback = decisions[2]
    assert fallback["priority"] == 0
    assert fallback["tier"] == 2
    assert fallback["modelRefs"] == [{"model": SMALL, "use_reasoning": False}]
    for decision in decisions:
        assert any(p["type"] == "router_replay" for p in decision["plugins"])

    # Contract endpoints replace the shared scaffold endpoint; pricing
    # follows cost_basis (unpriced models never get a fake zero).
    models = {m["name"]: m for m in config["providers"]["models"]}
    assert models[SMALL]["backend_refs"] == [
        {
            "name": f"{SMALL}-backend",
            "endpoint": "127.0.0.1:8001",
            "protocol": "http",
            "provider": "vllm",
            "weight": 1,
        }
    ]
    assert models[LARGE]["backend_refs"][0]["endpoint"] == "127.0.0.1:8002"
    assert models[SMALL]["pricing"] == {"prompt_per_1m": 0.1, "completion_per_1m": 0.4}
    assert "pricing" not in models[LARGE]
<<<<<<< HEAD
    # The models' card facts ride their providers.models[] entries verbatim
    # (the removed routing.modelCards form is gone).
=======
    # The scaffold's model cards merge onto their providers.models[] entries
    # (the modality keyword is a removed form and stays out).
>>>>>>> 1946016ca (feat(recipe-lab): emit canonical v0.5 seed and candidate configs)
    assert models[SMALL]["capabilities"] == ["chat"]
    assert "modality" not in models[LARGE]

    # Listeners, entrypoints and the Y2 shared classification catalog merge.
    assert config["listeners"] == [
        {"name": "http-gateway", "address": "0.0.0.0", "port": 8899, "timeout": "300s"}
    ]
    assert config["entrypoints"] == [
        {"model_names": ["vllm-sr/auto"], "recipe": "default"}
    ]
    catalog = config["global"]["model_catalog"]
    # Y2: the classification sections come from the shared source, the seed's
    # deployments/system pinning survives, and the scaffold's own copies
    # (embeddings/modules, plus the sneaky deployments/system) all lose.
    assert catalog["deployments"] == {
        "primary": {"provider": "model_runtime", "artifact": "vllm-sr/d3-lite"}
    }
    assert catalog["system"] == {"decision_model": {"deployment": "primary"}}
    assert catalog["embeddings"] == CLASSIFICATION["embeddings"]

    # Cache and tools run on their plugin-based v0.5 defaults.
    assert config["global"]["stores"]["response_cache"] == {"backend_type": "memory"}
    assert selection["disabled"]["response_cache"]
    assert selection["disabled"]["tools"]


def test_complete_candidate_decision_name_collision(tmp_path):
    spec = _spec(tmp_path)
    scaffold = _golden_scaffold()
    # "x/y" and "x-y" are distinct contract aliases but collide after the
    # route-<alias> sanitization.
    scaffold["providers"]["models"] = [
        {"name": "x/y", "backend_refs": []},
        {"name": "x-y", "backend_refs": []},
    ]
    scaffold["recipes"][0]["signals"]["domains"] = [
        {
            "name": "math",
            "description": "d",
            "mmlu_categories": ["math"],
            "model_scores": [_score("x/y", 0.9), _score("x-y", 0.1)],
        },
        {
            "name": "business",
            "description": "d",
            "mmlu_categories": ["business"],
            "model_scores": [_score("x-y", 0.9), _score("x/y", 0.1)],
        },
    ]
    with pytest.raises(ContractError, match="collision"):
        complete_candidate(
            scaffold=scaffold,
            spec=spec,
            label_counts={"math": 10},
            lam=0.0,
            classification=CLASSIFICATION,
        )


def test_bootstrap_runs_real_result_to_config(tmp_path):
    profile_dir = _write_profile(tmp_path)
    out = tmp_path / "cands"
    classification_path = _write_classification_source(tmp_path)
    summary = run_bootstrap(
        spec=_spec(tmp_path),
        profile_dir=profile_dir,
        model_eval_dir=REAL_MODEL_EVAL,
        lambdas="0,0.1",
        out_dir=out,
        python_bin=sys.executable,
        validate_bin=_write_validator(tmp_path, ok=True),
        classification=CLASSIFICATION,
        classification_path=classification_path,
        dev_cases=_dev_cases(),
        env=_label_check_env(tmp_path),
    )
    assert [c["lambda"] for c in summary["candidates"]] == [0.0, 0.1]
    # Y2: the seed config the label check observed and its source identity.
    assert summary["seed_config"].endswith("seed-config.yaml")
    assert (out / "seed-config.yaml").is_file()
    assert summary["classification_source"] == classification_path
    assert summary["classification_source_sha256"].startswith("sha256:")
    assert all(
        candidate["label_agreement"] == 1.0 for candidate in summary["candidates"]
    )

    # C4: the real script's cost-aware ranking - lambda 0.1 flips history.
    # The script emits recipes[].signals.domains (the canonical layout).
    def scaffold_domains(scaffold: dict) -> list[dict]:
        return [
            domain
            for recipe in scaffold.get("recipes") or []
            for domain in ((recipe.get("signals") or {}).get("domains") or [])
        ]

    def top_model(scaffold: dict, label: str) -> str:
        by_name = {d["name"]: d for d in scaffold_domains(scaffold)}
        return by_name[label]["model_scores"][0]["model"]

    scaffold0 = yaml.safe_load(
        (out / "scaffold-lambda-0.yaml").read_text(encoding="utf-8")
    )
    scaffold1 = yaml.safe_load(
        (out / "scaffold-lambda-0.1.yaml").read_text(encoding="utf-8")
    )
    assert top_model(scaffold0, "history") == LARGE
    assert top_model(scaffold1, "history") == SMALL
    # math's margin (0.3) survives lambda 0.1.
    assert top_model(scaffold1, "math") == LARGE

    # C5: decisions per first-place group (+ fallback), traffic order.
    candidate0 = _load_candidate(out, 0)
    names0 = [d["name"] for d in candidate0["recipes"][0]["decisions"]]
    assert names0 == ["route-mock-large", "route-mock-small", FALLBACK_DECISION_NAME]
    candidate1 = _load_candidate(out, 1)
    names1 = [d["name"] for d in candidate1["recipes"][0]["decisions"]]
    assert names1 == ["route-mock-small", "route-mock-large", FALLBACK_DECISION_NAME]
    # lambda-adjusted mean default (large wins under both lambdas here).
    assert candidate1["providers"]["defaults"]["model"] == LARGE

    # Cache/tools on their plugin-based defaults; endpoints from the contract.
    assert candidate1["global"]["stores"]["response_cache"] == {
        "backend_type": "memory"
    }
    refs = {
        m["name"]: m["backend_refs"][0]["endpoint"]
        for m in candidate1["providers"]["models"]
    }
    assert refs == {SMALL: "127.0.0.1:8001", LARGE: "127.0.0.1:8002"}
    pricing = {m["name"]: m.get("pricing") for m in candidate1["providers"]["models"]}
    assert pricing[SMALL] == {"prompt_per_1m": 0.1, "completion_per_1m": 0.4}
    assert pricing[LARGE] == {"prompt_per_1m": 0.1, "completion_per_1m": 0.4}
    # Y2: the candidate classifies with the shared source's sections, not the
    # scaffold's own catalog copies.
    catalog1 = candidate1["global"]["model_catalog"]
    assert catalog1["embeddings"] == CLASSIFICATION["embeddings"]
    assert catalog1["system"] == {"decision_model": {"deployment": "primary"}}

    # Sidecar: proposer, lambda, per-label table, hashes, command.
    sidecar = json.loads(
        (out / "candidate-lambda-0.1.provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["proposer"] == "bootstrap"
    assert sidecar["lambda"] == 0.1
    assert sidecar["scaffold"]["sha256"].startswith("sha256:")
    assert sidecar["candidate"]["sha256"].startswith("sha256:")
    assert sidecar["scaffold"]["script"]["sha256"].startswith("sha256:")
    assert "--cost-lambda" in " ".join(sidecar["scaffold"]["command"])
    assert sidecar["labels"]["history"]["selected_model"] == SMALL
    assert sidecar["labels"]["history"]["n"] == 20
    assert sidecar["labels"]["history"]["accuracy"] == pytest.approx(0.78)
    assert sidecar["default_model"] == LARGE
    # profile identity lands in the sidecar
    assert sidecar["profile"]["sha256"].startswith("sha256:")
    # Y2 ③: the classification source identity and the candidate's projection
    # hash land in the sidecar (bundle verifies the packaged config against it).
    assert sidecar["classification"]["path"] == classification_path
    assert sidecar["classification"]["sha256"].startswith("sha256:")
    assert sidecar["classification"]["projection_sha256"].startswith("sha256:")
    assert sidecar["dev_label_agreement"] == {
        "cases": 2,
        "mismatched_count": 0,
        "mismatched": [],
        "agreement": 1.0,
    }

    # Deterministic: a rerun reproduces identical candidate bytes.
    first = (out / "candidate-lambda-0.yaml").read_bytes()
    run_bootstrap(
        spec=_spec(tmp_path),
        profile_dir=profile_dir,
        model_eval_dir=REAL_MODEL_EVAL,
        lambdas="0,0.1",
        out_dir=out,
        python_bin=sys.executable,
        validate_bin=_write_validator(tmp_path, ok=True),
        classification=CLASSIFICATION,
        classification_path=classification_path,
        dev_cases=_dev_cases(),
        env=_label_check_env(tmp_path, work_dir="label-check-rerun"),
    )
    assert (out / "candidate-lambda-0.yaml").read_bytes() == first


def _load_candidate(out: Path, index: int) -> dict:
    path = out / f"candidate-lambda-{lambda_tag([0.0, 0.1][index])}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_bootstrap_rejects_silent_lambda_degradation(tmp_path):
    profile_dir = _write_profile(tmp_path)
    for token_file in (profile_dir / "calib" / "token_costs").iterdir():
        token_file.unlink()
    out = tmp_path / "cands"
    with pytest.raises(ContractError, match="ignored --cost-lambda"):
        run_bootstrap(
            spec=_spec(tmp_path),
            profile_dir=profile_dir,
            model_eval_dir=REAL_MODEL_EVAL,
            lambdas="0.1",
            out_dir=out,
            python_bin=sys.executable,
            validate_bin=_write_validator(tmp_path, ok=True),
            classification=CLASSIFICATION,
            classification_path=_write_classification_source(tmp_path),
            dev_cases=_dev_cases(),
            env=_label_check_env(tmp_path),
        )
    assert not (out / "candidate-lambda-0.1.yaml").exists()


def test_bootstrap_rejects_rejected_candidate(tmp_path):
    profile_dir = _write_profile(tmp_path)
    out = tmp_path / "cands"
    with pytest.raises(ContractError, match="rejected by config validate"):
        run_bootstrap(
            spec=_spec(tmp_path),
            profile_dir=profile_dir,
            model_eval_dir=REAL_MODEL_EVAL,
            lambdas="0",
            out_dir=out,
            python_bin=sys.executable,
            validate_bin=_write_validator(tmp_path, ok=False),
            classification=CLASSIFICATION,
            classification_path=_write_classification_source(tmp_path),
            dev_cases=_dev_cases(),
            env=_label_check_env(tmp_path),
        )
    # A rejected candidate never lands in the output directory.
    assert not list(out.glob("candidate-lambda-*.yaml"))
    assert not list(out.glob("*.provenance.json"))


def test_bootstrap_label_agreement_rejects_divergent_candidate(tmp_path):
    # Y2 ②: the seed's stack (no state file) catch-alls every dev case, while
    # the candidate's stack routes d1 to biology - the profile's grouping
    # premise does not hold, so the command fails and ships no sidecar.
    profile_dir = _write_profile(tmp_path)
    out = tmp_path / "cands"
    _write_state(
        tmp_path / "label-check" / "lambda-0" / "fake_state.json",
        {"d1": {"decision": "biology"}},
    )
    with pytest.raises(ContractError, match=r"disagrees with the observe seed on 1/2"):
        run_bootstrap(
            spec=_spec(tmp_path),
            profile_dir=profile_dir,
            model_eval_dir=REAL_MODEL_EVAL,
            lambdas="0",
            out_dir=out,
            python_bin=sys.executable,
            validate_bin=_write_validator(tmp_path, ok=True),
            classification=CLASSIFICATION,
            classification_path=_write_classification_source(tmp_path),
            dev_cases=_dev_cases(),
            env=_label_check_env(tmp_path),
        )
    assert (out / "candidate-lambda-0.yaml").is_file()
    assert not list(out.glob("*.provenance.json"))


def test_bootstrap_cli_end_to_end(tmp_path, capsys):
    profile_dir = _write_profile(tmp_path)
    out = tmp_path / "cands"
    router, mock = _write_fake_stack(tmp_path)
    classification_source = _write_classification_source(tmp_path)
    rc = main(
        [
            "bootstrap",
            "--legacy-domain-template",
            "--spec",
            _write_spec(tmp_path),
            "--profile",
            str(profile_dir),
            "--model-eval-dir",
            str(REAL_MODEL_EVAL),
            "--lambdas",
            "0",
            "--out",
            str(out),
            "--validate-bin",
            _write_validator(tmp_path, ok=True),
            "--classification-config",
            classification_source,
            "--router-bin",
            router,
            "--mock-backend",
            mock,
            "--backend-port",
            str(_free_port()),
            "--ready-timeout",
            "30",
        ]
    )
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["candidates"]) == 1
    assert parsed["candidates"][0]["candidate"].endswith("candidate-lambda-0.yaml")
    assert parsed["candidates"][0]["label_agreement"] == 1.0
    assert parsed["classification_source"] == classification_source
    assert parsed["seed_config"].endswith("seed-config.yaml")
    assert (out / "scaffold-lambda-0.yaml").is_file()
    assert (out / "candidate-lambda-0.yaml").is_file()
    assert (out / "candidate-lambda-0.provenance.json").is_file()
    sidecar = json.loads(
        (out / "candidate-lambda-0.provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["classification"]["path"] == classification_source
    assert sidecar["classification"]["projection_sha256"].startswith("sha256:")
    assert sidecar["dev_label_agreement"]["agreement"] == 1.0


def test_bootstrap_cli_refuses_missing_profile(tmp_path, capsys):
    rc = main(
        [
            "bootstrap",
            "--legacy-domain-template",
            "--spec",
            _write_spec(tmp_path),
            "--profile",
            str(tmp_path / "nope"),
            "--model-eval-dir",
            str(REAL_MODEL_EVAL),
            "--classification-config",
            _write_classification_source(tmp_path),
            "--router-bin",
            "unused-router",
            "--mock-backend",
            "unused-mock",
            "--out",
            str(tmp_path / "cands"),
        ]
    )
    assert rc == 2
    assert "profile not found" in capsys.readouterr().err


def test_bootstrap_cli_refuses_negative_lambda(tmp_path, capsys):
    rc = main(
        [
            "bootstrap",
            "--legacy-domain-template",
            "--spec",
            _write_spec(tmp_path),
            "--profile",
            str(_write_profile(tmp_path)),
            "--model-eval-dir",
            str(REAL_MODEL_EVAL),
            "--lambdas",
            "-0.5",
            "--out",
            str(tmp_path / "cands"),
            "--validate-bin",
            _write_validator(tmp_path, ok=True),
            "--classification-config",
            _write_classification_source(tmp_path),
            "--router-bin",
            "unused-router",
            "--mock-backend",
            "unused-mock",
        ]
    )
    assert rc == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_bootstrap_cli_requires_dev_split(tmp_path, capsys):
    # Y2 ②: without a dev split there is nothing to verify the labels against.
    rc = main(
        [
            "bootstrap",
            "--legacy-domain-template",
            "--spec",
            _write_spec(tmp_path, dev_split=False),
            "--profile",
            str(_write_profile(tmp_path)),
            "--model-eval-dir",
            str(REAL_MODEL_EVAL),
            "--lambdas",
            "0",
            "--out",
            str(tmp_path / "cands"),
            "--validate-bin",
            _write_validator(tmp_path, ok=True),
            "--classification-config",
            _write_classification_source(tmp_path),
            "--router-bin",
            "unused-router",
            "--mock-backend",
            "unused-mock",
        ]
    )
    assert rc == 2
    assert "declares no dev split" in capsys.readouterr().err


def _real_validator() -> str | None:
    """The real config validator, or None when this checkout has none."""
    return os.environ.get("RECIPE_LAB_VLLM_SR_BIN") or shutil.which("vllm-sr")


def test_bootstrap_candidates_pass_real_validate(tmp_path):
    """C5: with the real vllm-sr validator the completed candidates are
    canonical configs, not just fixtures that pass a stub (skipped without
    the CLI; the cold-start chain exercises the same gate on every phase)."""
    validator = _real_validator()
    if validator is None:
        pytest.skip("vllm-sr CLI not available; real config validate required")
    out = tmp_path / "cands"
    summary = run_bootstrap(
        spec=_spec(tmp_path),
        profile_dir=_write_profile(tmp_path),
        model_eval_dir=REAL_MODEL_EVAL,
        lambdas="0",
        out_dir=out,
        python_bin=sys.executable,
        validate_bin=validator,
        classification=CLASSIFICATION,
        classification_path=_write_classification_source(tmp_path),
        dev_cases=_dev_cases(),
        env=_label_check_env(tmp_path),
    )
    assert [candidate["lambda"] for candidate in summary["candidates"]] == [0.0]
    # validate_config returns None only when the validator accepted the file;
    # re-check the emitted candidate to pin the contract, not the side effect.
    assert validate_config(out / "candidate-lambda-0.yaml", validator) is None
