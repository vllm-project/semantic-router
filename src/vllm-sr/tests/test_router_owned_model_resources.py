"""The Router owns model deployments and bindings; the CLI asks it for its verdict.

The CLI refuses only what its schema and its own rules refuse, so nothing the
Router reads is turned away before the Router sees it.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml
from cli import router_validation
from cli.commands.runtime_paths import STATE_ROOT_DIR_ENV
from cli.commands.validate import collect_config_errors
from cli.main import main
from cli.models import UserConfig
from cli.parser import parse_user_config_data
from cli.router_validation import ROUTER_BINARY_ENV, RouterVerdict
from click.testing import CliRunner

BASE = {
    "version": "v0.3",
    "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
    "providers": {
        "defaults": {"model": "small"},
        "models": [{"name": "small", "backend_refs": [{"endpoint": "small:8000"}]}],
    },
    "routing": {"modelCards": [{"name": "small"}]},
    "global": {
        "model_catalog": {
            "deployments": {
                "local-vela": {
                    "provider": "model_runtime",
                    "artifact": "vllm-sr/vela3-fast",
                    "device": "cpu",
                },
                "chat-judge": {
                    "provider": "openai_chat",
                    "endpoint": "https://models.example.com/v1",
                    "model": "judge",
                    "auth": {"api_key_env": "JUDGE_KEY"},
                    "timeout_ms": 1500,
                },
                "classifier": {
                    "provider": "classify",
                    "endpoint": "https://classify.example.com",
                    "auth": {
                        "header": "X-Api-Key",
                        "prefix": "",
                        "api_key_file": "/run/secrets/classify",
                    },
                },
                "embedder": {
                    "provider": "openai_embeddings",
                    "endpoint": "https://embed.example.com/v1",
                    "model": "embed-1",
                },
                "ranker": {
                    "provider": "rerank",
                    "endpoint": "https://rank.example.com/v1",
                    "model": "rank-1",
                },
                "d3-remote": {
                    "provider": "systemone",
                    "endpoint": "https://systemone.example.com/v1/systemone",
                    "model": "vllm-sr/d3-ultra",
                    "auth": {"api_key_env": "SYSTEMONE_KEY"},
                    "input": {"overflow": "reject"},
                },
            },
            "system": {"decision_model": {"deployment": "d3-remote"}},
        }
    },
}

DEPRECATED = {
    "code": "deprecated_model_field",
    "field": "global.model_catalog.bindings.pii",
    "message": "contract: the signal decides the contract; write pii: local-vela",
}
UNCALIBRATED = {
    "code": "uncalibrated_threshold",
    "field": "global.model_catalog.modules.prompt_guard.threshold",
    "message": "jailbreak asks d3-remote, which has no calibrated threshold",
}


def _document(*mutations) -> dict:
    document = copy.deepcopy(BASE)
    for mutate in mutations:
        mutate(document)
    return document


def _catalog(document: dict) -> dict:
    return document["global"]["model_catalog"]


def _cli_errors(document: dict) -> list[str]:
    """What the CLI refuses on its own: its schema, typed model and rules."""
    config = parse_user_config_data(
        copy.deepcopy(document), "config.yaml", log_summary=False
    )
    return [str(error) for error in collect_config_errors(config)]


def _cli_rule_errors(document: dict) -> list[str]:
    """The typed model and the CLI's rules, without the generated schema."""
    config = UserConfig.model_validate(copy.deepcopy(document))
    return [str(error) for error in collect_config_errors(config)]


def _signal_bindings(document):
    _catalog(document)["bindings"] = {
        "jailbreak": {"deployment": "chat-judge"},
        "pii": {"deployment": "local-vela", "head": "pii"},
        "domain": {"deployment": "d3-remote"},
        "fact_check": {"deployment": "local-vela"},
        "feedback_detector": {"deployment": "local-vela"},
    }


def _rag_rerank(document):
    document["routing"]["decisions"] = [
        {
            "name": "retrieve",
            "priority": 1,
            "rules": {"operator": "AND", "conditions": []},
            "modelRefs": [{"model": "small"}],
            "plugins": [
                {
                    "type": "rag",
                    "configuration": {
                        "enabled": True,
                        "backend": "vectorstore",
                        "top_k": 10,
                        "rerank": {
                            "deployment": "ranker",
                            "top_k": 3,
                            "layer": 22,
                            "dimension": 768,
                        },
                        "backend_config": {"vector_store_id": "vs-test"},
                    },
                }
            ],
        }
    ]


def _string_bindings(document):
    _catalog(document)["bindings"] = {
        "pii": "local-vela",
        "domain": "d3-remote",
        "jailbreak": "chat-judge",
        "hallucination": "chat-judge",
        "fact_check": "local-vela",
    }
    document["recipes"] = [
        {
            "name": "private",
            "routing": {
                "model_bindings": {
                    "preference": "chat-judge",
                    "reask": "local-vela",
                    "user_feedback": "local-vela",
                    "modality_detector": "local-vela",
                }
            },
        }
    ]


def _connection_block(document):
    deployments = _catalog(document)["deployments"]
    deployments["chat-judge"].update(
        {
            "headers": {"X-Org": "acme"},
            "max_retries": 1,
            "tls": {"ca_file": "/etc/ssl/ca.pem", "server_name": "models.example.com"},
            "limits": {"max_request_bytes": 262144, "max_response_bytes": 4194304},
        }
    )
    deployments["d3-remote"]["capabilities"] = {
        "question_types": ["choice", "noul"],
        "presets": ["pii", "halu"],
    }
    deployments["embedder"]["capabilities"] = {"dimensions": 1024}
    deployments["attached"] = {
        "provider": "model_runtime",
        "endpoint": "https://runtime.example.com",
        "model": "served",
        "auth": {"api_key_env": "RUNTIME_KEY"},
    }
    deployments["workers"] = {
        "provider": "model_runtime",
        "replicas": [
            {"endpoint": "http://worker-a:8100", "model": "first"},
            {"endpoint": "http://worker-b:8100", "model": "second"},
        ],
    }


def _deployment_settings(document):
    deployments = _catalog(document)["deployments"]
    deployments["local-vela"]["operating_point"] = {
        "path": "operating_point.json",
        "sha256": "a" * 64,
    }
    deployments["chat-judge"]["generation"] = {"max_tokens": 64, "temperature": 0}


def _rule_deployments(document):
    document["routing"]["signals"] = {
        "classifiers": [
            {
                "name": "risk",
                "deployment": "chat-judge",
                "labels": ["safe", "unsafe"],
                "instructions": "Is the request risky?",
            }
        ],
        "safety": [
            {
                "name": "harm",
                "deployment": "classifier",
                "threshold": 0.5,
                "hazard": {
                    "deployment": "local-vela",
                    "labels": ["violence", "privacy"],
                    "categories": ["privacy"],
                    "threshold": 0.5,
                },
            }
        ],
        "complexity": [{"name": "hard", "deployment": "local-vela"}],
        "decision": [
            {
                "name": "billing",
                "deployment": "d3-remote",
                "question": {"type": "noul", "instructions": "About billing?"},
            }
        ],
    }
    document["routing"]["decisions"] = [
        {
            "name": "choose",
            "priority": 1,
            "rules": {
                "operator": "AND",
                "conditions": [{"type": "decision", "name": "billing"}],
            },
            "modelRefs": [{"model": "small"}, {"model": "large"}],
            "algorithm": {
                "type": "decision",
                "decision": {
                    "deployment": "local-vela",
                    "instructions": "Which model should answer?",
                },
            },
        }
    ]
    document["providers"]["models"].append(
        {"name": "large", "backend_refs": [{"endpoint": "large:8000"}]}
    )
    document["routing"]["modelCards"].append({"name": "large"})


def _embedding_defaults(document):
    _catalog(document)["embeddings"] = {"default": "embedder", "multimodal": "omni"}
    _catalog(document)["deployments"]["omni"] = {
        "provider": "model_runtime",
        "artifact": "vllm-sr/Vela-1.0-Omni-Nano",
        "device": "cpu",
    }
    document["routing"]["signals"] = {
        "embeddings": [
            {
                "name": "photos",
                "threshold": 0.5,
                "image_candidates": ["./cat.png"],
                "query_modality": "image",
            }
        ]
    }


@pytest.mark.parametrize("scope", ["global", "recipe"])
@pytest.mark.parametrize(
    "key",
    [
        "pii_classifier",
        "pii",
        "hallucination_detector",
        "hallucination",
        "preference",
        "reask",
    ],
)
def test_decision_task_bindings_are_the_routers_to_judge(scope, key):
    binding = {"deployment": "local-vela", "contract": "decision.v1"}

    def bind(document):
        if scope == "global":
            _catalog(document)["bindings"] = {key: binding}
        else:
            document["recipes"] = [
                {"name": "private", "routing": {"model_bindings": {key: binding}}}
            ]

    assert _cli_errors(_document(bind)) == []


def test_the_cli_admits_new_providers_and_signal_keyed_bindings():
    assert _cli_errors(_document(_signal_bindings)) == []


SCHEMA_SHAPES = [
    pytest.param(_string_bindings, id="string bindings"),
    pytest.param(_connection_block, id="connection block"),
    pytest.param(_deployment_settings, id="deployment settings"),
    pytest.param(_rule_deployments, id="rule deployments"),
    pytest.param(_embedding_defaults, id="embedding defaults"),
    pytest.param(_rag_rerank, id="rag rerank deployment"),
]


@pytest.mark.parametrize("mutate", SCHEMA_SHAPES)
def test_the_cli_rules_admit_the_shapes_the_router_reads(mutate):
    assert _cli_rule_errors(_document(mutate)) == []


@pytest.mark.parametrize("mutate", SCHEMA_SHAPES)
def test_the_generated_schema_admits_the_shapes_the_router_reads(mutate):
    assert _cli_errors(_document(mutate)) == []


def test_a_string_binding_names_its_deployment():
    config = UserConfig.model_validate(_document(_string_bindings))
    binding = config.recipes[0].routing.model_bindings["preference"]

    assert binding.deployment == "chat-judge" and binding.contract is None
    assert binding.model_dump() == "chat-judge"


def _write(tmp_path: Path, document: dict) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return config


def _fake_router(tmp_path: Path, report: dict, code: int = 0) -> Path:
    """A Router binary that records how it was asked, then prints its report."""

    script = tmp_path / "router"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "config = open(sys.argv[sys.argv.index('-config') + 1]).read()\n"
        f"with open({str(tmp_path / 'router-call.json')!r}, 'w') as record:\n"
        "    json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'config': config}, record)\n"
        f"print({json.dumps(report)!r})\n"
        f"sys.exit({code})\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _router_call(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "router-call.json").read_text(encoding="utf-8"))


def _validate(*arguments: str):
    return CliRunner().invoke(main, ["config", "validate", *arguments])


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_config_validate_reports_the_router_binarys_refusal_and_warnings(
    monkeypatch, tmp_path
):
    config = _write(tmp_path, BASE)
    refusal = 'global.model_catalog.deployments.chat-judge: unsupported provider "x"'
    router = _fake_router(
        tmp_path,
        {"valid": False, "error": refusal, "warnings": [DEPRECATED]},
        code=1,
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    result = _validate("--config", str(config))

    assert result.exit_code == 1, result.output
    assert "Configuration is valid" not in result.stdout
    assert f"the Router binary {router} refuses it: {refusal}" in result.stderr
    assert "the signal decides the contract" in _flat(result.stderr)
    call = _router_call(tmp_path)
    assert call["argv"] == [
        "-validate-config",
        "-config",
        "/dev/stdin",
        "-gateway",
        "standalone",
    ]
    assert call["config"] == config.read_text(encoding="utf-8")


def test_config_validate_prints_the_router_binarys_warnings(monkeypatch, tmp_path):
    monkeypatch.delenv(STATE_ROOT_DIR_ENV, raising=False)
    stack = tmp_path / "stack"
    (stack / "models").mkdir(parents=True)
    config = _write(stack, BASE)
    router = _fake_router(
        tmp_path, {"valid": True, "warnings": [DEPRECATED, UNCALIBRATED]}
    )

    result = _validate(
        "--config", str(config), "--router-binary", str(router), "--gateway", "extproc"
    )

    assert result.exit_code == 0, result.output
    assert f"Checked by the Router binary {router} as well as the CLI." in (
        result.stdout
    )
    warnings = _flat(result.stderr)
    assert "the signal decides the contract" in warnings
    assert "has no calibrated threshold" in warnings
    call = _router_call(tmp_path)
    assert call["argv"][-2:] == ["-gateway", "extproc"]
    assert Path(call["cwd"]).resolve() == stack.resolve()


def test_the_router_verdict_runs_whatever_the_cli_finds(monkeypatch, tmp_path):
    document = _document()
    document["entrypoints"] = [{"model_names": ["private"], "recipe": "missing"}]
    config = _write(tmp_path, document)
    router = _fake_router(
        tmp_path,
        {"valid": False, "error": "entrypoints: unknown recipe", "warnings": []},
        code=1,
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    result = _validate("--config", str(config))

    assert result.exit_code == 1
    assert "Entrypoint references unknown recipe 'missing'" in result.stderr
    assert f"the Router binary {router} refuses it" in result.stderr
    assert _router_call(tmp_path)["config"] == config.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "bindings, refusal",
    [
        ({"pii": {"deployment": "missing"}}, 'pii: unknown deployment "missing"'),
        (
            {"safety": {"deployment": "local-vela"}},
            "bindings.safety: safety rules name their deployment",
        ),
    ],
    ids=["unknown deployment", "rule-level signal key"],
)
def test_the_router_alone_refuses_a_model_resource(
    monkeypatch, tmp_path, bindings, refusal
):
    def bind(document):
        _catalog(document)["bindings"] = bindings

    document = _document(bind)
    assert _cli_errors(document) == []
    router = _fake_router(
        tmp_path, {"valid": False, "error": refusal, "warnings": []}, code=1
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    result = _validate("--config", str(_write(tmp_path, document)))

    assert result.exit_code == 1
    assert f"refuses it: {refusal}" in result.stderr


@pytest.mark.router_image
def test_a_named_binary_wins_over_the_image_and_an_explicit_image_over_the_env(
    monkeypatch, tmp_path
):
    images = []

    def image(explicit=None):
        images.append(explicit)
        return explicit or "router:present"

    monkeypatch.setattr(router_validation, "validation_image", image)
    monkeypatch.setattr(
        router_validation,
        "validate_with_image",
        lambda _path, name, **_kwargs: RouterVerdict(
            source=f"the Router in {name}", valid=True
        ),
    )
    config = _write(tmp_path, BASE)
    router = _fake_router(tmp_path, {"valid": True, "warnings": []})
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    by_env = _validate("--config", str(config))
    by_image = _validate("--config", str(config), "--image", "router:pinned")
    by_flag = _validate(
        "--config",
        str(config),
        "--image",
        "router:pinned",
        "--router-binary",
        str(router),
    )

    assert f"Checked by the Router binary {router}" in by_env.stdout
    assert "Checked by the Router in router:pinned" in by_image.stdout
    assert f"Checked by the Router binary {router}" in by_flag.stdout
    assert images == ["router:pinned"]


def test_a_router_binary_that_cannot_run_is_reported(monkeypatch, tmp_path):
    config = _write(tmp_path, BASE)
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(tmp_path / "absent-router"))

    result = _validate("--config", str(config))

    assert result.exit_code == 0, result.output
    output = _flat(result.stdout)
    assert "Only the CLI's structural checks ran" in output
    assert f"{ROUTER_BINARY_ENV} names" in output and "not an executable" in output


def test_offline_runs_only_the_structural_checks(monkeypatch, tmp_path):
    config = _write(tmp_path, BASE)
    monkeypatch.setenv(
        ROUTER_BINARY_ENV, str(_fake_router(tmp_path, {"valid": False}, code=1))
    )

    result = _validate("--config", str(config), "--offline")

    assert result.exit_code == 0, result.output
    assert "Only the CLI's structural checks ran (--offline)" in _flat(result.stdout)
    assert not (tmp_path / "router-call.json").exists()


def test_migrate_writes_the_routers_rewrite_of_the_model_fields(monkeypatch, tmp_path):
    source = _write(tmp_path, BASE)
    rewritten = yaml.safe_dump(BASE, sort_keys=False) + "# the Router's rewrite\n"
    kept = {
        "code": "deprecated_model_field",
        "field": 'global.model_catalog.external["rewriter"]',
        "message": "external models are deprecated",
    }
    router = tmp_path / "router"
    router.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        "report = {'valid': True, 'warnings': []}\n"
        "if '-migrate-config' in sys.argv:\n"
        f"    report.update(migrated={rewritten!r}, kept=[{kept!r}])\n"
        "print(json.dumps(report))\n",
        encoding="utf-8",
    )
    router.chmod(0o755)
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    result = CliRunner().invoke(main, ["config", "migrate", "--config", str(source)])

    assert result.exit_code == 0, result.output
    output = tmp_path / "config.migrated.yaml"
    assert output.read_text(encoding="utf-8") == rewritten
    assert "written as the deployments, bindings and rule deployments" in _flat(
        result.stdout
    )
    assert "kept: external models are deprecated" in _flat(result.stderr)


def test_migrate_holds_its_output_to_the_routers_verdict(monkeypatch, tmp_path):
    source = _write(tmp_path, BASE)
    output = tmp_path / "config.migrated.yaml"
    _fake_router(
        tmp_path,
        {"valid": False, "error": "deployments: refused", "warnings": []},
        code=1,
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(tmp_path / "router"))

    refused = CliRunner().invoke(main, ["config", "migrate", "--config", str(source)])

    assert refused.exit_code == 1
    assert "refuses it: deployments: refused" in refused.stderr
    assert "No output was written" in refused.stderr
    assert not output.exists()

    (tmp_path / "router-call.json").unlink()
    offline = CliRunner().invoke(
        main, ["config", "migrate", "--config", str(source), "--offline"]
    )

    assert offline.exit_code == 0, offline.output
    assert "Only the CLI's structural checks ran (--offline)" in _flat(offline.stdout)
    assert not (tmp_path / "router-call.json").exists()
    output.unlink()

    router = _fake_router(tmp_path, {"valid": True, "warnings": [DEPRECATED]})
    accepted = CliRunner().invoke(main, ["config", "migrate", "--config", str(source)])

    assert accepted.exit_code == 0, accepted.output
    assert f"Checked by the Router binary {router}" in accepted.stdout
    assert "the signal decides the contract" in _flat(accepted.stderr)
    assert _router_call(tmp_path)["config"] == output.read_text(encoding="utf-8")
