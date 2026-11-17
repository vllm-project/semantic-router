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
from cli.gateway_mode import GATEWAY_ENV
from cli.main import main
from cli.models import UserConfig
from cli.parser import ConfigParseError, parse_user_config_data
from cli.router_validation import ROUTER_BINARY_ENV, RouterVerdict
from click.testing import CliRunner

BASE = {
    "version": "v0.5",
    "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
    "providers": {
        "defaults": {"model": "small"},
        "models": [{"name": "small", "backend_refs": [{"endpoint": "small:8000"}]}],
    },
    "recipes": [{"name": "default"}],
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
                "embedder": {
                    "provider": "openai_embeddings",
                    "endpoint": "https://embed.example.com/v1",
                    "model": "embed-1",
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

RENAMED = {
    "code": "renamed_signal_type",
    "field": "recipes[default].signals.decision",
    "message": "the decision signal is named systemone; run `vllm-sr config "
    "migrate` (recipes[default].signals.systemone)",
}
UNCALIBRATED = {
    "code": "uncalibrated_preset",
    "field": "recipes[default].signals.jailbreak[attack]",
    "message": 'resolves to deployment "d3-remote", whose model has no calibrated '
    "threshold for the jailbreak preset",
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
    config = parse_user_config_data(copy.deepcopy(document), log_summary=False)
    return [str(error) for error in collect_config_errors(config)]


def _cli_rule_errors(document: dict) -> list[str]:
    """The typed model and the CLI's rules, without the generated schema."""
    config = UserConfig.model_validate(copy.deepcopy(document))
    return [str(error) for error in collect_config_errors(config)]


def _signal_bindings(document):
    _catalog(document)["bindings"] = {
        "jailbreak": {"deployment": "chat-judge"},
        "pii": "local-vela",
        "domain": {"deployment": "d3-remote"},
        "fact_check": {"deployment": "local-vela"},
        "user_feedback": {"deployment": "local-vela"},
    }


def _rag_judge(document):
    document["recipes"][0]["decisions"] = [
        {
            "name": "retrieve",
            "priority": 1,
            "rules": {"operator": "AND", "conditions": []},
            "modelRefs": [{"model": "small"}],
            "plugins": [
                {
                    "type": "context",
                    "configuration": {
                        "enabled": True,
                        "sources": [
                            {
                                "type": "rag",
                                "backend": "vectorstore",
                                "match": {
                                    "recall": {"method": "embedding", "top_k": 10},
                                    "judge": {
                                        "preset": "relevance",
                                        "deployment": "d3-remote",
                                        "top_k": 3,
                                    },
                                },
                                "backend_config": {"vector_store_id": "vs-test"},
                            }
                        ],
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
            "model_bindings": {
                "turn_relation": "chat-judge",
                "complexity": "local-vela",
                "user_feedback": "local-vela",
                "modality": "local-vela",
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
    deployments["chat-judge"]["generation"] = {"max_tokens": 64, "temperature": 0}


def _rule_deployments(document):
    document["recipes"][0]["signals"] = {
        "jailbreak": [{"name": "attack", "deployment": "d3-remote"}],
        "safety": [
            {
                "name": "harm",
                "deployment": "local-vela",
                "labels": ["Harassment"],
                "predicate": {"gte": 0.5},
            }
        ],
        "complexity": [{"name": "hard", "deployment": "local-vela"}],
        "systemone": [
            {
                "name": "billing",
                "deployment": "d3-remote",
                "question": {"type": "noul", "instructions": "About billing?"},
            }
        ],
    }
    document["recipes"][0]["decisions"] = [
        {
            "name": "choose",
            "priority": 1,
            "rules": {
                "operator": "AND",
                "conditions": [{"type": "systemone", "name": "billing"}],
            },
            "modelRefs": [{"model": "small"}, {"model": "large"}],
            "algorithm": {
                "type": "systemone",
                "systemone": {
                    "deployment": "local-vela",
                    "instructions": "Which model should answer?",
                },
            },
        }
    ]
    document["providers"]["models"].append(
        {"name": "large", "backend_refs": [{"endpoint": "large:8000"}]}
    )


def _embedding_defaults(document):
    _catalog(document)["embeddings"] = {"default": {"deployment": "embedder"}}


def _bind(scope: str, key: str):
    binding = {"deployment": "local-vela", "contract": "decision.v1"}

    def bind(document):
        if scope == "global":
            _catalog(document)["bindings"] = {key: binding}
        else:
            document["recipes"] = [
                {"name": "private", "model_bindings": {key: binding}}
            ]

    return bind


@pytest.mark.parametrize("scope", ["global", "recipe"])
@pytest.mark.parametrize("key", ["pii", "hallucination"])
def test_a_binding_object_with_a_contract_is_refused_with_the_routers_line(scope, key):
    prefix = (
        "global.model_catalog.bindings"
        if scope == "global"
        else "recipes[private].model_bindings"
    )

    with pytest.raises(ConfigParseError) as refused:
        _cli_errors(_document(_bind(scope, key)))

    assert str(refused.value) == (
        f"{prefix}.{key}: removed_field: {prefix}.{key}.contract is removed; "
        f"run `vllm-sr config migrate` ({key}: local-vela; the signal decides "
        "what the deployment answers)"
    )


@pytest.mark.parametrize("scope", ["global", "recipe"])
@pytest.mark.parametrize(
    "key, becomes",
    [
        ("pii_classifier", "(pii)"),
        ("hallucination_detector", "(hallucination)"),
        ("preference", "(drop it; the decision model answers the signal)"),
        ("reask", "(drop it; the decision model answers the signal)"),
    ],
)
def test_bindings_of_removed_modules_and_signals_are_refused(scope, key, becomes):
    with pytest.raises(ConfigParseError) as refused:
        _cli_errors(_document(_bind(scope, key)))
    prefix = (
        "global.model_catalog.bindings"
        if scope == "global"
        else "recipes[private].model_bindings"
    )
    assert str(refused.value).startswith(
        f"{prefix}.{key}: removed_field: binding {key} "
    )
    assert str(refused.value).endswith(becomes)


def test_the_cli_admits_new_providers_and_signal_keyed_bindings():
    assert _cli_errors(_document(_signal_bindings)) == []


SCHEMA_SHAPES = [
    pytest.param(_string_bindings, id="string bindings"),
    pytest.param(_connection_block, id="connection block"),
    pytest.param(_deployment_settings, id="deployment settings"),
    pytest.param(_rule_deployments, id="rule deployments"),
    pytest.param(_embedding_defaults, id="embedding defaults"),
    pytest.param(_rag_judge, id="rag relevance judge deployment"),
]


@pytest.mark.parametrize("mutate", SCHEMA_SHAPES)
def test_the_cli_rules_admit_the_shapes_the_router_reads(mutate):
    assert _cli_rule_errors(_document(mutate)) == []


@pytest.mark.parametrize("mutate", SCHEMA_SHAPES)
def test_the_generated_schema_admits_the_shapes_the_router_reads(mutate):
    assert _cli_errors(_document(mutate)) == []


def test_a_string_binding_names_its_deployment():
    config = UserConfig.model_validate(_document(_string_bindings))
    binding = config.recipes[0].model_bindings["turn_relation"]

    assert binding.deployment == "chat-judge"
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
        {"valid": False, "error": refusal, "warnings": [RENAMED]},
        code=1,
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    result = _validate("--config", str(config))

    assert result.exit_code == 1, result.output
    assert "Configuration is valid" not in result.stdout
    assert f"the Router binary {router} refuses it: {refusal}" in result.stderr
    assert "the decision signal is named systemone" in _flat(result.stderr)
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
        tmp_path, {"valid": True, "warnings": [RENAMED, UNCALIBRATED]}
    )

    result = _validate(
        "--config", str(config), "--router-binary", str(router), "--gateway", "extproc"
    )

    assert result.exit_code == 0, result.output
    assert f"Checked by the Router binary {router} as well as the CLI." in (
        result.stdout
    )
    warnings = _flat(result.stderr)
    assert "the decision signal is named systemone" in warnings
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


LEGACY = {
    **BASE,
    "global": {
        "model_catalog": {
            "deployments": {"local-vela": {"artifact": "vllm-sr/vela3-fast"}},
            "bindings": {"pii": {"deployment": "local-vela", "contract": "x"}},
        }
    },
}
REFUSAL = (
    "global.model_catalog.bindings.pii: removed_field: "
    "global.model_catalog.bindings.pii.contract is removed; run `vllm-sr config "
    "migrate` (pii: local-vela; the signal decides what the deployment answers)"
)
CHANGED = {
    "code": "migration_change",
    "field": "global.model_catalog.bindings.pii",
    "message": "global.model_catalog.bindings.pii.contract is removed; written as "
    "pii: local-vela",
}
ACTION = {
    "code": "migration_action",
    "field": "global.model_catalog.deployments.rewriter.auth.api_key_env",
    "message": "set VLLM_SR_REWRITER_API_KEY to the key access_key held",
}


def _migrating_router(tmp_path: Path, migration: dict, verdict: dict) -> Path:
    """A Router binary that answers -migrate-config with migration and any
    other call with verdict, recording what each call was asked."""

    script = tmp_path / "router"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "config = open(sys.argv[sys.argv.index('-config') + 1]).read()\n"
        f"with open({str(tmp_path / 'router-calls.jsonl')!r}, 'a') as record:\n"
        "    record.write(json.dumps({'argv': sys.argv[1:], 'config': config}) + '\\n')\n"
        f"report = {migration!r} if '-migrate-config' in sys.argv else {verdict!r}\n"
        "print(json.dumps(report))\n"
        "sys.exit(0 if report['valid'] else 1)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _router_calls(tmp_path: Path) -> list[dict]:
    calls = (tmp_path / "router-calls.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in calls.splitlines()]


def _migrate(*arguments: str):
    return CliRunner().invoke(main, ["config", "migrate", *arguments])


def test_migrate_writes_what_the_router_rewrites(monkeypatch, tmp_path):
    source = _write(tmp_path, LEGACY)
    rewritten = yaml.safe_dump(BASE, sort_keys=False)
    router = _migrating_router(
        tmp_path,
        {
            "valid": False,
            "error": REFUSAL,
            "warnings": [],
            "migrated": rewritten,
            "notes": [CHANGED, ACTION],
        },
        {"valid": True, "warnings": [UNCALIBRATED]},
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))

    result = _migrate("--config", str(source))

    assert result.exit_code == 0, result.output
    output = tmp_path / "config.migrated.yaml"
    assert output.read_text(encoding="utf-8") == rewritten
    asked, checked = _router_calls(tmp_path)
    assert "-migrate-config" in asked["argv"]
    assert asked["config"] == source.read_text(encoding="utf-8")
    assert "-migrate-config" not in checked["argv"]
    assert checked["config"] == rewritten
    stdout, stderr = _flat(result.stdout), _flat(result.stderr)
    assert f"Checked by the Router binary {router}" in stdout
    assert "Changes to review global.model_catalog.bindings.pii" in stdout
    assert "written as pii: local-vela" in stdout
    assert "set VLLM_SR_REWRITER_API_KEY to the key access_key held" in stderr
    assert "has no calibrated threshold" in stderr


@pytest.mark.parametrize(
    "migration, verdict, refusal",
    [
        (
            {
                "valid": False,
                "error": REFUSAL,
                "warnings": [],
                "migration_error": "default_model: removed_section: it has no "
                "mechanical rewrite",
            },
            {"valid": True, "warnings": []},
            "The Router could not migrate the configuration: default_model: "
            "removed_section: it has no mechanical rewrite",
        ),
        (
            {"valid": True, "warnings": []},
            {"valid": True, "warnings": []},
            "The Router could not migrate the configuration: it wrote nothing",
        ),
        (
            {
                "valid": False,
                "error": REFUSAL,
                "warnings": [],
                "migrated": yaml.safe_dump(BASE, sort_keys=False),
            },
            {"valid": False, "error": "deployments: refused", "warnings": []},
            "refuses it: deployments: refused",
        ),
    ],
    ids=["refused form", "no migration", "output refused"],
)
def test_migrate_writes_nothing_the_router_does_not_accept(
    monkeypatch, tmp_path, migration, verdict, refusal
):
    source = _write(tmp_path, LEGACY)
    output = tmp_path / "out.yaml"
    output.write_text("previous: output\n", encoding="utf-8")
    monkeypatch.setenv(
        ROUTER_BINARY_ENV, str(_migrating_router(tmp_path, migration, verdict))
    )

    result = _migrate("--config", str(source), "--output", str(output), "--force")

    assert result.exit_code == 1
    assert refusal in _flat(result.stderr)
    assert "No output was written" in result.stderr
    assert output.read_text(encoding="utf-8") == "previous: output\n"


def test_migrate_needs_a_router(monkeypatch, tmp_path):
    source = _write(tmp_path, LEGACY)
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(tmp_path / "absent-router"))

    result = _migrate("--config", str(source))

    assert result.exit_code == 1
    assert "No Router could migrate the configuration" in _flat(result.stderr)
    assert not (tmp_path / "config.migrated.yaml").exists()
    assert _migrate("--config", str(source), "--offline").exit_code == 2


IDENTITY_REFUSAL = (
    "global.services.authz.providers, which resolve per-user API keys and reads "
    "the client identity an authenticating gateway sets in x-authz-user-id and "
    "x-authz-user-groups, needs an identity source in standalone mode; behind an "
    "authenticating proxy, set listeners[].identity.trust_headers: true, or serve "
    "with --gateway extproc"
)
UNTRUSTED_IDENTITY_HINT = (
    "Hint: Re-run with `--gateway extproc` when Envoy or another authenticating "
    "gateway sits in front of the Router, or set "
    "`listeners[].identity.trust_headers: true` only behind a trusted proxy."
)


def _migrating_to(monkeypatch, tmp_path: Path, verdict: dict) -> str:
    """Route migrate to a Router that rewrites LEGACY into BASE and answers the
    check of BASE with verdict; return what it rewrites."""

    rewritten = yaml.safe_dump(BASE, sort_keys=False)
    migration = {
        "valid": False,
        "error": REFUSAL,
        "warnings": [],
        "migrated": rewritten,
    }
    router = _migrating_router(tmp_path, migration, verdict)
    monkeypatch.setenv(ROUTER_BINARY_ENV, str(router))
    return rewritten


@pytest.mark.parametrize(
    "arguments, environment, mode",
    [
        ((), None, "standalone"),
        (("--gateway", "extproc"), None, "extproc"),
        ((), "extproc", "extproc"),
        (("--gateway", "standalone"), "extproc", "standalone"),
    ],
    ids=["default", "option", "environment", "option over environment"],
)
def test_migrate_checks_its_output_in_the_gateway_mode(
    monkeypatch, tmp_path, arguments, environment, mode
):
    source = _write(tmp_path, LEGACY)
    rewritten = _migrating_to(monkeypatch, tmp_path, {"valid": True, "warnings": []})
    if environment is None:
        monkeypatch.delenv(GATEWAY_ENV, raising=False)
    else:
        monkeypatch.setenv(GATEWAY_ENV, environment)

    result = _migrate("--config", str(source), *arguments)

    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.migrated.yaml").read_text(encoding="utf-8") == rewritten
    calls = _router_calls(tmp_path)
    modes = [call["argv"][call["argv"].index("-gateway") + 1] for call in calls]
    assert modes == [mode, mode]


def test_migrate_takes_only_a_known_gateway_mode(tmp_path):
    source = _write(tmp_path, LEGACY)

    result = _migrate("--config", str(source), "--gateway", "envoy")

    assert result.exit_code == 2
    assert "--gateway" in result.output
    assert not (tmp_path / "config.migrated.yaml").exists()


@pytest.mark.parametrize(
    "arguments, error, hinted",
    [
        ((), IDENTITY_REFUSAL, True),
        ((), "deployments: refused", False),
        (("--gateway", "extproc"), IDENTITY_REFUSAL, False),
    ],
    ids=["standalone identity", "another refusal", "extproc"],
)
def test_a_standalone_refusal_of_identity_headers_names_the_way_out(
    monkeypatch, tmp_path, arguments, error, hinted
):
    source = _write(tmp_path, LEGACY)
    monkeypatch.delenv(GATEWAY_ENV, raising=False)
    rewritten = _migrating_to(monkeypatch, tmp_path, {"valid": False, "error": error})

    result = _migrate("--config", str(source), *arguments)

    assert result.exit_code == 1
    stderr = _flat(result.stderr)
    assert f"refuses it: {error}" in stderr
    assert (UNTRUSTED_IDENTITY_HINT in stderr) is hinted
    assert "No output was written" in stderr
    assert not (tmp_path / "config.migrated.yaml").exists()
    # The Router checked the migration as written, once: migrate never trusts
    # the identity headers itself.
    _asked, checked = _router_calls(tmp_path)
    assert checked["config"] == rewritten
