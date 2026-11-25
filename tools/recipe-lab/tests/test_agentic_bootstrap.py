"""AG5: agentic cold start - profile aggregation, the shared unmeasured-lane
selection (Z10), and the client identity contract. The golden test
synthesizes a three-model profile and asserts the shared ranking, the
claude-code identity headers and the privacy lane's bypass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from recipe_lab.agentic_bootstrap import (
    PROFILE_KIND,
    bootstrap_agentic_config,
    profile_from_reports,
)

MAINTAINED_RECIPES = Path(__file__).resolve().parents[3] / "config" / "recipes"


def _report(model: str, *, rate: float, cost: float, tasks: int = 10) -> dict:
    return {
        "summary": {
            "targets": [
                {
                    "id": model,
                    "kind": "single",
                    "total": tasks,
                    "accuracy": rate,
                    "accuracy_ci95": [max(0.0, rate - 0.1), min(1.0, rate + 0.1)],
                    "total_spend_usd": cost * tasks,
                    "cache_neutral_cost_usd": cost * tasks * 1.2,
                    "latency_p50_s": 10.0,
                    "output_diagnostics": {
                        "planned_cases": tasks,
                        "result_cases": tasks,
                        "output_limit_cases": 0,
                        "subject_calls": {"total": tasks * 4},
                    },
                }
            ]
        }
    }


@pytest.fixture()
def three_model_profile():
    return profile_from_reports(
        [
            _report("local/cheap", rate=0.60, cost=0.5),
            _report("local/mid", rate=0.86, cost=2.0),
            _report("local/strong", rate=0.90, cost=5.0),
        ]
    )


# ---- profile aggregation ----


def test_profile_from_reports_aggregates_per_model_arms():
    profile = profile_from_reports(
        [_report("m/a", rate=0.8, cost=1.0), _report("m/b", rate=0.9, cost=3.0)]
    )
    assert profile["kind"] == PROFILE_KIND
    by_model = {m["model"]: m for m in profile["models"]}
    assert by_model["m/a"]["success_rate"] == pytest.approx(0.8)
    assert by_model["m/a"]["cost_per_task_usd"] == pytest.approx(1.0)
    assert by_model["m/a"]["cache_neutral_cost_per_task_usd"] == pytest.approx(1.2)
    assert by_model["m/a"]["mean_calls_per_task"] == pytest.approx(4.0)
    assert by_model["m/a"]["truncation_rate"] == 0.0
    assert by_model["m/a"]["missing"] == []


def test_profile_flags_missing_evidence_instead_of_zero():
    report = _report("m/a", rate=0.8, cost=1.0)
    del report["summary"]["targets"][0]["total_spend_usd"]
    profile = profile_from_reports([report])
    entry = profile["models"][0]
    assert entry["cost_per_task_usd"] is None
    assert "cost_per_task_usd" in entry["missing"]


# ---- the shared unmeasured-lane selection (Z10) ----


def test_ranked_lanes_take_the_lambda_adjusted_model(three_model_profile):
    # Z10: the agentic profile measures models, not lanes - every ranked
    # lane goes through the shared selection core's unmeasured branch. At
    # lambda 0 the strongest model wins every ranked lane.
    lanes = {
        lane["decision"]: lane
        for lane in _bootstrap_sidecar(three_model_profile)["lanes"]
    }
    assert lanes["domain_code_complex"]["model"] == "local/strong"
    assert lanes["simple_general"]["model"] == "local/strong"
    # the shared unmeasured marking, not a hand-rolled rule
    assert "default rule (unmeasured lane)" in lanes["domain_code_complex"]["basis"]
    assert lanes["domain_code_complex"]["untested"] is True
    assert lanes["domain_code_complex"]["scores"]["local/strong"][
        "overall_accuracy"
    ] == pytest.approx(0.90)


def _bootstrap_sidecar(profile, **kwargs):
    template = _template()
    _, sidecar = bootstrap_agentic_config(
        template, profile, client="generic", local_models=["local/private"], **kwargs
    )
    return sidecar


def _template():
    return {
        "version": "v0.5",
        "global": {
            "router": {
                "learning": {
                    "adaptation": {"enabled": True},
                    "protection": {"enabled": True, "scope": "conversation"},
                }
            }
        },
        "model_catalog": {"system": {"decision_model": {"deployment": "p"}}},
        "recipes": [
            {
                "name": "agent",
                "decisions": [
                    {
                        "name": "domain_code_complex",
                        "modelRefs": [{"model": "template/a"}],
                    },
                    {"name": "simple_general", "modelRefs": [{"model": "template/a"}]},
                    {
                        "name": "local_privacy_policy",
                        "modelRefs": [{"model": "template/a"}],
                    },
                ],
            }
        ],
    }


def test_lambda_flips_the_ranking_to_the_cheap_model(three_model_profile):
    # normalized costs (max 5.0): cheap 0.1, mid 0.4, strong 1.0 - at
    # lambda 1 the cheap model's adjusted score (0.5) beats the strong
    # model's (-0.1) on every ranked lane
    lanes = {
        lane["decision"]: lane
        for lane in _bootstrap_sidecar(three_model_profile, lam=1.0)["lanes"]
    }
    assert lanes["domain_code_complex"]["model"] == "local/cheap"
    assert lanes["simple_general"]["model"] == "local/cheap"


def test_unpriced_models_rank_as_free_only_at_lambda_zero(three_model_profile):
    # missing cost is "cannot be price-compared": the shared core refuses a
    # lambda > 0 over an unpriced candidate rather than ranking it as free
    profile = profile_from_reports(
        [
            _report("local/cheap", rate=0.60, cost=0.5),
            _report("local/mid", rate=0.86, cost=2.0),
        ]
    )
    profile["models"][1]["cost_per_task_usd"] = None
    profile["models"][1]["missing"] = ["cost_per_task_usd"]
    with pytest.raises(Exception, match="cost"):
        _bootstrap_sidecar(profile, lam=1.0)


# ---- golden test: template rewrite ----


def test_bootstrap_rewrites_the_shipped_agent_recipe(three_model_profile):
    # the CLI's default template is the canonical v0.5 agent recipe: the
    # rewrite edits that recipe in place and every other section rides along
    template = yaml.safe_load(
        (MAINTAINED_RECIPES / "agent" / "config.yaml").read_text(encoding="utf-8")
    )
    config, sidecar = bootstrap_agentic_config(
        template, three_model_profile, client="generic", local_models=["local/private"]
    )
    assert len(config["recipes"]) == len(template["recipes"]) == 1
    recipe = config["recipes"][0]
    by_name = {d["name"]: d for d in recipe["decisions"]}
    assert by_name["domain_code_complex"]["modelRefs"][0]["model"] == "local/strong"
    assert by_name["local_privacy_policy"]["modelRefs"][0]["model"] == "local/private"
    assert by_name["omni"]["modelRefs"][0]["model"] == "local/omni"
    # the candidate requirements sit on the recipe, next to its decisions
    assert recipe["candidate_requirements"] == {"capabilities": "declared"}
    assert recipe["signals"] == template["recipes"][0]["signals"]
    assert config["entrypoints"] == template["entrypoints"]
    assert config["providers"] == template["providers"]
    assert sidecar["recipe"] == "default"
    assert sidecar["decision_model"] == {"deployment": "primary"}
    # the protection bootstrap enables is dead under a learning-disabled
    # template - the rewrite forces learning on
    assert config["global"]["router"]["learning"]["enabled"] is True


def test_bootstrap_selects_the_recipe_a_public_model_serves(three_model_profile):
    # a document may ship one recipe per public model: without a selection
    # the rewrite is ambiguous, --public-model names the recipe to rewrite
    template = _template()
    template["recipes"].append(
        {
            "name": "agent-fast",
            "decisions": [
                {"name": "simple_general", "modelRefs": [{"model": "template/a"}]}
            ],
        }
    )
    template["entrypoints"] = [
        {"model_names": ["vllm-sr/agent"], "recipe": "agent"},
        {"model_names": ["vllm-sr/agent-flash"], "recipe": "agent-fast"},
    ]
    with pytest.raises(ValueError, match="--recipe"):
        bootstrap_agentic_config(
            template,
            three_model_profile,
            client="generic",
            local_models=["local/private"],
        )
    config, sidecar = bootstrap_agentic_config(
        template,
        three_model_profile,
        client="generic",
        local_models=["local/private"],
        public_model="vllm-sr/agent-flash",
    )
    assert sidecar["recipe"] == "agent-fast"
    assert [lane["decision"] for lane in sidecar["lanes"]] == ["simple_general"]
    assert sidecar["lanes"][0]["model"] == "local/strong"
    untouched, rewritten = config["recipes"]
    assert rewritten["decisions"][0]["modelRefs"][0]["model"] == "local/strong"
    assert rewritten["candidate_requirements"] == {"capabilities": "declared"}
    assert untouched == template["recipes"][0]
    assert "candidate_requirements" not in untouched


def test_bootstrap_rejects_a_template_outside_the_canonical_layout(
    three_model_profile,
):
    with pytest.raises(ValueError, match=r"v0\.5 'recipes'"):
        bootstrap_agentic_config(
            {"decisions": []}, three_model_profile, client="generic"
        )


@pytest.fixture()
def agent_template():
    return {
        "version": "v0.5",
        "global": {
            "router": {
                "learning": {
                    "adaptation": {"enabled": True, "strategy": "routing_sampling"},
                    "protection": {
                        "enabled": True,
                        "scope": "conversation",
                        "identity": {
                            "headers": {
                                "session": "x-session-id",
                                "conversation": "x-conversation-id",
                            }
                        },
                        "tuning": {"switch_margin": 0.05},
                    },
                }
            }
        },
        "model_catalog": {"system": {"decision_model": {"deployment": "primary"}}},
        "recipes": [
            {
                "name": "agent",
                "decisions": [
                    {
                        "name": "domain_code_complex",
                        "modelRefs": [{"model": "template/a", "use_reasoning": False}],
                    },
                    {
                        "name": "medium_general",
                        "modelRefs": [{"model": "template/a", "use_reasoning": False}],
                    },
                    {
                        "name": "simple_general",
                        "modelRefs": [{"model": "template/a", "use_reasoning": False}],
                    },
                    {
                        "name": "local_privacy_policy",
                        "modelRefs": [{"model": "template/a", "use_reasoning": False}],
                        "adaptations": {"mode": "apply"},
                    },
                    {
                        "name": "local_security_containment",
                        "modelRefs": [{"model": "template/a", "use_reasoning": False}],
                    },
                    {"name": "omni", "modelRefs": [{"model": "local/omni"}]},
                ],
            }
        ],
    }


def test_golden_bootstrap_writes_lanes_identity_and_bypass(
    agent_template, three_model_profile
):
    config, sidecar = bootstrap_agentic_config(
        agent_template,
        three_model_profile,
        client="claude-code",
        local_models=["local/private"],
    )
    recipe = config["recipes"][0]
    by_name = {d["name"]: d for d in recipe["decisions"]}
    # the ranked lanes all take the shared ranking's winner at lambda 0
    assert by_name["domain_code_complex"]["modelRefs"][0]["model"] == "local/strong"
    assert by_name["medium_general"]["modelRefs"][0]["model"] == "local/strong"
    assert by_name["simple_general"]["modelRefs"][0]["model"] == "local/strong"
    # the privacy lane is exempt from learning and pinned to the local model
    privacy = by_name["local_privacy_policy"]
    assert privacy["modelRefs"][0]["model"] == "local/private"
    assert privacy["adaptations"]["mode"] == "bypass"
    assert by_name["local_security_containment"]["modelRefs"][0]["model"] == (
        "local/private"
    )
    assert by_name["local_security_containment"]["adaptations"]["mode"] == "bypass"
    # the omni modality lane is not a profile choice
    assert by_name["omni"]["modelRefs"][0]["model"] == "local/omni"
    # claude-code writes the two verified identity headers and session scope
    protection = config["global"]["router"]["learning"]["protection"]
    assert protection["scope"] == "session"
    assert protection["identity"]["headers"] == {
        "session": "x-claude-code-session-id",
        "agent": "x-claude-code-agent-id",
    }
    # adaptation off, progress gate observing, tool-capable candidates only
    learning = config["global"]["router"]["learning"]
    assert learning["adaptation"]["enabled"] is False
    assert protection["tuning"]["progress_gate"] == {"mode": "observe"}
    assert recipe["candidate_requirements"] == {"capabilities": "declared"}
    assert sidecar["recipe"] == "agent"
    # the decision model stays pinned to the template's deployment
    assert (
        config["model_catalog"]["system"]["decision_model"]["deployment"] == "primary"
    )
    # the sidecar carries every lane's basis and evidence
    lanes = {lane["decision"]: lane for lane in sidecar["lanes"]}
    assert lanes["domain_code_complex"]["model"] == "local/strong"
    assert lanes["domain_code_complex"]["untested"] is True
    assert lanes["local_privacy_policy"]["basis"].startswith("contract-declared local")
    assert sidecar["client"] == "claude-code"
    assert sidecar["adaptation_enabled"] is False
    assert sidecar["lambda"] == 0.0


def test_golden_bootstrap_neutral_client_uses_conversation_scope(
    agent_template, three_model_profile
):
    config, sidecar = bootstrap_agentic_config(
        agent_template,
        three_model_profile,
        client="generic",
        local_models=["local/private"],
    )
    protection = config["global"]["router"]["learning"]["protection"]
    assert protection["scope"] == "conversation"
    assert protection["identity"]["headers"] == {
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    }
    assert "agent" not in sidecar["identity_headers"]


def test_bootstrap_requires_local_models_for_privacy_lanes(
    agent_template, three_model_profile
):
    with pytest.raises(ValueError, match="local"):
        bootstrap_agentic_config(agent_template, three_model_profile, client="generic")


# ---- CLI end to end (validate via a stub bin) ----


VALID_SPEC_YAML = (
    "api_version: recipe-lab/v1alpha1\n"
    "objective:\n"
    "  metric_key: capability.score\n"
    "  direction: maximize\n"
    "  min_improvement_pct: 1.0\n"
    "  replay_metric: routed_accuracy\n"
    "workload:\n"
    "  manifest: cases.jsonl\n"
    "  content_hash: " + "a" * 64 + "\n"
    "attribution:\n"
    "  oracle_policy:\n"
    "    domain_code: local/model-a\n"
    "local_models:\n"
    "  - local/private\n"
)


def _write_spec(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(VALID_SPEC_YAML, encoding="utf-8")
    return spec_path


def _write_template(path, *recipes):
    path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.5",
                "global": {
                    "router": {
                        "learning": {
                            "adaptation": {"enabled": True},
                            "protection": {"enabled": True, "scope": "conversation"},
                        }
                    }
                },
                "model_catalog": {"system": {"decision_model": {"deployment": "p"}}},
                "recipes": list(recipes),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            profile_from_reports(
                [
                    _report("local/cheap", rate=0.60, cost=0.5),
                    _report("local/mid", rate=0.86, cost=2.0),
                    _report("local/strong", rate=0.90, cost=5.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    return path


def test_cli_profile_and_bootstrap_end_to_end(tmp_path, capsys):
    from recipe_lab.cli import main

    reports = [tmp_path / f"report-{name}.json" for name in ("cheap", "mid", "strong")]
    for path, (name, rate, cost) in zip(
        reports,
        [("cheap", 0.60, 0.5), ("mid", 0.86, 2.0), ("strong", 0.90, 5.0)],
        strict=False,
    ):
        path.write_text(
            json.dumps(_report(name, rate=rate, cost=cost)), encoding="utf-8"
        )
    profile_path = tmp_path / "profile.json"
    assert (
        main(
            [
                "profile-agentic",
                "--reports",
                str(reports[0]),
                "--reports",
                str(reports[1]),
                "--reports",
                str(reports[2]),
                "--output",
                str(profile_path),
            ]
        )
        == 0
    )
    assert json.loads(profile_path.read_text(encoding="utf-8"))["kind"] == PROFILE_KIND

    template = _write_template(
        tmp_path / "template.yaml",
        {
            "name": "agent",
            "decisions": [
                {"name": "domain_code_complex", "modelRefs": [{"model": "template/a"}]},
                {
                    "name": "local_privacy_policy",
                    "modelRefs": [{"model": "template/a"}],
                },
            ],
        },
    )
    out = tmp_path / "cands"
    capsys.readouterr()
    assert (
        main(
            [
                "bootstrap-agentic",
                "--spec",
                str(_write_spec(tmp_path)),
                "--profile",
                str(profile_path),
                "--template",
                str(template),
                "--client",
                "claude-code",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["validated"] is False
    assert payload["recipe"] == "agent"
    assert payload["lanes"]["local_privacy_policy"] == "local/private"
    config = yaml.safe_load((out / "agentic-config.yaml").read_text(encoding="utf-8"))
    protection = config["global"]["router"]["learning"]["protection"]
    assert protection["identity"]["headers"]["session"] == "x-claude-code-session-id"
    sidecar = json.loads(
        (out / "agentic-config.bootstrap.json").read_text(encoding="utf-8")
    )
    assert sidecar["identity_headers"]["agent"] == "x-claude-code-agent-id"


def test_cli_bootstrap_defaults_to_the_shipped_agent_recipe(tmp_path, capsys):
    # without --template the command rewrites config/recipes/agent/config.yaml,
    # so the canonical document itself must round-trip
    from recipe_lab.cli import main

    out = tmp_path / "cands"
    assert (
        main(
            [
                "bootstrap-agentic",
                "--spec",
                str(_write_spec(tmp_path)),
                "--profile",
                str(_write_profile(tmp_path)),
                "--client",
                "generic",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["recipe"] == "default"
    assert payload["lanes"]["domain_code_complex"] == "local/strong"
    config = yaml.safe_load((out / "agentic-config.yaml").read_text(encoding="utf-8"))
    assert config["recipes"][0]["candidate_requirements"] == {
        "capabilities": "declared"
    }


def test_cli_bootstrap_names_the_recipe_to_rewrite(tmp_path, capsys):
    from recipe_lab.cli import main

    template = _write_template(
        tmp_path / "template.yaml",
        {
            "name": "agent",
            "decisions": [
                {"name": "domain_code_complex", "modelRefs": [{"model": "a"}]}
            ],
        },
        {
            "name": "agent-fast",
            "decisions": [{"name": "medium_general", "modelRefs": [{"model": "a"}]}],
        },
    )
    args = [
        "bootstrap-agentic",
        "--spec",
        str(_write_spec(tmp_path)),
        "--profile",
        str(_write_profile(tmp_path)),
        "--template",
        str(template),
        "--client",
        "generic",
        "--out",
        str(tmp_path / "cands"),
    ]
    # several recipes and no selection: a contract error, the reason on stderr
    assert main(args) == 2
    assert "--recipe" in capsys.readouterr().err
    assert main([*args, "--recipe", "agent-fast"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recipe"] == "agent-fast"
    assert list(payload["lanes"]) == ["medium_general"]
