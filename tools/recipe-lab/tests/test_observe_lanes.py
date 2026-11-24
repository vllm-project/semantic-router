"""observe-lanes (C2 template path) unit tests.

All subprocess boundaries are faked (recipe builtin list/init via run_json,
probe_records via monkeypatch), so the suite is deterministic and
platform-independent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from recipe_lab import observe_lanes as ol
from recipe_lab.contract import ContractError


def fake_list_document():
    """The shape `vllm-sr recipe builtin list` prints (upstream contract)."""
    return {
        "catalog_version": "latest",
        "bundles": [
            {
                "name": "mom-v1",
                "sha256": "sha256:bundle",
                "files": ["recipe.yaml"],
                "recipes": [
                    {
                        "name": "balance",
                        "description": "ten lanes",
                        "decisions": [
                            {
                                "name": "long_context",
                                "algorithm": "static",
                                "minimum_candidates": 1,
                            },
                            {
                                "name": "vision",
                                "algorithm": "static",
                                "minimum_candidates": 1,
                            },
                            {
                                "name": "frontier",
                                "algorithm": "decision",
                                "minimum_candidates": 2,
                            },
                            {
                                "name": "standard",
                                "algorithm": "multi_factor",
                                "minimum_candidates": 1,
                            },
                            {
                                "name": "fast",
                                "algorithm": None,
                                "minimum_candidates": 0,
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_template_info_parses_list_and_selects_recipe():
    info = ol.template_info(
        bundle="mom-v1",
        recipe="balance",
        run_json=lambda argv, timeout_s, what: fake_list_document(),
    )
    assert info["sha256"] == "sha256:bundle"
    assert info["lanes"]["frontier"]["minimum_candidates"] == 2
    assert info["lanes"]["fast"]["immediate"] is True
    assert info["lanes"]["vision"]["immediate"] is False


def test_template_info_refuses_unknown_bundle_or_recipe():
    with pytest.raises(ContractError, match="not installed"):
        ol.template_info(
            bundle="nope",
            recipe="balance",
            run_json=lambda argv, timeout_s, what: fake_list_document(),
        )
    with pytest.raises(ContractError, match="has no recipe"):
        ol.template_info(
            bundle="mom-v1",
            recipe="nope",
            run_json=lambda argv, timeout_s, what: fake_list_document(),
        )


def test_observation_bindings_allocate_distinct_placeholders():
    template = ol.template_info(
        bundle="mom-v1",
        recipe="balance",
        run_json=lambda argv, timeout_s, what: fake_list_document(),
    )
    bindings, count = ol.observation_bindings(template)
    # Immediate lane binds to []; frontier needs two distinct placeholders.
    assert bindings["fast"] == []
    assert len({ref["model"] for ref in bindings["frontier"]}) == 2
    assert count == 2
    # Lanes are processed in sorted order: the first sorted backend lane
    # takes placeholder-1.
    first_backend = min(lane for lane, refs in bindings.items() if refs)
    assert bindings[first_backend][0]["model"] == f"{ol.PLACEHOLDER_PREFIX}-1"


def test_observation_providers_config_declares_no_decision_model():
    config = ol.observation_providers_config(
        placeholder_count=2, backend_endpoint="127.0.0.1:8000"
    )
    names = [model["name"] for model in config["providers"]["models"]]
    assert names == [f"{ol.PLACEHOLDER_PREFIX}-1", f"{ol.PLACEHOLDER_PREFIX}-2"]
    # The response cache is off (a warm cache would swallow repeated cases):
    # canonical v0.5 runs no response_cache plugin unless a decision mounts one.
    assert config["global"]["stores"]["response_cache"] == {"backend_type": "memory"}
    # The placeholder models carry their card facts themselves
    # (candidate_requirements: capabilities declared).
    assert {
        model["name"]
        for model in config["providers"]["models"]
        if "capabilities" in model
    } == set(names)
    # No decision model here: init copies the bundle's pinned deployment.
    assert "model_catalog" not in config["global"]


def test_build_observation_config_invokes_init_and_disables_cache(tmp_path):
    template = ol.template_info(
        bundle="mom-v1",
        recipe="balance",
        catalog_version="v0.4",
        run_json=lambda argv, timeout_s, what: fake_list_document(),
    )
    captured: list[list[str]] = []

    def run_json(argv, *, timeout_s, what):
        captured.append(list(argv))
        if "init" in argv:
            out = Path(argv[argv.index("--output") + 1])
            providers = yaml.safe_load(
                Path(argv[argv.index("--config") + 1]).read_text()
            )
            assert providers["global"]["services"]["router_replay"] == {
                "store_backend": "memory"
            }
            out.write_text(
                yaml.safe_dump(
                    {
                        "version": "v0.3",
                        "global": {
                            "services": providers["global"]["services"],
                            "stores": {
                                "response_cache": {
                                    "enabled": True,
                                    "backend_type": "memory",
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {"valid": True}
        return {"bundles": []}

    result = ol.build_observation_config(
        template=template,
        out_dir=tmp_path,
        backend_endpoint="127.0.0.1:8000",
        vllm_sr_bin="fake-vllm-sr",
        run_json=run_json,
    )
    inits = [argv for argv in captured if "init" in argv]
    assert len(inits) == 1
    argv = inits[0]
    assert argv[argv.index("--bundle") + 1] == "mom-v1"
    assert argv[argv.index("--catalog-version") + 1] == "v0.4"
    assert argv[argv.index("--model-name") + 1] == ol.OBSERVE_MODEL_NAME
    # Bindings file: the top level IS the {lane: refs} mapping.
    bindings_doc = yaml.safe_load(
        Path(argv[argv.index("--bindings") + 1]).read_text(encoding="utf-8")
    )
    assert bindings_doc["frontier"][0]["model"]
    # The generated config's cache was forced off.
    config = yaml.safe_load(Path(result["config"]).read_text(encoding="utf-8"))
    assert config["global"]["stores"]["response_cache"]["enabled"] is False
    assert config["global"]["services"]["router_replay"]["store_backend"] == "memory"
    assert result["cache_disabled"] is True
    assert result["bundle_sha256"] == "sha256:bundle"


def test_build_observation_config_fails_without_output(tmp_path):
    template = ol.template_info(
        bundle="mom-v1",
        recipe="balance",
        run_json=lambda argv, timeout_s, what: fake_list_document(),
    )
    with pytest.raises(ContractError, match="wrote no config"):
        ol.build_observation_config(
            template=template,
            out_dir=tmp_path,
            backend_endpoint="127.0.0.1:8000",
            run_json=lambda argv, timeout_s, what: {"valid": True},
        )


def test_lane_rows_join_by_request_id_and_count_missing():
    cases = [{"case_id": f"case-{i}"} for i in range(4)]
    records = [
        {"request_id": f"{ol.REQUEST_ID_PREFIX}case-{i}", "decision": lane}
        for i, lane in enumerate(["code", "fast", "standard"])
    ]
    rows, stats = ol.lane_rows_from_records(records, cases)
    assert [(row["case_id"], row["lane"]) for row in rows] == [
        ("case-0", "code"),
        ("case-1", "fast"),
        ("case-2", "standard"),
    ]
    assert stats["missing"] == 1  # case-3 produced no record
    # The lane is the record's decision name verbatim - no `other` folding.
    assert all(row["lane"] != "other" for row in rows)


def test_write_lanes_is_byte_stable(tmp_path):
    rows = [
        {"case_id": "b", "lane": "fast", "confidence": None},
        {"case_id": "a", "lane": "code", "confidence": 0.5},
    ]
    text1 = ol.write_lanes(rows, tmp_path / "lanes-1.jsonl")
    text2 = ol.write_lanes(list(reversed(rows)), tmp_path / "lanes-2.jsonl")
    assert text1.read_bytes() == text2.read_bytes()
    assert (
        text1.read_text(encoding="utf-8").splitlines()[0].startswith('{"case_id": "a"')
    )
