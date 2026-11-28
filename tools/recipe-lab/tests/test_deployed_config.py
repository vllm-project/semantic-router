"""Check both byte identities and all authored fields across a CLI projection."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recipe_lab.deployed_config import config_identity
from recipe_lab.loop_evidence import digest


def materialized(tmp_path, *, edit=None, corrupt=None, tracing=None):
    source = tmp_path / "proposal.yaml"
    active = tmp_path / "runtime-config.dev.yaml"
    doc = {
        "version": "v0.5",
        "providers": {"models": [{"name": "original"}]},
        "recipes": [{"name": "test", "decisions": [{"plugins": ["safety"]}]}],
    }
    if tracing is not None:
        doc["global"] = {"services": {"observability": {"tracing": tracing}}}
    source.write_text(yaml.safe_dump(doc), encoding="utf-8")
    deployed = copy.deepcopy(doc)
    deployed["global"] = {
        "services": {
            "response_api": {
                "store_backend": "redis",
                "redis": {"address": "dev-vllm-sr-redis:6379"},
            },
            "management_api": {"port": 8080, "bind_address": "0.0.0.0"},
        }
    }
    if tracing is not None:
        deployed["global"]["services"]["observability"] = {
            "tracing": copy.deepcopy(tracing)
        }
    if edit is not None:
        edit(deployed)
    active.write_text(yaml.safe_dump(deployed), encoding="utf-8")
    receipt = {
        "schema_version": "vllm-sr/runtime-config-provenance/v1",
        "source_digest": "sha256:" + digest(source.read_bytes()),
        "last_materialized_active_digest": "sha256:" + digest(active.read_bytes()),
    }
    if corrupt:
        receipt[corrupt] = "sha256:" + "a" * 64
    active.with_suffix(".provenance.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return source, active


def test_binds_cli_derived_bytes_and_preserves_authored_identity(tmp_path):
    source, active = materialized(tmp_path)
    raw, doc, observed = config_identity(source, active)
    assert raw == source.read_bytes() and doc == yaml.safe_load(raw)
    assert observed == digest(active.read_bytes()) != digest(raw)
    assert config_identity(source) == (raw, doc, digest(raw))


@pytest.mark.parametrize("field", ["source_digest", "last_materialized_active_digest"])
def test_wrong_sidecar_hash_is_rejected(tmp_path, field):
    with pytest.raises(ValueError, match="provenance"):
        config_identity(*materialized(tmp_path, corrupt=field))


@pytest.mark.parametrize(
    "edit",
    [
        lambda dr: dr["recipes"][0]["decisions"][0].update(plugins=["other"]),
        lambda dr: dr["providers"]["models"][0].update(name="other"),
        lambda dr: dr.update(listeners=[]),
        lambda dr: dr["global"]["services"]["response_api"]["redis"].update(
            address="other:6379"
        ),
        lambda dr: dr["global"]["services"].update(unknown_service={}),
    ],
)
def test_matching_receipt_cannot_hide_a_different_policy_or_backend(tmp_path, edit):
    with pytest.raises(ValueError):
        config_identity(*materialized(tmp_path, edit=edit))


@pytest.mark.parametrize(
    "source,active",
    [
        ({}, {"enabled": False, "exporter": {}}),
        ({"enabled": True}, {"enabled": False, "exporter": {}}),
        (
            {"exporter": {"endpoint": "vllm-sr-jaeger:4317"}},
            {"exporter": {"endpoint": "dev-vllm-sr-jaeger:4317"}},
        ),
    ],
)
def test_local_tracing_projections(tmp_path, source, active):
    def project(document):
        document["global"]["services"]["observability"]["tracing"] = active

    src, dst = materialized(tmp_path, tracing=source, edit=project)
    assert config_identity(src, dst)[2] == digest(dst.read_bytes())


@pytest.mark.parametrize(
    "source,active",
    [
        ({}, {"enabled": False, "exporter": {"unknown": {}}}),
        (
            {"exporter": {"endpoint": "external.example:4317"}},
            {"exporter": {"endpoint": "dev-vllm-sr-jaeger:4317"}},
        ),
        (
            {"enabled": False, "exporter": {"endpoint": "vllm-sr-jaeger:4317"}},
            {"enabled": False, "exporter": {"endpoint": "dev-vllm-sr-jaeger:4317"}},
        ),
        ({}, {"exporter": {}}),
    ],
)
def test_unknown_tracing_changes_stay_disqualified(tmp_path, source, active):
    def project(document):
        document["global"]["services"]["observability"]["tracing"] = active

    with pytest.raises(ValueError):
        config_identity(*materialized(tmp_path, tracing=source, edit=project))
