import stat

import pytest
from cli.masking_identity import (
    LOCAL_NAMESPACE,
    inject_local_masking_identity,
    provision_local_masking,
)


def _masking(**configuration):
    return {"type": "masking", "configuration": {"enabled": True, **configuration}}


def test_provision_creates_a_private_key_once(tmp_path):
    key = provision_local_masking(tmp_path, "default")
    assert key == tmp_path / "masking" / "default" / "key"
    assert len(key.read_bytes()) == 32
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    first = key.read_bytes()
    assert provision_local_masking(tmp_path, "default").read_bytes() == first


def test_provision_refuses_a_registry_without_its_key(tmp_path):
    key = provision_local_masking(tmp_path, "default")
    (key.parent / "registry.log").write_bytes(b"history")
    key.unlink()
    with pytest.raises(RuntimeError, match="Restore the key"):
        provision_local_masking(tmp_path, "default")
    assert not key.exists()


def test_provision_refuses_a_key_of_the_wrong_size(tmp_path):
    key = provision_local_masking(tmp_path, "default")
    key.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="32 bytes"):
        provision_local_masking(tmp_path, "default")


def test_inject_names_the_local_namespace_for_unset_plugins():
    unset = _masking(signal="pii")
    sequence = _masking(signal="pii", index="sequence")
    named = _masking(signal="pii", identity_namespace="team")
    config = {
        "routing": {"decisions": [{"plugins": [unset, sequence]}]},
        "recipes": [{"decisions": [{"plugins": [named]}]}],
    }
    assert inject_local_masking_identity(config, "default")
    assert unset["configuration"]["identity_namespace"] == LOCAL_NAMESPACE
    assert "identity_namespace" not in sequence["configuration"]
    assert named["configuration"]["identity_namespace"] == "team"
    namespace = config["global"]["services"]["masking_identity"]["namespaces"][
        LOCAL_NAMESPACE
    ]
    assert namespace == {
        "key_file": "/app/.vllm-sr/masking/default/key",
        "scope": "principal_or_installation",
        "registry": {
            "backend": "file",
            "path": "/app/.vllm-sr/masking/default/registry.log",
        },
    }


def test_inject_leaves_configs_without_masking_alone():
    config = {"routing": {"decisions": [{"plugins": [{"type": "system_prompt"}]}]}}
    assert not inject_local_masking_identity(config, "default")
    assert "global" not in config


def test_inject_keeps_a_local_namespace_the_user_wrote():
    own = {"key_env": "MASKING_KEY", "scope": "principal_or_installation"}
    config = {
        "global": {"services": {"masking_identity": {"namespaces": {"local": own}}}},
        "routing": {"decisions": [{"plugins": [_masking(signal="pii")]}]},
    }
    assert inject_local_masking_identity(config, "default")
    assert (
        config["global"]["services"]["masking_identity"]["namespaces"]["local"] is own
    )
