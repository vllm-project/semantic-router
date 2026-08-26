import pytest
import yaml
from cli.yaml_contract import load_yaml


def test_yaml12_boolean_rules_preserve_router_condition_keys() -> None:
    document = load_yaml(
        """
retry:
  on: [unavailable, timeout]
flags:
  enabled: true
  disabled: FALSE
  yes_word: yes
  no_word: no
"""
    )

    assert document["retry"] == {"on": ["unavailable", "timeout"]}
    assert document["flags"] == {
        "enabled": True,
        "disabled": False,
        "yes_word": "yes",
        "no_word": "no",
    }


def test_canonical_yaml_loader_rejects_duplicate_mapping_keys() -> None:
    with pytest.raises(
        yaml.constructor.ConstructorError, match="duplicate key 'count'"
    ):
        load_yaml("retry:\n  count: 1\n  count: 2\n")


def test_canonical_yaml_loader_preserves_merge_semantics() -> None:
    document = load_yaml(
        "defaults: &defaults\n  request: 30s\ntimeout:\n  <<: *defaults\n  stream: 2m\n"
    )

    assert document["timeout"] == {"request": "30s", "stream": "2m"}
