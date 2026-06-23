import os
import tempfile

import yaml

from cli.config_migration import migrate_config_data
from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _parse_config(config_yaml: str):
    data = yaml.safe_load(config_yaml)
    providers = data.get("providers") if isinstance(data, dict) else None
    models = providers.get("models", []) if isinstance(providers, dict) else []
    if isinstance(data, dict) and (
        "signals" in data
        or "decisions" in data
        or any(isinstance(model, dict) and "endpoints" in model for model in models)
    ):
        data = migrate_config_data(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        temp_path = handle.name

    try:
        return parse_user_config(temp_path)
    finally:
        os.unlink(temp_path)


def _hybrid_config_yaml(hybrid_block: str) -> str:
    return f"""
version: v0.1
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  domains:
    - name: "general"
      description: "General domain"
decisions:
  - name: "hybrid_route"
    description: "Hybrid route"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "general"
    modelRefs:
      - model: "test_model"
    algorithm:
      type: "hybrid"
      hybrid:
{hybrid_block}
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""


def test_validate_user_config_rejects_all_zero_hybrid_weights():
    config = _parse_config(
        _hybrid_config_yaml(
            "        experience_weight: 0\n        router_dc_weight: 0\n"
            "        automix_weight: 0\n        cost_weight: 0"
        )
    )
    errors = validate_user_config(config)

    assert any("all zero" in e.message for e in errors)


def test_validate_user_config_accepts_partial_hybrid_weights():
    config_yaml = """
version: v0.1
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  domains:
    - name: "general"
      description: "General domain"
decisions:
  - name: "hybrid_route"
    description: "Hybrid route"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "general"
    modelRefs:
      - model: "test_model"
    algorithm:
      type: "hybrid"
      hybrid:
        experience_weight: 0.6
        router_dc_weight: 0.4
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

    config = _parse_config(config_yaml)
    errors = validate_user_config(config)

    assert errors == []
