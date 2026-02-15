"""Tests for legacy latency validation behavior."""

import os
import tempfile

from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _parse_config(config_yaml: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        temp_path = f.name

    try:
        return parse_user_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_validate_rejects_unknown_legacy_latency_signal_reference():
    config_yaml = """
version: v0.1
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  domains:
    - name: "math"
      description: "Math domain"
  latency:
    - name: "low_latency"
      tpot_percentile: 20
      ttft_percentile: 20
      description: "Prefer lower latency"
decisions:
  - name: "math_decision"
    description: "Math decision"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
        - type: "latency"
          name: "low_latecy"
    modelRefs:
      - model: "test_model"
    algorithm:
      type: "static"
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

    assert any(
        "unknown legacy latency signal 'low_latecy'" in str(error) for error in errors
    )


def test_validate_accepts_known_legacy_latency_signal_reference():
    config_yaml = """
version: v0.1
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  domains:
    - name: "math"
      description: "Math domain"
  latency:
    - name: "low_latency"
      tpot_percentile: 20
      ttft_percentile: 20
      description: "Prefer lower latency"
decisions:
  - name: "math_decision"
    description: "Math decision"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
        - type: "latency"
          name: "low_latency"
    modelRefs:
      - model: "test_model"
    algorithm:
      type: "static"
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
