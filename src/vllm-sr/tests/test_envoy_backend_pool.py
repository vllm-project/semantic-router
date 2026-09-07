"""Regression coverage for the shared validate/Envoy backend seam."""

import re
import sys
from pathlib import Path

import pytest

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402


@pytest.mark.parametrize(
    ("backend_refs", "expected_error"),
    (
        (
            """
        - provider: vllm
          base_url: http://query.example.test/v1?region=west
""",
            "query parameters are not supported by Envoy routing",
        ),
        (
            """
        - provider: vllm
          endpoint: local.example:65536/v1
""",
            "expected 1..65535",
        ),
        (
            """
        - name: replica
          provider: vllm
          endpoint: local-a.example:8000/v1
        - name: replica
          provider: vllm
          endpoint: local-b.example:8000/v1
""",
            "name 'replica' is duplicated",
        ),
        (
            """
        - provider: vllm
          base_url: http://base.example/v1
          endpoint: endpoint.example:8000/v1
""",
            "cannot set both base_url and endpoint",
        ),
    ),
)
def test_envoy_generation_rejects_invalid_backend_group_at_shared_seam(
    tmp_path, backend_refs, expected_error
):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(
        f"""
version: v0.3
providers:
  defaults:
    model: local-model
  models:
    - name: local-model
      api_format: openai
      backend_refs:
{backend_refs}
routing: {{}}
"""
    )

    config = parse_user_config(str(config_path), log_summary=False)
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        generate_envoy_config_from_user_config(
            config,
            str(output_path),
            log_summary=False,
        )
