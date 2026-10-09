"""Regression coverage for bounded decimal scalar conversion."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_yaml import safe_load_router_config  # noqa: E402
from cli.context_bands import parse_token_count  # noqa: E402
from cli.goyaml_scalars import router_scalar_text  # noqa: E402


@pytest.mark.parametrize("sign", ["", "+", "-"])
@pytest.mark.parametrize("separator", ["", "_"])
def test_long_decimal_scalar_keeps_string_fallback(sign, separator):
    scalar = sign + separator.join("9" * 4301)
    assert router_scalar_text(scalar) == scalar


@pytest.mark.parametrize(
    ("scalar", "expected"),
    [
        ("18446744073709551615", "18446744073709551615"),
        ("18446744073709551616", "1.8446744073709552e+19"),
        ("99999999999999999999", "1e+20"),
        ("100000000000000000000", "1e+20"),
        ("+100000000000000000000", "1e+20"),
        ("-100000000000000000000", "-1e+20"),
    ],
)
def test_decimal_scalar_preserves_uint64_and_float_fallbacks(scalar, expected):
    assert router_scalar_text(scalar) == expected


@pytest.mark.parametrize("field", ["min_tokens", "max_tokens"])
def test_long_decimal_token_count_reaches_normal_validation(field):
    scalar = "9" * 4301
    document = (
        "routing:\n"
        "  signals:\n"
        "    context:\n"
        "      - name: long_decimal\n"
        f"        {field}: {scalar}\n"
    )
    data = safe_load_router_config(document)
    value = data["routing"]["signals"]["context"][0][field]
    assert value == scalar
    with pytest.raises(ValueError, match=r"^invalid token count format:"):
        parse_token_count(value)
