"""ContextRule schema tests: bounded, exact-match, open-ended, and invalid bands.

These mirror the Router's ContextRule.Bounds contract so a configuration the
CLI accepts also loads in the Router.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_yaml import safe_load_router_config  # noqa: E402
from cli.context_bands import parse_go_float, parse_token_count  # noqa: E402
from cli.goyaml_scalars import format_go_float, router_scalar_text  # noqa: E402
from cli.models import ContextRule, UserConfig  # noqa: E402
from cli.parser import (  # noqa: E402
    ConfigParseError,
    load_config_file,
    parse_user_config,
)

# The Router's TokenCount test loads the same file, so the CLI and Router
# parsers are pinned to one contract and cannot drift apart silently.
ROUTER_TOKEN_COUNT_CASES = (
    PROJECT_ROOT.parents[1]
    / "src"
    / "semantic-router"
    / "pkg"
    / "config"
    / "testdata"
    / "token_count_cases.json"
)


def router_token_count_cases(section="cases", key="input"):
    if not ROUTER_TOKEN_COUNT_CASES.is_file():
        pytest.fail(f"shared token count contract missing: {ROUTER_TOKEN_COUNT_CASES}")
    cases = json.loads(ROUTER_TOKEN_COUNT_CASES.read_text(encoding="utf-8"))[section]
    return [pytest.param(case, id=repr(case[key])) for case in cases]


def test_bounded_band_still_parses():
    rule = ContextRule(name="medium", min_tokens="8001", max_tokens="64K")

    assert rule.min_tokens == "8001"
    assert rule.max_tokens == "64K"


def test_open_ended_band_omits_max_tokens():
    rule = ContextRule(name="long", min_tokens="64001")

    assert rule.max_tokens is None


def test_empty_max_tokens_is_open_ended():
    rule = ContextRule(name="long", min_tokens="64001", max_tokens="")

    assert rule.max_tokens == ""


def test_band_with_only_max_tokens_defaults_min_to_zero():
    rule = ContextRule(name="short", max_tokens="4K")

    assert rule.min_tokens is None


def test_exact_match_band_accepts_equal_limits():
    rule = ContextRule(name="exact", min_tokens="4096", max_tokens="4096")

    assert rule.min_tokens == rule.max_tokens


@pytest.mark.parametrize("value", [8001, 8001.0, 0])
def test_plain_yaml_numbers_are_accepted(value):
    rule = ContextRule(name="numeric", min_tokens=value)

    assert rule.min_tokens == str(value)


def test_band_with_neither_limit_is_rejected():
    with pytest.raises(ValidationError, match="min_tokens or max_tokens must be set"):
        ContextRule(name="empty")


def test_min_above_max_is_rejected():
    with pytest.raises(ValidationError, match="must not exceed max_tokens"):
        ContextRule(name="inverted", min_tokens="64K", max_tokens="8K")


@pytest.mark.parametrize("value", ["abc", "nan", "inf", "-1", "1.5G"])
def test_unparsable_or_negative_max_tokens_is_rejected(value):
    with pytest.raises(ValidationError, match="max_tokens"):
        ContextRule(name="bad", min_tokens="0", max_tokens=value)


@pytest.mark.parametrize(
    "value", ["1e100", "9223372036854775807", "9223372036854775808K", "1e19M"]
)
def test_oversized_token_count_is_rejected_like_router(value):
    """TokenCount.Value() rejects scaled values at math.MaxInt; so must the CLI."""
    with pytest.raises(ValidationError, match="min_tokens: token count is too large"):
        ContextRule(name="bad", min_tokens=value)
    with pytest.raises(ValidationError, match="max_tokens: token count is too large"):
        ContextRule(name="bad", min_tokens="0", max_tokens=value)


def test_boolean_token_count_is_rejected():
    with pytest.raises(ValidationError, match="min_tokens must be a token count"):
        ContextRule(name="bad", min_tokens=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("", 0),
        ("  ", 0),
        ("4096", 4096),
        ("4k", 4000),
        ("1.5K", 1500),
        ("0.5M", 500000),
        (" 64K ", 64000),
        ("9e18", 9_000_000_000_000_000_000),
    ],
)
def test_parse_token_count_matches_router(value, expected):
    assert parse_token_count(value) == expected


@pytest.mark.parametrize("value", ["1e100", "9223372036854775807", "9.3e18"])
def test_parse_token_count_overflow_matches_router(value):
    with pytest.raises(ValueError, match="token count is too large"):
        parse_token_count(value)


def test_user_config_loads_open_ended_final_band():
    config = UserConfig(
        version="0.3",
        routing={
            "signals": {
                "context": [
                    {"name": "short", "min_tokens": 0, "max_tokens": "8K"},
                    {"name": "medium", "min_tokens": 8001, "max_tokens": "64K"},
                    {"name": "long", "min_tokens": 64001},
                ]
            }
        },
    )

    bands = config.routing.signals.context
    assert [band.name for band in bands] == ["short", "medium", "long"]
    assert bands[-1].max_tokens is None


def test_user_config_reports_context_band_path_on_error():
    with pytest.raises(ValidationError) as excinfo:
        UserConfig(
            version="0.3",
            routing={"signals": {"context": [{"name": "empty"}]}},
        )

    assert excinfo.value.errors()[0]["loc"][:4] == ("routing", "signals", "context", 0)


@pytest.mark.parametrize("case", router_token_count_cases())
def test_parse_token_count_matches_router_contract(case):
    """Every input the Router accepts or rejects must behave the same in the CLI."""
    assert ("value" in case) != (
        "error" in case
    ), "case must set exactly one of value or error"
    if "error" in case:
        with pytest.raises(ValueError, match=f"^{case['error']}: "):
            parse_token_count(case["input"])
    else:
        assert parse_token_count(case["input"]) == case["value"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0x1p10", 1024.0),
        ("0X1.8P3", 12.0),
        ("1_000.5", 1000.5),
        ("1e1_0", 1e10),
        ("0x1p-2", 0.25),
        ("1e400", float("inf")),
        ("0x1p99999", float("inf")),
    ],
)
def test_parse_go_float_accepts_go_syntax(text, expected):
    assert parse_go_float(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "\uff11\uff12\uff18",
        "0x10",
        "1__0",
        "_1",
        "1_",
        "1_.5",
        "1e_5",
        "inf",
        "nan",
        "1,000",
        "",
    ],
)
def test_parse_go_float_rejects_non_go_syntax(text):
    with pytest.raises(ValueError):
        parse_go_float(text)


def test_hex_float_band_accepted_like_router():
    """A pre-existing config using Go hex floats must keep loading in the CLI."""
    rule = ContextRule(name="hex", min_tokens="0x1p10", max_tokens="0x1p20")

    assert parse_token_count(rule.min_tokens) == 1024
    assert parse_token_count(rule.max_tokens) == 1048576


def test_full_width_digits_rejected_like_router():
    with pytest.raises(ValidationError, match="min_tokens: invalid token count format"):
        ContextRule(name="wide", min_tokens="\uff11\uff12\uff18")


# ---------------------------------------------------------------------------
# YAML boundary: the Router loader decodes config.yaml untyped with yaml.v2,
# re-marshals it, and only then decodes the typed config, so TokenCount sees
# yaml.v2's rendering of a plain scalar. cli.config_yaml reproduces that for
# the token-count fields so Pydantic validates the same text.
# ---------------------------------------------------------------------------


def _context_band_document(scalar: str) -> str:
    """The yaml_cases document the Go test loads, under the default profile."""
    return (
        "version: '0.3'\n"
        "routing:\n"
        "  signals:\n"
        "    context:\n"
        "      - name: probe\n"
        f"        min_tokens: {scalar}\n"
        "        max_tokens: 100M\n"
    )


@pytest.mark.parametrize("case", router_token_count_cases("yaml_cases", "scalar"))
def test_yaml_scalar_reaches_pydantic_as_the_router_reads_it(case):
    """YAML -> cli.config_yaml -> ContextRule must match YAML -> Router loader -> TokenCount."""
    assert ("value" in case) != (
        "error" in case
    ), "case must set exactly one of value or error"
    data = safe_load_router_config(_context_band_document(case["scalar"]))
    loaded = data["routing"]["signals"]["context"][0]["min_tokens"]
    assert ("" if loaded is None else loaded) == case["text"]
    if "error" in case:
        with pytest.raises(ValidationError, match=f"min_tokens: {case['error']}"):
            UserConfig(**data)
        return
    config = UserConfig(**data)
    assert (
        parse_token_count(config.routing.signals.context[0].min_tokens) == case["value"]
    )


@pytest.mark.parametrize(
    ("scalar", "expected"),
    [
        ("1:30", "1:30"),
        ("0o17", "15"),
        ("0X10", "16"),
        ("1_e3", "1000"),
        ("2001-12-14", "2001-12-14"),
        ("y", "true"),
        ("~", None),
        ("\uff11\uff12\uff18", "\uff11\uff12\uff18"),
    ],
)
def test_router_scalar_text_follows_yaml_v2_not_pyyaml(scalar, expected):
    """The forms where PyYAML and yaml.v2 implicit typing disagree."""
    assert router_scalar_text(scalar) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (-0.0, "-0"),
        (5.0, "5"),
        (0.5, "0.5"),
        (1000.5, "1000.5"),
        (123456.0, "123456"),
        (1e6, "1e+06"),
        (1234567.0, "1.234567e+06"),
        (0.0001, "0.0001"),
        (0.00001, "1e-05"),
        (1e20, "1e+20"),
        (1.8446744073709552e19, "1.8446744073709552e+19"),
        (float("inf"), ".inf"),
        (float("-inf"), "-.inf"),
    ],
)
def test_format_go_float_matches_strconv_g(value, expected):
    assert format_go_float(value) == expected


def test_safe_load_router_config_types_token_counts_only_in_context_rules():
    data = safe_load_router_config(
        "routing:\n"
        "  signals:\n"
        "    context:\n"
        "      - {name: a, min_tokens: 0123, max_tokens: 1:30}\n"
        "      - {name: b, min_tokens: ~, max_tokens: 8K}\n"
        "recipes:\n"
        "  - name: r\n"
        "    routing:\n"
        "      signals:\n"
        "        context:\n"
        "          - {name: c, min_tokens: 0o17}\n"
        "elsewhere: {min_tokens: 0123, other: 1:30}\n"
    )

    default_bands = data["routing"]["signals"]["context"]
    assert default_bands[0] == {"name": "a", "min_tokens": "83", "max_tokens": "1:30"}
    assert default_bands[1] == {"name": "b", "min_tokens": None, "max_tokens": "8K"}
    recipe_bands = data["recipes"][0]["routing"]["signals"]["context"]
    assert recipe_bands[0] == {"name": "c", "min_tokens": "15"}
    # Keys of the same name outside a context rule keep PyYAML's typing.
    assert data["elsewhere"] == {"min_tokens": 83, "other": 90}


def test_safe_load_router_config_leaves_quoted_and_tagged_scalars_literal():
    data = safe_load_router_config(
        "routing:\n"
        "  signals:\n"
        "    context:\n"
        "      - {name: a, min_tokens: '0123', max_tokens: \"1:30\"}\n"
        "      - {name: b, min_tokens: !!str 0123, max_tokens: !!int 0123}\n"
    )

    quoted, tagged = data["routing"]["signals"]["context"]
    assert (quoted["min_tokens"], quoted["max_tokens"]) == ("0123", "1:30")
    # An explicit tag that matches implicit typing re-emits like yaml.v2 too.
    assert (tagged["min_tokens"], tagged["max_tokens"]) == ("0123", "83")


def test_safe_load_router_config_matches_safe_load_for_empty_document():
    assert safe_load_router_config("") is None
    assert safe_load_router_config("# only a comment\n") is None


def test_merge_key_token_counts_are_typed_like_router():
    data = safe_load_router_config(
        "version: '0.3'\n"
        "routing:\n"
        "  signals:\n"
        "    context:\n"
        "      - &base {name: base, min_tokens: 0123, max_tokens: 0x1p10}\n"
        "      - <<: *base\n"
        "        name: inherited\n"
    )

    base, inherited = UserConfig(**data).routing.signals.context
    assert (base.min_tokens, base.max_tokens) == ("83", "0x1p10")
    assert (inherited.min_tokens, inherited.max_tokens) == ("83", "0x1p10")
    assert parse_token_count(inherited.min_tokens) == 83


def test_recipe_context_band_is_validated_as_the_router_reads_it():
    data = safe_load_router_config(
        "version: '0.3'\n"
        "recipes:\n"
        "  - name: long-form\n"
        "    routing:\n"
        "      signals:\n"
        "        context:\n"
        "          - name: sexagesimal\n"
        "            min_tokens: 1:30\n"
    )

    with pytest.raises(ValidationError) as excinfo:
        UserConfig(**data)

    error = excinfo.value.errors()[0]
    assert error["loc"][:6] == ("recipes", 0, "routing", "signals", "context", 0)
    assert "min_tokens: invalid token count format: 1:30" in error["msg"]


def _minimal_config_with_context_band(min_tokens: str, max_tokens: str) -> str:
    return (
        "version: v0.3\n"
        "listeners:\n"
        "  - {name: http-8899, address: 0.0.0.0, port: 8899}\n"
        "providers:\n"
        "  defaults: {default_model: demo-model}\n"
        "  models:\n"
        "    - name: demo-model\n"
        "      backend_refs: [{endpoint: 127.0.0.1:8000}]\n"
        "routing:\n"
        "  modelCards: [{name: demo-model}]\n"
        "  signals:\n"
        "    context:\n"
        "      - name: probe\n"
        f"        min_tokens: {min_tokens}\n"
        f"        max_tokens: {max_tokens}\n"
        "  decisions:\n"
        "    - name: default-route\n"
        "      description: fallback\n"
        "      priority: 100\n"
        "      rules: {operator: AND, conditions: []}\n"
        "      modelRefs: [{model: demo-model}]\n"
    )


def test_parse_user_config_rejects_base60_token_count_like_router(tmp_path):
    """PyYAML reads 1:30 as 90; the Router, which gets the forwarded file, rejects it."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _minimal_config_with_context_band("0", "1:30"), encoding="utf-8"
    )

    with pytest.raises(
        ConfigParseError, match="max_tokens: invalid token count format: 1:30"
    ):
        parse_user_config(str(config_path), log_summary=False)


def test_parse_user_config_reads_yaml_typed_token_count_like_router(tmp_path):
    """Unquoted 0123 is YAML 1.1 octal to the Router's decoder, so the CLI sees 83."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _minimal_config_with_context_band("0123", "64K"), encoding="utf-8"
    )

    config = parse_user_config(str(config_path), log_summary=False)
    band = config.routing.signals.context[0]
    assert (band.min_tokens, band.max_tokens) == ("83", "64K")
    assert parse_token_count(band.min_tokens) == 83

    raw = load_config_file(str(config_path))
    assert raw["routing"]["signals"]["context"][0]["min_tokens"] == "83"
