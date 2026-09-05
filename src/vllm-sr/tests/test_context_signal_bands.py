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

from cli.context_bands import parse_go_float, parse_token_count  # noqa: E402
from cli.models import ContextRule, UserConfig  # noqa: E402

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


def router_token_count_cases():
    if not ROUTER_TOKEN_COUNT_CASES.is_file():
        pytest.fail(f"shared token count contract missing: {ROUTER_TOKEN_COUNT_CASES}")
    cases = json.loads(ROUTER_TOKEN_COUNT_CASES.read_text(encoding="utf-8"))["cases"]
    return [pytest.param(case, id=repr(case["input"])) for case in cases]


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
