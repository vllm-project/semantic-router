"""ContextRule schema tests: bounded, exact-match, open-ended, and invalid bands.

These mirror the Router's ContextRule.Bounds contract so a configuration the
CLI accepts also loads in the Router.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.context_bands import parse_token_count  # noqa: E402
from cli.models import ContextRule, UserConfig  # noqa: E402


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
