from cli.config_contract import (
    LEGACY_SIGNAL_KEY_TO_CANONICAL,
    build_signal_reference_index,
    signal_reference_exists,
)
from cli.models import Signals


def test_legacy_signal_inventory_covers_flat_authz_and_context_blocks():
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["role_bindings"] == "role_bindings"
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["context_rules"] == "context"


def test_build_signal_reference_index_expands_complexity_levels_and_authz_names():
    signals = Signals(
        complexity=[
            {
                "name": "difficulty",
                "easy": {"candidates": ["simple"]},
                "hard": {"candidates": ["complex"]},
            }
        ],
        role_bindings=[
            {
                "name": "admin-access",
                "role": "admin",
                "subjects": [{"kind": "User", "name": "alice"}],
            }
        ],
    )

    signal_names = build_signal_reference_index(signals)

    assert "difficulty:easy" in signal_names
    assert "difficulty:medium" in signal_names
    assert "difficulty:hard" in signal_names
    assert "admin-access" in signal_names


def test_signal_reference_exists_strips_suffixes_for_non_complexity_signals():
    signal_names = {"security", "admin-access"}

    assert signal_reference_exists(signal_names, "keyword", "security:match")
    assert signal_reference_exists(signal_names, "authz", "admin-access")
    assert not signal_reference_exists(signal_names, "complexity", "security:match")
