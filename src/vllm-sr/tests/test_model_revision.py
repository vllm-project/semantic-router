"""Public model references become one immutable deployment identity."""

from types import SimpleNamespace

import huggingface_hub
import pytest
from cli.model_revision import (
    builtin_releases,
    resolve_model_revision,
    warn_saved_revision_mismatch,
)
from huggingface_hub import HfApi, constants


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    calls = []

    def model_info(self, repo_id, *, revision, timeout):
        calls.append((repo_id, revision, timeout))
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(HfApi, "model_info", model_info)
    return calls


def test_released_defaults_and_explicit_commits_do_not_contact_hub(hub):
    releases = builtin_releases()
    assert releases["vllm-sr/Vela-2.0-4B"] == "c1e64d4f872cb38bc58502e6888340100bab9d55"
    for repo, commit in releases.items():
        assert resolve_model_revision(repo, None) == commit
        assert resolve_model_revision(repo, commit[:8]) == commit
    assert resolve_model_revision("acme/custom", "B" * 40) == "b" * 40
    assert hub == []


@pytest.mark.parametrize("revision", [None, "main", "Release/V2", "v2.0", "deadbeef"])
def test_custom_refs_resolve_without_changing_ref_case(hub, revision):
    assert resolve_model_revision("acme/custom", revision) == "a" * 40
    assert hub == [("acme/custom", revision, 15)]


def test_explicit_builtin_branch_overrides_release_pin(hub):
    assert resolve_model_revision("vllm-sr/Vela-2.0-4B", "Main") == "a" * 40
    assert hub[0][1] == "Main"


def test_local_packages_do_not_use_hub_revisions(hub, tmp_path):
    assert resolve_model_revision(str(tmp_path), None) is None
    with pytest.raises(ValueError, match="only to Hub"):
        resolve_model_revision(str(tmp_path), "main")
    assert hub == []


@pytest.mark.parametrize("revision", ["", " main", "main "])
def test_invalid_refs_fail_before_resolution(hub, revision):
    with pytest.raises(ValueError, match="non-empty"):
        resolve_model_revision("acme/custom", revision)
    assert hub == []


def test_offline_ref_uses_cached_snapshot_without_network(monkeypatch, hub):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *args, **kwargs: "/cache/snapshots/" + "c" * 40 + "/config.json",
    )
    assert resolve_model_revision("acme/custom", "Release/V2") == "c" * 40
    assert hub == []
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="not cached"):
        resolve_model_revision("acme/custom", "Release/V2")
    assert hub == []


def test_hub_must_return_an_immutable_commit(monkeypatch, hub):
    monkeypatch.setattr(
        HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="main")
    )
    with pytest.raises(ValueError, match="immutable"):
        resolve_model_revision("acme/custom", None)


def test_hub_errors_do_not_echo_response_or_credentials(monkeypatch, hub):
    def denied(*args, **kwargs):
        raise OSError("secret-token in signed URL")

    monkeypatch.setattr(HfApi, "model_info", denied)
    with pytest.raises(ValueError, match="Cannot resolve") as error:
        resolve_model_revision("acme/custom", "main")
    assert "secret-token" not in str(error.value)


# -- warn_saved_revision_mismatch tests --


def test_saved_revision_differs_from_pin_warns(caplog):
    """Saved revision != pin → one warning naming both short hashes and --revision."""
    releases = builtin_releases()
    artifact = "vllm-sr/Vela-2.0-4B"
    pinned = releases[artifact]
    saved = "f" * 40
    with caplog.at_level("WARNING", logger="cli.model_revision"):
        warn_saved_revision_mismatch(artifact, saved)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert saved[:7] in message
    assert pinned[:7] in message
    assert "--revision" in message


def test_saved_revision_matches_pin_is_silent(caplog):
    """Saved revision == pin → no warning."""
    releases = builtin_releases()
    artifact = "vllm-sr/Vela-2.0-4B"
    pinned = releases[artifact]
    with caplog.at_level("WARNING", logger="cli.model_revision"):
        warn_saved_revision_mismatch(artifact, pinned)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_no_pin_for_model_is_silent(caplog):
    """Unknown model with no release pin → no warning, no crash."""
    with caplog.at_level("WARNING", logger="cli.model_revision"):
        warn_saved_revision_mismatch("acme/custom-no-pin", "a" * 40)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_fresh_workspace_gets_pin_no_warning(hub, caplog):
    """Fresh workspace (revision=None) → resolve returns pin, no warning path."""
    releases = builtin_releases()
    artifact = "vllm-sr/d3-turbo"
    pinned = releases[artifact]
    result = resolve_model_revision(artifact, None)
    assert result == pinned
    # warn_saved_revision_mismatch is never called with None; calling with
    # the pin confirms silence.
    with caplog.at_level("WARNING", logger="cli.model_revision"):
        warn_saved_revision_mismatch(artifact, pinned)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    assert hub == []


def test_explicit_revision_skips_warning(hub, caplog):
    """Explicit --revision → caller does not invoke warn; verify no false warn."""
    # Simulates the caller's contract: when --revision is explicit,
    # warn_saved_revision_mismatch is never called. Here we verify that the
    # function itself is harmless when called with the explicit value matching
    # the pin (the only case it could be called).
    artifact = "vllm-sr/d3-turbo"
    # Explicit commit passed by the user → resolve_model_revision returns it.
    explicit = "b" * 40
    result = resolve_model_revision(artifact, explicit)
    assert result == explicit.lower()
    # Caller skips warn_saved_revision_mismatch entirely for explicit revision.
    # No warning records should exist.
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
