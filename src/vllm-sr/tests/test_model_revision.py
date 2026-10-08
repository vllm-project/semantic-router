"""Public model references become one immutable deployment identity."""

from types import SimpleNamespace

import pytest

from cli.model_revision import builtin_releases, resolve_model_revision


@pytest.fixture
def hub(monkeypatch):
    from huggingface_hub import HfApi, constants

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
    import huggingface_hub
    from huggingface_hub import constants

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
    from huggingface_hub import HfApi

    monkeypatch.setattr(
        HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="main")
    )
    with pytest.raises(ValueError, match="immutable"):
        resolve_model_revision("acme/custom", None)


def test_hub_errors_do_not_echo_response_or_credentials(monkeypatch, hub):
    from huggingface_hub import HfApi

    def denied(*args, **kwargs):
        raise OSError("secret-token in signed URL")

    monkeypatch.setattr(HfApi, "model_info", denied)
    with pytest.raises(ValueError, match="Cannot resolve") as error:
        resolve_model_revision("acme/custom", "main")
    assert "secret-token" not in str(error.value)
