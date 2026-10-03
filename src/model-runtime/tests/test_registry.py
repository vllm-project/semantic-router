import pytest
from vllm_sr_runtime.errors import PackageError
from vllm_sr_runtime.registry import builtin, policy
from vllm_sr_runtime.registry.resolve import pinned_revision, resolve


def test_builtin_table_pins_every_phase1_model():
    names = [model.repo_id for model in builtin.all_models()]
    assert names == [
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "vllm-sr/Decision-2.0-Eos-0.8B",
        "vllm-sr/Decision-2.0-Sol-2B",
        "vllm-sr/Decision-2.0-Nox-4B",
        "vllm-sr/Decision-2.0-Lux-9B",
        "vllm-sr/Decision-2.0-Vega-27B",
    ]
    for model in builtin.all_models():
        assert (
            len(model.revision) == 40
            and len(model.model_sha256) == 64
            and len(model.manifest_sha256) == 64
        )
    assert builtin.lookup("decision-2.0-kai-0.6b").revision.startswith("881bee41")
    assert builtin.lookup("vllm-sr/Decision-2.0-Vega-27B").base[0] == "Qwen/Qwen3.8-27B"


def test_revisions_are_always_pinned():
    assert pinned_revision("vllm-sr/Decision-2.0-Eos-0.8B", None).startswith("ad0aa724")
    assert (
        pinned_revision("vllm-sr/Decision-2.0-Eos-0.8B", "ad0aa724")
        == builtin.lookup("Decision-2.0-Eos-0.8B").revision
    )
    assert pinned_revision("acme/model", "a" * 40) == "a" * 40
    with pytest.raises(PackageError, match="not a built-in"):
        pinned_revision("acme/model", None)
    with pytest.raises(PackageError, match="40-hex"):
        pinned_revision("acme/model", "main")
    with pytest.raises(PackageError, match="40-hex"):
        pinned_revision("vllm-sr/Decision-2.0-Eos-0.8B", "deadbeef")


def test_local_directories_resolve_without_revision(qwen3_package):
    assert resolve(str(qwen3_package)).root == qwen3_package.resolve()
    with pytest.raises(PackageError):
        resolve(str(qwen3_package), revision="a" * 40)


def test_offline_resolution_without_cache_fails_cleanly(tmp_path):
    with pytest.raises((OSError, ValueError, PackageError)):
        resolve("vllm-sr/Decision-2.0-Kai-0.6B", cache_dir=tmp_path, offline=True)


def test_licence_policy():
    assert (
        policy.check(
            {
                "licence": {
                    "spdx": "apache-2.0",
                    "components": [{"licence": "apache-2.0"}],
                }
            }
        )
        == "apache-2.0"
    )
    with pytest.raises(PackageError):
        policy.check(
            {
                "licence": {
                    "spdx": "apache-2.0",
                    "components": [{"licence": "cc-by-nc-4.0"}],
                }
            }
        )
    assert (
        policy.check({"licence": {"spdx": "cc-by-nc-4.0"}}, ("CC-BY-NC-4.0",))
        == "cc-by-nc-4.0"
    )


def test_every_builtin_model_has_cpu_golden_answers():
    from vllm_sr_runtime.supervision.readiness import well_formed

    for model in builtin.all_models():
        answers = model.golden_answers.get("cpu")
        assert answers, model.repo_id
        assert set(answers) == {"domain", "reasoning", "difficulty"}
        assert all(well_formed(answer) for answer in answers.values()), model.repo_id
