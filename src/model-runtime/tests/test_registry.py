from dataclasses import replace

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
    assert builtin.lookup("decision-2.0-kai-0.6b").revision.startswith("cd49ea38")
    assert builtin.lookup("vllm-sr/Decision-2.0-Vega-27B").base[0] == "Qwen/Qwen3.8-27B"


def test_revisions_are_always_pinned():
    assert pinned_revision("vllm-sr/Decision-2.0-Eos-0.8B", None).startswith("3594047d")
    assert (
        pinned_revision("vllm-sr/Decision-2.0-Eos-0.8B", "3594047d")
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


def test_every_builtin_model_has_cpu_and_rocm_golden_answers():
    from vllm_sr_runtime.supervision.readiness import well_formed

    for model in builtin.all_models():
        for device_class in ("cpu", "rocm"):
            answers = model.golden_answers.get(device_class)
            assert answers, (model.repo_id, device_class)
            assert set(answers) == {"domain", "reasoning", "difficulty"}
            assert all(
                well_formed(answer) for answer in answers.values()
            ), model.repo_id


def test_a_later_revision_with_the_same_identity_gets_the_table_references(tmp_path):
    from vllm_sr_runtime.families.decision2.family import Decision2Family
    from vllm_sr_runtime.plugins.base import (
        PackageRef,
        RegistryOptions,
        VerifiedPackage,
    )

    known = builtin.lookup("vllm-sr/Decision-2.0-Eos-0.8B")
    package = VerifiedPackage(
        ref=PackageRef(root=tmp_path, repo_id=known.repo_id, revision="b" * 40),
        family="decision2",
        model_name="Decision-2.0-Eos-0.8B",
        manifest={},
        manifest_sha256="c" * 64,
        model_sha256=known.model_sha256,
        max_input_tokens=4096,
        licence="apache-2.0",
    )
    family = Decision2Family(RegistryOptions())
    (golden,) = family.golden(package)
    assert golden["expected"] == known.golden_answers
    (other,) = family.golden(replace(package, model_sha256="d" * 64))
    assert other["expected"] == {}
