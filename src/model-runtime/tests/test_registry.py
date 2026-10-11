from dataclasses import replace

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.registry import builtin, policy
from vllm_srun.registry.artifacts import read_json
from vllm_srun.registry.resolve import pinned_revision, resolve


def test_builtin_table_pins_every_phase1_model():
    names = [model.repo_id for model in builtin.all_models("decision2")]
    assert names == [
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "vllm-sr/Decision-2.0-Eos-0.8B",
        "vllm-sr/Decision-2.0-Sol-2B",
        "vllm-sr/Decision-2.0-Nox-4B",
        "vllm-sr/Decision-2.0-Lux-9B",
        "vllm-sr/Decision-2.0-Vega-27B",
    ]
    for model in builtin.all_models("decision2"):
        assert (
            len(model.revision) == 40
            and len(model.model_sha256) == 64
            and len(model.manifest_sha256) == 64
        )
    assert builtin.lookup("decision-2.0-kai-0.6b").revision.startswith("cd49ea38")
    assert builtin.lookup("vllm-sr/Decision-2.0-Vega-27B").base[0] == "Qwen/Qwen3.8-27B"


def test_every_family_reads_package_json_one_way(tmp_path):
    (tmp_path / "list.json").write_text("[1]", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert read_json(tmp_path / "list.json") == [1]
    with pytest.raises(PackageError, match=r"list\.json must hold a JSON object"):
        read_json(tmp_path / "list.json", mapping=True)
    for name in ("bad.json", "missing.json"):
        with pytest.raises(PackageError, match=f"cannot read {name}"):
            read_json(tmp_path / name)


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
    from vllm_srun.plugins.decisions import well_formed

    for model in builtin.all_models("decision2"):
        for device_class in ("cpu", "rocm"):
            answers = model.golden_answers.get(device_class)
            assert answers, (model.repo_id, device_class)
            assert set(answers) == {"domain", "reasoning", "difficulty"}
            assert all(
                well_formed(answer) for answer in answers.values()
            ), model.repo_id


def test_a_later_revision_with_the_same_identity_gets_the_table_references(tmp_path):
    from vllm_srun.families.decision2.family import Decision2Family
    from vllm_srun.plugins.base import (
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


def test_every_family_has_a_table_and_lookups_cover_them():
    families = {model.family for model in builtin.all_models()}
    assert families <= {
        "decision2",
        "decision1",
        "task_heads",
        "vela2",
        "multimodal_embedding",
    }
    assert all(model.family == "decision2" for model in builtin.all_models("decision2"))
    assert builtin.all_models("nobody") == ()


def test_pinned_files_are_the_whole_download(monkeypatch, tmp_path):
    from vllm_srun.registry import resolve as resolver

    pinned = replace(
        builtin.lookup("vllm-sr/Decision-2.0-Kai-0.6B"),
        repo_id="vllm-sr/Pinned-Files",
        files={"config.json": "0" * 64, "model.safetensors": "1" * 64},
    )
    monkeypatch.setattr(
        builtin, "lookup", lambda name: pinned if name == pinned.repo_id else None
    )
    calls = []

    def snapshot_download(allow_patterns, **kwargs):
        calls.append(sorted(allow_patterns))
        root = tmp_path / pinned.revision
        root.mkdir(exist_ok=True)
        return str(root)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    resolver.download(pinned.repo_id, pinned.revision)
    assert calls == [["config.json", "model.safetensors"]]


def test_a_hub_package_without_a_manifest_keeps_its_pointer_files(
    monkeypatch, tmp_path
):
    from vllm_srun.plugins.base import PackageRef
    from vllm_srun.registry import resolve as resolver

    calls = []

    def snapshot_download(allow_patterns, **kwargs):
        calls.append(sorted(allow_patterns))
        root = tmp_path / kwargs["revision"]
        root.mkdir(exist_ok=True)
        return str(root)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    revision = "e" * 40
    root = resolver.download("acme/encoder", revision)
    assert calls == [sorted(resolver.POINTER_FILES)] and root.name == revision
    ref = resolver.fetch(
        PackageRef(root=root, repo_id="acme/encoder", revision=revision),
        ["model.safetensors"],
    )
    assert calls[-1] == ["model.safetensors"] and ref.revision == revision


def test_parallel_hashing_matches_one_file_at_a_time(tmp_path):
    from vllm_srun.registry.artifacts import inventory, sha256_file, sha256_files

    for index in range(12):
        (tmp_path / f"part-{index}.bin").write_bytes(
            bytes([index]) * (index * 4099 + 1)
        )
    paths = {path.name: path for path in sorted(tmp_path.iterdir())}
    expected = {name: sha256_file(path) for name, path in paths.items()}
    assert sha256_files(paths) == expected
    assert inventory(tmp_path) == expected
    assert sha256_files({}) == {}


def test_named_files_hash_only_what_a_family_loads(tmp_path):
    from vllm_srun.registry.artifacts import named_files, sha256_file

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "large.onnx").write_bytes(b"\0" * 64)
    assert named_files(tmp_path, ["config.json"]) == {
        "config.json": sha256_file(tmp_path / "config.json")
    }
    with pytest.raises(PackageError, match="missing"):
        named_files(tmp_path, ["model.safetensors"])
    with pytest.raises(PackageError, match="unsafe"):
        named_files(tmp_path, ["../config.json"])
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "linked.json").symlink_to(outside)
    with pytest.raises(PackageError, match="link"):
        named_files(tmp_path, ["linked.json"])


def test_downloads_retry_transient_hub_failures_only(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from vllm_srun.registry import resolve as resolver

    revision = "f" * 40
    failures = [LocalEntryNotFoundError("Temporary failure in name resolution")] * 2
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs["revision"])
        if failures:
            raise failures.pop()
        root = tmp_path / revision
        root.mkdir(exist_ok=True)
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(resolver.time, "sleep", lambda seconds: None)
    assert resolver.download("acme/encoder", revision).name == revision
    assert len(calls) == 3

    class ForbiddenError(Exception):
        response = type("Response", (), {"status_code": 403})()

    def refused(**kwargs):
        calls.append(kwargs["revision"])
        raise ForbiddenError("gated")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refused)
    calls.clear()
    with pytest.raises(ForbiddenError):
        resolver.download("acme/encoder", revision)
    assert len(calls) == 1
    failures.append(LocalEntryNotFoundError("not in the cache"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    calls.clear()
    with pytest.raises(LocalEntryNotFoundError):
        resolver.download("acme/encoder", revision, offline=True)
    assert len(calls) == 1
