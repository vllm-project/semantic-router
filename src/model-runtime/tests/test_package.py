import json
import os
import shutil
import sys

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.decision2.family import Decision2Family
from vllm_srun.plugins.base import PackageRef, RegistryOptions


def verify(root, **options):
    return Decision2Family(RegistryOptions(**options)).verify(PackageRef(root=root))


def rewrite_manifest(root, change):
    path = root / "MODEL_MANIFEST.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_valid_package_verifies_without_running_package_code(qwen3_package):
    verified = verify(qwen3_package)
    assert verified.family == "decision2"
    assert verified.model_name == "Decision-2.0-Tiny-Qwen3"
    assert verified.max_input_tokens == 2048
    assert verified.details["package"].score_bias[3] == [0.05, -0.02, 0.01]
    # The bundled decision2/ package raises on import; verification never touched it.
    assert "decision2" not in [name.split(".")[0] for name in list(sys.modules)]


def test_detect_reads_only_the_pointer(qwen3_package, tmp_path):
    family = Decision2Family()
    assert family.detect(PackageRef(root=qwen3_package))
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    assert not family.detect(PackageRef(root=tmp_path))


def test_tampered_file_is_refused(package_copy):
    with (package_copy / "decision_head.safetensors").open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        last = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([last[0] ^ 1]))
    with pytest.raises(PackageError, match="differ from MODEL_MANIFEST"):
        verify(package_copy)


def test_extra_and_missing_files_are_refused(package_copy):
    (package_copy / "extra.txt").write_text("x")
    with pytest.raises(PackageError, match="extra"):
        verify(package_copy)
    (package_copy / "extra.txt").unlink()
    (package_copy / "README.md").unlink()
    with pytest.raises(PackageError, match="missing"):
        verify(package_copy)


def test_links_are_refused(package_copy):
    (package_copy / "link.json").symlink_to(package_copy / "config.json")
    with pytest.raises(PackageError, match="link"):
        verify(package_copy)


def hub_snapshot(package, cache):
    """Lay ``package`` out as a Xet-backed Hub cache: snapshot -> repo blob -> shared store."""
    repository = cache / "models--vllm-sr--Decision-2.0-Tiny"
    snapshot = repository / "snapshots" / ("a" * 40)
    for index, source in enumerate(
        sorted(p for p in package.rglob("*") if p.is_file())
    ):
        shared = cache / "blobs" / f"{index:02d}" / f"content-{index}"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, shared)
        blob = repository / "blobs" / f"etag-{index}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.symlink_to(os.path.relpath(shared, blob.parent))
        target = snapshot / source.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(blob, target.parent))
    return snapshot


def test_hub_cache_links_resolve_inside_the_cache(qwen3_package, tmp_path):
    snapshot = hub_snapshot(qwen3_package, tmp_path / "hub")
    assert verify(snapshot).model_name == "Decision-2.0-Tiny-Qwen3"


def test_hub_cache_links_must_not_leave_the_cache(qwen3_package, tmp_path):
    snapshot = hub_snapshot(qwen3_package, tmp_path / "hub")
    outside = tmp_path / "outside.md"
    outside.write_text((snapshot / "README.md").read_text())
    (snapshot / "README.md").unlink()
    (snapshot / "README.md").symlink_to(outside)
    with pytest.raises(PackageError, match="link"):
        verify(snapshot)


def test_hub_added_gitattributes_is_ignored(package_copy):
    (package_copy / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    assert verify(package_copy).model_name


def test_pointer_must_name_the_format(package_copy):
    pointer = json.loads((package_copy / "config.json").read_text())
    pointer["format_version"] = 1
    (package_copy / "config.json").write_text(json.dumps(pointer))
    with pytest.raises(PackageError, match="pointer"):
        verify(package_copy)


def test_identity_must_match_the_manifest(package_copy):
    def change(manifest):
        manifest["identity"]["model_sha256"] = "0" * 64

    rewrite_manifest(package_copy, change)
    with pytest.raises(PackageError, match=r"identity|differ"):
        verify(package_copy)


def test_header_counts_must_match(package_copy):
    def change(manifest):
        manifest["parameters"]["packaged"]["head"] += 1

    rewrite_manifest(package_copy, change)
    with pytest.raises(PackageError, match="differ"):
        verify(package_copy)


def test_unsupported_profiles_are_reported(package_copy):
    def change(manifest):
        manifest["profile"] = "kai-native"

    rewrite_manifest(package_copy, change)
    with pytest.raises(PackageError, match="not served"):
        verify(package_copy)


def test_restricted_licences_need_acceptance(package_copy):
    def change(manifest):
        manifest["licence"] = {"spdx": "cc-by-nc-4.0", "components": []}

    rewrite_manifest(package_copy, change)
    with pytest.raises(PackageError, match="accept-licence"):
        verify(package_copy)


def test_adapter_package_verifies_against_its_pinned_base(adapter_package):
    base = adapter_package.parent / f"{adapter_package.name}-base"
    verified = verify(adapter_package, base_path=base)
    assert verified.details["package"].profile == "qwen-adapter"
    (base / "config.json").write_text((base / "config.json").read_text() + " ")
    try:
        with pytest.raises(PackageError, match="base file differs"):
            verify(adapter_package, base_path=base)
    finally:
        (base / "config.json").write_text((base / "config.json").read_text()[:-1])
