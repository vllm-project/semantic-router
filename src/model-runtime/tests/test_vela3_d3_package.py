"""Detecting and verifying Vela 3.0 d3-size packages (``vela3/3``) and their pinned d3 base."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.decision3.family import Decision3Family
from vllm_srun.families.vela3.family import Vela3Family
from vllm_srun.families.vela3_d3.family import Vela3D3Family
from vllm_srun.plugins.base import DeviceInfo, PackageRef, RegistryOptions
from vllm_srun.registry import builtin
from vllm_srun.testing import vela3_d3 as fixture
from vllm_srun.testing.fixtures import FAKE_BASE_REVISION


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Path:
    return fixture.write_fixture(
        tmp_path_factory.mktemp("v3d3") / "vela3-fixture", "text", 3
    )


def base_of(root: Path) -> Path:
    return root.parent / f"{root.name}-base"


def copy(root: Path, tmp_path: Path, name: str = "copy") -> Path:
    target = tmp_path / name
    shutil.copytree(root, target)
    shutil.copytree(base_of(root), base_of(target))
    return target


def family(root: Path) -> Vela3D3Family:
    return Vela3D3Family(RegistryOptions(base_path=base_of(root)))


def edit_config(root: Path, change) -> None:
    path = root / "vela3_config.json"
    config = json.loads(path.read_text())
    change(config)
    path.write_text(json.dumps(config))
    fixture.write_sums(root)


def test_detects_only_vela3_3_packages(written) -> None:
    assert family(written).detect(PackageRef(written))
    assert not Vela3Family().detect(PackageRef(written))
    assert not Decision3Family().detect(PackageRef(written))
    assert not family(written).detect(PackageRef(base_of(written)))


def test_verifies_the_package_its_adapters_and_its_base(written) -> None:
    verified = family(written).verify(PackageRef(written))
    details = verified.details["package"]
    assert verified.family == "vela3_d3"
    assert details.base.repo_id == fixture.BASE_REPO
    assert details.base.revision == FAKE_BASE_REVISION
    base = verified.details["base"]
    assert base.family == "decision3"
    assert set(details.adapters) == set(fixture.ADAPTERS)
    decision = details.adapters["decision"]
    assert (decision.rank, decision.alpha) == (4, 8)
    assert set(details.span.heads) == {"span", "ner", "relation"}
    assert verified.max_input_tokens == base.max_input_tokens


def test_the_package_runs_its_base_kernel_choices(written) -> None:
    verified = family(written).verify(PackageRef(written))
    gfx942 = DeviceInfo(accelerator="rocm", index=0, name="MI325X", arch="gfx942")
    assert family(written).kernel_choices(verified, gfx942) == {}
    turbo = builtin.lookup("vllm-sr/d3-turbo")
    base = replace(verified.details["base"], model_sha256=turbo.model_sha256)
    on_turbo = replace(verified, details={**verified.details, "base": base})
    recorded = family(written).kernel_choices(on_turbo, gfx942)
    assert recorded == turbo.kernel_choices["rocm:gfx942"]
    assert (
        family(written).kernel_choices(on_turbo, replace(gfx942, arch="gfx90a")) == {}
    )
    # A base pinned at another revision of a built-in model runs that model's choices.
    other = replace(
        verified.details["base"],
        ref=PackageRef(base_of(written), "vllm-sr/d3-turbo", "d" * 40),
    )
    on_other = replace(verified, details={**verified.details, "base": other})
    assert family(written).kernel_choices(on_other, gfx942) == recorded


def test_identity_covers_the_base(written, tmp_path) -> None:
    first = family(written).verify(PackageRef(written)).model_sha256
    other = fixture.write_fixture(tmp_path / "other", "text", 4)
    shutil.rmtree(base_of(other))
    shutil.copytree(base_of(written), base_of(other))
    assert family(other).verify(PackageRef(other)).model_sha256 != first
    rebased = copy(written, tmp_path)
    shutil.rmtree(base_of(rebased))
    from vllm_srun.testing import decision3

    decision3.write_fixture(base_of(rebased), "text", 99)
    assert family(rebased).verify(PackageRef(rebased)).model_sha256 != first


def test_refuses_a_file_that_differs_from_sha256sums(written, tmp_path) -> None:
    root = copy(written, tmp_path)
    (root / "router_questions.json").write_text("{}")
    with pytest.raises(PackageError, match="SHA256SUMS"):
        family(root).verify(PackageRef(root))


def test_refuses_an_adapter_whose_metadata_differs_from_the_config(
    written, tmp_path
) -> None:
    root = copy(written, tmp_path)
    edit_config(root, lambda c: c["adapters"]["span"].update(r=8))
    with pytest.raises(PackageError, match="span"):
        family(root).verify(PackageRef(root))
    root = copy(written, tmp_path, "modules")
    edit_config(root, lambda c: c["adapters"]["ner"].update(modules=1))
    with pytest.raises(PackageError, match="ner"):
        family(root).verify(PackageRef(root))


def test_refuses_unpaired_lora_factors(written, tmp_path) -> None:
    from safetensors.torch import load_file, save_file

    root = copy(written, tmp_path)
    path = root / "decision_lora.safetensors"
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    tensors = load_file(str(path))
    tensors.pop(sorted(k for k in tensors if k.endswith(".B"))[0])
    save_file(tensors, str(path), metadata=metadata)
    fixture.write_sums(root)
    with pytest.raises(PackageError, match="unpaired"):
        family(root).verify(PackageRef(root))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda span: span.pop("default_question"),
            "span.default_question must be a string",
        ),
        (lambda span: span.update(threshold="high"), "span.threshold must be a number"),
        (lambda span: span.update(threshold=True), "span.threshold must be a number"),
        (
            lambda span: span["schemas"]["pii"].pop("labels"),
            "span.schemas.pii.labels must be an object",
        ),
        (
            lambda span: span["router_label_sets"]["pii"].append(3),
            "span.router_label_sets.pii must be a list of strings",
        ),
        (
            lambda span: span["ner"].pop("threshold_names_only"),
            "span.ner.threshold_names_only must be a number",
        ),
        (
            lambda span: span["ner"].pop("head_file"),
            "span.ner.head_file must name a file",
        ),
        (
            lambda span: span["relation"].update(threshold_dev_pair="0.4"),
            "span.relation.threshold_dev_pair must be a number",
        ),
        (
            lambda span: span.pop("threshold_names_only"),
            "span.threshold_names_only must be a number",
        ),
        (lambda span: span.update(ner="auto"), "span.ner must be an object"),
        (
            lambda span: span["pii_buckets"].pop(),
            "span.pii_buckets must reach the span budget",
        ),
    ],
)
def test_refuses_a_malformed_span_section(written, tmp_path, change, message) -> None:
    root = copy(written, tmp_path)
    edit_config(root, lambda config: change(config["span"]))
    with pytest.raises(PackageError, match=re.escape(message)):
        family(root).verify(PackageRef(root))


def test_refuses_an_unpinned_base(written, tmp_path) -> None:
    root = copy(written, tmp_path)
    edit_config(root, lambda c: c["base"].update(revision="main"))
    with pytest.raises(PackageError, match="40-hex"):
        family(root).verify(PackageRef(root))


def test_refuses_a_base_that_fails_its_own_manifest(written, tmp_path) -> None:
    root = copy(written, tmp_path)
    (base_of(root) / "readout.safetensors").write_bytes(b"0")
    with pytest.raises(PackageError):
        family(root).verify(PackageRef(root))


def pin(root: Path, monkeypatch) -> builtin.BuiltinModel:
    """Register root's verified files, identity and base as a built-in entry."""
    verified = family(root).verify(PackageRef(root))
    entry = builtin.BuiltinModel(
        repo_id="vllm-sr/vela3-fixture",
        revision="a" * 40,
        family="vela3_d3",
        model_sha256=verified.model_sha256,
        manifest_sha256="",
        loaded_parameters=0,
        backbone="qwen3_5_text",
        min_device_memory_gib=1,
        base=(fixture.BASE_REPO, FAKE_BASE_REVISION),
        files=dict(verified.details["package"].files),
    )
    monkeypatch.setattr(
        builtin, "lookup", lambda repo: entry if repo == entry.repo_id else None
    )
    return entry


def test_a_pinned_entry_holds_files_identity_and_base(written, monkeypatch) -> None:
    entry = pin(written, monkeypatch)
    ref = PackageRef(written, entry.repo_id, entry.revision)
    assert family(written).verify(ref).model_sha256 == entry.model_sha256
    for wrong in (
        replace(entry, model_sha256="0" * 64),
        replace(entry, base=(fixture.BASE_REPO, "b" * 40)),
        replace(entry, files={**entry.files, "span_head.safetensors": "0" * 64}),
    ):
        monkeypatch.setattr(builtin, "lookup", lambda repo, e=wrong: e)
        with pytest.raises(PackageError):
            family(written).verify(ref)


def test_a_pinned_download_needs_no_sha256sums(written, tmp_path, monkeypatch) -> None:
    # A pinned revision downloads only the files its entry lists.
    root = copy(written, tmp_path)
    entry = pin(root, monkeypatch)
    (root / "SHA256SUMS").unlink()
    ref = PackageRef(root, entry.repo_id, entry.revision)
    assert family(root).verify(ref).model_sha256 == entry.model_sha256


def test_an_unpinned_package_needs_sha256sums(written, tmp_path) -> None:
    root = copy(written, tmp_path)
    (root / "SHA256SUMS").unlink()
    with pytest.raises(PackageError, match="SHA256SUMS"):
        family(root).verify(PackageRef(root))
