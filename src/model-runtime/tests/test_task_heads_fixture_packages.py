"""`vllm-srun fixture --family task_heads` writes embedders and rerankers too (E2E uses them)."""

from __future__ import annotations

import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.cli import main
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.families.task_heads.family import TaskHeadsFamily
from vllm_srun.plugins.base import (
    DeviceInfo,
    EngineOptions,
    PackageRef,
    RegistryOptions,
)

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


@pytest.mark.parametrize(
    ("variant", "surface"), [("embedding", "embeddings"), ("reranker", "rerank")]
)
def test_fixture_command_writes_a_task_heads_package(tmp_path, variant, surface):
    root = tmp_path / variant

    assert (
        main(["fixture", str(root), "--family", "task_heads", "--variant", variant])
        == 0
    )

    family = TaskHeadsFamily(RegistryOptions())
    assert family.detect(PackageRef(root))
    package = family.verify(PackageRef(root))
    spec = family.describe(package)
    engine = NativeEngine()
    model = family.load(
        package,
        spec,
        engine.load(spec, CPUAccelerator(), CPU, EngineOptions(threads=1)),
    )
    assert tuple(model.info.surfaces) == (surface,)


def test_fixture_command_names_every_variant_it_refuses(tmp_path):
    with pytest.raises(ValueError, match="embedding, reranker"):
        main(
            [
                "fixture",
                str(tmp_path / "x"),
                "--family",
                "task_heads",
                "--variant",
                "nope",
            ]
        )
