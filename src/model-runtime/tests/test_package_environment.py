"""The environment the runtime sets before anything loads PyTorch.

Importing the package sets fixed defaults. libgomp's spin count depends on
what a process serves, so ``vllm-srun serve`` chooses it from its models.
"""

import json
import logging
import os
import subprocess
import sys

import pytest
from vllm_srun.config import ModelConfig, ServeConfig, spin_count
from vllm_srun.registry import builtin
from vllm_srun.runtime import Runtime, planned_engine
from vllm_srun.testing import omni

DEFAULTS = {
    "THP_MEM_ALLOC_ENABLE": "1",
    "ONEDNN_PRIMITIVE_CACHE_CAPACITY": "8192",
    "MIOPEN_FIND_MODE": "FAST",
    "MIOPEN_LOG_LEVEL": "3",
    "OPENBLAS_NUM_THREADS": "1",
}
SPIN = "GOMP_SPINCOUNT"
# `vllm-srun serve` with the server replaced: what the server would start with.
SERVE_PROBE = """
import json, os, sys, types
sys.modules["vllm_srun.api.server"] = types.SimpleNamespace(
    serve=lambda config: print(json.dumps([os.environ.get("GOMP_SPINCOUNT"), "torch" in sys.modules]))
)
from vllm_srun.cli import main
main(sys.argv[1:])
"""


def clean_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in {*DEFAULTS, SPIN}}


def environment_after_import(
    env: dict[str, str], machine: str = "x86_64"
) -> dict[str, str | None]:
    probe = (
        f"import platform; platform.machine = lambda: {machine!r}; "
        "import json, os, sys; import vllm_srun; "
        f"print(json.dumps({{k: os.environ.get(k) for k in {sorted({*DEFAULTS, SPIN})!r}}})); "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out[1] == "False", "importing the package must not load PyTorch"
    return json.loads(out[0])


def served_spin_count(arguments: list[str], env: dict[str, str]) -> str | None:
    out = subprocess.run(
        [sys.executable, "-c", SERVE_PROBE, "serve", *arguments],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    value, torch_loaded = json.loads(out[-1])
    assert not torch_loaded, "the spin count must be chosen before PyTorch loads"
    return value


def test_import_sets_every_default_before_pytorch_loads() -> None:
    assert environment_after_import(clean_environment()) == {**DEFAULTS, SPIN: None}


def test_openblas_threads_are_capped_only_on_x86_64() -> None:
    env = {k: v for k, v in os.environ.items() if k not in DEFAULTS}
    # PyTorch's aarch64 wheels run their own matrix products on OpenBLAS.
    assert environment_after_import(env, "aarch64")["OPENBLAS_NUM_THREADS"] is None
    assert environment_after_import(env, "AMD64")["OPENBLAS_NUM_THREADS"] == "1"


def test_a_value_the_caller_set_is_kept() -> None:
    env = clean_environment()
    env["MIOPEN_FIND_MODE"] = "NORMAL"
    assert environment_after_import(env)["MIOPEN_FIND_MODE"] == "NORMAL"


@pytest.mark.parametrize(
    ("models", "expected"),
    [
        ((ModelConfig(model="m", device="cpu"),), None),
        ((ModelConfig(model="m", device="cpu", engine="native"),), None),
        ((ModelConfig(model="m", device="cpu", engine="onnxruntime"),), "10000"),
        ((ModelConfig(model="m", engine="onnxruntime"),), "10000"),
        ((ModelConfig(model="m", device="rocm:0", engine="onnxruntime"),), None),
        (
            (
                ModelConfig(model="a", device="cpu"),
                ModelConfig(model="b", device="CPU", engine="onnxruntime"),
            ),
            "10000",
        ),
    ],
)
def test_onnx_runtime_on_the_cpu_keeps_the_short_spin(models, expected) -> None:
    assert spin_count(models) == expected


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["vllm-sr/Decision-1.0-Kai-0.6B", "--device", "cpu"], None),
        (["package", "--engine", "onnxruntime", "--device", "cpu"], "10000"),
        (["package", "--engine", "onnxruntime", "--device", "rocm"], None),
    ],
)
def test_serve_chooses_the_spin_count_before_pytorch_loads(arguments, expected) -> None:
    assert served_spin_count(arguments, clean_environment()) == expected


def test_serve_reads_the_models_file_and_keeps_a_value_the_caller_set(
    tmp_path,
) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n"
        "  - {model: vllm-sr/Decision-1.0-Kai-0.6B, name: kai, device: cpu}\n"
        "  - {model: ./omni, name: omni, device: cpu, engine: onnxruntime}\n"
    )
    env = clean_environment()
    assert served_spin_count(["--models", str(models)], env) == "10000"
    assert served_spin_count(["--models", str(models)], {**env, SPIN: "300"}) == "300"


def test_auto_plans_no_built_in_model_on_onnx_runtime_on_the_cpu() -> None:
    """``spin_count`` counts ``engine: auto`` as native; the runtime's own plan agrees for every built-in."""
    for model in builtin.all_models():
        config = ModelConfig(model=model.repo_id, device="cpu")
        assert planned_engine(config) != "onnxruntime", model.repo_id


def test_onnx_runtime_beside_native_cpu_models_warns_on_the_default_spin(
    tmp_path, qwen3_package, monkeypatch, caplog
) -> None:
    monkeypatch.delenv(SPIN, raising=False)
    source = {"repo_id": "example/omni-fixture", "revision": "0" * 40}
    bundle = omni.write_bundle(tmp_path / "omni", source=source)
    models = (
        ModelConfig(model=str(qwen3_package), name="native", device="cpu"),
        ModelConfig(model=str(bundle), name="omni", device="cpu", engine="onnxruntime"),
    )
    runtime = Runtime(ServeConfig(models=models))
    with caplog.at_level(logging.WARNING, logger="vllm_srun"):
        runtime.start(background=False)
    runtime.stop()
    warned = [r.getMessage() for r in caplog.records if SPIN in r.getMessage()]
    assert len(warned) == 1 and warned[0].startswith("omni runs ONNX Runtime")
