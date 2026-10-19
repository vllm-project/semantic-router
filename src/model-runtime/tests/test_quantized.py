"""Decision 3.0 packages with compressed-tensors int4 (W4A16), FP8 (W8A8) and NVFP4 weights: loading, numerics, refusals."""

import asyncio
import json

import pytest
import torch
from torch import nn
from vllm_srun.engines.native.quantized import Int4Linear
from vllm_srun.errors import PackageError
from vllm_srun.quantization import parse
from vllm_srun.testing.decision3 import pack_int4, write_quantized_fixture

from tests.conftest import start_runtime

QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {"code": None, "math": None, "other": None},
    },
    "reasoning": {"type": "noul", "instructions": "Does this need reasoning?"},
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", "Moderate", "Expert"],
    },
}
STATES = (
    "Write a Python function that merges two sorted lists.",
    "What is 17 times 23?",
)


def answers(package):
    """The answers to ``STATES`` and the engine's (int4 layers, parameters, bytes)."""
    runtime = start_runtime(package)
    try:
        out = []
        for state in STATES:
            status, body = asyncio.run(
                runtime.call("decisions", {"state": state, "questions": QUESTIONS})
            )
            assert status == 200, body
            out.append(body["answers"])
        engine = runtime.lookup(None).model.engine_model
        return out, (
            len(engine._quantized()),
            engine.parameter_count(),
            engine.memory_bytes(),
        )
    finally:
        runtime.stop()


@pytest.mark.parametrize("symmetric", [True, False])
def test_packed_weights_unpack_to_the_values_they_hold(symmetric):
    torch.manual_seed(0)
    values = torch.randint(-8, 8, (24, 64), dtype=torch.int8)
    zero = torch.randint(-8, 8, (24, 2), dtype=torch.int8)
    scale = torch.rand(24, 2) + 0.5
    layer = Int4Linear(nn.Linear(64, 24, bias=False), 32, symmetric)
    layer.weight_packed.copy_(pack_int4(values))
    layer.weight_scale.copy_(scale)
    want = values.float()
    if not symmetric:
        layer.weight_zero_point.copy_(pack_int4(zero, dim=0))
        want = want - zero.float().repeat_interleave(32, 1)
    want = (want * scale.repeat_interleave(32, 1)).to(torch.bfloat16)
    assert torch.equal(layer.unpacked(torch.bfloat16), want)


@pytest.mark.parametrize("symmetric", [True, False])
def test_an_int4_package_answers_as_its_dequantized_weights_do(
    tmp_path, d3_text_package, symmetric
):
    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "int4", symmetric=symmetric
    )
    plain = write_quantized_fixture(
        d3_text_package, tmp_path / "plain", symmetric=symmetric, dequantized=True
    )
    got, (layers, parameters, size) = answers(packed)
    want, (plain_layers, plain_parameters, plain_size) = answers(plain)
    assert got == want
    # MLPs in all four layers, GDN projections in three, attention in one
    assert (layers, plain_layers) == (4 * 3 + 3 * 3 + 4, 0)
    assert parameters == plain_parameters
    assert size < plain_size


def test_the_bf16_package_differs_from_the_original(tmp_path, d3_text_package):
    """The int4 fixture is not the BF16 checkpoint: the equality above compares rounded weights."""
    plain = write_quantized_fixture(
        d3_text_package, tmp_path / "plain", dequantized=True
    )
    assert answers(plain)[0] != answers(d3_text_package)[0]


def config(**weights):
    group = {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "group",
            "group_size": 128,
            "dynamic": False,
            "actorder": None,
        }
        | weights,
        "input_activations": None,
    }
    return {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": group},
        }
    }


def test_a_checkpoint_without_quantization_config_is_plain():
    assert parse({"model_type": "qwen3_5"}) is None


def test_awq_and_gptq_weights_parse():
    assert parse(config()).symmetric
    assert not parse(config(symmetric=False, actorder="weight")).symmetric


@pytest.mark.parametrize(
    "change",
    [
        {"num_bits": 8},
        {"type": "float"},
        {"strategy": "channel"},
        {"actorder": "group"},
        {"group_size": 100},
        {"dynamic": True},
    ],
)
def test_other_weight_schemes_are_refused(change):
    with pytest.raises(PackageError):
        parse(config(**change))


def test_quantized_int4_activations_are_refused():
    w4a8 = config()
    w4a8["quantization_config"]["config_groups"]["group_0"]["input_activations"] = {
        "num_bits": 8
    }
    with pytest.raises(PackageError):
        parse(w4a8)


def test_a_config_that_targets_nothing_is_refused(tmp_path, d3_text_package):
    packed = write_quantized_fixture(d3_text_package, tmp_path / "int4")
    path = packed / "config.json"
    data = json.loads(path.read_text())
    data["quantization_config"]["config_groups"]["group_0"]["targets"] = ["re:nothing"]
    path.write_text(json.dumps(data))
    from vllm_srun.testing.decision3 import write_manifest

    write_manifest(
        packed, json.loads((packed / "MODEL_MANIFEST.json").read_text())["parameters"]
    )
    with pytest.raises(Exception, match="targets no linear layer"):
        start_runtime(packed)


def float8_config(**weights):
    return {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": False,
                    }
                    | weights,
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
        }
    }


def test_fp8_dynamic_weights_parse():
    assert parse(float8_config()).scheme == "float8"


@pytest.mark.parametrize("change", [{"strategy": "tensor"}, {"num_bits": 4}])
def test_other_fp8_schemes_are_refused(change):
    with pytest.raises(PackageError, match="FP8_DYNAMIC"):
        parse(float8_config(**change))


def test_static_fp8_activations_are_refused():
    static = float8_config()
    group = static["quantization_config"]["config_groups"]["group_0"]
    group["input_activations"] |= {"strategy": "tensor", "dynamic": False}
    with pytest.raises(PackageError, match="FP8_DYNAMIC"):
        parse(static)


def test_an_fp8_package_needs_an_fp8_gpu(tmp_path, d3_text_package):
    packed = write_quantized_fixture(d3_text_package, tmp_path / "fp8", scheme="float8")
    with pytest.raises(Exception, match="FP8 weights need"):
        start_runtime(packed)


@pytest.mark.parametrize(
    "safe, ada, unscaled, captured",
    [
        (False, True, False, False),
        (False, True, True, True),
        (False, False, False, True),
        (True, True, False, True),
    ],
)
def test_graphs_skip_adas_per_row_fp8_product_before_pytorch_2_12(
    monkeypatch, safe, ada, unscaled, captured
):
    from vllm_srun.engines.native import quantized

    layer = quantized.Float8Linear(nn.Linear(16, 16, bias=False))
    layer.ada, layer.unscaled_product = ada, unscaled
    monkeypatch.setattr(quantized, "ROWWISE_CAPTURE_SAFE", safe)
    assert quantized.capture_safe(nn.Sequential(layer)) is captured


FP4 = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "tensor_group",
    "group_size": 16,
    "scale_dtype": "torch.float8_e4m3fn",
    "block_structure": None,
    "actorder": None,
}


def nvfp4_config(weights=None, inputs=None):
    """What compressed-tensors' NVFP4 scheme exports, with ``weights`` / ``inputs`` changes."""
    return {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": FP4 | {"dynamic": False} | (weights or {}),
                    "input_activations": FP4 | {"dynamic": "local"} | (inputs or {}),
                    "output_activations": None,
                }
            },
        }
    }


def test_nvfp4_weights_parse():
    weights = parse(nvfp4_config())
    assert (weights.scheme, weights.group_size) == ("nvfp4", 16)


@pytest.mark.parametrize(
    ("weights", "inputs"),
    [
        ({"group_size": 32}, None),  # MXFP4-like groups
        ({"strategy": "group"}, None),
        ({"scale_dtype": "torch.uint8"}, None),  # E8M0 scales
        ({"dynamic": True}, None),
        ({"symmetric": False}, None),
        (None, {"dynamic": False}),  # static activation block scales
        (None, {"num_bits": 8}),
        (None, {"group_size": 32}),
    ],
)
def test_other_fp4_schemes_are_refused(weights, inputs):
    with pytest.raises(PackageError, match="only NVFP4"):
        parse(nvfp4_config(weights, inputs))


def test_nvfp4_weights_without_quantized_activations_are_refused():
    config = nvfp4_config()
    config["quantization_config"]["config_groups"]["group_0"][
        "input_activations"
    ] = None
    with pytest.raises(PackageError, match="only NVFP4"):
        parse(config)


def test_an_nvfp4_package_counts_the_parameters_of_its_weights(
    tmp_path, d3_text_package
):
    from vllm_srun.families.decision3 import package as pkg

    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "nvfp4", scheme="nvfp4"
    )
    names = ("model.safetensors",)
    assert (
        pkg.tensor_groups(packed, names)[0]
        == pkg.tensor_groups(d3_text_package, names)[0]
    )


def test_an_nvfp4_package_requires_fp4_tensor_cores_on_every_accelerator(
    tmp_path, d3_text_package
):
    from vllm_srun.families.decision3.family import Decision3Family
    from vllm_srun.plugins.base import PackageRef, RegistryOptions
    from vllm_srun.plugins.registry import names

    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "nvfp4", scheme="nvfp4"
    )
    family = Decision3Family(RegistryOptions(base_path=None, offline=True))
    spec = family.describe(family.verify(PackageRef(root=packed)))
    assert set(spec.requires) == set(names("accelerators"))
    assert spec.requires["cpu"] == ("lapack", "nvfp4")
    assert all("nvfp4" in needs for needs in spec.requires.values())


def test_an_nvfp4_package_is_refused_on_a_cpu_before_its_weights_load(
    tmp_path, d3_text_package, monkeypatch
):
    from vllm_srun.engines.native import engine

    packed = write_quantized_fixture(
        d3_text_package, tmp_path / "nvfp4", scheme="nvfp4"
    )

    def never(*args, **kwargs):
        raise AssertionError("weights loaded")

    monkeypatch.setattr(engine, "load_backbone", never)
    with pytest.raises(Exception, match="NVFP4 packages need FP4 tensor cores"):
        start_runtime(packed)
