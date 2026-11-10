import torch
from vllm_srun.engines.native.models.lora import LoRALinear, adapter_key, fold_scaling

from .conftest import QUESTIONS, STATE, start_runtime


def test_lora_linear_matches_peft_formula():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 6, bias=False)
    layer = LoRALinear(base, rank=2, scaling=2.0)
    torch.nn.init.normal_(layer.lora_B.weight)
    x = torch.randn(3, 8)
    expected = base(x) + layer.lora_B(layer.lora_A(x)) * 2.0
    assert torch.equal(layer(x), expected)


def test_power_of_two_scaling_folds_bit_for_bit():
    torch.manual_seed(0)
    layers = torch.nn.Sequential(
        LoRALinear(torch.nn.Linear(64, 96, bias=False), rank=8, scaling=2.0),
        LoRALinear(torch.nn.Linear(96, 64, bias=True), rank=8, scaling=0.25),
        LoRALinear(torch.nn.Linear(64, 64, bias=False), rank=8, scaling=3.0),
    )
    for layer in layers:
        torch.nn.init.normal_(layer.lora_B.weight)
    x = torch.randn(5, 37, 64)
    expected = layers(x)
    assert fold_scaling(layers) == 2
    assert [layer.scaling for layer in layers] == [1.0, 1.0, 3.0]
    assert torch.equal(layers(x), expected)
    assert fold_scaling(layers) == 0


def test_adapter_keys():
    assert adapter_key("base_model.model.layers.0.mlp.up_proj.lora_A.weight") == (
        "layers.0.mlp.up_proj",
        "lora_A",
    )


def test_adapter_package_serves_on_its_pinned_base(adapter_package):
    base = adapter_package.parent / f"{adapter_package.name}-base"
    runtime = start_runtime(adapter_package, base_path=str(base))
    try:
        assert runtime.health.ready
        model = runtime.lookup(None).model
        engine_model = model.engine_model
        wrapped = [
            m for m in engine_model.backbone.modules() if isinstance(m, LoRALinear)
        ]
        assert len(wrapped) == 12
        plan = model.plan(STATE, QUESTIONS)
        with_adapter = model.run(plan.items)
        for module in wrapped:
            module.lora_B.weight.data.zero_()
        without = model.run(plan.items)
        assert with_adapter != without
        package = runtime.lookup(None).package
        assert model.info.parameters == package.loaded_parameters
    finally:
        runtime.stop()
