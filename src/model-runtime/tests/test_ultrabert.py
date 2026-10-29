"""The native UltraBERT backbone and LoRA adapter networks over it (tiny configs, CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from vllm_srun.accel.kernels import reference_kernels
from vllm_srun.engines.native import models
from vllm_srun.engines.native.adapters import load_adapter_network, module_name
from vllm_srun.engines.native.models import ultrabert
from vllm_srun.engines.native.models.modernbert import FULL, SLIDING, Group, Layout
from vllm_srun.engines.native.models.ultrabert import additive_masks
from vllm_srun.engines.native.weights import load_backbone
from vllm_srun.families.vela3.package import LORA_TARGETS
from vllm_srun.plugins.base import AdapterSpec, BackboneSpec
from vllm_srun.testing.fixtures import save
from vllm_srun.testing.vela3 import ultrabert_config, write_encoder_package

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    return write_encoder_package(tmp_path_factory.mktemp("ultrabert") / "fast")


def backbone(package: Path) -> torch.nn.Module:
    import json

    config = json.loads((package / "config.json").read_text())
    module = models.build("ultrabert", config)
    module.rotary_emb = type(module.rotary_emb)(config)
    load_backbone(module, [package / "model.safetensors"], "encoder.")
    module.kernels = reference_kernels("cpu")
    return module.eval()


def spec(package: Path, **adapters: AdapterSpec) -> BackboneSpec:
    import json

    config = json.loads((package / "config.json").read_text())
    return BackboneSpec(
        "ultrabert",
        config,
        (package / "model.safetensors",),
        "encoder.",
        adapters=adapters,
    )


def adapter(package: Path, file: str, rank: int = 4) -> AdapterSpec:
    return AdapterSpec((package / file,), rank, 2.0 * rank, LORA_TARGETS, "encoder.")


def test_local_layers_read_keys_within_their_window_only() -> None:
    valid = torch.tensor([[True] * 6 + [False] * 2])
    masks = additive_masks(valid, 1, 8, 2, torch.float32)
    reads = masks[SLIDING][0, 0] == 0
    assert reads[3].tolist() == [False, True, True, True, True, True, False, False]
    assert (masks[FULL][0, 0] == 0)[0].tolist() == [True] * 6 + [False] * 2
    assert masks[FULL].dtype == torch.float32
    assert masks[SLIDING].min() == torch.finfo(torch.float32).min


def test_padding_never_changes_a_rows_hidden_states(package) -> None:
    module = backbone(package)
    torch.manual_seed(0)
    ids = torch.randint(1, 300, (1, 9))
    with torch.no_grad():
        alone = module(ids, torch.ones_like(ids))
        padded = torch.cat([ids, torch.zeros(1, 5, dtype=torch.long)], dim=1)
        other = torch.randint(1, 300, (1, 14))
        batch = torch.cat([padded, other])
        mask = torch.cat([torch.tensor([[1] * 9 + [0] * 5]), torch.ones(1, 14)]).long()
        together = module(batch, mask)
    assert torch.allclose(together[0, :9], alone[0], atol=1e-5)


def test_cpu_key_masks_match_the_dense_masks_bit_for_bit(package, monkeypatch) -> None:
    module = backbone(package)
    monkeypatch.setattr(ultrabert, "LOCAL_QUERIES", 8)
    torch.manual_seed(3)
    ids = torch.randint(1, 300, (3, 21))
    valid = torch.ones(3, 21, dtype=torch.bool)
    valid[1, 13:], valid[2, 5:] = False, False
    dense = additive_masks(valid, 3, 21, module.window, torch.float32)
    with torch.no_grad():
        expected = module.encode(ids, Layout((Group(3, 21, dict(dense)),)))
        layout = module.padded(valid.long(), 3, 21, "cpu")
        found = module.encode(ids, layout)
    assert layout.groups[0].masks[FULL].shape == (3, 1, 1, 21)
    assert torch.equal(found[module.num_layers], expected[module.num_layers])


def test_tokens_past_the_local_window_reach_a_token_only_through_global_layers(
    package,
) -> None:
    import json

    config = json.loads((package / "config.json").read_text())
    local_only = models.build("ultrabert", {**config, "attention_windows": [4, 4, 4]})
    local_only.rotary_emb = type(local_only.rotary_emb)(local_only.config)
    load_backbone(local_only, [package / "model.safetensors"], "encoder.")
    local_only.kernels = reference_kernels("cpu")
    torch.manual_seed(1)
    ids = torch.randint(1, 300, (1, 40))
    changed = ids.clone()
    changed[0, 39] = (ids[0, 39] + 7) % 300 + 1
    with torch.no_grad():
        first, second = local_only(ids), local_only(changed)
    # Three local layers of window 4 carry a change at most 12 positions back.
    assert torch.equal(first[0, :27], second[0, :27])
    assert not torch.equal(first[0, 30:], second[0, 30:])


def test_packed_rows_match_padded_rows(package) -> None:
    module = backbone(package)
    torch.manual_seed(2)
    rows = [torch.randint(1, 300, (n,)) for n in (5, 17, 11)]
    with torch.no_grad():
        layout = module.packed([len(r) for r in rows], "cpu")
        packed = module.encode(torch.cat(rows), layout)[module.num_layers]
        start = 0
        for row in rows:
            alone = module(row[None])[0]
            assert torch.allclose(packed[start : start + len(row)], alone, atol=1e-5)
            start += len(row)


def test_adapter_network_shares_the_backbones_weights(package) -> None:
    module = backbone(package)
    network = load_adapter_network(
        module, spec(package, span=adapter(package, "span_adapter.safetensors")), "span"
    )
    assert network.embed_tokens is module.embed_tokens
    assert network.norm is module.norm
    assert network.rotary_emb is module.rotary_emb
    bottom = network.layers[0].self_attn.q_proj
    assert bottom.base_layer.weight is module.layers[0].self_attn.q_proj.weight
    top = network.layers[2].self_attn.q_proj
    assert top.base_layer.weight is not module.layers[2].self_attn.q_proj.weight
    assert (
        network.layers[0].input_layernorm.weight
        is module.layers[0].input_layernorm.weight
    )
    shared = {id(p) for p in module.parameters()}
    own = [p for p in network.parameters() if id(p) not in shared]
    assert own and all(not p.is_meta for p in own)


def test_an_adapter_changes_the_hidden_states_only_through_its_own_tensors(
    package,
) -> None:
    module = backbone(package)
    network = load_adapter_network(
        module, spec(package, span=adapter(package, "span_adapter.safetensors")), "span"
    )
    network.kernels = module.kernels
    for layer in network.modules():
        if hasattr(layer, "lora_B"):
            torch.nn.init.zeros_(layer.lora_B.weight)
    # Give the adapter's own top layer the backbone's weights: a wrapped projection holds them under base_layer.
    for index in range(3):
        mine, theirs = (
            network.layers[index].state_dict(),
            module.layers[index].state_dict(),
        )
        for name, value in theirs.items():
            wrapped = name.replace(".weight", ".base_layer.weight")
            mine[wrapped if wrapped in mine else name].copy_(value)
    ids = torch.randint(1, 300, (1, 12))
    with torch.no_grad():
        assert torch.allclose(network(ids), module(ids), atol=1e-6)


def test_adapter_files_must_cover_every_lora_factor(package, tmp_path) -> None:
    from safetensors.torch import load_file

    tensors = load_file(str(package / "span_adapter.safetensors"))
    missing = next(name for name in tensors if name.endswith("lora_B"))
    save({k: v for k, v in tensors.items() if k != missing}, tmp_path / "a.safetensors")
    module = backbone(package)
    broken = AdapterSpec(
        (tmp_path / "a.safetensors",), 4, 8.0, LORA_TARGETS, "encoder."
    )
    with pytest.raises(ValueError, match="does not hold"):
        load_adapter_network(module, spec(package, span=broken), "span")
    save(
        {**tensors, "encoder.layers.0.extra": torch.zeros(1)},
        tmp_path / "b.safetensors",
    )
    extra = AdapterSpec((tmp_path / "b.safetensors",), 4, 8.0, LORA_TARGETS, "encoder.")
    with pytest.raises(ValueError, match="has no parameter"):
        load_adapter_network(module, spec(package, span=extra), "span")


def test_ultrabert_rejects_configs_it_does_not_implement() -> None:
    config = ultrabert_config(64)
    with pytest.raises(ValueError, match="one local window"):
        models.build("ultrabert", {**config, "attention_windows": [4, 8, None]})
    with pytest.raises(ValueError, match="key / value heads"):
        models.build("ultrabert", {**config, "num_key_value_heads": 1})
    with pytest.raises(ValueError, match="every layer"):
        models.build("ultrabert", {**config, "rope_theta": [1.0, 2.0]})


def test_adapter_tensor_names_map_to_wrapped_parameters() -> None:
    assert (
        module_name("layers.0.mlp.up_proj.lora_A")
        == "layers.0.mlp.up_proj.lora_A.weight"
    )
    assert (
        module_name("layers.2.self_attn.o_proj.base.weight")
        == "layers.2.self_attn.o_proj.base_layer.weight"
    )
    assert (
        module_name("layers.2.input_layernorm.weight")
        == "layers.2.input_layernorm.weight"
    )
