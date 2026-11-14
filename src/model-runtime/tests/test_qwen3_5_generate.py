"""Cached greedy decoding of the native Qwen3.5 backbone against its own full forward."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm_srun.accel.kernels import reference_kernels
from vllm_srun.engines.native import models
from vllm_srun.engines.native.models.qwen3_5_decode import (
    decode_mask,
    greedy,
    logits_of,
    pad_batch,
    prefill,
    rotary_positions,
    step,
)
from vllm_srun.testing.fixtures import qwen3_5_config


def backbone():
    config = qwen3_5_config(128)
    torch.manual_seed(0)
    module = models.build(config["model_type"], config)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("A_log"):
                parameter.copy_(torch.log(torch.rand(parameter.shape) * 15 + 1))
            elif name.endswith("dt_bias") or name.endswith("linear_attn.norm.weight"):
                parameter.copy_(torch.ones(parameter.shape))
            else:
                parameter.normal_(0, 0.02)
    module.kernels = reference_kernels("cpu")
    module.requires_grad_(False)
    module.eval()
    return module


def argmax_at(module, weight, ids: torch.Tensor) -> int:
    mask = torch.ones_like(ids)
    hidden = module(ids, mask)
    logits = F.linear(hidden[:, -1].to(weight.dtype), weight).float()
    return int(torch.argmax(logits))


def test_prefill_logits_match_the_causal_forward() -> None:
    module = backbone()
    weight = module.embed_tokens.weight
    ids = torch.randint(0, 128, (1, 6))
    hidden, _ = prefill(module, ids, torch.ones_like(ids))
    want = F.linear(module(ids, torch.ones_like(ids))[:, -1].to(weight.dtype), weight)
    assert torch.equal(logits_of(hidden, weight), want.float())


def test_a_cached_step_matches_the_full_forward_across_a_chunk() -> None:
    module = backbone()
    weight = module.embed_tokens.weight
    ids = torch.randint(0, 128, (1, 70))
    mask = torch.ones_like(ids)
    tokens = greedy(module, weight, ids, mask, 2, None, 0)
    assert tokens[0][0] == argmax_at(module, weight, ids)
    extended = torch.cat([ids, torch.tensor([[tokens[0][0]]])], dim=1)
    assert tokens[0][1] == argmax_at(module, weight, extended)


def test_equal_rows_match_each_row_alone() -> None:
    module = backbone()
    weight = module.embed_tokens.weight
    rows = [[3, 9, 11, 4], [8, 2, 7, 15]]
    ids, mask = pad_batch(rows, 0, torch.device("cpu"))
    batched = greedy(module, weight, ids, mask, 1, None, 0)
    for index, row in enumerate(rows):
        alone = torch.tensor(row)[None]
        got = greedy(module, weight, alone, torch.ones_like(alone), 1, None, 0)
        assert batched[index] == got[0]


def test_the_longest_right_padded_row_matches_alone() -> None:
    module = backbone()
    weight = module.embed_tokens.weight
    short, long = [4, 6, 8], [1, 2, 3, 4, 5, 9]
    ids, mask = pad_batch([short, long], 0, torch.device("cpu"))
    batched = greedy(module, weight, ids, mask, 2, None, 0)
    alone = torch.tensor(long)[None]
    got = greedy(module, weight, alone, torch.ones_like(alone), 2, None, 0)
    assert batched[1] == got[0]


def test_padding_positions_are_zero() -> None:
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    positions = rotary_positions(mask)
    assert torch.equal(positions[0, 0], torch.tensor([0, 1, 2, 0, 0]))
    assert torch.equal(positions[0, 1], torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(positions[1], positions[0])
    kept = decode_mask(mask)
    assert kept is not None and kept.shape == (2, 1, 1, 5)
    assert kept[0, 0, 0].tolist() == [True, True, True, False, False]
    assert decode_mask(torch.ones(2, 5, dtype=torch.long)) is None


def test_right_padding_is_read_at_its_last_position_and_then_masked() -> None:
    """A shorter row's first token is the padding position, and the next step drops those keys."""
    module = backbone()
    weight = module.embed_tokens.weight
    short, long = [4, 6, 8], [1, 2, 3, 4, 5, 9, 7]
    ids, mask = pad_batch([short, long], 0, torch.device("cpu"))
    hidden, past = prefill(module, ids, mask)
    batched = logits_of(hidden, weight)
    alone_ids = torch.tensor(short)[None]
    alone = logits_of(prefill(module, alone_ids, torch.ones_like(alone_ids))[0], weight)
    assert not torch.equal(batched[0], alone[0])

    positions = rotary_positions(mask)[0]
    cursor = positions[:, -1] + 1
    assert cursor.tolist() == [1, len(long)]
    tokens = torch.argmax(batched, dim=-1)
    extended = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=1)
    masked, _ = step(module, tokens, past, cursor, extended)
    opened, _ = step(module, tokens, past, cursor, torch.ones_like(extended))
    assert not torch.allclose(masked[0], opened[0])
    assert torch.allclose(masked[1], opened[1])


def test_eos_ends_a_row_and_is_not_returned() -> None:
    module = backbone()
    weight = module.embed_tokens.weight
    ids = torch.randint(0, 128, (1, 4))
    first = argmax_at(module, weight, ids)
    tokens = greedy(module, weight, ids, torch.ones_like(ids), 5, first, 0)
    assert tokens == [[]]
