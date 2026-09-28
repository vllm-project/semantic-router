"""Generate synthetic RoPE/model goldens with pinned official Transformers.

Run --mode tf4 under Transformers 4.57.6, then --mode tf5 under 5.3.0, using
the same output directory. No downloads, training, tokenization, or dataset inputs.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
import transformers.modeling_rope_utils as rope_utils
from safetensors.torch import load_file, save_file
from transformers import ModernBertConfig, ModernBertModel
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertRotaryEmbedding,
    apply_rotary_pos_emb,
)

POSITIONS = [0, 1, 255, 256, 2047, 2048, 8191, 8192, 16383, 16384, 32766, 32767]
RECIPES = [
    (
        10000.0,
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "truncate": True,
        },
    ),
    (
        160000.0,
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 16.0,
            "beta_slow": 2.0,
            "attention_factor": 1.3,
        },
    ),
    (
        20000.0,
        {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 8192},
    ),
]
REFERENCE_FIELDS = {
    "rope.json": ("inv_freq", "cos", "sin", "q", "k", "rotated_q", "rotated_k"),
    "tiny-output.json": ("valid_hidden", "masked_mean"),
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_shared(path, value):
    """Store one reference only when both official versions agree exactly."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Official reference versions disagree: {path.name}")
    else:
        path.write_text(encoded, encoding="utf-8")


def write_references(output, references):
    """Keep case metadata readable and store every official F32 bit losslessly."""
    tensors = {}
    for filename, fields in REFERENCE_FIELDS.items():
        for index, case in enumerate(references[filename]["cases"]):
            for field in fields:
                tensor = torch.tensor(case[field], dtype=torch.float32)
                if not torch.isfinite(tensor).all():
                    raise ValueError("Reference values must be finite float32")
                name = f"{Path(filename).stem}.{index}.{field}"
                tensors[name] = tensor.contiguous()
                case[field] = {"tensor": name, "shape": list(tensor.shape)}
    path = output / "reference.safetensors.fixture"
    if path.exists():
        previous = load_file(path)
        if set(previous) != set(tensors) or any(
            previous[name].dtype != tensor.dtype
            or previous[name].shape != tensor.shape
            or not torch.equal(
                previous[name].view(torch.int32), tensor.view(torch.int32)
            )
            for name, tensor in tensors.items()
        ):
            raise ValueError("Official reference versions disagree: tensor bits")
    else:
        save_file(tensors, path)
    for filename, value in references.items():
        write_shared(output / filename, value)


def cache_goldens(mode):
    cases = []
    for theta, recipe in RECIPES:
        kwargs = {
            "hidden_size": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "max_position_embeddings": 32768,
            "global_attn_every_n_layers": 2,
        }
        if mode == "tf4":
            config = ModernBertConfig(**kwargs, rope_scaling=recipe)
            config.rope_theta = theta
        else:
            params = {"rope_theta": theta, **recipe}
            # TF5.3's nested YaRN validator reads the original length from the
            # outer dictionary and raises KeyError. Keep official model math:
            # create the config normally, then set the fully specified recipe.
            config = ModernBertConfig(**kwargs)
            config.rope_parameters = {
                "full_attention": params,
                "sliding_attention": dict(params),
            }
        rope = ModernBertRotaryEmbedding(config, device="cpu")
        q = (
            torch.arange(len(POSITIONS) * 64, dtype=torch.float32).reshape(
                1, 1, len(POSITIONS), 64
            )
            / 500
            - 0.7
        )
        k = q.flip(-1).contiguous()
        if mode == "tf4":
            cos, sin = rope(q, torch.tensor([POSITIONS]))
            inv_freq, attention_factor = rope.inv_freq, rope.attention_scaling
        else:
            cos, sin = rope(q, torch.tensor([POSITIONS]), layer_type="full_attention")
            inv_freq = rope.full_attention_inv_freq
            attention_factor = rope.full_attention_attention_scaling
        rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cos, sin)
        cases.append(
            {
                "theta": theta,
                "recipe": recipe,
                "head_dim": 64,
                "positions": POSITIONS,
                "inv_freq": inv_freq.tolist(),
                "attention_factor": attention_factor,
                "cos": cos[0, :, :32].tolist(),
                "sin": sin[0, :, :32].tolist(),
                "q": q.flatten().tolist(),
                "k": k.flatten().tolist(),
                "rotated_q": rotated_q.flatten().tolist(),
                "rotated_k": rotated_k.flatten().tolist(),
            }
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tf4", "tf5"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {"tf4": "4.57.6", "tf5": "5.3.0"}[args.mode]
    if transformers.__version__ != expected:
        raise RuntimeError(
            f"Expected Transformers {expected}, got {transformers.__version__}"
        )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260913)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    variant = output / args.mode
    variant.mkdir(exist_ok=False)
    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "max_position_embeddings": 32768,
        "global_attn_every_n_layers": 2,
        "local_attention": 8,
        "pad_token_id": 0,
        "norm_eps": 1e-5,
        "classifier_bias": False,
        "attention_bias": False,
        "mlp_bias": False,
        "norm_bias": False,
        "reference_compile": False,
    }
    recipe = RECIPES[0][1]
    if args.mode == "tf4":
        config = ModernBertConfig(
            **kwargs,
            global_rope_theta=10000.0,
            local_rope_theta=20000.0,
            rope_scaling=recipe,
        )
    else:
        config = ModernBertConfig(
            **kwargs, layer_types=["full_attention", "sliding_attention"]
        )
        config.rope_parameters = {
            "full_attention": {"rope_theta": 10000.0, **recipe},
            "sliding_attention": {"rope_theta": 20000.0, **recipe},
        }
    config._attn_implementation = "sdpa"
    model = ModernBertModel(config).float().eval()
    weights = output / "weights.safetensors.fixture"
    if args.mode == "tf4":
        save_file(
            {name: tensor.contiguous() for name, tensor in model.state_dict().items()},
            weights,
        )
    else:
        model.load_state_dict(load_file(weights), strict=True)
    write_json(variant / "config.json", config.to_dict())
    cases = []
    for length in [1, 17, 65]:
        ids = (torch.arange(2 * length).reshape(2, length) % 63 + 1).long()
        mask = torch.ones_like(ids)
        if length > 1:
            ids[1, -3:] = 0
            mask[1, -3:] = 0
        with torch.inference_mode():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        assert hidden.dtype == torch.float32 and torch.isfinite(hidden).all()
        cases.append(
            {
                "input_ids": ids.tolist(),
                "attention_mask": mask.tolist(),
                "valid_hidden": hidden[mask.bool()].flatten().tolist(),
                "masked_mean": pooled.tolist(),
            }
        )
    write_references(
        output,
        {
            "tiny-output.json": {
                "cases": cases,
                "compared_positions": "valid tokens and their masked mean; padded query hidden states are outside this assertion",
            },
            "rope.json": {"cases": cache_goldens(args.mode)},
        },
    )
    source = Path(inspect.getsourcefile(ModernBertRotaryEmbedding))

    utility = Path(inspect.getsourcefile(rope_utils))
    write_json(
        variant / "manifest.json",
        {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float32",
            "attention": "sdpa",
            "seed": 20260913,
            "script_sha256": digest(Path(__file__)),
            "implementation_sha256": {
                source.name: digest(source),
                utility.name: digest(utility),
            },
            "weights_sha256": digest(weights),
            "files": {
                "config.json": digest(variant / "config.json"),
                "../rope.json": digest(output / "rope.json"),
                "../tiny-output.json": digest(output / "tiny-output.json"),
                "../reference.safetensors.fixture": digest(
                    output / "reference.safetensors.fixture"
                ),
            },
            "configuration_path": (
                "direct official TF4 constructor"
                if args.mode == "tf4"
                else "official TF5 config then explicit nested parameters; upstream nested YaRN validator KeyError preserved separately"
            ),
            "scope": "Generated random tiny backbone and numerical arrays; no trained checkpoint, data, or task-quality claim",
        },
    )


if __name__ == "__main__":
    main()
