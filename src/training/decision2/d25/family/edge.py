"""d3-edge: a Qwen3-VL code-readout model from Qwen3-0.6B-Base (text) and the Qwen3.5-0.8B vision tower.

    python -m d25.family.edge build --text TEXT_DIR --vision VISION_DIR --out OUT
    python -m d25.family.edge check --ckpt OUT --text TEXT_DIR --vision VISION_DIR --rows DEV.jsonl.gz \
        --report REPORT.json

``build`` writes the Omni code-readout v1 layout with a ``Qwen3VLModel`` backbone (``config.json``,
``model-*.safetensors`` + index, ``readout.safetensors``, ``decision_config.json``, processor files):

- ``language_model.*``: the tensors of Qwen/Qwen3-0.6B-Base (28 layers, hidden 1024, tied embeddings).
  Its text config keeps Qwen3's RoPE base; M-RoPE sections [24, 20, 20] cover the 64 rotary
  frequencies, and with equal t/h/w positions (every text token) M-RoPE is the 1-D RoPE Qwen3 was
  trained with.
- ``visual.*``: the vision tower and merger of Qwen/Qwen3.5-0.8B (12 blocks x 768, merger into 1024
  = the Qwen3 width; same tensor names as the Qwen3-VL ViT), DeepStack off.
- readout: the embedding rows of the 255 answer codes (the tied head), FP32.
- tokenizer: Qwen3's; chat template: Qwen3's with Qwen3-VL image/video placeholders
  (``edge_chat_template.jinja``; string content renders byte for byte as with Qwen3's template);
  image and video processor configs: Qwen3.5-0.8B's.

Both sources must be snapshots of the pinned revisions (checked from the Hub download metadata).
Every written tensor is re-read and checked bit-equal to its source.

``check`` (CPU, FP32 unless noted) writes a JSON report and exits non-zero on any failure: parameter
counts; text and vision tensors bit-equal to their sources; the vision tower's output equal to
Qwen3.5-0.8B's on a test image; full-vocabulary text logits equal to Qwen3-0.6B-Base's; left-padded
batches equal to single rows; text rendering and answer codes equal to the base tokenizer's; image
placeholders before the text; the Omni vision engine and the Vega text engine load the checkpoint,
answer text and image requests, and agree on text requests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from d25.omni.common import vision_format
from d25.omni.model import checkpoint
from d25.vega.common import decision_format as text_format

TEXT_BASE = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
VISION_BASE = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
MROPE_SECTION = [24, 20, 20]
MAX_LENGTH = 16_384
TEMPLATE = Path(__file__).with_name("edge_chat_template.jinja")
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt")
IMAGE_PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")
VISION_TOKENS = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}
TEXT_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
)
VISION_KEYS = (
    "depth",
    "hidden_act",
    "hidden_size",
    "in_channels",
    "initializer_range",
    "intermediate_size",
    "num_heads",
    "num_position_embeddings",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
)


def snapshot_revision(directory: Path, files: list[str]) -> str:
    """The single Hub commit the files of a ``snapshot_download(local_dir=...)`` copy came from."""
    commits = set()
    for name in files:
        meta = directory / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        if not meta.exists():
            raise FileNotFoundError(
                f"{meta}: no download metadata; cannot verify the revision"
            )
        commits.add(meta.read_text().splitlines()[0].strip())
    if len(commits) != 1:
        raise ValueError(
            f"{directory}: files come from several revisions {sorted(commits)}"
        )
    return commits.pop()


def verify_source(
    directory: Path, repo: str, revision: str, files: list[str]
) -> dict[str, str]:
    found = snapshot_revision(directory, files)
    if found != revision:
        raise ValueError(
            f"{directory}: {repo} snapshot is at {found}, expected {revision}"
        )
    return {name: checkpoint.file_sha256(directory / name) for name in sorted(files)}


def chat_template() -> str:
    return TEMPLATE.read_text()


def edge_config(
    text_cfg: dict[str, Any], vision_cfg: dict[str, Any], token_ids: dict[str, int]
):
    from transformers import Qwen3VLConfig

    if text_cfg.get("model_type") != "qwen3" or not text_cfg.get("tie_word_embeddings"):
        raise ValueError("the text source must be a Qwen3 model with tied embeddings")
    if text_cfg.get("rope_scaling") or text_cfg.get("use_sliding_window"):
        raise ValueError(
            "the text source uses RoPE scaling or sliding windows; not supported"
        )
    if 2 * sum(MROPE_SECTION) != text_cfg["head_dim"]:
        raise ValueError(
            f"M-RoPE sections {MROPE_SECTION} do not cover head_dim {text_cfg['head_dim']}"
        )
    text = {key: text_cfg[key] for key in TEXT_KEYS}
    text["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": float(text_cfg["rope_theta"]),
        "mrope_section": list(MROPE_SECTION),
        "mrope_interleaved": True,
    }
    text["tie_word_embeddings"] = True
    vision = {key: vision_cfg[key] for key in VISION_KEYS}
    vision["deepstack_visual_indexes"] = []
    if vision["out_hidden_size"] != text["hidden_size"]:
        raise ValueError(
            f"merger output {vision['out_hidden_size']} != text hidden size {text['hidden_size']}"
        )
    config = Qwen3VLConfig(
        text_config=text, vision_config=vision, tie_word_embeddings=True, **token_ids
    )
    config.architectures = ["Qwen3VLModel"]
    return config


def tensor_plan(text_dir: Path, vision_dir: Path) -> dict[str, tuple[Path, str]]:
    """Edge tensor name -> (source directory, stored name)."""
    plan: dict[str, tuple[Path, str]] = {}
    for key in checkpoint.weight_files(text_dir):
        if key == "lm_head.weight":
            continue
        if not key.startswith("model."):
            raise ValueError(f"{text_dir}: unexpected tensor {key}")
        plan["language_model." + key[len("model.") :]] = (text_dir, key)
    for key in checkpoint.weight_files(vision_dir):
        if key.startswith("model.visual."):
            plan["visual." + key[len("model.visual.") :]] = (vision_dir, key)
    return plan


def read_sources(plan: dict[str, tuple[Path, str]]) -> dict[str, torch.Tensor]:
    by_dir: dict[Path, dict[str, str]] = {}
    for name, (directory, key) in plan.items():
        by_dir.setdefault(directory, {})[key] = name
    tensors: dict[str, torch.Tensor] = {}
    for directory, keys in by_dir.items():
        for key, tensor in checkpoint.iter_tensors(directory, set(keys)):
            tensors[keys[key]] = tensor
    return tensors


def write_processor_files(
    text_dir: Path, vision_dir: Path, out: Path
) -> dict[str, str]:
    template = chat_template()
    for name in TOKENIZER_FILES:
        (out / name).write_bytes((text_dir / name).read_bytes())
    tokenizer_config = json.loads((text_dir / "tokenizer_config.json").read_text())
    tokenizer_config.pop("chat_template", None)
    checkpoint.write_json(out / "tokenizer_config.json", tokenizer_config)
    (out / "chat_template.jinja").write_text(template)
    for name in IMAGE_PROCESSOR_FILES:
        (out / name).write_bytes((vision_dir / name).read_bytes())
    return {
        name: checkpoint.file_sha256(out / name)
        for name in (
            *TOKENIZER_FILES,
            "tokenizer_config.json",
            "chat_template.jinja",
            *IMAGE_PROCESSOR_FILES,
        )
    }


def parameter_counts(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    counts = {"language": 0, "vision_encoder": 0, "vision_merger": 0}
    for name, tensor in tensors.items():
        counts[checkpoint.component(name)] += tensor.numel()
    counts["backbone"] = sum(counts.values())
    return counts


def build(
    text_dir: Path, vision_dir: Path, out: Path, max_length: int = MAX_LENGTH
) -> dict[str, Any]:
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLModel

    started = time.time()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    text_files = sorted(
        {
            *checkpoint.weight_files(text_dir).values(),
            "config.json",
            *TOKENIZER_FILES,
            "tokenizer_config.json",
        }
    )
    vision_files = sorted(
        {
            *checkpoint.weight_files(vision_dir).values(),
            "config.json",
            checkpoint.INDEX_FILE,
            *IMAGE_PROCESSOR_FILES,
        }
    )
    text_hashes = verify_source(text_dir, *TEXT_BASE, text_files)
    vision_hashes = verify_source(vision_dir, *VISION_BASE, vision_files)
    text_cfg = json.loads((text_dir / "config.json").read_text())
    vision_cfg = json.loads((vision_dir / "config.json").read_text())["vision_config"]

    base_tokenizer = AutoTokenizer.from_pretrained(str(text_dir))
    token_ids = {}
    for key, token in VISION_TOKENS.items():
        index = base_tokenizer.convert_tokens_to_ids(token)
        if not isinstance(index, int) or index == base_tokenizer.unk_token_id:
            raise ValueError(f"{token} is not a single token of the text tokenizer")
        token_ids[key] = index
    config = edge_config(text_cfg, vision_cfg, token_ids)

    plan = tensor_plan(text_dir, vision_dir)
    with torch.device("meta"):
        skeleton = Qwen3VLModel(config)
    expected = {name: tuple(t.shape) for name, t in skeleton.state_dict().items()}
    del skeleton
    if set(plan) != set(expected):
        raise ValueError(
            f"tensor names differ from Qwen3VLModel: missing {sorted(set(expected) - set(plan))[:5]}, "
            f"unexpected {sorted(set(plan) - set(expected))[:5]}"
        )
    tensors = read_sources(plan)
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {expected[name]}")
    if "lm_head.weight" in checkpoint.weight_files(text_dir):
        ((_, head),) = checkpoint.iter_tensors(text_dir, {"lm_head.weight"})
        if not torch.equal(head, tensors["language_model.embed_tokens.weight"]):
            raise ValueError(f"{text_dir}: lm_head differs from the tied embeddings")

    out.mkdir(parents=True, exist_ok=True)
    writer = checkpoint.ShardWriter(out)
    digests = {}
    for name in sorted(tensors):
        writer.add(name, tensors[name])
        digests[name] = checkpoint.tensor_digest(tensors[name])
    shard_hashes = writer.close()
    reread = {
        key: checkpoint.tensor_digest(t) for key, t in checkpoint.iter_tensors(out)
    }
    bad = sorted(name for name in digests if reread.get(name) != digests[name])
    if bad or set(reread) != set(digests):
        raise ValueError(
            f"written tensors are not bit-equal to their sources: {bad[:5]}"
        )
    config.save_pretrained(str(out))
    processor_hashes = write_processor_files(text_dir, vision_dir, out)

    tokenizer = AutoTokenizer.from_pretrained(str(out))
    codes, code_ids = text_format.answer_codes(tokenizer)
    if (codes, code_ids) != text_format.answer_codes(base_tokenizer):
        raise ValueError("answer codes differ from the base tokenizer's")
    readout = tensors["language_model.embed_tokens.weight"][code_ids].float().clone()
    readout_sha256 = checkpoint.save_readout(out, readout)
    AutoProcessor.from_pretrained(str(out))

    counts = parameter_counts(tensors)
    counts["readout"] = readout.numel()
    vision_digests = {k: v for k, v in digests.items() if k.startswith("visual.")}
    decision = {
        "format_version": 1,
        "format_id": vision_format.FORMAT_ID,
        "prompt": "d25-vega",
        "base_model": TEXT_BASE[0],
        "revision": TEXT_BASE[1],
        "vision_base_model": VISION_BASE[0],
        "vision_revision": VISION_BASE[1],
        "architecture": "Qwen3VLModel",
        "codes": codes,
        "token_ids": code_ids,
        "temperature": 1.0,
        "attention_mode": "causal",
        "pooling": "last",
        "max_length": int(max_length),
        "readout_dtype": "float32",
        "modalities": ["text", "image"],
        "images": {
            "max_images": vision_format.MAX_IMAGES,
            "min_pixels": vision_format.MIN_PIXELS,
            "max_pixels": vision_format.MAX_PIXELS,
            "order": "before_text",
        },
        "provenance": {
            "assembled": {
                "tool": "d25.family.edge",
                "code_sha256": {
                    "edge.py": checkpoint.file_sha256(__file__),
                    "edge_chat_template.jinja": checkpoint.file_sha256(TEMPLATE),
                },
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - started, 1),
            },
            "init": {
                "kind": "edge",
                "text": {
                    "repo": TEXT_BASE[0],
                    "revision": TEXT_BASE[1],
                    "files_sha256": text_hashes,
                },
                "vision": {
                    "repo": VISION_BASE[0],
                    "revision": VISION_BASE[1],
                    "files_sha256": vision_hashes,
                },
            },
            "vision": {
                "status": "stock",
                "source": f"{VISION_BASE[0]}@{VISION_BASE[1]}",
                "tensors": len(vision_digests),
                "parameters": counts["vision_encoder"] + counts["vision_merger"],
                "digest": checkpoint.digest_of_digests(vision_digests),
                "bit_equal_to_stock": True,
            },
            "readout": {
                "source": "embed_tokens rows of the answer codes (tied lm_head)",
                "sha256": readout_sha256,
            },
            "parameters": counts,
            "processor_sha256": processor_hashes,
            "shards_sha256": shard_hashes,
        },
    }
    checkpoint.write_json(out / checkpoint.DECISION_CONFIG, decision)
    return decision


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def sample_rows(paths: list[str], limit: int) -> list[dict[str, Any]]:
    from d25.omni.eval.shards import read_jsonl

    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(Path(path)):
            rows.append({"state": row["state"], "question": row["question"]})
            if len(rows) >= limit:
                return rows
    return rows


def test_image(width: int, height: int, seed: int):
    """A deterministic RGB test image (gradients, shapes and noise)."""
    from PIL import Image, ImageDraw

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randint(
        0, 64, (height, width, 3), generator=generator, dtype=torch.uint8
    )
    ys = torch.linspace(0, 191, height)[:, None].expand(height, width)
    xs = torch.linspace(0, 191, width)[None, :].expand(height, width)
    base = torch.stack([ys, xs, (ys + xs) / 2], dim=-1).to(torch.uint8) + noise
    image = Image.fromarray(base.numpy(), "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [width // 8, height // 8, width // 3, height // 3], fill=(220, 30, 30)
    )
    draw.ellipse(
        [width // 2, height // 2, width - width // 8, height - height // 8],
        fill=(30, 30, 220),
    )
    return image


def check(
    ckpt: Path,
    text_dir: Path,
    vision_dir: Path,
    row_files: list[str],
    n_rows: int,
    engines: bool,
) -> dict[str, Any]:
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
        Qwen3ForCausalLM,
        Qwen3VLModel,
    )

    torch.manual_seed(0)
    report: dict[str, Any] = {"ckpt": str(ckpt), "checks": {}}
    checks = report["checks"]
    decision = checkpoint.read_decision_config(ckpt)

    # Tensors and parameter counts.
    plan = tensor_plan(text_dir, vision_dir)
    source = {
        name: checkpoint.tensor_digest(t) for name, t in read_sources(plan).items()
    }
    written = {
        name: checkpoint.tensor_digest(t) for name, t in checkpoint.iter_tensors(ckpt)
    }
    vision_names = [n for n in source if n.startswith("visual.")]
    text_names = [n for n in source if n.startswith("language_model.")]
    checks["vision_tensors_bit_equal"] = {
        "tensors": len(vision_names),
        "ok": set(written) == set(source)
        and all(written[n] == source[n] for n in vision_names),
    }
    checks["text_tensors_bit_equal"] = {
        "tensors": len(text_names),
        "ok": all(written.get(n) == source[n] for n in text_names),
    }
    shapes = {name: t.numel() for name, t in checkpoint.iter_tensors(ckpt)}
    counts = {"language": 0, "vision_encoder": 0, "vision_merger": 0}
    for name, numel in shapes.items():
        counts[checkpoint.component(name)] += numel
    counts["backbone"] = sum(counts.values())
    counts["readout"] = checkpoint.load_readout(ckpt).numel()
    model = Qwen3VLModel.from_pretrained(
        str(ckpt), dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    counts["module_parameters"] = sum(p.numel() for p in model.parameters())
    checks["parameters"] = {
        **counts,
        "ok": counts["module_parameters"] == counts["backbone"],
    }

    # Text rendering and answer codes.
    edge_tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    base_tokenizer = AutoTokenizer.from_pretrained(str(text_dir))
    rows = sample_rows(row_files, n_rows)
    codes = decision["codes"]
    renders = [
        text_format.render(edge_tokenizer, r["state"], r["question"], codes)
        for r in rows
    ]
    base_renders = [
        text_format.render(base_tokenizer, r["state"], r["question"], codes)
        for r in rows
    ]
    processor = AutoProcessor.from_pretrained(str(ckpt))
    image_text = vision_format.render(
        processor, rows[0]["state"], rows[0]["question"], codes, 2
    )
    user_turn = image_text.split("<|im_start|>user\n", 1)[1]
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    checks["template"] = {
        "rows": len(rows),
        "tokenizer_and_processor_use_edge_template": edge_tokenizer.chat_template
        == chat_template()
        and processor.chat_template == chat_template(),
        "text_renders_equal_base": renders == base_renders,
        "processor_text_render_equal": all(
            vision_format.render(processor, r["state"], r["question"], codes, 0) == t
            for r, t in zip(rows, renders)
        ),
        "answer_codes_equal_base": text_format.answer_codes(base_tokenizer)
        == (decision["codes"], decision["token_ids"]),
        "images_before_text": user_turn.startswith(placeholder * 2 + "State:"),
    }
    checks["template"]["ok"] = all(
        v for k, v in checks["template"].items() if k != "rows"
    )

    # Full-vocabulary text logits against Qwen3-0.6B-Base (single rows, no padding).
    base = Qwen3ForCausalLM.from_pretrained(
        str(text_dir), dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    embed = model.language_model.embed_tokens.weight
    readout = checkpoint.load_readout(ckpt).float()
    plain = [
        "The capital of France is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n",
    ]
    sequences = [
        edge_tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in renders + plain
    ]
    diffs, code_diffs, agree, last_hidden = [], [], [], []
    with torch.inference_mode():
        for ids in sequences:
            tensor = torch.tensor([ids])
            reference = base(input_ids=tensor).logits[0]
            hidden = model(input_ids=tensor, use_cache=False).last_hidden_state[0]
            logits = hidden @ embed.T
            diffs.append(float((logits - reference).abs().max()))
            code_logits = hidden[-1] @ readout.T
            code_diffs.append(
                float((code_logits - reference[-1, decision["token_ids"]]).abs().max())
            )
            agree.append(bool(torch.equal(logits.argmax(-1), reference.argmax(-1))))
            last_hidden.append(hidden[-1])
    checks["text_logits_vs_base"] = {
        "prompts": len(sequences),
        "tokens": [len(s) for s in sequences],
        "max_abs_diff_full_vocab": max(diffs),
        "max_abs_diff_codes": max(code_diffs),
        "argmax_equal_all_positions": all(agree),
        "ok": max(diffs) <= 1e-4 and max(code_diffs) <= 1e-4 and all(agree),
    }
    del base

    # Left-padded batch (the engines' layout) against single rows.
    width = max(len(s) for s in sequences)
    pad = edge_tokenizer.pad_token_id
    ids = torch.full((len(sequences), width), pad)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, width - len(seq) :] = torch.tensor(seq)
        mask[i, width - len(seq) :] = 1
    with torch.inference_mode():
        batched = model(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state[:, -1]
    single = torch.stack(last_hidden)
    checks["left_padded_batch"] = {
        "max_abs_diff_code_logits": float(((batched - single) @ readout.T).abs().max()),
        "ok": float(((batched - single) @ readout.T).abs().max()) <= 1e-3,
    }

    # Vision tower output against Qwen3.5-0.8B's on test images.
    stock = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(vision_dir), dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    stock_processor = AutoProcessor.from_pretrained(str(vision_dir))
    images = [test_image(640, 480, 1), test_image(333, 517, 2)]
    vision_format.configure_processor(processor)
    vision_format.configure_processor(stock_processor)
    ours = processor.image_processor(images=images, return_tensors="pt")
    theirs = stock_processor.image_processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        out_edge = model.visual(
            ours["pixel_values"], grid_thw=ours["image_grid_thw"]
        ).pooler_output
        out_stock = stock.model.visual(
            theirs["pixel_values"], grid_thw=theirs["image_grid_thw"]
        ).pooler_output
    checks["vision_forward_vs_stock"] = {
        "pixel_values_equal": bool(
            torch.equal(ours["pixel_values"], theirs["pixel_values"])
        ),
        "grid_thw": ours["image_grid_thw"].tolist(),
        "max_abs_diff": float((out_edge - out_stock).abs().max()),
        "ok": bool(torch.equal(ours["pixel_values"], theirs["pixel_values"]))
        and float((out_edge - out_stock).abs().max()) == 0.0,
    }
    del stock

    # M-RoPE positions of an image request (sanity, reported).
    encoded = processor(text=[image_text], images=images, return_tensors="pt")
    positions, _ = model.get_rope_index(
        encoded["input_ids"],
        mm_token_type_ids=encoded["mm_token_type_ids"],
        image_grid_thw=encoded["image_grid_thw"],
        attention_mask=encoded["attention_mask"],
    )
    image_tokens = int((encoded["input_ids"] == model.config.image_token_id).sum())
    checks["image_request"] = {
        "tokens": int(encoded["input_ids"].shape[1]),
        "image_tokens": image_tokens,
        "expected_image_tokens": int(
            sum(g.prod() // 4 for g in encoded["image_grid_thw"])
        ),
        "max_position": int(positions.max()),
        "ok": image_tokens
        == int(sum(g.prod() // 4 for g in encoded["image_grid_thw"])),
    }
    del model

    if engines:
        checks["engines"] = check_engines(ckpt, rows[:4], images)
    report["ok"] = all(part.get("ok") for part in checks.values())
    return report


def check_engines(ckpt: Path, rows: list[dict[str, Any]], images) -> dict[str, Any]:
    from d25.omni.eval.engine import VisionCodeReadoutModel
    from d25.vega.eval.engine import CodeReadoutModel

    requests = [{**r, "images": []} for r in rows]
    requests += [
        {**rows[0], "images": [images[0]]},
        {**rows[1], "images": [images[0], images[1]]},
    ]
    vision = VisionCodeReadoutModel(
        ckpt, device="cpu", dtype="bfloat16", prefetch=False
    )
    outcomes = vision.score(requests)
    text = CodeReadoutModel(ckpt, device="cpu").predict(rows)
    statuses = [o.status for o in outcomes]
    sums = [sum(o.probabilities) for o in outcomes if o.probabilities]
    text_diff = max(
        max(abs(a - b) for a, b in zip(o.probabilities, p))
        for o, p in zip(outcomes[: len(rows)], text)
    )
    return {
        "statuses": statuses,
        "probability_sums": [round(s, 6) for s in sums],
        "image_probabilities": [o.probabilities for o in outcomes[len(rows) :]],
        "text_vision_vs_vega_max_abs_diff": text_diff,
        "ok": all(s == "ok" for s in statuses)
        and all(abs(s - 1) < 1e-4 for s in sums)
        and text_diff < 1e-2,
    }


def smoke(ckpt: Path, row_files: list[str], work: Path) -> dict[str, Any]:
    """Two E2-style updates (merger + language model + readout) on text rows and synthetic image rows,
    then the saved checkpoint through both engines. Runs on CPU in FP32."""
    import gzip
    import io

    from d25.family import edge_train
    from d25.omni.eval.engine import VisionCodeReadoutModel
    from d25.omni.eval.shards import read_jsonl
    from d25.vega.eval.engine import CodeReadoutModel

    work.mkdir(parents=True, exist_ok=True)
    text_rows = []
    for path in row_files:
        for row in read_jsonl(Path(path)):
            if len(json.dumps(row["state"])) < 1500:
                text_rows.append(row)
            if len(text_rows) == 8:
                break
    text_rows, dev_rows = text_rows[:6], text_rows[6:]
    colours = {"red": (220, 30, 30), "blue": (30, 30, 220), "green": (30, 200, 30)}
    image_rows = []
    for index, (name, rgb) in enumerate(list(colours.items()) * 2):
        image = test_image(224 + 32 * index, 192 + 16 * index, 10 + index)
        image.paste(rgb, (40, 40, 160, 140))
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        ref = vision_format.store_image(payload.getvalue(), work, "png")
        keys = list(colours)
        image_rows.append(
            {
                "id": f"edge-smoke-image-{index}",
                "source": "edge-smoke",
                "family": "edge-smoke",
                "state": "A photo is attached.",
                "question": {
                    "type": "choice",
                    "instructions": "Which colour is the large rectangle in the image?",
                    "criteria": dict.fromkeys(keys),
                },
                "target": [1.0 if k == name else 0.0 for k in keys],
                "label": keys.index(name),
                "images": [ref],
                "meta": {
                    "image_sha256": [vision_format.sha256_bytes(payload.getvalue())]
                },
            }
        )
    for name, rows in (
        ("images.jsonl.gz", image_rows),
        ("text.jsonl.gz", text_rows),
        ("dev.jsonl.gz", dev_rows),
    ):
        with gzip.open(work / name, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    out = work / "run"
    edge_train.main(
        [
            "--init",
            str(ckpt),
            "--arm",
            "E2-joint",
            "--rows",
            str(work / "images.jsonl.gz"),
            "--replay-rows",
            str(work / "text.jsonl.gz"),
            "--replay-ratio",
            "0.5",
            "--dev-rows",
            str(work / "dev.jsonl.gz"),
            "--lr",
            "1e-5",
            "--warmup-ratio",
            "0.5",
            "--effective-batch-size",
            "6",
            "--token-budget",
            "4096",
            "--max-rows-per-microbatch",
            "4",
            "--stop-after",
            "2",
            "--save-every",
            "2",
            "--eval-every",
            "2",
            "--out",
            str(out),
        ]
    )
    saved = sorted((out / "checkpoints").glob("step-*"))[-1]
    training = [
        json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()
    ]
    requests = [
        {"state": r["state"], "question": r["question"], "images": []}
        for r in text_rows[:2]
    ]
    requests += [
        {
            "state": r["state"],
            "question": r["question"],
            "images": [str(work / r["images"][0])],
        }
        for r in image_rows[:2]
    ]
    outcomes = VisionCodeReadoutModel(saved, device="cpu", prefetch=False).score(
        requests
    )
    text = CodeReadoutModel(saved, device="cpu").predict(requests[:2])
    decision = checkpoint.read_decision_config(saved)
    return {
        "checkpoint": str(saved),
        "steps": [
            {k: r[k] for k in ("step", "loss", "grad_norm", "rows", "images", "tokens")}
            for r in training
        ],
        "statuses": [o.status for o in outcomes],
        "probabilities": [o.probabilities for o in outcomes],
        "vega_text_probabilities": text,
        "init_kind": ((decision.get("provenance") or {}).get("init") or {}).get("kind"),
        "ok": len(training) == 2
        and all(r["images"] > 0 for r in training)
        and all(o.status == "ok" for o in outcomes),
    }


def forward_parity(ckpt: Path, row_files: list[str]) -> dict[str, Any]:
    """Logits of the trainer's left-padded masked forward vs ``edge_train.right_padded_forward`` on one
    mixed batch (CPU, FP32)."""
    from transformers import AutoProcessor

    from d25.family.edge_train import right_padded_forward
    from d25.omni.model import inputs
    from d25.omni.train.collator import MultimodalCollator
    from d25.omni.train.model import OmniDecisionModel
    from d25.omni.train.rows import measure, read_rows

    decision = checkpoint.read_decision_config(ckpt)
    processor = inputs.setup_processor(AutoProcessor.from_pretrained(str(ckpt)))
    rows, _ = read_rows(row_files, "main")
    for row, (tokens, _) in zip(
        rows, measure(rows, processor, decision["codes"], "d25-vega", 16_384)
    ):
        row.tokens = tokens
    batch = MultimodalCollator(processor, decision["codes"]).__call__(rows)
    model = OmniDecisionModel.from_checkpoint(ckpt).eval()
    with torch.no_grad():
        masked = model(**batch.inputs)
        unmasked = right_padded_forward(model, **batch.inputs)
    diff = float((masked - unmasked).abs().max())
    return {
        "rows": len(rows),
        "images": batch.images,
        "lengths": batch.inputs["attention_mask"].sum(-1).tolist(),
        "max_abs_diff_logits": diff,
        "argmax_equal": bool(torch.equal(masked.argmax(-1), unmasked.argmax(-1))),
        "ok": diff <= 1e-3,
    }


BENCH_CONFIGS = (
    ("masked-ckpt-49152", True, True, 49_152, "--masked-forward --token-budget 49152"),
    ("right-ckpt-49152", False, True, 49_152, "--token-budget 49152"),
    ("right-ckpt-98304", False, True, 98_304, "--token-budget 98304"),
    (
        "right-nockpt-24576",
        False,
        False,
        24_576,
        "--no-gradient-checkpointing --token-budget 24576",
    ),
)


def bench(ckpt: Path, row_file: str, rows_per_step: int) -> dict[str, Any]:
    """Seconds and peak memory of one update's forward + backward (BF16 autocast, one GPU) per layout:
    the Omni trainer's left-padded masked forward or the right-padded one, with or without gradient
    checkpointing, at a few token budgets. Prints and returns ``configs`` (``args`` = trainer flags).
    """
    from transformers import AutoProcessor

    from d25.family.edge_train import right_padded_forward
    from d25.omni.model import inputs
    from d25.omni.model.patch_embed import linearize_patch_embed
    from d25.omni.train import schedule
    from d25.omni.train.collator import MultimodalCollator
    from d25.omni.train.loss import row_losses
    from d25.omni.train.model import OmniDecisionModel
    from d25.omni.train.rows import measure, read_rows

    device = torch.device("cuda:0")
    decision = checkpoint.read_decision_config(ckpt)
    processor = inputs.setup_processor(AutoProcessor.from_pretrained(str(ckpt)))
    rows, _ = read_rows([row_file], "main")
    rows = rows[:rows_per_step]
    for row, (tokens, reason) in zip(
        rows, measure(rows, processor, decision["codes"], "d25-vega", 8192)
    ):
        row.tokens = 0 if reason else tokens
    rows = [r for r in rows if r.tokens]
    tokens = [r.tokens for r in rows]
    collator = MultimodalCollator(processor, decision["codes"])
    model = OmniDecisionModel.from_checkpoint(ckpt).to(device)
    linearize_patch_embed(model.backbone)
    model.train()
    results = []
    for name, masked, ckpt_on, budget, flags in BENCH_CONFIGS:
        plan = schedule.plan_step(range(len(rows)), tokens, 1, budget, 64)[0]
        if ckpt_on:
            model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.backbone.gradient_checkpointing_disable()
        forward = (
            model.forward
            if masked
            else (lambda **kw: right_padded_forward(model, **kw))
        )
        record: dict[str, Any] = {
            "name": name,
            "args": flags,
            "microbatches": len(plan),
        }
        try:
            for repeat in range(2):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for indices in plan:
                    batch = collator([rows[i] for i in indices]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hidden_logits = forward(**batch.inputs)
                    loss = row_losses(
                        hidden_logits, batch.targets, batch.counts
                    ).cross_entropy.mean()
                    loss.backward()
                torch.cuda.synchronize()
                record["step_seconds"] = round(time.perf_counter() - started, 3)
                model.zero_grad(set_to_none=True)
            record["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            record["ok"] = True
        except torch.OutOfMemoryError as error:
            record.update(ok=False, error=f"OOM: {str(error)[:200]}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(json.dumps(record), flush=True)
        results.append(record)
    return {"rows": len(rows), "tokens": sum(tokens), "configs": results}


def bench_choice(path: Path) -> str:
    """Trainer flags of the fastest benchmarked layout (the masked default when nothing ran)."""
    try:
        configs = [c for c in json.loads(path.read_text())["configs"] if c.get("ok")]
    except (OSError, ValueError, KeyError):
        configs = []
    return (
        min(configs, key=lambda c: c["step_seconds"])["args"]
        if configs
        else BENCH_CONFIGS[0][4]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    bn = sub.add_parser("bench")
    bn.add_argument("--ckpt", required=True)
    bn.add_argument("--rows", required=True)
    bn.add_argument("--rows-per-step", type=int, default=256)
    bn.add_argument("--report", required=True)
    bc = sub.add_parser("bench-choice")
    bc.add_argument("--report", required=True)
    f = sub.add_parser("forward")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--rows", nargs="+", required=True)
    f.add_argument("--report", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--report", required=True)
    b = sub.add_parser("build")
    c = sub.add_parser("check")
    for p in (b, c):
        p.add_argument(
            "--text",
            required=True,
            help="Qwen/Qwen3-0.6B-Base snapshot (pinned revision)",
        )
        p.add_argument(
            "--vision",
            required=True,
            help="Qwen/Qwen3.5-0.8B snapshot (pinned revision)",
        )
    b.add_argument("--out", required=True)
    b.add_argument("--max-length", type=int, default=MAX_LENGTH)
    c.add_argument("--ckpt", required=True)
    c.add_argument(
        "--rows",
        nargs="+",
        required=True,
        help="training-format rows for the text checks",
    )
    c.add_argument("--n-rows", type=int, default=8)
    c.add_argument("--report", required=True)
    c.add_argument("--no-engines", action="store_true")
    args = parser.parse_args()
    if args.cmd == "bench":
        result = bench(Path(args.ckpt), args.rows, args.rows_per_step)
        checkpoint.write_json(args.report, result)
        return
    if args.cmd == "bench-choice":
        print(bench_choice(Path(args.report)))
        return
    if args.cmd == "forward":
        result = forward_parity(Path(args.ckpt), args.rows)
        checkpoint.write_json(args.report, result)
        print(json.dumps(result), flush=True)
        sys.exit(0 if result["ok"] else 1)
    if args.cmd == "smoke":
        result = smoke(Path(args.ckpt), args.rows, Path(args.work))
        checkpoint.write_json(args.report, result)
        print(
            json.dumps({k: result[k] for k in ("steps", "statuses", "ok")}), flush=True
        )
        sys.exit(0 if result["ok"] else 1)
    if args.cmd == "build":
        decision = build(
            Path(args.text), Path(args.vision), Path(args.out), args.max_length
        )
        print(
            json.dumps(
                {"out": args.out, "parameters": decision["provenance"]["parameters"]},
                indent=1,
            )
        )
        return
    report = check(
        Path(args.ckpt),
        Path(args.text),
        Path(args.vision),
        args.rows,
        args.n_rows,
        not args.no_engines,
    )
    checkpoint.write_json(args.report, report)
    print(
        json.dumps(
            {k: v.get("ok") for k, v in report["checks"].items()} | {"ok": report["ok"]}
        ),
        flush=True,
    )
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
