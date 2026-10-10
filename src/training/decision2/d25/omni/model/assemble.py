"""Build a Decision 2.5 Omni init checkpoint in the code-readout v1 layout.

Sources:

- ``--vega DIR``: a Decision 2.5 Vega code-readout v1 checkpoint. Its language weights and readout
  are copied bit for bit. Its vision tower is kept only when every encoder and merger tensor is
  bit-equal to stock Qwen3.8-27B; when any is missing or differs, the stock tower is re-attached.
- no ``--vega``: stock Qwen3.8-27B, readout initialised from the ``lm_head`` rows of the codes.

``lm_head`` and ``mtp.*`` are dropped. After writing, every output tensor is re-read and checked
bit-equal to its source (vision: stock), and the key set must equal a ``Qwen3_5Model``'s. The
provenance (init kind, source file sha256, vision status and digest, output shard sha256) goes into
``decision_config.json``.

    python -m d25.omni.model.assemble --stock STOCK_DIR --vega VEGA_CKPT --out OUT
    python -m d25.omni.model.assemble --stock STOCK_DIR --out OUT --attention-mode noncausal_full_attention

``STOCK_DIR`` is a local snapshot of Qwen/Qwen3.8-27B at the pinned revision (default: the local
Hugging Face cache); its shards are checked against the pinned sha256 unless ``--stock-pin skip``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from d25.omni.common import vision_format
from d25.omni.model import checkpoint, inputs
from d25.omni.model.attention import ATTENTION_MODES
from d25.vega.common import decision_format as text_format

DEFAULT_MAX_LENGTH = 16_384
STRUCTURE_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "vocab_size",
)


def resolve_stock(path: str | None) -> Path:
    if path:
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            checkpoint.BASE_MODEL,
            revision=checkpoint.BASE_REVISION,
            local_files_only=True,
        )
    )


def canonical_map(directory: Path) -> dict[str, str]:
    """Canonical name -> stored name for every kept tensor of a checkpoint."""
    names: dict[str, str] = {}
    for key in checkpoint.weight_files(directory):
        name = checkpoint.canonical_name(key)
        if name is None:
            continue
        if name in names:
            raise ValueError(
                f"{directory}: {key} and {names[name]} map to the same tensor {name}"
            )
        names[name] = key
    return names


def verify_stock_pin(stock: Path, shards: set[str]) -> dict[str, str]:
    """Check the read shards and the tokenizer/config files against the pinned revision."""
    pinned = {**checkpoint.STOCK_SHARDS_SHA256, **checkpoint.STOCK_FILES_SHA256}
    hashes: dict[str, str] = {}
    for name in sorted(shards | set(checkpoint.STOCK_FILES_SHA256)):
        if name not in pinned:
            raise ValueError(
                f"{name} is not a file of {checkpoint.BASE_MODEL}@{checkpoint.BASE_REVISION}"
            )
        actual = checkpoint.file_sha256(stock / name)
        if actual != pinned[name]:
            raise ValueError(
                f"{stock / name}: sha256 {actual} differs from the pinned {pinned[name]}"
            )
        hashes[name] = actual
    for name, expected in checkpoint.STOCK_FILES_GIT_OID.items():
        actual = checkpoint.git_blob_oid(stock / name)
        if actual != expected:
            raise ValueError(
                f"{stock / name}: git blob {actual} differs from the pinned {expected}"
            )
        hashes[name] = checkpoint.file_sha256(stock / name)
    return hashes


def check_structure(
    stock_config: dict[str, Any], other: dict[str, Any], label: str
) -> None:
    stock_text, other_text = (
        stock_config["text_config"],
        other.get("text_config") or other,
    )
    for key in STRUCTURE_KEYS:
        if key in stock_text and stock_text[key] != other_text.get(key):
            raise ValueError(
                f"{label}: text_config.{key} differs from the stock config"
            )


def stock_readout(
    stock: Path, stock_names: dict[str, str], token_ids: list[int], tied: bool
) -> torch.Tensor:
    if "lm_head.weight" in checkpoint.weight_files(stock):
        key = "lm_head.weight"
    elif tied:
        key = stock_names["language_model.embed_tokens.weight"]
    else:
        raise ValueError(f"{stock}: no lm_head.weight to initialise the readout from")
    ((_, head),) = checkpoint.iter_tensors(stock, {key})
    return head[token_ids].float()


def assemble(
    stock: str | Path,
    out: str | Path,
    vega: str | Path | None = None,
    *,
    attention_mode: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    stock_pin: str = "verify",
    max_shard_bytes: int = 5 * 1024**3,
) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3_5Config

    stock, out = Path(stock), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = Qwen3_5Config.from_pretrained(str(stock))
    stock_config = json.loads((stock / "config.json").read_text())
    stock_names = canonical_map(stock)
    stock_files = checkpoint.weight_files(stock)
    vision_names = sorted(
        n for n in stock_names if checkpoint.component(n).startswith("vision")
    )
    language_names = sorted(
        n for n in stock_names if checkpoint.component(n) == "language"
    )
    if not vision_names:
        raise ValueError(
            f"{stock}: no visual.* tensors; not a Qwen3.5 vision-language checkpoint"
        )

    processor = inputs.setup_processor(AutoProcessor.from_pretrained(str(stock)))
    codes, token_ids = text_format.answer_codes(processor.tokenizer)

    vision_files = {stock_files[stock_names[n]] for n in vision_names}
    pinned_files = set(stock_files.values()) if vega is None else vision_files
    if stock_pin == "verify":
        if vega is None and "lm_head.weight" in stock_files:
            pinned_files.add(stock_files["lm_head.weight"])
        stock_hashes, pin_verified = verify_stock_pin(stock, pinned_files), True
    elif stock_pin == "skip":
        stock_hashes = {
            name: checkpoint.file_sha256(stock / name) for name in sorted(pinned_files)
        }
        pin_verified = False
    else:
        raise ValueError("stock_pin must be verify or skip")

    stock_vision = {
        checkpoint.canonical_name(k): t
        for k, t in checkpoint.iter_tensors(
            stock, {stock_names[n] for n in vision_names}
        )
    }
    stock_vision_digests = {
        name: checkpoint.tensor_digest(t) for name, t in stock_vision.items()
    }

    if vega is None:
        source, source_names = stock, {n: stock_names[n] for n in language_names}
        readout = stock_readout(
            stock, stock_names, token_ids, bool(config.tie_word_embeddings)
        )
        decision: dict[str, Any] = {
            "format_version": 1,
            "prompt": "d25-vega",
            "base_model": checkpoint.BASE_MODEL,
            "revision": checkpoint.BASE_REVISION,
            "codes": codes,
            "token_ids": token_ids,
            "temperature": 1.0,
            "attention_mode": attention_mode or "causal",
            "pooling": "last",
        }
        init = {
            "kind": "stock",
            "path": str(stock),
            "files_sha256": stock_hashes,
            "pin_verified": pin_verified,
        }
        vision_status: dict[str, Any] = {"status": "stock"}
    else:
        source = Path(vega)
        decision = checkpoint.read_decision_config(source)
        if (decision.get("base_model"), decision.get("revision")) != (
            checkpoint.BASE_MODEL,
            checkpoint.BASE_REVISION,
        ):
            raise ValueError(
                f"{source}: base model/revision differs from {checkpoint.BASE_MODEL}@{checkpoint.BASE_REVISION}"
            )
        if decision.get("prompt", "d25-vega") != "d25-vega":
            raise ValueError(
                f"{source}: Omni builds on the d25-vega prompt, got {decision.get('prompt')!r}"
            )
        inputs.check_codes(
            processor.tokenizer, decision["codes"], decision["token_ids"]
        )
        if attention_mode and attention_mode != decision.get(
            "attention_mode", "causal"
        ):
            raise ValueError(
                "the attention mode of a Vega init is fixed by its decision_config"
            )
        check_structure(
            stock_config, json.loads((source / "config.json").read_text()), str(source)
        )
        vega_names = canonical_map(source)
        extra = sorted(
            n
            for n in vega_names
            if checkpoint.component(n).startswith("vision") and n not in stock_vision
        )
        if extra:
            raise ValueError(
                f"{source}: vision tensors unknown to the stock tower: {extra[:5]}"
            )
        missing_language = sorted(set(language_names) - set(vega_names))
        if missing_language:
            raise ValueError(
                f"{source}: missing language tensors {missing_language[:5]}"
            )
        source_names = {n: vega_names[n] for n in language_names}
        changed: list[str] = []
        present = {vega_names[n]: n for n in vision_names if n in vega_names}
        for key, tensor in checkpoint.iter_tensors(source, set(present)):
            name = present[key]
            if checkpoint.tensor_digest(tensor) != stock_vision_digests[name]:
                changed.append(name)
        missing = [n for n in vision_names if n not in vega_names]
        vision_status = {
            "status": "kept" if not missing and not changed else "reattached",
            "missing": len(missing),
            "changed": sorted(changed),
        }
        if (source / checkpoint.READOUT_FILE).exists():
            readout = checkpoint.load_readout(source)
        elif "readout.weight" in vega_names:
            ((_, readout),) = checkpoint.iter_tensors(
                source, {vega_names["readout.weight"]}
            )
        else:
            raise ValueError(f"{source}: no readout")
        source_files = sorted(
            {checkpoint.weight_files(source)[k] for k in vega_names.values()}
            | {
                name
                for name in (
                    checkpoint.READOUT_FILE,
                    checkpoint.DECISION_CONFIG,
                    "config.json",
                    checkpoint.INDEX_FILE,
                )
                if (source / name).exists()
            }
        )
        init = {
            "kind": "vega",
            "path": str(source),
            "files_sha256": {
                name: checkpoint.file_sha256(source / name) for name in source_files
            },
            "decision_config": {
                k: v
                for k, v in decision.items()
                if k not in ("codes", "token_ids", "provenance")
            },
            "stock_files_sha256": stock_hashes,
            "stock_pin_verified": pin_verified,
        }

    if readout.shape != (checkpoint.NUM_CODES, config.text_config.hidden_size):
        raise ValueError(
            f"readout shape {tuple(readout.shape)} does not match the backbone"
        )

    expected = checkpoint.expected_backbone_shapes(config)
    writer = checkpoint.ShardWriter(out, max_shard_bytes)
    source_digests: dict[str, str] = {}
    by_key = {key: name for name, key in source_names.items()}
    for key, tensor in checkpoint.iter_tensors(source, set(by_key)):
        name = by_key[key]
        if tuple(tensor.shape) != expected.get(name):
            raise ValueError(
                f"{source}: {key} has shape {tuple(tensor.shape)}, expected {expected.get(name)}"
            )
        writer.add(name, tensor)
        source_digests[name] = checkpoint.tensor_digest(tensor)
    for name in vision_names:
        writer.add(name, stock_vision[name])
        source_digests[name] = stock_vision_digests[name]
    shard_hashes = writer.close()

    written = set(writer.digests)
    if written != set(expected):
        raise ValueError(
            f"output tensors differ from Qwen3_5Model: missing {sorted(set(expected) - written)[:5]}, "
            f"unexpected {sorted(written - set(expected))[:5]}"
        )
    reread = {
        key: checkpoint.tensor_digest(t) for key, t in checkpoint.iter_tensors(out)
    }
    mismatched = sorted(
        name for name, digest in reread.items() if source_digests.get(name) != digest
    )
    if mismatched or set(reread) != set(source_digests):
        raise ValueError(
            f"written tensors are not bit-equal to their sources: {mismatched[:5]}"
        )
    vision_parameters = sum(stock_vision[name].numel() for name in vision_names)

    config.architectures = ["Qwen3_5Model"]
    config.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    readout_sha256 = checkpoint.save_readout(out, readout.float())

    decision.update(
        {
            "format_id": vision_format.FORMAT_ID,
            "modalities": ["text", "image"],
            "images": {
                "max_images": vision_format.MAX_IMAGES,
                "min_pixels": vision_format.MIN_PIXELS,
                "max_pixels": vision_format.MAX_PIXELS,
                "order": "before_text",
            },
            "max_length": max_length,
        }
    )
    if decision["attention_mode"] not in ATTENTION_MODES:
        raise ValueError(f"unknown attention mode {decision['attention_mode']!r}")
    vision_status.update(
        {
            "source": f"{checkpoint.BASE_MODEL}@{checkpoint.BASE_REVISION}",
            "tensors": len(vision_names),
            "parameters": vision_parameters,
            "digest": checkpoint.digest_of_digests(stock_vision_digests),
            "bit_equal_to_stock": True,
        }
    )
    previous = decision.get("provenance")
    decision["provenance"] = {
        "assembled": {
            "tool": "d25.omni.model.assemble",
            "code_sha256": checkpoint.file_sha256(__file__),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - started, 1),
        },
        "init": init,
        "init_provenance": previous,
        "vision": vision_status,
        "readout": {"source": "vega" if vega else "lm_head", "sha256": readout_sha256},
        "shards_sha256": shard_hashes,
    }
    checkpoint.write_json(out / checkpoint.DECISION_CONFIG, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stock", help="local Qwen/Qwen3.8-27B snapshot (default: Hugging Face cache)"
    )
    parser.add_argument(
        "--vega",
        help="Decision 2.5 Vega code-readout v1 checkpoint; omit for a stock init",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--attention-mode", choices=ATTENTION_MODES, help="stock init only"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stock-pin", choices=("verify", "skip"), default="verify")
    parser.add_argument("--max-shard-gb", type=float, default=5.0)
    args = parser.parse_args()
    decision = assemble(
        resolve_stock(args.stock),
        args.out,
        args.vega,
        attention_mode=args.attention_mode,
        max_length=args.max_length,
        stock_pin=args.stock_pin,
        max_shard_bytes=int(args.max_shard_gb * 1024**3),
    )
    provenance = decision["provenance"]
    print(
        json.dumps(
            {
                "out": args.out,
                "init": provenance["init"]["kind"],
                "vision": provenance["vision"],
                "shards_sha256": provenance["shards_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
