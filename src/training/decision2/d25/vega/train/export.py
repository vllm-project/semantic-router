"""Offline export and export-parity tools for the Vega trainer (single process).

``dcp``: turn a trainer DCP checkpoint into a "code-readout v1" export (when no export was
scheduled at that update):

    python -m d25.vega.train.export dcp --run-dir /data/d25/vega/ckpt/<run> --step 1200

``parity``: reload an export and compare it with the probabilities the in-memory FSDP model wrote
at export time (``parity_rows.jsonl``), through (a) the trainer's packed model code and (b) an
independent left-padded SDPA reference that follows Perplexity's released ``DecisionModel``
(unpatched transformers Qwen3_5Model, their noncausal mask hook, last-token pooling, FP32 readout):

    python -m d25.vega.train.export parity --ckpt <export dir> --dev <dev.jsonl> [--rows 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from d25.vega.common import decision_format as fmt
from d25.vega.train import data as D
from d25.vega.train import model as M


def dcp_to_export(args: argparse.Namespace) -> None:
    import torch.distributed.checkpoint as dcp

    run_dir = Path(args.run_dir)
    run_config = json.loads((run_dir / "run_config.json").read_text())
    targs = run_config["args"]
    path = run_dir / "dcp" / f"step-{args.step:06d}"
    trainer_state = json.loads((path / "trainer_state.json").read_text())
    init = targs["init"]
    tokenizer_dir = targs.get("tokenizer") or init
    tokenizer = M.load_tokenizer(tokenizer_dir)
    codes, token_ids = fmt.answer_codes(tokenizer)
    config = M.load_config(init)
    skeleton = M.build_skeleton(config, targs["attention_mode"], device="meta")
    state = {
        "model": {
            name: torch.empty(p.shape, dtype=torch.float32)
            for name, p in skeleton.named_parameters()
        }
    }
    began = time.time()
    dcp.load(state, checkpoint_id=str(path), no_dist=True)
    model_state = state["model"]
    text = {
        k[len("text.") :]: v for k, v in model_state.items() if k.startswith("text.")
    }
    readout = model_state["readout.weight"]
    visual = M.read_checkpoint(init, token_ids, parts=("visual",))["visual"]
    provenance = {
        "run": targs["run"],
        "step": args.step,
        "total_steps": run_config["total_steps"],
        "rows_seen": trainer_state["rows_seen"],
        "trainer": run_config["trainer"],
        "init": {"kind": targs["init_kind"], "path": init},
        "data_sha256": run_config["data_sha256"],
        "code_sha256": run_config["code_sha256"],
        "converted_from": str(path),
    }
    decision = M.decision_config(
        codes=codes,
        token_ids=token_ids,
        attention_mode=targs["attention_mode"],
        max_length=targs["max_length"],
        provenance=provenance,
    )
    destination = Path(args.out) if args.out else run_dir / f"step-{args.step:06d}"
    M.export_checkpoint(
        destination,
        config=config,
        text_state=text,
        visual_state=visual,
        readout=readout,
        tokenizer_source=tokenizer_dir,
        decision_config=decision,
    )
    print(json.dumps({"export": str(destination), "seconds": time.time() - began}))


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------


def reference_model(ckpt: Path, device: torch.device):
    """Perplexity-style reference: unpatched Qwen3_5Model (SDPA) + FP32 readout from the export."""
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    backbone = Qwen3_5Model.from_pretrained(
        str(ckpt), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    backbone.eval()
    readout = load_file(str(ckpt / "readout.safetensors"))["weight"].float().to(device)
    return backbone, readout


@torch.no_grad()
def reference_probs(
    backbone, readout, tokenizer, rows, attention_mode, codes, batch_size, device
):
    from transformers.masking_utils import create_recurrent_attention_mask

    text_model = backbone.language_model
    if attention_mode == "noncausal_full_attention":

        def hook(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                embeds = module.embed_tokens(kwargs["input_ids"])
                kwargs["inputs_embeds"] = embeds
                kwargs["input_ids"] = None
            padding = kwargs["attention_mask"]
            kwargs["attention_mask"] = {
                "full_attention": padding[:, None, None, :].bool(),
                "linear_attention": create_recurrent_attention_mask(
                    config=module.config, inputs_embeds=embeds, attention_mask=padding
                ),
            }
            return args, kwargs

        handle = text_model.register_forward_pre_hook(hook, with_kwargs=True)
    else:
        handle = None
    tokenizer.padding_side = "left"
    out = []
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            texts = [
                fmt.render(tokenizer, r["state"], r["question"], codes) for r in chunk
            ]
            enc = tokenizer(
                texts, padding=True, return_tensors="pt", add_special_tokens=False
            ).to(device)
            hidden = backbone(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=False,
            ).last_hidden_state[:, -1]
            logits = hidden.float() @ readout.T
            for i, r in enumerate(chunk):
                count = len(fmt.options(r["question"])[0])
                out.append(torch.softmax(logits[i, :count], dim=-1).cpu())
    finally:
        if handle is not None:
            handle.remove()
    return out


@torch.no_grad()
def packed_probs(
    ckpt: Path, rows, tokenizer, codes, token_ids, attention_mode, device, budget=16384
):
    config = M.load_config(ckpt)
    model = M.build_skeleton(config, attention_mode, device="meta")
    model.text.to(torch.bfloat16)
    model.to_empty(device=device)
    M.reset_rotary(model, device)
    weights = M.read_checkpoint(ckpt, token_ids, parts=("text", "readout"))
    model.text.load_state_dict(weights["text"], strict=True)
    model.readout.weight.copy_(weights["readout"].to(device))
    del weights
    model.eval()
    encoder = D.Encoder(
        tokenizer,
        codes,
        teacher=None,
        teacher_weight=0.0,
        shuffle_options=False,
        seed=0,
        max_length=10**9,
    )
    items = [encoder.encode(r, epoch=None, index=i) for i, r in enumerate(rows)]
    out = []
    chunk: list = []
    used = 0

    def flush():
        if not chunk:
            return
        inputs, _, _ = M.PackedBatch(
            [it.ids for it in chunk],
            [it.count for it in chunk],
            [it.target for it in chunk],
            [1.0] * len(chunk),
        ).to(device)
        probs = torch.softmax(model(inputs).float(), dim=-1).cpu()
        for j, it in enumerate(chunk):
            out.append(probs[j, : it.count])

    for it in items:
        if used + len(it.ids) > budget and chunk:
            flush()
            chunk, used = [], 0
        chunk.append(it)
        used += len(it.ids)
    flush()
    del model
    torch.cuda.empty_cache()
    return out


def compare(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict:
    diffs = [float((x - y).abs().max()) for x, y in zip(a, b)]
    agree = [int(x.argmax()) == int(y.argmax()) for x, y in zip(a, b)]
    return {
        "rows": len(diffs),
        "max_abs_dp": max(diffs),
        "mean_max_abs_dp": sum(diffs) / len(diffs),
        "argmax_agreement": sum(agree) / len(agree),
    }


def parity(args: argparse.Namespace) -> None:
    ckpt = Path(args.ckpt)
    device = torch.device("cuda")
    config = json.loads((ckpt / "decision_config.json").read_text())
    tokenizer = M.load_tokenizer(ckpt)
    codes, token_ids = fmt.answer_codes(tokenizer)
    if codes != config["codes"] or token_ids != config["token_ids"]:
        raise SystemExit("export codes differ from decision_format.answer_codes()")
    attention_mode = config["attention_mode"]
    store = D.RowStore(args.dev, args.cache_dir, build=True)
    rows_by_id = {}
    for i in range(len(store)):
        row = store.get(i)
        if row is not None:
            rows_by_id[str(row["id"])] = row
    recorded = []
    parity_file = ckpt / "parity_rows.jsonl"
    if parity_file.exists():
        recorded = [
            json.loads(line)
            for line in parity_file.read_text().splitlines()
            if line.strip()
        ]
    if recorded:
        ids = [r["id"] for r in recorded][: args.rows]
    else:
        ids = list(rows_by_id)[: args.rows]
    rows = [rows_by_id[i] for i in ids]
    result = {"ckpt": str(ckpt), "attention_mode": attention_mode, "rows": len(rows)}
    began = time.time()
    packed = packed_probs(
        ckpt, rows, tokenizer, codes, token_ids, attention_mode, device
    )
    result["packed_seconds"] = time.time() - began
    began = time.time()
    backbone, readout = reference_model(ckpt, device)
    padded = reference_probs(
        backbone,
        readout,
        tokenizer,
        rows,
        attention_mode,
        codes,
        args.batch_size,
        device,
    )
    single = reference_probs(
        backbone,
        readout,
        tokenizer,
        rows[: args.single_rows],
        attention_mode,
        codes,
        1,
        device,
    )
    result["reference_seconds"] = time.time() - began
    if recorded:
        memory = [torch.tensor(r["probs"]) for r in recorded[: len(rows)]]
        result["export_packed_vs_inmemory_fsdp"] = compare(packed, memory)
        result["export_padded_reference_vs_inmemory_fsdp"] = compare(padded, memory)
    result["packed_vs_padded_reference"] = compare(packed, padded)
    result["padded_batch_vs_single_row"] = compare(padded[: len(single)], single)
    if args.engine:
        try:
            from d25.vega.eval.engine import CodeReadoutModel

            engine = CodeReadoutModel(str(ckpt), device="cuda:0")
            eng = [
                torch.tensor(p)
                for p in engine.predict(
                    [{"state": r["state"], "question": r["question"]} for r in rows]
                )
            ]
            result["engine_vs_padded_reference"] = compare(eng, padded)
            result["engine_vs_packed"] = compare(eng, packed)
            if recorded:
                result["engine_vs_inmemory_fsdp"] = compare(eng, memory)
        except Exception as exc:  # noqa: BLE001
            result["engine_error"] = f"{type(exc).__name__}: {exc}"
    out = Path(args.out) if args.out else ckpt.parent / f"{ckpt.name}.parity.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dcp")
    d.add_argument("--run-dir", required=True)
    d.add_argument("--step", type=int, required=True)
    d.add_argument("--out")
    p = sub.add_parser("parity")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--single-rows", type=int, default=8)
    p.add_argument("--cache-dir", default="/data/d25/vega/train/cache")
    p.add_argument(
        "--engine", action="store_true", help="Also run ws-eval's d25.vega.eval.engine"
    )
    p.add_argument("--out")
    args = parser.parse_args()
    {"dcp": dcp_to_export, "parity": parity}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
