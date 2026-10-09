"""Environment probe for the Vega trainer (run in the training image on a GPU node).

Single process (1 GPU): versions; transformers resolves FLA for Qwen3.5; FLA varlen gated-delta and
causal-conv kernels match a per-row reference in forward and backward (and autograd goes through
FLA's Function); flash-attention varlen (head dim 256, GQA) matches SDPA; a tiny random Qwen3.5
packed through ``DecisionReadout`` matches the left-padded HF/SDPA reference (causal and
noncausal, Perplexity's mask hook); tokenizer and answer codes.

Under ``d25.vega.train.launch --nproc N``: FSDP2 mechanics on the tiny model (fused AdamW on
DTensors, clipping, DCP save/load round trip, full-tensor gather).

    python -m d25.vega.train.probe_env --base /models/Qwen3.8-27B --out /data/d25/vega/train/probe.json
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

RESULTS: dict = {"checks": {}}


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            began = time.time()
            try:
                value = fn(*args, **kwargs)
                RESULTS["checks"][name] = {
                    "ok": True,
                    "seconds": round(time.time() - began, 2),
                    **(value or {}),
                }
            except Exception as exc:  # noqa: BLE001
                RESULTS["checks"][name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-3000:],
                }
            print(name, json.dumps(RESULTS["checks"][name])[:2000], flush=True)

        return run

    return wrap


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


@check("versions")
def versions():
    out = {}
    for name in (
        "torch",
        "triton",
        "transformers",
        "flash_attn",
        "fla",
        "einops",
        "causal_conv1d",
        "kernels",
        "safetensors",
        "tokenizers",
        "numpy",
    ):
        try:
            module = importlib.import_module(name)
            out[name] = (
                f"{getattr(module, '__version__', '?')} @ {os.path.dirname(getattr(module, '__file__', '') or '')}"
            )
        except Exception as exc:  # noqa: BLE001
            out[name] = f"MISSING ({type(exc).__name__})"
    out["hip"] = torch.version.hip
    out["device"] = torch.cuda.get_device_name(0)
    out["gpus"] = torch.cuda.device_count()
    out["python"] = sys.version.split()[0]
    return out


@check("transformers_resolves_fla")
def hf_resolution():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    out = {}
    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        fn = getattr(mq, name)
        impl = None
        for cell in fn.__closure__ or ():
            value = cell.cell_contents
            if (
                callable(value)
                and getattr(value, "__module__", "")
                and value is not getattr(fn, "__wrapped__", None)
            ):
                if getattr(value, "__name__", "") not in (
                    "wrapped",
                ) and not isinstance(value, type):
                    impl = value
        out[name] = (
            f"{getattr(impl, '__module__', None)}.{getattr(impl, '__name__', None)}"
        )
    if not out["torch_chunk_gated_delta_rule"].startswith("fla."):
        raise RuntimeError(f"transformers does not resolve FLA: {out}")
    return out


def reference_gdr(q, k, v, g, beta, bounds):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    torch_fn = mq.torch_chunk_gated_delta_rule.__wrapped__
    outs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        o, _ = torch_fn(
            q[:, s:e],
            k[:, s:e],
            v[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(o)
    return torch.cat(outs, dim=1)


@check("fla_gated_delta_varlen_fwd_bwd")
def fla_gdr():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    bounds = [0, 300, 517, 1200, 1201 + 63]
    T, H, K, V = bounds[-1], 6, 128, 128
    dev = "cuda"
    q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
    g = (-F.softplus(torch.randn(1, T, H, device=dev))).requires_grad_(True)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16).requires_grad_(True)
    cu = torch.tensor(bounds, device=dev, dtype=torch.long)
    o, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu.cpu(),
    )
    grad_fn = type(o.grad_fn).__name__
    go = torch.randn_like(o)
    grads = torch.autograd.grad(o, (q, k, v, g, beta), go)
    ref = reference_gdr(q, k, v, g, beta, bounds)
    ref_grads = torch.autograd.grad(ref, (q, k, v, g, beta), go)
    scale = float(ref.float().abs().max())
    out = {"grad_fn": grad_fn, "fwd_maxdiff": maxdiff(o, ref), "fwd_scale": scale}
    for name, a, b in zip("qkvgb", grads, ref_grads):
        out[f"grad_{name}_maxdiff"] = maxdiff(a, b)
        out[f"grad_{name}_scale"] = float(b.float().abs().max())
    if "ChunkGatedDeltaRule" not in grad_fn:
        raise RuntimeError(f"autograd does not go through FLA: {grad_fn}")
    if out["fwd_maxdiff"] > 0.05 * max(scale, 1):
        raise RuntimeError("FLA varlen output differs from the reference")
    return out


@check("fla_causal_conv1d_varlen_fwd_bwd")
def fla_conv():
    from fla.modules.convolution import causal_conv1d

    torch.manual_seed(1)
    bounds = [0, 5, 300, 1027, 2048]
    T, C, W = bounds[-1], 1024, 4
    x = torch.randn(1, T, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(C, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor(bounds, device="cuda", dtype=torch.long)
    y, _ = causal_conv1d(
        x,
        w,
        None,
        activation="silu",
        backend="triton",
        cu_seqlens=cu,
        cu_seqlens_cpu=cu.cpu(),
    )
    grad_fn = type(y.grad_fn).__name__
    gy = torch.randn_like(y)
    gx, gw = torch.autograd.grad(y, (x, w), gy)
    refs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = x[:, s:e].transpose(1, 2)
        r = F.conv1d(
            seg.float(), w.float().unsqueeze(1), None, padding=W - 1, groups=C
        )[..., : e - s]
        refs.append(F.silu(r).transpose(1, 2))
    ref = torch.cat(refs, dim=1)
    rgx, rgw = torch.autograd.grad(ref, (x, w), gy.float())
    from d25.vega.train import model as M

    mine = M._conv_torch(x, w, cu, bounds)
    y_full, _ = causal_conv1d(x, w, None, activation="silu", backend="triton")
    ref_full = F.silu(
        F.conv1d(
            x.transpose(1, 2).float(),
            w.float().unsqueeze(1),
            None,
            padding=W - 1,
            groups=C,
        )[..., :T]
    ).transpose(1, 2)
    flipped = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = x[:, s:e].transpose(1, 2)
        r = F.conv1d(
            seg.float(), w.float().flip(-1).unsqueeze(1), None, padding=W - 1, groups=C
        )[..., : e - s]
        flipped.append(F.silu(r).transpose(1, 2))
    flipped = torch.cat(flipped, dim=1)
    per_row = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        per_row.append(maxdiff(y[:, s:e], ref[:, s:e]))
    out = {
        "grad_fn": grad_fn,
        "scale": float(ref.abs().max()),
        "fwd_maxdiff": maxdiff(y, ref),
        "fwd_maxdiff_per_row": per_row,
        "fwd_maxdiff_vs_flipped_taps": maxdiff(y, flipped),
        "nonvarlen_maxdiff": maxdiff(y_full, ref_full),
        "gx_maxdiff": maxdiff(gx, rgx),
        "gw_maxdiff": maxdiff(gw, rgw),
        "gw_scale": float(rgw.abs().max()),
        "torch_impl_maxdiff": maxdiff(mine, ref),
    }
    RESULTS["checks"]["fla_causal_conv1d_varlen_detail"] = out
    if out["fwd_maxdiff"] > 0.02 * max(1.0, out["scale"]) or out[
        "torch_impl_maxdiff"
    ] > 0.02 * max(1.0, out["scale"]):
        raise RuntimeError(f"varlen conv differs from the per-row reference: {out}")
    return out


@check("flash_attn_varlen_hdim256")
def flash_varlen():
    from flash_attn import flash_attn_varlen_func

    torch.manual_seed(2)
    bounds = [0, 333, 1024, 1500, 3000]
    T, Hq, Hk, D = bounds[-1], 24, 4, 256
    q = torch.randn(T, Hq, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, Hk, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, Hk, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    out = {}
    for causal in (True, False):
        o = flash_attn_varlen_func(
            q, k, v, cu, cu, 1500, 1500, softmax_scale=D**-0.5, causal=causal
        )
        go = torch.randn_like(o)
        g = torch.autograd.grad(o, (q, k, v), go)
        refs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            qi, ki, vi = (t[s:e].transpose(0, 1)[None].float() for t in (q, k, v))
            ri = F.scaled_dot_product_attention(
                qi, ki, vi, is_causal=causal, scale=D**-0.5, enable_gqa=True
            )
            refs.append(ri[0].transpose(0, 1))
        ref = torch.cat(refs)
        rg = torch.autograd.grad(ref, (q, k, v), go.float())
        tag = "causal" if causal else "noncausal"
        out[f"{tag}_fwd_rel"] = float((o.float() - ref).norm() / ref.norm())
        for name, a, b in zip("qkv", g, rg):
            out[f"{tag}_d{name}_rel"] = float((a.float() - b).norm() / b.norm())
    if max(out.values()) > 0.01:
        raise RuntimeError(f"flash varlen differs: {out}")
    return out


def tiny_config(base: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base)
    t = config.text_config
    t.hidden_size = 512
    t.intermediate_size = 1024
    t.num_hidden_layers = 4
    t.layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    t.num_attention_heads = 4
    t.num_key_value_heads = 2
    t.linear_num_key_heads = 2
    t.linear_num_value_heads = 6
    return config


def reference_hidden(ref_model, rows, attention_mode):
    """Left-padded SDPA forward as Perplexity's DecisionModel runs it; returns last-token hidden."""
    from transformers.masking_utils import create_recurrent_attention_mask

    length = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), length, dtype=torch.long, device="cuda")
    mask = torch.zeros(len(rows), length, dtype=torch.long, device="cuda")
    for i, r in enumerate(rows):
        ids[i, length - len(r) :] = torch.tensor(r, device="cuda")
        mask[i, length - len(r) :] = 1
    if attention_mode == "noncausal_full_attention":
        embeds = ref_model.embed_tokens(ids)
        masks = {
            "full_attention": mask[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=ref_model.config, inputs_embeds=embeds, attention_mask=mask
            ),
        }
        out = ref_model(inputs_embeds=embeds, attention_mask=masks, use_cache=False)
    else:
        out = ref_model(input_ids=ids, attention_mask=mask, use_cache=False)
    return out.last_hidden_state[:, -1]


@check("tiny_packed_vs_padded")
def tiny_equivalence(base: str):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from d25.vega.train import model as M

    torch.manual_seed(3)
    config = tiny_config(base)
    out = {}
    rows = [
        list(torch.randint(0, 5000, (n,)).tolist()) for n in (37, 200, 64, 513, 130)
    ]
    for mode, conv in (
        ("causal", "fla"),
        ("noncausal_full_attention", "fla"),
        ("causal", "torch"),
    ):
        M.CONV_BACKEND["name"] = conv
        mode_key = f"{mode}/{conv}"
        ref_cfg = copy.deepcopy(config.text_config)
        ref_cfg._attn_implementation = "sdpa"
        ref = Qwen3_5TextModel(ref_cfg).to("cuda", torch.bfloat16).eval()
        mine = M.build_skeleton(config, mode, device="cuda")
        mine.text.load_state_dict(ref.state_dict())
        mine.text.to(torch.bfloat16)
        mine.eval()
        batch = M.PackedBatch(
            rows, [4] * len(rows), [[0.25] * 4] * len(rows), [1.0] * len(rows)
        )
        inputs, _, _ = batch.to(torch.device("cuda"))
        with torch.no_grad():
            packed = mine.text(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask={"full_attention": None, "linear_attention": None},
                use_cache=False,
                cu_seq_lens_q=inputs["cu_seq_lens_q"],
                cu_seq_lens_k=inputs["cu_seq_lens_k"],
                max_length_q=inputs["max_length_q"],
                max_length_k=inputs["max_length_k"],
                fla_cu_seqlens=inputs["fla_cu_seqlens"],
                fla_cu_seqlens_cpu=inputs["fla_cu_seqlens_cpu"],
                cu_seqlens_list=inputs["cu_seqlens_list"],
            ).last_hidden_state[0, inputs["last"]]
            if mode == "noncausal_full_attention":
                for module in ref.modules():
                    if hasattr(module, "is_causal"):
                        module.is_causal = False
            padded = reference_hidden(ref, rows, mode)
            single = torch.stack([reference_hidden(ref, [r], mode)[0] for r in rows])
        rel = float((packed.float() - padded.float()).norm() / padded.float().norm())
        rel_single = float(
            (padded.float() - single.float()).norm() / single.float().norm()
        )
        rel_packed_single = float(
            (packed.float() - single.float()).norm() / single.float().norm()
        )
        cos = float(F.cosine_similarity(packed.float(), padded.float(), dim=-1).min())
        out[mode_key] = {
            "rel_packed_vs_padded": rel,
            "rel_padded_vs_single": rel_single,
            "rel_packed_vs_single": rel_packed_single,
            "min_cos": cos,
        }
        # gradient path through the packed model
        mine.train()
        inputs, target, w = batch.to(torch.device("cuda"))
        logits = mine(inputs)
        loss = (M.row_losses(logits, target)["ce"] * w).mean()
        loss.backward()
        norms = [
            p.grad.float().norm().item()
            for p in mine.parameters()
            if p.grad is not None
        ]
        out[mode_key]["grad_params"] = len(norms)
        out[mode_key]["grad_finite"] = all(
            map(lambda x: x == x and x != float("inf"), norms)
        )
    M.CONV_BACKEND["name"] = "fla"
    RESULTS["checks"]["tiny_packed_vs_padded_detail"] = out
    if any(
        v["rel_packed_vs_padded"] > 0.03 or not v["grad_finite"] for v in out.values()
    ):
        raise RuntimeError(f"packed forward differs from padded reference: {out}")
    return out


@check("tokenizer_and_codes")
def tokenizer_check(base: str):
    from d25.vega.common import decision_format as fmt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    codes, ids = fmt.answer_codes(tok)
    question = {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {"a": "first", "b": None, "c": {"x": 1}},
    }
    text = fmt.render(tok, {"k": "v"}, question, codes)
    plain = tok(text, add_special_tokens=False)["input_ids"]
    default = tok(text)["input_ids"]
    return {
        "codes_head": codes[:30],
        "codes_tail": codes[-5:],
        "token_ids_head": ids[:8],
        "n_codes": len(codes),
        "special_tokens_added_by_default": default != plain,
        "prompt_tokens": len(plain),
        "prompt_tail": repr(text[-60:]),
        "pad_token_id": tok.pad_token_id,
        "eos": tok.eos_token,
        "tokenizer_class": type(tok).__name__,
    }


def fsdp_mechanics(base: str, out_path: Path) -> None:
    """Tiny-model FSDP2 + fused AdamW + DCP round trip under the launcher."""
    import torch.distributed as dist

    from d25.vega.train import model as M
    from d25.vega.train import train as T

    args = T.parse_args(
        [
            "--run",
            "probe",
            "--train",
            "x",
            "--output",
            str(out_path.parent / "probe-fsdp"),
            "--init",
            base,
            "--save-every",
            "0",
        ]
    )
    rt = T.Runtime(10)
    result = {"world": rt.world}
    try:
        config = tiny_config(base)
        model = M.build_skeleton(config, "noncausal_full_attention", device="meta")
        T.configure_ac(model.text, "every:2")
        T.shard_model(model, rt, args)
        model.to_empty(device=rt.device)
        M.reset_rotary(model, rt.device)
        torch.manual_seed(0)
        full = None
        if rt.main:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

            ref = Qwen3_5TextModel(copy.deepcopy(config.text_config))
            full = {
                f"text.{k}": v.to(torch.bfloat16) for k, v in ref.state_dict().items()
            }
            full["readout.weight"] = (
                torch.randn(255, config.text_config.hidden_size) * 0.02
            )
        T.load_initial_weights(model, rt, full)
        emb = model.text.embed_tokens.weight.full_tensor()
        if rt.main:
            result["load_exact"] = bool(
                torch.equal(
                    emb.cpu().to(torch.bfloat16), full["text.embed_tokens.weight"]
                )
            )
        opt = T.build_optimizer(model, args)
        result["optimizer"] = "fused" if opt.defaults.get("fused") else "foreach"
        rows = [list(range(10 + 7 * rt.rank, 10 + 7 * rt.rank + n)) for n in (100, 250)]
        batch = M.PackedBatch(rows, [3, 5], [[1, 0, 0], [0, 0, 0, 1, 0]], [1.0, 1.0])
        losses = []
        for _ in range(3):
            inputs, target, w = batch.to(rt.device)
            opt.zero_grad(set_to_none=True)
            loss = (M.row_losses(model(inputs), target)["ce"] * w).sum()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters()], 1.0, foreach=True
            )
            norm = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
            opt.step()
            losses.append(float(loss))
        result["losses_rank_view"] = losses
        result["grad_norm"] = norm
        root = out_path.parent / "probe-fsdp" / "dcp"
        T.save_dcp(
            model,
            opt,
            rt,
            root,
            3,
            {"step": 3, "rows_seen": 6},
            1,
            T.Logger(root.parent / "logs", rt.main),
        )
        before = model.text.layers[0].mlp.up_proj.weight.full_tensor().clone()
        with torch.no_grad():
            for p in model.parameters():
                p.to_local().zero_()
        state = T.load_dcp(model, opt, rt, root / "step-000003")
        after = model.text.layers[0].mlp.up_proj.weight.full_tensor()
        result["dcp_roundtrip_exact"] = bool(torch.equal(before, after))
        result["dcp_state_step"] = state["step"]
        full_state = T.gather_full_state(model, rt)
        if rt.main:
            result["gathered_params"] = len(full_state)
            result["gathered_readout_dtype"] = str(full_state["readout.weight"].dtype)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-4000:]
    gathered = [None] * rt.world
    dist.all_gather_object(gathered, result)
    if rt.main:
        path = out_path.with_name(out_path.stem + "-fsdp.json")
        path.write_text(json.dumps(gathered, indent=2))
        print(json.dumps(gathered, indent=2)[:6000], flush=True)
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        fsdp_mechanics(args.base, out)
        return 0
    versions()
    hf_resolution()
    fla_gdr()
    fla_conv()
    flash_varlen()
    tiny_equivalence(args.base)
    tokenizer_check(args.base)
    RESULTS["ok"] = all(c.get("ok", True) for c in RESULTS["checks"].values())
    out.write_text(json.dumps(RESULTS, indent=2))
    print("PROBE", "OK" if RESULTS["ok"] else "FAILED", flush=True)
    return 0 if RESULTS["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
