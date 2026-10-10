"""Does the image reach the language model? Image ablation + plumbing checks for a d3 package (run in a GPU lane).

    python -m d25.family.edge_diag <out.json> <suite dir> <n rows> <package> [<reference package> ...]

Per package: on n shuffled vision-suite rows, answers with the real images, gray images of the same size and no
images (per-option probability shifts, argmax agreement, accuracy vs expected); then, for a Qwen3-VL backbone, the
image token id (tokenizer vs config), image tokens in input_ids vs rows of image features, merger output width vs
the text hidden size, embedding norms (image features vs text token embeddings) and the image-token positions.
"""

import gzip
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def ablation(m, suite: Path, rows: list[dict]) -> dict:
    stats = {
        "real_vs_blank": [],
        "real_vs_none": [],
        "agree_blank": 0,
        "agree_none": 0,
        "n": 0,
    }
    correct = {"real": 0, "blank": 0, "none": 0}
    for r in rows:
        paths = [str(suite / p) for p in r["images"]]
        variants = {
            "real": paths,
            "blank": [
                Image.new("RGB", Image.open(p).convert("RGB").size, (128, 128, 128))
                for p in paths
            ],
            "none": [],
        }
        answers = {
            k: m.system_one(state=r["state"], questions=r["questions"], images=v)[
                "answers"
            ]
            for k, v in variants.items()
        }
        for q, want in r["expected"].items():
            got = {k: answers[k].get(q, {}) for k in variants}
            if any("probabilities" not in g for g in got.values()):
                continue
            keys = list(got["real"]["probabilities"])
            vec = {k: [got[k]["probabilities"][o] for o in keys] for k in variants}
            stats["real_vs_blank"].append(
                max(abs(a - b) for a, b in zip(vec["real"], vec["blank"]))
            )
            stats["real_vs_none"].append(
                max(abs(a - b) for a, b in zip(vec["real"], vec["none"]))
            )
            arg = {
                k: keys[max(range(len(keys)), key=vec[k].__getitem__)] for k in variants
            }
            stats["agree_blank"] += arg["real"] == arg["blank"]
            stats["agree_none"] += arg["real"] == arg["none"]
            stats["n"] += 1
            for k in variants:
                correct[k] += arg[k] == want
    n = max(stats["n"], 1)
    return {
        "questions": stats["n"],
        "max_dp_real_vs_blank": {
            "median": round(statistics.median(stats["real_vs_blank"]), 4),
            "mean": round(statistics.mean(stats["real_vs_blank"]), 4),
        },
        "max_dp_real_vs_none": {
            "median": round(statistics.median(stats["real_vs_none"]), 4),
            "mean": round(statistics.mean(stats["real_vs_none"]), 4),
        },
        "argmax_agreement": {
            "blank": round(stats["agree_blank"] / n, 3),
            "none": round(stats["agree_none"] / n, 3),
        },
        "accuracy": {k: round(v / n, 3) for k, v in correct.items()},
    }


def plumbing(m, image_path: Path) -> dict:
    model, proc = m.backbone, m.processor
    cfg = model.config
    tok = proc.tokenizer
    out = {
        "backbone": type(model).__name__,
        "image_token_id": {
            "config": cfg.image_token_id,
            "tokenizer": tok.convert_tokens_to_ids("<|image_pad|>"),
        },
        "vision_start": {
            "config": getattr(cfg, "vision_start_token_id", None),
            "tokenizer": tok.convert_tokens_to_ids("<|vision_start|>"),
        },
    }
    image = Image.open(image_path).convert("RGB")
    prepared = m.prepare(
        {},
        {"q": {"type": "noul", "instructions": "Is there a person in the image?"}},
        [image],
    )
    enc = proc(
        text=[prepared.texts["q"]], images=list(prepared.images), return_tensors="pt"
    )
    ids = enc["input_ids"][0]
    n_tokens = int((ids == cfg.image_token_id).sum())
    with torch.inference_mode():
        pv = enc["pixel_values"].to(
            m.device,
            model.visual.dtype if hasattr(model.visual, "dtype") else torch.bfloat16,
        )
        grid = enc["image_grid_thw"].to(m.device)
        feats = model.get_image_features(pv, grid)
        if isinstance(feats, (tuple, list)):
            image_embeds = feats[0]
        else:
            image_embeds = getattr(feats, "pooler_output", None)
            if image_embeds is None:
                image_embeds = feats.last_hidden_state
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), 0)
        emb = model.get_input_embeddings().weight
        text_norm = emb.float().norm(dim=-1)
        img_norm = image_embeds.float().norm(dim=-1)
        out.update(
            image_tokens_in_ids=n_tokens,
            image_feature_rows=int(image_embeds.shape[0]),
            image_feature_width=int(image_embeds.shape[-1]),
            text_hidden=int(cfg.text_config.hidden_size),
            norms={
                "text_embedding_median": round(float(text_norm.median()), 3),
                "image_feature_median": round(float(img_norm.median()), 3),
                "image_feature_p90": round(float(img_norm.quantile(0.9)), 3),
            },
        )
        if hasattr(model, "get_rope_index"):
            mm = enc.get("mm_token_type_ids")
            if mm is None:
                mm = (enc["input_ids"] == cfg.image_token_id).int()
            pos, _ = model.get_rope_index(
                enc["input_ids"].to(m.device),
                mm.to(m.device),
                grid,
                None,
                enc["attention_mask"].to(m.device),
            )
            out["processor_outputs"] = sorted(enc.keys())
            span = (ids == cfg.image_token_id).nonzero().flatten()
            first, last = int(span[0]), int(span[-1])
            p = pos[:, 0].tolist()
            out["positions"] = {
                "image_span": [first, last],
                "t_h_w_at_first_image_token": [p[d][first] for d in range(3)],
                "t_h_w_at_last_image_token": [p[d][last] for d in range(3)],
                "t_h_w_at_last_token": [p[d][-1] for d in range(3)],
                "sequence_length": len(ids),
            }
    return out


def main() -> None:
    out, suite, n, packages = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4:],
    )
    rows = [json.loads(line) for line in gzip.open(suite / "rows.jsonl.gz", "rt")]
    random.Random(20261011).shuffle(rows)
    rows = rows[:n]
    report = {"rows": n, "suite": suite.name, "packages": {}}
    for package in packages:
        sys.path.insert(0, package)
        from d3_runtime import D3  # noqa: E402

        started = time.time()
        m = D3.from_pretrained(package, device="cuda:0", verify="none")
        entry = {"ablation": ablation(m, suite, rows)}
        if getattr(m, "backbone_type", "qwen3_5") == "qwen3_vl":
            try:
                entry["plumbing"] = plumbing(m, suite / rows[0]["images"][0])
            except Exception as exc:  # noqa: BLE001 - keep the ablation result
                entry["plumbing"] = {"error": f"{type(exc).__name__}: {exc}"}
        entry["seconds"] = round(time.time() - started, 1)
        report["packages"][Path(package).parent.name + "/" + Path(package).name] = entry
        print(json.dumps({package: entry}), flush=True)
        del m
        torch.cuda.empty_cache()
        sys.path.pop(0)
        sys.modules.pop("d3_runtime", None)
        sys.modules.pop("d3_format", None)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
