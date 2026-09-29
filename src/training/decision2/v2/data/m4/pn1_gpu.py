"""PN1 GPU jobs on one GPU (prereg ``v2/data/records/m4-pn1-prereg-2026-09-29.md`` sec. 2).

    python3 -m v2.data.m4.pn1_gpu generate --model-dir SNAPSHOT --repo-id Qwen/Qwen3.5-27B \
        --revision REV --seeds WORK/seeds.jsonl --languages ja,zh --output-dir DIR \
        --budget-seconds S [--expect-pci-bus 9b]
    python3 -m v2.data.m4.pn1_gpu judge --model-dir SNAPSHOT --repo-id Qwen/Qwen3.8-27B \
        --revision REV --items WORK/judgeset/items.gpu3.jsonl --output-dir DIR --budget-seconds S

Both load the official checkpoint with ``Qwen3_5ForConditionalGeneration`` (BF16, SDPA,
as the ~27B track does), render one user turn with the chat template and thinking
disabled, and run length-sorted fixed-size batches padded on the left to a multiple of
32 tokens. ``generate`` decodes greedily (at most 192 new tokens); ``judge`` takes one
forward pass and P(yes) = softmax over the next-token logits of "Yes" and "No".

The first 256 items are a timed preflight (after loading). From its steady rate the job
keeps as many items as the budget allows (``generate``: the same fraction of every
language, as prefixes in seed order; ``judge``: whole rows in rank order), and it stops
before a batch that would overrun the budget. The snapshot must be the pinned revision
with an Apache-2.0 card and licence file, or the job stops before loading.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

STARTED = time.time()
PREFLIGHT_ITEMS = 256
BUCKET = 32
MAX_NEW_TOKENS = 192
SAFETY = 1.15
SHAPE_ALLOWANCE = 20.0
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"


def elapsed() -> float:
    return time.time() - STARTED


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def code_identity() -> dict[str, str]:
    for parent in Path(__file__).resolve().parents:
        receipt = parent / ".dev2-mirror.json"
        if receipt.is_file():
            data = json.loads(receipt.read_text(encoding="utf-8"))
            return {"commit": data["commit"], "tree": data["tree"]}
    return {"commit": "unknown", "tree": "unknown"}


def licence_check(model_dir: Path, repo_id: str, revision: str) -> dict[str, Any]:
    readme = (model_dir / "README.md").read_text(encoding="utf-8")
    front: dict[str, str] = {}
    if readme.startswith("---"):
        for line in readme[3 : readme.find("\n---", 3)].splitlines():
            match = re.match(
                r"^(license|license_link|pipeline_tag|library_name):\s*(.*)$",
                line.strip(),
            )
            if match:
                front[match.group(1)] = match.group(2).strip().strip("'\"")
    licence = (model_dir / "LICENSE").read_text(encoding="utf-8")
    result = {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_is_revision": model_dir.name == revision
        and model_dir.parent.name == "snapshots",
        "card_front_matter": front,
        "licence_file_apache_2": "Apache License" in licence
        and "Version 2.0" in licence,
        "readme_sha256": file_sha256(model_dir / "README.md"),
        "licence_sha256": file_sha256(model_dir / "LICENSE"),
    }
    result["passed"] = (
        result["snapshot_is_revision"]
        and front.get("license") == "apache-2.0"
        and result["licence_file_apache_2"]
    )
    return result


def device_info(expect_bus: str | None) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible GPU is required")
    props = torch.cuda.get_device_properties(0)
    info = {
        "name": props.name,
        "total_memory": props.total_memory,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "pci_bus_id": getattr(props, "pci_bus_id", None),
        "visible": {
            k: os.environ.get(k)
            for k in (
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    if expect_bus is not None:
        if info["pci_bus_id"] is None:
            raise SystemExit("cannot read the PCI bus of the visible GPU")
        if int(info["pci_bus_id"]) != int(expect_bus, 16):
            raise SystemExit(
                f"visible GPU is on bus {info['pci_bus_id']:x}, expected {expect_bus}"
            )
    return info


def load(model_dir: Path) -> tuple[Any, Any, float]:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise SystemExit(
            f"incomplete official load: { {k: len(v) for k, v in info.items()} }"
        )
    model.eval()
    return model, tokenizer, time.time() - started


def render(tokenizer: Any, content: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not text.endswith(THINK_OFF_SUFFIX):
        raise SystemExit("chat template did not close an empty thinking block")
    return tokenizer(text, add_special_tokens=False).input_ids


def batches(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(items, key=lambda item: (len(item["ids"]), item["key"]))
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def dry_summary(
    items: list[dict[str, Any]], size: int, tokens: dict[str, Any]
) -> dict[str, Any]:
    groups = batches(items, size)
    widths = collections.Counter(
        BUCKET * math.ceil(max(len(i["ids"]) for i in g) / BUCKET) for g in groups
    )
    lengths = sorted(len(item["ids"]) for item in items)
    return {
        "dry_run": True,
        "items": len(items),
        "batches": len(groups),
        "padded_widths": dict(sorted(widths.items())),
        "prompt_tokens": (
            {"p50": lengths[len(lengths) // 2], "max": lengths[-1]} if lengths else None
        ),
        "token_ids": tokens,
    }


def padded(batch: list[dict[str, Any]], size: int, pad_id: int) -> tuple[Any, Any, int]:
    import torch

    width = BUCKET * math.ceil(max(len(item["ids"]) for item in batch) / BUCKET)
    rows = [item["ids"] for item in batch] + [batch[-1]["ids"]] * (size - len(batch))
    ids = torch.tensor([[pad_id] * (width - len(r)) + r for r in rows], device="cuda:0")
    mask = torch.tensor(
        [[0] * (width - len(r)) + [1] * len(r) for r in rows], device="cuda:0"
    )
    return ids, mask, width


def width_of(batch: list[dict[str, Any]]) -> int:
    return BUCKET * math.ceil(max(len(item["ids"]) for item in batch) / BUCKET)


class Runner:
    """Timed fixed-size batches, a projected cost and a hard budget stop.

    Every batch is padded to the same row count, so the cost unit is a batch: the mean
    time of batches at an already-seen padded width (else all batches but the first),
    times SAFETY, plus SHAPE_ALLOWANCE seconds per padded width not seen yet.
    """

    def __init__(self, budget: float, size: int) -> None:
        self.budget = budget
        self.size = size
        self.log: list[dict[str, Any]] = []
        self.shapes: set[int] = set()
        self.stopped = False

    def batch_seconds(self) -> float:
        warm = [b for b in self.log if not b["new_shape"]] or self.log[1:] or self.log
        return sum(b["seconds"] for b in warm) / max(len(warm), 1)

    def cost(self, groups: list[list[dict[str, Any]]]) -> float:
        new = {width_of(batch) for batch in groups} - self.shapes
        return SAFETY * len(groups) * self.batch_seconds() + SHAPE_ALLOWANCE * len(new)

    def run(self, groups: list[list[dict[str, Any]]], step, phase: str) -> list[Any]:
        import torch

        out = []
        for batch in groups:
            if self.log and elapsed() + self.cost([batch]) > self.budget:
                self.stopped = True
                break
            torch.cuda.synchronize()
            started = time.time()
            results, width = step(batch)
            torch.cuda.synchronize()
            seconds = time.time() - started
            self.log.append(
                {
                    "phase": phase,
                    "items": len(batch),
                    "width": width,
                    "seconds": round(seconds, 3),
                    "new_shape": width not in self.shapes,
                }
            )
            self.shapes.add(width)
            out.extend(results)
        return out


def write_outputs(
    directory: Path, name: str, records: list[dict[str, Any]], receipt: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists() or (directory / "receipt.json").exists():
        raise FileExistsError(f"{directory} already has outputs")
    pending = path.with_name(name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    receipt["outputs_sha256"] = file_sha256(pending)
    os.replace(pending, path)
    receipt["timing"]["total_seconds"] = round(elapsed(), 1)
    with (directory / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def common_receipt(args: argparse.Namespace, job: str) -> dict[str, Any]:
    return {
        "schema": f"dev2-m4-pn1-gpu/{job}/1",
        "job": job,
        "code": code_identity(),
        "model": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "snapshot": str(args.model_dir),
            "loader": "transformers.Qwen3_5ForConditionalGeneration, BF16, sdpa, device_map cuda:0",
        },
        "parameters": {
            "batch_size": args.batch_size,
            "bucket": BUCKET,
            "preflight_items": PREFLIGHT_ITEMS,
            "budget_seconds": args.budget_seconds,
            "safety": SAFETY,
        },
        "timing": {},
    }


def prepare(args: argparse.Namespace, job: str) -> tuple[dict[str, Any], Any, Any]:
    receipt = common_receipt(args, job)
    receipt["model"]["licence"] = licence_check(
        args.model_dir, args.repo_id, args.revision
    )
    if not receipt["model"]["licence"]["passed"]:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "STOPPED.json").write_text(
            json.dumps(receipt, indent=1) + "\n"
        )
        raise SystemExit(3)
    for name in (
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        receipt["model"][name.replace(".", "_") + "_sha256"] = file_sha256(
            args.model_dir / name
        )
    if args.dry_run:
        from transformers import AutoTokenizer

        return (
            receipt,
            None,
            AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True),
        )
    from v2.dec.runtime_check import require_runtime

    receipt["runtime"] = require_runtime()
    receipt["device"] = device_info(args.expect_pci_bus)
    import torch

    free, total = torch.cuda.mem_get_info()
    receipt["device"]["free_bytes_at_start"] = free
    receipt["device"]["total_bytes"] = total
    if args.min_free_gb is not None and free < args.min_free_gb * 1e9:
        raise SystemExit(f"only {free / 1e9:.0f} GB free VRAM (< {args.min_free_gb})")
    model, tokenizer, seconds = load(args.model_dir)
    receipt["timing"]["load_seconds"] = round(seconds, 1)
    receipt["timing"]["ready_after_seconds"] = round(elapsed(), 1)
    return receipt, model, tokenizer


def cmd_generate(args: argparse.Namespace) -> int:
    from v2.data.m4.pn1_text import twin_prompt

    languages = args.languages.split(",")
    excluded = {r["seed"] for path in args.exclude for r in read_jsonl(path)}
    seeds = [
        s
        for s in read_jsonl(args.seeds)
        if s["language"] in languages and s["seed"] not in excluded
    ]
    receipt, model, tokenizer = prepare(args, "generate")
    receipt["inputs_sha256"] = {
        "seeds.jsonl": file_sha256(args.seeds),
        **{f"exclude/{p.parent.name}/{p.name}": file_sha256(p) for p in args.exclude},
    }
    from transformers import GenerationConfig

    defaults = GenerationConfig.from_pretrained(args.model_dir, local_files_only=True)
    eos = defaults.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    pad_id = defaults.pad_token_id
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos,
        pad_token_id=pad_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    receipt["parameters"].update(
        max_new_tokens=MAX_NEW_TOKENS,
        greedy=True,
        eos_token_id=eos,
        pad_token_id=pad_id,
        languages=languages,
    )
    per_lang = {
        lang: sorted(
            (s for s in seeds if s["language"] == lang), key=lambda s: s["rank"]
        )
        for lang in languages
    }
    items = {}
    for lang, group in per_lang.items():
        for seed in group:
            items[seed["seed"]] = {
                "key": seed["seed"],
                "seed": seed,
                "ids": render(tokenizer, twin_prompt(lang, seed["text"])),
            }
    if args.dry_run:
        print(
            json.dumps(
                dry_summary(
                    list(items.values()), args.batch_size, {"eos": eos, "pad": pad_id}
                )
            )
        )
        return 0
    import torch

    def step(batch):
        ids, mask, width = padded(batch, args.batch_size, pad_id)
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids,
                attention_mask=mask,
                generation_config=config,
                do_sample=False,
            )
        results = []
        for item, tokens in zip(batch, output[: len(batch), width:].tolist()):
            cut = next((i for i, t in enumerate(tokens) if t in eos), None)
            kept = tokens if cut is None else tokens[:cut]
            results.append(
                {
                    "seed": item["seed"]["seed"],
                    "language": item["seed"]["language"],
                    "raw": tokenizer.decode(kept, skip_special_tokens=True),
                    "new_tokens": len(kept),
                    "finished": cut is not None,
                    "prompt_tokens": len(item["ids"]),
                }
            )
        return results, width

    interleaved = [
        group[i]
        for i in range(max((len(g) for g in per_lang.values()), default=0))
        for group in per_lang.values()
        if i < len(group)
    ][:PREFLIGHT_ITEMS]
    first = {
        lang: sum(s["language"] == lang for s in interleaved) for lang in languages
    }
    preflight = [items[s["seed"]] for s in interleaved]
    runner = Runner(args.budget_seconds, args.batch_size)
    results = runner.run(batches(preflight, args.batch_size), step, "preflight")
    spent = elapsed()

    def plan(scale: float) -> tuple[dict[str, int], list[dict[str, Any]]]:
        keep = {
            lang: max(first[lang], math.floor(len(group) * scale))
            for lang, group in per_lang.items()
        }
        rest = [
            items[s["seed"]]
            for lang, group in per_lang.items()
            for s in group[first[lang] : keep[lang]]
        ]
        return keep, rest

    scale = 1.0
    keep, rest = plan(scale)
    projected = spent + runner.cost(batches(rest, args.batch_size))
    while (
        scale > 0
        and spent + runner.cost(batches(rest, args.batch_size)) > args.budget_seconds
    ):
        scale = round(scale - 0.005, 6)
        keep, rest = plan(max(scale, 0.0))
    receipt["timing"]["preflight"] = {
        "items": len(preflight),
        "seconds": round(sum(b["seconds"] for b in runner.log), 1),
        "steady_batch_seconds": round(runner.batch_seconds(), 3),
        "projected_total_seconds_all_planned": round(projected, 1),
    }
    results += runner.run(batches(rest, args.batch_size), step, "main")
    done = {r["seed"] for r in results}
    receipt["volumes"] = {
        "excluded_already_generated": len(excluded),
        "planned": {lang: len(group) for lang, group in per_lang.items()},
        "kept_after_preflight": keep,
        "scale": round(scale, 6),
        "generated": {
            lang: sum(r["language"] == lang for r in results) for lang in languages
        },
        "not_generated_due_to_budget_stop": sum(
            i["seed"]["seed"] not in done for i in rest
        ),
    }
    receipt["timing"]["batches"] = runner.log
    receipt["timing"]["stopped_at_budget"] = runner.stopped
    receipt["counts"] = {
        "generated": len(results),
        "finished": sum(r["finished"] for r in results),
        "mean_new_tokens": round(
            sum(r["new_tokens"] for r in results) / max(len(results), 1), 2
        ),
    }
    write_outputs(
        args.output_dir,
        "generations.jsonl",
        sorted(results, key=lambda r: r["seed"]),
        receipt,
    )
    print(
        json.dumps(
            {
                "generated": len(results),
                "scale": scale,
                "batch_seconds": runner.batch_seconds(),
                "seconds": round(elapsed(), 1),
            }
        )
    )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    records = read_jsonl(args.items)
    receipt, model, tokenizer = prepare(args, "judge")
    receipt["inputs_sha256"] = {args.items.name: file_sha256(args.items)}
    from transformers import GenerationConfig

    yes = tokenizer.encode("Yes", add_special_tokens=False)
    no = tokenizer.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise SystemExit("Yes / No are not single tokens")
    pad_id = GenerationConfig.from_pretrained(
        args.model_dir, local_files_only=True
    ).pad_token_id
    receipt["parameters"].update(
        yes_token_id=yes[0], no_token_id=no[0], pad_token_id=pad_id
    )
    items = [
        {"key": r["item"], "record": r, "ids": render(tokenizer, r["prompt"])}
        for r in records
    ]
    if args.dry_run:
        print(
            json.dumps(
                dry_summary(
                    items, args.batch_size, {"yes": yes[0], "no": no[0], "pad": pad_id}
                )
            )
        )
        return 0
    import torch

    def step(batch):
        ids, mask, width = padded(batch, args.batch_size, pad_id)
        with torch.inference_mode():
            logits = model(
                input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1
            ).logits[:, -1, :]
        pair = logits[: len(batch), [yes[0], no[0]]].float()
        probs = torch.softmax(pair, dim=-1)[:, 0].tolist()
        results = []
        for item, p, (ly, ln) in zip(batch, probs, pair.tolist()):
            record = item["record"]
            results.append(
                {
                    "item": record["item"],
                    "cid": record["cid"],
                    "kind": record["kind"],
                    "language": record["language"],
                    "order": record.get("order"),
                    "p_yes": p,
                    "logit_yes": ly,
                    "logit_no": ln,
                    "tokens": len(item["ids"]),
                }
            )
        return results, width

    rows: list[list[dict[str, Any]]] = []
    for item in items:
        if rows and rows[-1][0]["record"]["cid"] == item["record"]["cid"]:
            rows[-1].append(item)
        else:
            rows.append([item])
    preflight = []
    while rows and len(preflight) + len(rows[0]) <= PREFLIGHT_ITEMS:
        preflight += rows.pop(0)
    runner = Runner(args.budget_seconds, args.batch_size)
    results = runner.run(batches(preflight, args.batch_size), step, "preflight")
    spent = elapsed()

    def fits(count: int) -> bool:
        chosen = [item for row in rows[:count] for item in row]
        return (
            spent + runner.cost(batches(chosen, args.batch_size)) <= args.budget_seconds
        )

    projected = spent + runner.cost(
        batches([item for row in rows for item in row], args.batch_size)
    )
    low, high = 0, len(rows)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    rest = [item for row in rows[:low] for item in row]
    receipt["timing"]["preflight"] = {
        "items": len(preflight),
        "seconds": round(sum(b["seconds"] for b in runner.log), 1),
        "steady_batch_seconds": round(runner.batch_seconds(), 3),
        "projected_total_seconds_all_items": round(projected, 1),
    }
    results += runner.run(batches(rest, args.batch_size), step, "main")
    receipt["volumes"] = {
        "items": len(items),
        "kept_after_preflight": len(preflight) + len(rest),
        "judged": len(results),
        "dropped_for_budget": len(items) - len(preflight) - len(rest),
    }
    receipt["timing"]["batches"] = runner.log
    receipt["timing"]["stopped_at_budget"] = runner.stopped
    receipt["counts"] = {
        "judged": len(results),
        "by_kind": {
            k: sum(r["kind"] == k for r in results) for k in ("label", "fluency")
        },
        "mean_tokens": round(
            sum(r["tokens"] for r in results) / max(len(results), 1), 2
        ),
    }
    write_outputs(
        args.output_dir,
        "judgments.jsonl",
        sorted(results, key=lambda r: r["item"]),
        receipt,
    )
    print(
        json.dumps(
            {
                "judged": len(results),
                "batch_seconds": runner.batch_seconds(),
                "seconds": round(elapsed(), 1),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, default_batch in (("generate", 128), ("judge", 64)):
        sub = commands.add_parser(name)
        sub.add_argument("--model-dir", type=Path, required=True)
        sub.add_argument("--repo-id", required=True)
        sub.add_argument("--revision", required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--budget-seconds", type=float, required=True)
        sub.add_argument("--batch-size", type=int, default=default_batch)
        sub.add_argument("--expect-pci-bus")
        sub.add_argument(
            "--min-free-gb", type=float, help="co-tenant: required free VRAM"
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="CPU only: licence, tokenizer, prompts, batches",
        )
        if name == "generate":
            sub.add_argument("--seeds", type=Path, required=True)
            sub.add_argument("--languages", required=True)
            sub.add_argument(
                "--exclude",
                type=Path,
                action="append",
                default=[],
                help="generations.jsonl of an earlier pass; its seeds are skipped",
            )
        else:
            sub.add_argument("--items", type=Path, required=True)
    args = parser.parse_args(argv)
    return cmd_generate(args) if args.command == "generate" else cmd_judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
