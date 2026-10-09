"""GPU self-test of the code-readout engine.

    python -m d25.vega.eval.selftest --ckpt CKPT --kit KIT --suite-dir SUITE [--reference-src SRC] --out OUT.json

Checks, on a deterministic sample of suite requests (a few per benchmark):
1. which Gated DeltaNet / conv kernels transformers bound (fla or the torch fallback);
2. tokenizer rendering equals the processor's chat-template rendering (Perplexity renders with the processor);
3. parity against a reference implementation: with ``--reference-src`` (Perplexity's ``source/src``), the
   reference ``autojev.model.DecisionModel`` and this engine score the same requests in the same batches
   (each request's questions in batches of 8, as Perplexity's server does); reports the largest logit
   difference and answer agreement;
4. determinism (same batch twice) and throughput of length-sorted batches under several token budgets.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path


def sample_rows(suite_dir: str, per_benchmark: int, max_tokens_chars: int = 40000):
    from decision_index.suite.io import Suite

    by = collections.defaultdict(list)
    for row in Suite(suite_dir, "0.3").rows(apply_exclusions=True):
        n = row["_evaluation"]["catalog_id"]
        if len(by[n]) < per_benchmark * 4:
            by[n].append(row)
    picked = []
    for n in sorted(by):
        rows = sorted(
            by[n],
            key=lambda r: hashlib.sha256(
                r["_evaluation"]["run_id"].encode()
            ).hexdigest(),
        )
        rows = [r for r in rows if len(json.dumps(r["state"])) < max_tokens_chars][
            :per_benchmark
        ]
        picked += rows
    return picked


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--reference-src")
    ap.add_argument("--readout-dtype", default="float32")
    ap.add_argument("--attention-mode")
    ap.add_argument("--prompt")
    ap.add_argument("--per-benchmark", type=int, default=3)
    ap.add_argument("--budgets", default="16384,32768,65536,131072")
    ap.add_argument("--throughput-questions", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="torch.profiler table of one batch at the largest budget",
    )
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.kit)
    import torch

    from d25.vega.common import decision_format as df
    from d25.vega.eval.engine import CodeReadoutModel, kernel_report, option_count

    report: dict = {"kernels_before_load": kernel_report()}
    rows = sample_rows(a.suite_dir, a.per_benchmark)
    report["sample_requests"] = len(rows)
    t0 = time.time()
    model = CodeReadoutModel(
        a.ckpt,
        device=a.device,
        readout_dtype=a.readout_dtype,
        attention_mode=a.attention_mode,
        prompt=a.prompt,
    )
    report["load_seconds"] = round(time.time() - t0, 1)
    report["provenance"] = model.provenance()
    report["runtime"] = model.runtime()
    codec = model.codec
    questions = [(r, k, q) for r in rows for k, q in r["questions"].items()]
    report["sample_questions"] = len(questions)

    try:
        if a.skip_parity:
            raise RuntimeError("--skip-parity")
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(codec.dir))
        same = 0
        for r, _, q in questions[:200]:
            if codec.prompt == "pplx":
                from d25.vega.eval.engine import pplx_messages

                msgs = pplx_messages(r["state"], q, codec.codes)
            else:
                msgs = df.messages(r["state"], q, codec.codes)
            text_p = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids_p = processor(text=[text_p], return_tensors=None)["input_ids"][0]
            same += int(
                text_p == codec.text(r["state"], q)
                and list(ids_p)
                == codec.encode([{"state": r["state"], "question": q}])[0]
            )
        report["processor_rendering_equal"] = f"{same}/{min(200, len(questions))}"
    except Exception as exc:  # noqa: BLE001
        report["processor_rendering_equal"] = f"skipped: {type(exc).__name__}: {exc}"

    # request-order batches of 8 (Perplexity's server path)
    seqs = codec.encode([{"state": r["state"], "question": q} for r, _, q in questions])
    counts = [option_count(q) for _, _, q in questions]
    groups, start = [], 0
    for r in rows:
        n = len(r["questions"])
        for s in range(start, start + n, 8):
            groups.append(list(range(s, min(s + 8, start + n))))
        start += n
    groups = [g for g in groups if not codec.over_limit(max(len(seqs[i]) for i in g))]
    if a.skip_parity:
        groups = groups[:2]
    t0 = time.time()
    mine = {}
    for g in groups:
        logits = model.logits([seqs[i] for i in g], [counts[i] for i in g]).cpu()
        for i, row in zip(g, logits):
            mine[i] = row[: counts[i]]
    report["engine_seconds_request_batches"] = round(time.time() - t0, 1)
    again = model.logits(
        [seqs[i] for i in groups[0]], [counts[i] for i in groups[0]]
    ).cpu()
    report["deterministic_repeat"] = bool(
        all(
            torch.equal(again[j][: counts[i]], mine[i]) for j, i in enumerate(groups[0])
        )
    )

    if a.reference_src and not a.skip_parity:
        sys.path.insert(0, a.reference_src)
        from autojev.model import DecisionModel

        t0 = time.time()
        ref = DecisionModel(checkpoint=str(codec.dir), device=a.device)
        report["reference_load_seconds"] = round(time.time() - t0, 1)
        max_diff, agree, n = 0.0, 0, 0
        flips = []
        with torch.inference_mode():
            for g in groups:
                batch = ref.prepare(
                    [
                        {"state": questions[i][0]["state"], "question": questions[i][2]}
                        for i in g
                    ],
                    max_length=10**9,
                )
                logits = ref(batch).float().cpu()
                for i, row in zip(g, logits):
                    a_row, b_row = mine[i], row[: counts[i]]
                    max_diff = max(max_diff, float((a_row - b_row).abs().max()))
                    same = int(a_row.argmax()) == int(b_row.argmax())
                    agree += same
                    n += 1
                    if not same:
                        flips.append(questions[i][0]["_evaluation"]["run_id"])
        report["reference_parity"] = {
            "questions": n,
            "argmax_agree": agree,
            "max_abs_logit_diff": max_diff,
            "flips": flips[:20],
        }
        del ref
        torch.cuda.empty_cache()

    # throughput: length-sorted batches over a pool of questions under several budgets
    pool = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    pool = [i for i in pool if not codec.over_limit(len(seqs[i]))]
    reps = (pool * (a.throughput_questions // max(1, len(pool)) + 1))[
        : a.throughput_questions
    ]
    tput = {}
    for budget in [int(b) for b in a.budgets.split(",")]:
        model.max_batch_tokens = budget
        lengths = [len(seqs[i]) for i in reps]
        batches = model.batches(lengths)
        model.logits([seqs[reps[batches[-1][0]]]], [counts[reps[batches[-1][0]]]])
        model.synchronize()
        t0 = time.time()
        padded = 0
        for b in batches:
            model.logits([seqs[reps[j]] for j in b], [counts[reps[j]] for j in b])
            padded += max(lengths[j] for j in b) * len(b)
        model.synchronize()
        dt = time.time() - t0
        real = sum(lengths)
        tput[str(budget)] = {
            "questions": len(reps),
            "tokens": real,
            "padded_tokens": padded,
            "seconds": round(dt, 2),
            "tokens_per_second": round(real / dt),
            "questions_per_second": round(len(reps) / dt, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated(model.device) / 2**30, 1),
        }
        torch.cuda.reset_peak_memory_stats(model.device)
    report["throughput"] = tput
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        model.max_batch_tokens = int(a.budgets.split(",")[-1])
        lengths = [len(seqs[i]) for i in reps]
        batch = model.batches(lengths)[0]
        model.logits([seqs[reps[j]] for j in batch], [counts[reps[j]] for j in batch])
        model.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model.logits(
                [seqs[reps[j]] for j in batch], [counts[reps[j]] for j in batch]
            )
            model.synchronize()
        table = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=25, max_name_column_width=60
        )
        report["profile"] = {
            "batch_rows": len(batch),
            "padded_tokens": max(lengths[j] for j in batch) * len(batch),
            "table": table.splitlines(),
        }
    report["kernels_after_load"] = kernel_report()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
