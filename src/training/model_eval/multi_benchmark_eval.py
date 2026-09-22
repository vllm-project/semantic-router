#!/usr/bin/env python3
"""Multi-benchmark MCQ evaluation client for vLLM OpenAI-compatible endpoints.

Reads a task JSONL file (each line: task_id, prompt_user, prompt_system?,
dataset?, category?, language?, ...) and sends concurrent chat completion
requests to a vLLM endpoint. Supports reasoning-mode injection via
--chat-template-kwargs for thinking models (DeepSeek/Qwen).

Output: records.jsonl with per-question metadata (completion, tokens, latency).

Resumable: if the output file already exists, completed task_ids are skipped.

Usage:
    python multi_benchmark_eval.py \\
        --split tasks/accept_office.jsonl \\
        --out results/dsv4-flash/accept_office/records.jsonl \\
        --endpoint http://127.0.0.1:8001/v1 \\
        --model dsv4-flash \\
        --params gen_params_dsv4.json \\
        --concurrency 8 \\
        [--chat-template-kwargs '{"chat_template_kwargs":{"enable_thinking":false}}']
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def load_params(path):
    """Load generation parameters from a JSON file."""
    with open(path) as f:
        p = json.load(f)
    return {
        "temperature": float(p.get("temperature", 0.0)),
        "top_p": float(p["top_p"]) if p.get("top_p") is not None else None,
        "max_tokens": int(p.get("max_tokens", 4096)),
        "chat_template_kwargs": p.get("chat_template_kwargs"),
    }


def gen_one(task, client, model, params, chat_template_kwargs, request_timeout):
    """Generate a response for a single task with retry logic."""
    messages = []
    if task.get("prompt_system"):
        messages.append({"role": "system", "content": task["prompt_system"]})
    messages.append({"role": "user", "content": task["prompt_user"]})

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        stream=True,
        stream_options={"include_usage": True},
    )
    if params["top_p"] is not None:
        kwargs["top_p"] = params["top_p"]
    # CLI --chat-template-kwargs overrides params file
    ctk = chat_template_kwargs or params["chat_template_kwargs"]
    if ctk:
        kwargs["extra_body"] = ctk

    err = None
    chunks, finish, usage, ttft, t0 = [], None, {}, None, 0.0
    for attempt in range(3):
        try:
            t0 = time.time()
            chunks, finish, usage, ttft = [], None, {}, None
            with client.chat.completions.create(**kwargs) as stream:
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    for ch in chunk.choices or []:
                        d = ch.delta
                        if d and d.content:
                            if ttft is None:
                                ttft = time.time() - t0
                            chunks.append(d.content)
                        if ch.finish_reason:
                            finish = ch.finish_reason
            err = None
            break
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(5 * (attempt + 1))

    return {
        "task_id": task["task_id"],
        "dataset": task.get("dataset"),
        "category": task.get("category"),
        "language": task.get("language"),
        "difficulty": task.get("difficulty"),
        "completion": "".join(chunks),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "latency_s": round(time.time() - t0, 3) if err is None else None,
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="Task JSONL file")
    ap.add_argument("--out", required=True, help="Output records JSONL")
    ap.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1",
        help="vLLM OpenAI-compatible endpoint",
    )
    ap.add_argument("--model", required=True, help="Served model name")
    ap.add_argument("--params", required=True, help="JSON file with generation params")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument(
        "--chat-template-kwargs",
        type=str,
        default="",
        help="JSON string for reasoning-mode injection, e.g. "
        '\'{"chat_template_kwargs":{"enable_thinking":false}}\'',
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run all tasks even if records.jsonl already exists "
        "(default: resume, skip already completed task_ids)",
    )
    args = ap.parse_args()

    params = load_params(args.params)
    ctk = None
    if args.chat_template_kwargs:
        ctk = json.loads(args.chat_template_kwargs)

    client = OpenAI(
        base_url=args.endpoint, api_key="dummy", timeout=args.request_timeout
    )

    with open(args.split, encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f]

    # Resumable: skip already completed task_ids (unless --force).
    # Only successful records (error is None) are skipped — failed records
    # remain eligible for retry so a transient endpoint failure does not
    # permanently leave a task unanswered. Failed records from a previous
    # run are stripped from the file so retried tasks are not duplicated.
    done = set()
    kept_lines: list[str] = []
    if args.force and os.path.exists(args.out):
        os.remove(args.out)
        print(f"[gen] --force: deleted existing {args.out}", flush=True)
    elif os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        done.add(rec["task_id"])
                        kept_lines.append(line.rstrip("\n"))
                except Exception:
                    pass
        # Rewrite file with only successful records, dropping failed ones
        # so retried tasks don't accumulate duplicates.
        if len(kept_lines) < sum(1 for _ in open(args.out, encoding="utf-8")):
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
            print(f"[gen] pruned failed records from {args.out}", flush=True)
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"[gen] total={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    lock = threading.Lock()
    n_ok = n_err = 0
    t_start = time.time()
    with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as ex:
        futs = [
            ex.submit(gen_one, t, client, args.model, params, ctk, args.request_timeout)
            for t in todo
        ]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if rec["error"]:
                    n_err += 1
                else:
                    n_ok += 1
                if (n_ok + n_err) % 50 == 0:
                    el = time.time() - t_start
                    print(
                        f"[gen] {n_ok + n_err}/{len(todo)} ok={n_ok} "
                        f"err={n_err} elapsed={el:.0f}s",
                        flush=True,
                    )
    print(
        f"[gen] DONE ok={n_ok} err={n_err} wall={time.time() - t_start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
