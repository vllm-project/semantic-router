"""Requests, answers and scores for the d3-turbo judge behind the escalation results.

The judge reads what a Router could see and answers fixed questions. Three
public sources are covered:

  tb2-end   one state per Terminal-Bench 2.0 trial of a cheap model (the rows of
            prepare_terminal_bench.py): the instruction, the commands, the last
            outputs and the agent's final message. Terminal agents leave no patch.
  tb2-task  one state per task: the instruction alone, for a decision taken
            before any attempt.
  twin      one state per TwinRouterBench static-track row: the benchmark's own
            classifier prompt, answered with one tier question.

Subcommands:
  requests  build request JSONL ({"id", "k", "state"}) from the sources
  answer    answer requests with a d3 package, such as vllm-sr/d3-turbo at
            revision 17a01e225d5e (needs the model and a GPU)
  scores    turn answers into policy_eval.py score files ({"id", "score"})

The answers that produced the published numbers are in data/, so policy_eval.py
and twinrouterbench_eval.py run without the model.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from .prepare_terminal_bench import (
        CHEAP,
        COLUMNS,
        FILES,
        REPO,
        REVISION,
        SCAFFOLD,
        build_rows,
        short,
    )
except ImportError:  # run as a script
    from prepare_terminal_bench import (
        CHEAP,
        COLUMNS,
        FILES,
        REPO,
        REVISION,
        SCAFFOLD,
        build_rows,
        short,
    )

HERE = Path(__file__).resolve().parent
QUESTIONS = {
    "tb2-end": HERE / "questions" / "end.json",
    "tb2-task": HERE / "questions" / "end.json",
    "twin": HERE / "questions" / "twinrouterbench.json",
}
TB2_TASKS_REPO = "harbor-framework/terminal-bench-2"
TB2_TASKS_COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
TWIN_REPO = "CommonstackAI/TwinRouterBench"
TWIN_COMMIT = "7cbb0deac8f697b5faa8489c309560e53d2ef088"
D3_TURBO_REVISION = "17a01e225d5e"

REF = re.compile(r"^\$\d+$")
END_AGENT = (
    "A low-cost terminal agent working in a Linux container through a shell. It has declared the task "
    "complete. Its commands, the outputs it saw and its final message are below."
)
TASK_AGENT = "A low-cost terminal agent that will work on this task in a Linux container through a shell."
TASK_CHARS = 6000
COMMAND_CHARS = 300
ACTION_CHARS = 600
FINAL_CHARS = 3000
LAST_ACTIONS = 4
STATE_CHARS = 24000
MIN_COMMANDS = 8
TWIN_HEAD, TWIN_TAIL = 8000, 52000


def clean(value, limit):
    """The scrape left some strings as unresolved references ("$123")."""
    value = "" if value is None or value == "None" else str(value)
    return "[not recorded]" if REF.match(value) else value[:limit]


def commands(step):
    try:
        tools = (
            ast.literal_eval(step["tools"])
            if step["tools"] not in (None, "None")
            else []
        )
    except (ValueError, SyntaxError):
        return clean(step["tools"], COMMAND_CHARS)
    return "\n".join(
        t.get("cmd", "") or t.get("fn", "") for t in tools if isinstance(t, dict)
    )


def end_state(instruction, steps):
    """The finished trial as the judge reads it, or None without an agent step."""
    agent = [s for s in steps if s["src"] == "agent"]
    if not agent:
        return None
    *work, final = agent
    body = {
        "task": instruction[:TASK_CHARS],
        "agent": END_AGENT,
        "commands": [clean(commands(s), COMMAND_CHARS) for s in work],
        "last_actions": [
            {
                "command": clean(commands(s), ACTION_CHARS),
                "output": clean(s["obs"], ACTION_CHARS),
            }
            for s in work[-LAST_ACTIONS:]
        ],
        "final_message": clean(final["msg"], FINAL_CHARS),
        "finished": True,
    }
    text = json.dumps(body, ensure_ascii=False)
    while len(text) > STATE_CHARS and len(body["commands"]) > MIN_COMMANDS:
        n = len(body["commands"])
        body["commands"] = [
            *body["commands"][: n // 4],
            "... commands omitted ...",
            *body["commands"][-n // 4 :],
        ]
        text = json.dumps(body, ensure_ascii=False)
    return text


def task_state(instruction):
    return json.dumps(
        {
            "task": instruction[:TASK_CHARS],
            "agent": TASK_AGENT,
            "actions_so_far": [],
            "finished": False,
        },
        ensure_ascii=False,
    )


def tb2_requests(records, instructions):
    """End requests for the rows prepare_terminal_bench.build_rows makes, and task requests."""
    row_ids = {r["id"] for r in build_rows(records)}
    trials = defaultdict(list)
    for record in records:
        if record["agent"] == SCAFFOLD:
            trials[(record["model"], record["task_name"])].append(record)
    end, tasks = [], set()
    for model in CHEAP:
        for task in sorted(t for m, t in trials if m == model):
            for i, trial in enumerate(trials[(model, task)]):
                rid = f"{task}::{short(model)}::{trial['trial_name'] or i}"
                if rid not in row_ids or task not in instructions:
                    continue
                state = end_state(instructions[task], json.loads(trial["steps"]))
                if state is not None:
                    end.append({"id": rid, "k": 99, "state": state})
                    tasks.add(task)
    task_requests = [
        {"id": t, "k": 0, "state": task_state(instructions[t])} for t in sorted(tasks)
    ]
    return end, task_requests


def tb2_instructions(tasks_dir):
    return {
        p.parent.name: p.read_text() for p in Path(tasks_dir).glob("*/instruction.md")
    }


def twin_requests(checkout):
    """Requests for the static track, prompted with the benchmark's own classifier prompt."""
    spec = importlib.util.spec_from_file_location(
        "chat_messages", Path(checkout) / "main" / "chat_messages.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = []
    bank = Path(checkout) / "data" / "static" / "question_bank.jsonl"
    for line in bank.open():
        r = json.loads(line)
        msgs = (
            r["messages"]
            if isinstance(r["messages"], list)
            else ast.literal_eval(r["messages"])
        )
        fns = r.get("functions")
        if isinstance(fns, str) and fns.strip().startswith("["):
            fns = ast.literal_eval(fns)
        elif not isinstance(fns, list):
            fns = None
        prompt = module.question_bank_messages_to_classifier_prompt(msgs, functions=fns)
        if len(prompt) > TWIN_HEAD + TWIN_TAIL:
            prompt = (
                prompt[:TWIN_HEAD]
                + "\n\n... earlier turns omitted ...\n\n"
                + prompt[-TWIN_TAIL:]
            )
        out.append({"id": r["id"], "k": 0, "state": prompt})
    return out


def answer(model_dir, requests_path, out_path, questions_path, device):
    """Answer requests with a d3 package's own runtime, resuming a partial file."""
    sys.path.insert(0, model_dir)
    from d3_runtime import D3  # noqa: PLC0415

    questions = json.loads(Path(questions_path).read_text())
    d3 = D3.from_pretrained(model_dir, device=device)
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"], r["k"]) for r in read_jsonl(out_path)}
    rows = [r for r in read_jsonl(requests_path) if (r["id"], r["k"]) not in done]
    start = time.time()
    with open(out_path, "a") as f:
        for n, r in enumerate(rows, 1):
            prepared, long = d3.prepare(r["state"], questions), False
            if (
                prepared.errors
                and "max_length_exceeded" in str(prepared.errors)
                and hasattr(d3, "question_cap")
            ):
                # Over the validated length: answer under the long-context cap and mark the row.
                d3.max_length, long = d3.question_cap(True), True
                prepared = d3.prepare(r["state"], questions)
                d3.max_length = d3.question_cap(False)
            if prepared.errors:
                raise RuntimeError(f"{r['id']}: {prepared.errors}")
            probs, _ = d3.run(prepared)
            row = {
                "id": r["id"],
                "k": r["k"],
                "probs": {k: [float(x) for x in v] for k, v in probs.items()},
            }
            f.write(json.dumps({**row, "long": True} if long else row) + "\n")
            f.flush()
            if n % 100 == 0:
                print(
                    f"{n}/{len(rows)} {n / (time.time() - start):.2f} req/s", flush=True
                )


def scores(answers, question, index, complement):
    """{"id", "score"} rows from one probability of one question."""
    out = []
    for a in answers:
        p = a["probs"][question][index]
        out.append({"id": a["id"], "score": 1 - p if complement else p})
    return out


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    rq = sub.add_parser("requests", help="build judge requests")
    rq.add_argument("--source", choices=["tb2", "twin"], required=True)
    rq.add_argument(
        "--tb2-tasks",
        help=f"checkout of {TB2_TASKS_REPO} at {TB2_TASKS_COMMIT}, for the instructions",
    )
    rq.add_argument(
        "--parquet-dir", help="the Terminal-Bench 2.0 parquet files, else downloaded"
    )
    rq.add_argument(
        "--twinrouterbench", help=f"checkout of {TWIN_REPO} at {TWIN_COMMIT}"
    )
    rq.add_argument("--out-dir", required=True)
    an = sub.add_parser("answer", help="answer requests with a d3 package")
    an.add_argument("--model-dir", required=True)
    an.add_argument("--requests", required=True)
    an.add_argument("--out", required=True)
    an.add_argument("--questions", required=True, help="one of questions/*.json")
    an.add_argument("--device", default="cuda:0")
    sc = sub.add_parser("scores", help="turn answers into score files")
    sc.add_argument("--answers", required=True)
    sc.add_argument("--question", required=True)
    sc.add_argument("--index", type=int, required=True)
    sc.add_argument("--complement", action="store_true", help="score 1 - p")
    sc.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "answer":
        answer(a.model_dir, a.requests, a.out, a.questions, a.device)
        return
    if a.cmd == "scores":
        rows = scores(read_jsonl(a.answers), a.question, a.index, a.complement)
        write_jsonl(a.out, rows)
        print(f"{len(rows)} scores written to {a.out}")
        return
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.source == "twin":
        rows = twin_requests(a.twinrouterbench)
        write_jsonl(out_dir / "twin.requests.jsonl", rows)
        print(f"{len(rows)} twin requests")
        return
    import pandas as pd  # noqa: PLC0415

    if a.parquet_dir:
        paths = [Path(a.parquet_dir) / Path(f).name for f in FILES]
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        paths = [
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)
            for f in FILES
        ]
    records = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in paths]).to_dict(
        "records"
    )
    end, task_requests = tb2_requests(records, tb2_instructions(a.tb2_tasks))
    write_jsonl(out_dir / "tb2_end.requests.jsonl", end)
    write_jsonl(out_dir / "tb2_task.requests.jsonl", task_requests)
    print(f"{len(end)} end requests, {len(task_requests)} task requests")


if __name__ == "__main__":
    main()
