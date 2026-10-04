"""Teacher reasoning graphs for natural-language System One rows (OpenAI-compatible vLLM replicas).

Per problem: two solutions with visible reasoning; the first solution that reaches the gold key is turned into a
graph of intermediate yes / no judgments (each with its dependencies and a true declarative statement); every node
is then re-asked without the reasoning and kept only when the fresh answer agrees. Records are appended to a JSONL
file and a rerun skips finished row ids.

usage: python3 -m v2.reasoning.teacher --rows ROWS.jsonl --out OUT.jsonl --ports 18100-18107 [--workers 384]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .render import parse_final_key, parse_yes_no, render_problem

TEACHER_VERSION = "reasoning-teacher-graph/1"

SOLVE_SUFFIX = (
    "\n\nWork through the problem step by step. Finish with exactly one line of the form\n"
    "FINAL: <option key>"
)

GRAPH_PROMPT = """Below is a decision problem and a worked solution that reaches the correct answer.

<problem>
{problem}
</problem>

<solution>
{reasoning}
</solution>

Rewrite the solution as a graph of the intermediate yes/no judgments it relies on.

Rules:
1. Each node is a yes/no question about this problem that can be settled from the problem text, possibly with reasoning. Use only the names, terms and quantities of the problem. Never mention the solution, the reasoning, option keys or the options list.
2. Do not ask the final question in any form, and do not ask whether a candidate answer is correct. Ask about intermediate facts, interpretations of the text, rule applications, computed quantities and checks.
3. "depends_on" lists the ids of earlier nodes whose conclusions are needed to settle this node ([] if none).
4. "answer" is the correct answer to the question, "yes" or "no". Phrase roughly half of the questions so that the correct answer is "no" (for example ask about a wrong value, the opposite relation or a rule that does not apply).
5. "statement" is one short declarative sentence that states the settled conclusion as a true fact (for example "Natalia sold 24 clips in May.").
6. Between 3 and 10 nodes, in the order the solution settles them. Skip plain restatements of given facts unless the text needs interpretation.

Return only JSON: {{"nodes": [{{"id": "n1", "depends_on": [], "question": "...", "answer": "yes", "statement": "..."}}]}}"""

VERIFY_SUFFIX = (
    "\n\nAnswer one intermediate question about this problem with a single word, yes or no.\n"
    "Question: {question}"
)

_LEAK = re.compile(
    r"\b(option|options|answer choice|correct answer|final answer|the answer|choice [a-z0-9]|key k\d|"
    r"the solution|the reasoning|the solver)\b|\[k\d+\]|\bk\d+\b",
    re.I,
)


class Pool:
    """Round-robin chat completions over local replicas, with bounded retries."""

    def __init__(
        self,
        ports: list[int],
        model: str,
        host: str = "127.0.0.1",
        timeout: float = 1800.0,
    ):
        self.urls = [f"http://{host}:{port}/v1/chat/completions" for port in ports]
        self.model = model
        self.timeout = timeout
        self._next = itertools.count()
        self._lock = threading.Lock()

    def chat(
        self,
        content: str,
        *,
        effort: str,
        max_tokens: int,
        temperature: float = 1.0,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": effort,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode()
        error = None
        for attempt in range(6):
            with self._lock:
                url = self.urls[next(self._next) % len(self.urls)]
            request = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    reply = json.load(response)
                choice = reply["choices"][0]
                message = choice.get("message") or {}
                return {
                    "content": message.get("content") or "",
                    "reasoning": message.get("reasoning_content")
                    or message.get("reasoning")
                    or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": reply.get("usage") or {},
                }
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                KeyError,
                ValueError,
            ) as exc:
                error = exc
                time.sleep(min(60, 5 * 2**attempt))
        raise RuntimeError(f"teacher call failed after retries: {error}")


def parse_graph(text: str) -> list[dict[str, Any]]:
    """Validated nodes: unique ids, yes/no answers, nonempty question and statement, dependencies on earlier kept
    nodes only, no leak wording."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0)).get("nodes", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    nodes: list[dict[str, Any]] = []
    kept: set[str] = set()
    for item in raw[:12]:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id", "")).strip()
        question = " ".join(str(item.get("question", "")).split())
        statement = " ".join(str(item.get("statement", "")).split())
        answer = str(item.get("answer", "")).strip().lower()
        if (
            not node_id
            or node_id in kept
            or answer not in ("yes", "no")
            or not question
            or not statement
        ):
            continue
        if (
            _LEAK.search(question)
            or _LEAK.search(statement)
            or len(question) > 400
            or len(statement) > 300
        ):
            continue
        deps = [str(d) for d in (item.get("depends_on") or []) if str(d) in kept]
        nodes.append(
            {
                "id": node_id,
                "depends_on": deps,
                "question": question,
                "answer": answer == "yes",
                "statement": statement,
            }
        )
        kept.add(node_id)
    return nodes


def run_problem(
    pool: Pool, row: dict[str, Any], *, solve_effort: str, solve_tokens: int
) -> dict[str, Any]:
    keys = [option["key"] for option in row["options"]]
    gold = keys[row["label"]]
    problem = render_problem(row)
    traces = []
    for _ in range(2):
        reply = pool.chat(
            problem + SOLVE_SUFFIX, effort=solve_effort, max_tokens=solve_tokens
        )
        key = parse_final_key(reply["content"], keys) or parse_final_key(
            reply["reasoning"], keys
        )
        traces.append(
            {
                "key": key,
                "correct": key == gold,
                "reasoning": reply["reasoning"],
                "finish_reason": reply["finish_reason"],
                "usage": reply["usage"],
            }
        )
    record: dict[str, Any] = {
        "id": row["id"],
        "input_sha256": row["input_sha256"],
        "gold": gold,
        "solve": [{k: v for k, v in t.items() if k != "reasoning"} for t in traces],
        "teacher_version": TEACHER_VERSION,
    }
    used = next((t for t in traces if t["correct"] and t["reasoning"]), None)
    if used is None:
        record["status"] = "unsolved"
        return record
    record["reasoning"] = used["reasoning"]
    reply = pool.chat(
        GRAPH_PROMPT.format(problem=problem, reasoning=used["reasoning"]),
        effort="low",
        max_tokens=6000,
        json_mode=True,
    )
    nodes = parse_graph(reply["content"])
    record["graph_usage"] = reply["usage"]
    for node in nodes:
        check = pool.chat(
            problem + VERIFY_SUFFIX.format(question=node["question"]),
            effort="low",
            max_tokens=3000,
        )
        fresh = parse_yes_no(check["content"])
        node["fresh"] = fresh
        node["verified"] = fresh is not None and fresh == node["answer"]
    record["nodes"] = nodes
    record["status"] = (
        "graph" if any(n["verified"] for n in nodes) else "no_verified_nodes"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ports", default="18100-18107")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=384)
    parser.add_argument("--solve-effort", default="medium")
    parser.add_argument("--solve-tokens", type=int, default=12000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    lo, _, hi = args.ports.partition("-")
    ports = list(range(int(lo), int(hi or lo) + 1))
    pool = Pool(ports, args.model)
    rows = [json.loads(line) for line in args.rows.open(encoding="utf-8")]
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [row for row in rows if row["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{len(rows)} rows, {len(done)} done, {len(todo)} to run on {len(ports)} replicas",
        file=sys.stderr,
    )
    lock = threading.Lock()
    started = time.time()
    counts: dict[str, int] = {}
    with args.out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(
        args.workers
    ) as executor:
        futures = {
            executor.submit(
                run_problem,
                pool,
                row,
                solve_effort=args.solve_effort,
                solve_tokens=args.solve_tokens,
            ): row["id"]
            for row in todo
        }
        for index, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except (
                Exception
            ) as exc:  # noqa: BLE001 - one failed problem must not stop the run
                record = {
                    "id": futures[future],
                    "status": "error",
                    "error": repr(exc)[:500],
                }
            with lock:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                counts[record["status"]] = counts.get(record["status"], 0) + 1
            if index % 200 == 0:
                rate = index / (time.time() - started)
                print(
                    f"{index}/{len(todo)} {rate:.2f}/s {counts}",
                    file=sys.stderr,
                    flush=True,
                )
    print(f"done {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
