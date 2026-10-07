#!/usr/bin/env python3
"""Router signal A/B on the router signal suite (docs/records/vela2-router-signals.md).

Compares two Router configurations on the evaluation rows of the built-in signals in
vllm-sr/router-signal-suite: the Vela 1.0 defaults (``vela1``) and every signal Vela 2.0
answers bound to one Vela 2.0 0.3B deployment (``vela2``).

config  writes an arm's Router configuration: ``full`` (every built-in request signal, with
        decisions that read each one) or ``latency`` (router-latency-cpu.yaml's signals).
rows    writes one routing-preview request per request-time row of the suite (a feedback row
        follows an assistant turn, since the Router reads feedback only then).
record  is a managed runtime command (VLLM_SRUN_COMMAND) that starts the real runtime on a
        side socket and serves the Router's socket itself, appending every exchange to a log.
run     sends the rows to POST /api/v1/routing/preview of a running Router (resumable).
join    reads a run and its recorded exchanges and writes, per task and file, the
        probabilities the Router received for each row (the suite's prediction layout).
halu    asks the hallucination rows as the Router's detector asks them; the routing preview
        never reaches that response-time signal.
score   compares two arms per file and per signal with group-bootstrap paired intervals.
verdicts  compares how often each arm's Router verdict at its default thresholds is right.
"""

from __future__ import annotations

import argparse
import collections
import copy
import glob
import http.client
import http.server
import json
import os
import queue
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TASKS = ["domain", "jailbreak", "safety", "fact_check", "modality", "pii", "feedback"]
DOMAINS = ["biology", "business", "chemistry", "computer science", "economics", "engineering", "health",
           "history", "law", "math", "other", "philosophy", "physics", "psychology"]  # fmt: skip
FEEDBACK = [
    "satisfied",
    "need_clarification",
    "wrong_answer",
    "want_different",
    "no_feedback",
]
PII_ALLOWED = [
    "AGE",
    "DATE_TIME",
    "DOMAIN_NAME",
    "GPE",
    "NRP",
    "ORGANIZATION",
    "TITLE",
    "ZIP_CODE",
]
SENSITIVE = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STREET_ADDRESS", "CREDIT_CARD", "IBAN_CODE",
             "US_SSN", "US_DRIVER_LICENSE", "IP_ADDRESS"}  # fmt: skip
ASSISTANT = "Here is my answer to your question."
VELA2 = {
    "provider": "model_runtime",
    "artifact": "vllm-sr/Vela-2.0-0.3B",
    "device": "cpu",
}
LABELS = "label_distribution.v1"
SPANS = "token_spans.v1"


# ---------------------------------------------------------------- config


def full_config(port: int) -> dict[str, Any]:
    signals: dict[str, list[dict[str, Any]]] = {
        "domains": [{"name": d, "description": f"{d} requests", "mmlu_categories": [d]} for d in DOMAINS],
        "jailbreak": [{"name": "prompt_attack", "threshold": 0.5, "description": "Prompt attacks."}],
        "pii": [{"name": "sensitive_pii", "threshold": 0.85, "pii_types_allowed": PII_ALLOWED,
                 "description": "Identifiers that name or reach a person."}],
        "fact_check": [{"name": "needs_fact_check", "description": "Needs fact checking."},
                       {"name": "no_fact_check_needed", "description": "No fact checking needed."}],
        "user_feedbacks": [{"name": f, "description": f"{f} feedback"} for f in FEEDBACK],
        "modality": [{"name": m, "description": f"{m} requests"} for m in ("AR", "DIFFUSION", "BOTH")],
        "safety": [{"name": "unsafe_request", "description": "Harmful requests.", "labels": ["safe", "unsafe"],
                    "unsafe_labels": ["unsafe"], "threshold": 0.5}],
    }  # fmt: skip
    kinds = {"domains": "domain", "jailbreak": "jailbreak", "pii": "pii", "fact_check": "fact_check",
             "user_feedbacks": "user_feedback", "modality": "modality", "safety": "safety"}  # fmt: skip
    decisions = []
    priority = 1000
    for key, rules in signals.items():
        for rule in rules:
            priority -= 1
            decisions.append({
                "name": f"{kinds[key]}-{rule['name']}".replace(" ", "-").replace("_", "-").lower(),
                "description": f"Reads {kinds[key]}:{rule['name']}.",
                "priority": priority,
                "rules": {"operator": "AND", "conditions": [{"type": kinds[key], "name": rule["name"]}]},
                "modelRefs": [{"model": "general-model"}],
            })  # fmt: skip
    backend = {
        "name": "local",
        "endpoint": "127.0.0.1:18000",
        "protocol": "http",
        "weight": 100,
    }
    return {
        "version": "v0.3",
        "listeners": [{"name": f"http-{port}", "address": "127.0.0.1", "port": port, "timeout": "300s"}],
        "providers": {"defaults": {"model": "general-model"}, "models": [
            {"name": "general-model", "provider_model_id": "general-model", "api_format": "openai", "backend_refs": [backend]}]},
        "routing": {"modelCards": [{"name": "general-model", "description": "General model.", "modality": "omni"}],
                    "signals": signals, "decisions": decisions},
        "global": {"model_catalog": {"modules": {"modality_detector": {
            "enabled": True, "method": "classifier", "confidence_threshold": 0.7,
            "classifier": {"model_path": "models/Vela-1.0-Encoder-307M-Modality", "use_cpu": True}}}}},
    }  # fmt: skip


def latency_config(port: int, record: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(record.read_text(encoding="utf-8"))
    config.pop("global", None)
    config["listeners"] = [
        {
            "name": f"http-{port}",
            "address": "127.0.0.1",
            "port": port,
            "timeout": "300s",
        }
    ]
    return config


def bind_vela2(config: dict[str, Any], consumers: dict[str, str]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    catalog = config.setdefault("global", {}).setdefault("model_catalog", {})
    catalog["deployments"] = {"vela2": dict(VELA2)}
    catalog["bindings"] = {
        name: {"contract": contract, "deployment": "vela2"}
        for name, contract in consumers.items()
    }
    return config


def cmd_config(args: argparse.Namespace) -> None:
    import yaml

    class NoAliases(yaml.SafeDumper):
        def ignore_aliases(self, data: Any) -> bool:
            return True

    if args.set == "full":
        config = full_config(args.port)
        consumers = {"domain_classifier": LABELS, "prompt_guard": LABELS, "fact_check_classifier": LABELS,
                     "feedback_detector": LABELS, "modality_detector": LABELS, "safety.unsafe_request": LABELS,
                     "pii_classifier": SPANS, "hallucination_detector": SPANS}  # fmt: skip
    else:
        config = latency_config(args.port, Path(args.latency_record))
        consumers = {"domain_classifier": LABELS, "prompt_guard": LABELS, "fact_check_classifier": LABELS,
                     "feedback_detector": LABELS, "pii_classifier": SPANS}  # fmt: skip
    if args.arm == "vela2":
        config = bind_vela2(config, consumers)
    yaml.dump(config, sys.stdout, Dumper=NoAliases, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------- rows and run


def suite_files(suite: str, task: str) -> dict[str, list[str]]:
    files: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(glob.glob(f"{suite}/*/{task}/*.jsonl")):
        stem = os.path.basename(path)[:-6]
        if not stem.startswith("train"):
            files[stem].append(path)
    return files


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cmd_rows(args: argparse.Namespace) -> None:
    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for task in TASKS:
            for stem, paths in sorted(suite_files(args.suite, task).items()):
                for path in paths:
                    for row in read_jsonl(path):
                        if task == "feedback":
                            body: dict[str, Any] = {"messages": [
                                {"role": "user", "content": "Can you help me with this?"},
                                {"role": "assistant", "content": ASSISTANT},
                                {"role": "user", "content": row["text"]}]}  # fmt: skip
                        else:
                            body = {"text": row["text"]}
                        out.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "task": task,
                                    "file": stem,
                                    "body": body,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
    print(f"{count} rows -> {args.out}")


class Client:
    def __init__(self, port: int):
        self.port = port
        self.local = threading.local()

    def post(
        self, path: str, body: dict[str, Any]
    ) -> tuple[int, float, dict[str, Any]]:
        data = json.dumps(body).encode()
        for attempt in range(3):
            conn = getattr(self.local, "conn", None)
            if conn is None:
                conn = self.local.conn = http.client.HTTPConnection(
                    "127.0.0.1", self.port, timeout=600
                )
            try:
                started = time.perf_counter_ns()
                conn.request("POST", path, data, {"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = response.read()
                ms = (time.perf_counter_ns() - started) / 1e6
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    parsed = {"raw": payload[:300].decode(errors="replace")}
                return response.status, ms, parsed
            except (http.client.HTTPException, OSError):
                conn.close()
                self.local.conn = None
                if attempt == 2:
                    raise
                time.sleep(1)
        raise RuntimeError("unreachable")


def cmd_run(args: argparse.Namespace) -> None:
    done = (
        {row["id"] for row in read_jsonl(args.out)}
        if os.path.exists(args.out)
        else set()
    )
    rows = [row for row in read_jsonl(args.rows) if row["id"] not in done]
    client = Client(args.port)
    lock = threading.Lock()
    out = open(args.out, "a", encoding="utf-8")
    count = [0]
    started = time.time()

    def one(row: dict[str, Any]) -> None:
        try:
            status, ms, parsed = client.post("/api/v1/routing/preview", row["body"])
        except Exception as exc:  # recorded, never dropped
            status, ms, parsed = 0, 0.0, {"error": str(exc)[:300]}
        parsed.pop("original_text", None)
        line = json.dumps({"id": row["id"], "task": row["task"], "file": row["file"], "status": status,
                           "ms": round(ms, 3), "response": parsed}, ensure_ascii=False) + "\n"  # fmt: skip
        with lock:
            out.write(line)
            count[0] += 1
            if count[0] % 5000 == 0:
                out.flush()
                print(
                    f"{count[0]}/{len(rows)} {count[0] / (time.time() - started):.1f}/s",
                    flush=True,
                )

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, rows))
    out.close()
    print(f"done {count[0]} rows in {time.time() - started:.0f}s", flush=True)


# ---------------------------------------------------------------- record


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("localhost", timeout=600)
        self.socket_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def recording_handler(upstream: str, log: Any, lock: threading.Lock) -> type:
    local = threading.local()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def address_string(self) -> str:
            return "router"

        def forward(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in ("host", "connection", "content-length")
            }
            for attempt in range(2):
                conn = getattr(local, "conn", None)
                if conn is None:
                    conn = local.conn = UnixConnection(upstream)
                try:
                    conn.request(self.command, self.path, body=body, headers=headers)
                    response = conn.getresponse()
                    payload = response.read()
                    break
                except (http.client.HTTPException, OSError):
                    conn.close()
                    local.conn = None
                    if attempt:
                        self.send_error(502)
                        return
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in (
                    "content-length",
                    "connection",
                    "transfer-encoding",
                ):
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            if self.path in ("/v1/bundle", "/v1/decisions", "/v1/classify") and body:
                entry: dict[str, Any] = {
                    "t": time.time(),
                    "path": self.path,
                    "status": response.status,
                }
                try:
                    entry["request"], entry["response"] = json.loads(body), json.loads(
                        payload
                    )
                except ValueError:
                    entry["raw"] = True
                with lock:
                    log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    log.flush()

        do_GET = forward
        do_POST = forward

    return Handler


def die_with_parent() -> None:
    import ctypes

    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def cmd_record(args: argparse.Namespace, rest: list[str]) -> int:
    rest = rest + args.append.split()
    socket_path = rest[rest.index("--uds") + 1]
    real = socket_path + ".real"
    rest[rest.index("--uds") + 1] = real
    child = subprocess.Popen(
        [args.real, *rest], preexec_fn=die_with_parent
    )  # noqa: S603
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, _frame: child.send_signal(number))
    while not os.path.exists(real):
        if child.poll() is not None:
            return child.returncode
        time.sleep(0.2)
    os.makedirs(args.log_dir, exist_ok=True)
    log = open(
        os.path.join(args.log_dir, os.path.basename(socket_path) + ".jsonl"),
        "a",
        encoding="utf-8",
    )
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    server = UnixServer(socket_path, recording_handler(real, log, threading.Lock()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    code = child.wait()
    server.shutdown()
    for path in (socket_path, real):
        if os.path.exists(path):
            os.unlink(path)
    log.close()
    return code


# ---------------------------------------------------------------- join


def recorded_answers(log_dirs: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """{(text, kind): {consumer: output}} from the recording runtime's exchanges.

    A Vela 2.0 answer depends on every question asked in the same sequence, so a decisions
    answer is kept under the kind of call that asked it: "feedback" when the call also asked
    the feedback question (a follow-up turn), "request" otherwise. A classify head answers
    alone, so its output is kept under "any".
    """
    answers: dict[tuple[str, str], dict[str, Any]] = collections.defaultdict(dict)
    for log_dir in log_dirs:
        for path in glob.glob(f"{log_dir}/*.jsonl"):
            for entry in read_jsonl(path):
                if entry.get("path") != "/v1/bundle" or "request" not in entry:
                    continue
                results = {r.get("id"): r for r in entry["response"].get("results", [])}
                for task in entry["request"].get("tasks", []):
                    result = results.get(task.get("id")) or {}
                    if task.get("decisions") and result.get("decisions"):
                        body, response = task["decisions"], result["decisions"]
                        if isinstance(body.get("state"), str):
                            questions = body.get("questions", {})
                            kind = (
                                "feedback"
                                if any(
                                    q.startswith("feedback_detector:")
                                    for q in questions
                                )
                                else "request"
                            )
                            for qid in questions:
                                answer = response.get("answers", {}).get(qid, {})
                                item = {
                                    "probabilities": answer.get("probabilities"),
                                    "noul": answer.get("noul"),
                                }
                                if qid in (response.get("spans") or {}):
                                    item["spans"] = response["spans"][qid]
                                answers[(body["state"], kind)][qid.split(":")[0]] = item
                    if task.get("classify") and result.get("classify"):
                        body, response = task["classify"], result["classify"]
                        inputs = body.get("input")
                        inputs = (
                            [inputs]
                            if isinstance(inputs, (str, dict))
                            else inputs or []
                        )
                        for index, text in enumerate(inputs):
                            text = text.get("text") if isinstance(text, dict) else text
                            if isinstance(text, str):
                                item = (response.get("results") or [{}])[index]
                                answers[(text, "any")][body.get("model") or "?"] = {
                                    "labels": response.get("labels"),
                                    **item,
                                }
    return answers


def vela1_scores(task: str, outputs: dict[str, Any]) -> dict[str, float] | None:
    """The probabilities a Vela 1.0 head gave the Router (windowed heads: the riskiest window)."""
    key = {"domain": "@domain_classifier", "jailbreak": "@prompt_guard", "fact_check": "@fact_check_classifier",
           "modality": "@modality_detector", "feedback": "@feedback_detector", "safety": "@safety",
           "pii": "@pii_classifier"}[task]  # fmt: skip
    item = outputs.get(key)
    if item is None:
        return None
    labels = item.get("labels") or []
    if task == "pii":
        best = max(
            (
                s.get("probability", 0.0)
                for s in item.get("spans") or []
                if s.get("label") in SENSITIVE
            ),
            default=0.0,
        )
        return {"yes": best, "no": 1 - best}
    probabilities = item.get("probabilities")
    if item.get("windows"):
        positive = {"jailbreak": "jailbreak", "safety": "unsafe"}.get(task, "")
        index = labels.index(positive) if positive in labels else 0
        probabilities = max(
            (w["probabilities"] for w in item["windows"]), key=lambda p: p[index]
        )
    return dict(zip(labels, probabilities, strict=True)) if probabilities else None


def vela2_scores(task: str, outputs: dict[str, Any]) -> dict[str, float] | None:
    key = {"domain": "domain_classifier", "jailbreak": "prompt_guard", "fact_check": "fact_check_classifier",
           "modality": "modality_detector", "feedback": "feedback_detector", "safety": "safety.unsafe_request",
           "pii": "pii_classifier"}[task]  # fmt: skip
    item = outputs.get(key)
    if item is None:
        return None
    if task == "pii":
        best = max(
            (
                s.get("probability", 0.0)
                for s in item.get("spans") or []
                if s.get("label") in SENSITIVE
            ),
            default=0.0,
        )
        return {"yes": best, "no": 1 - best}
    return item.get("probabilities")


def cmd_join(args: argparse.Namespace) -> None:
    rows = {row["id"]: row for row in read_jsonl(args.rows)}
    answers = recorded_answers(args.log_dir)
    scorer = vela1_scores if args.arm == "vela1" else vela2_scores
    handles: dict[tuple[str, str], Any] = {}
    missing: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    failed = 0
    for run in args.run:
        for result in read_jsonl(run):
            if result["id"] in seen:
                continue
            if result.get("status") != 200:
                failed += 1
                continue
            row = rows[result["id"]]
            body = row["body"]
            text = body.get("text") or body["messages"][-1]["content"]
            kind = "feedback" if row["task"] == "feedback" else "request"
            scores = scorer(
                row["task"],
                {**answers.get((text, "any"), {}), **answers.get((text, kind), {})},
            )
            if scores is None:
                missing[row["task"]] += 1
                continue
            seen.add(result["id"])
            decision = (result.get("response") or {}).get("decision_result") or {}
            entry = {
                "id": row["id"],
                "scores": scores,
                "ms": result["ms"],
                "matched": decision.get("matched_signals"),
            }
            key = (row["task"], row["file"])
            if key not in handles:
                os.makedirs(f"{args.out}/{row['task']}", exist_ok=True)
                handles[key] = open(
                    f"{args.out}/{row['task']}/{row['file']}.pred.jsonl",
                    "w",
                    encoding="utf-8",
                )
            handles[key].write(json.dumps(entry, ensure_ascii=False) + "\n")
    for handle in handles.values():
        handle.close()
    print(
        f"{len(seen)} rows joined; previews that failed: {failed}; without a recorded answer: {dict(missing)}"
    )


# ---------------------------------------------------------------- halu


def ask_halu(side: str, row: dict[str, Any], client: Client) -> dict[str, Any]:
    question = (row.get("question") or "").strip()
    if side == "vela1":
        item = {
            "context": row["text"],
            "answer": row["text_pair"],
            **({"question": question} if question else {}),
        }
        options = {
            "overflow": "reject",
            "max_tokens": 8192,
            "threshold": 0.5,
            "return_tokens": True,
        }
        status, _, response = client.post(
            "/v1/classify", {"input": [item], "options": options}
        )
        result = (response.get("results") or [{}])[0]
        if status != 200 or result.get("error"):
            return {"error": result.get("error") or response.get("error") or status}
        labels = response.get("labels") or []
        index = (
            labels.index("hallucinated")
            if "hallucinated" in labels
            else len(labels) - 1
        )
        best = max(
            (t["probabilities"][index] for t in result.get("tokens") or []), default=0.0
        )
        spans = result.get("spans") or []
    else:
        state = {
            "context": row["text"],
            "answer": row["text_pair"],
            **({"request": question} if question else {}),
        }
        qid = "hallucination_detector:halu"
        status, _, response = client.post(
            "/v1/decisions", {"state": state, "questions": {qid: {"preset": "halu"}}}
        )
        answer = (response.get("answers") or {}).get(qid) or {}
        if status != 200 or answer.get("error"):
            return {"error": answer.get("error") or response.get("error") or status}
        best = float(answer.get("noul", 0.0))
        spans = (response.get("spans") or {}).get(qid) or []
    return {"scores": {"hallucinated": best, "supported": 1 - best}, "spans": spans,
            "span_score": max((s.get("probability", 0.0) for s in spans), default=0.0)}  # fmt: skip


def cmd_halu(args: argparse.Namespace) -> None:
    os.makedirs(f"{args.out}/hallucination", exist_ok=True)
    jobs: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    for stem, paths in sorted(suite_files(args.suite, "hallucination").items()):
        target = f"{args.out}/hallucination/{stem}.pred.jsonl"
        done = (
            {row["id"] for row in read_jsonl(target)}
            if os.path.exists(target)
            else set()
        )
        for path in paths:
            for row in read_jsonl(path):
                if row["id"] not in done:
                    jobs.put((stem, row))
    total = jobs.qsize()
    lock = threading.Lock()
    handles: dict[str, Any] = {}

    def worker(port: int) -> None:
        client = Client(port)
        while True:
            try:
                stem, row = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                result = ask_halu(args.side, row, client)
            except Exception as exc:  # recorded, never dropped
                result = {"error": str(exc)[:300]}
            result["id"] = row["id"]
            with lock:
                if stem not in handles:
                    handles[stem] = open(
                        f"{args.out}/hallucination/{stem}.pred.jsonl",
                        "a",
                        encoding="utf-8",
                    )
                handles[stem].write(json.dumps(result, ensure_ascii=False) + "\n")

    threads = [
        threading.Thread(target=worker, args=(int(port),))
        for port in args.ports.split(",")
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for handle in handles.values():
        handle.close()
    print(f"done {total} rows")


# ---------------------------------------------------------------- score

POSITIVE = {"jailbreak": "jailbreak", "safety": "unsafe", "fact_check": "FACT_CHECK_NEEDED", "pii": "yes",
            "hallucination": "hallucinated", "modality": "DIFFUSION"}  # fmt: skip
CLASSES = {
    "domain": DOMAINS,
    "feedback": [
        "SAT",
        "NEED_CLARIFICATION",
        "WRONG_ANSWER",
        "WANT_DIFFERENT",
        "NO_FEEDBACK",
    ],
}
COMBOS = {
    "modality": {
        "hold-realmix": ["hold-arena-t2i-hard", "hold-parti", "hold-search-arena"]
    }
}
# The suite card's per-signal held-out sets, among the files whose text the suite publishes.
HELD_OUT = {
    "domain": [
        "hold-arena-expert",
        "hold-mmlu-cf",
        "hold-mmlu-pro",
        "hold-mmlu-prox",
        "hold-supergpqa",
    ],
    "fact_check": ["hold-no_robots", "hold-wildbench"],
    "hallucination": ["hold-hallumix", "hold-halubench", "hold-summedits"],
    "jailbreak": [
        "hold-bipia",
        "hold-llmail",
        "hold-promptshield-test",
        "hold-toxicchat",
    ],
    "modality": ["hold-realmix"],
    "pii": ["hold-kaggle-essays", "hold-pii-prompts"],
    "safety": ["hold-jbb", "hold-openai-moderation", "hold-toxicchat", "hold-xstest"],
    "feedback": ["hold-shipped-feedback"],
}


def scores_of(task: str, scores: dict[str, Any]) -> dict[str, float]:
    out = {k: float(v) for k, v in scores.items() if not k.startswith("_")}
    if task == "modality" and "BOTH" in out:
        out["DIFFUSION"] = out.get("DIFFUSION", 0.0) + out.pop("BOTH")
    return out


def auc(s: Any, y: Any) -> float:
    """Mann-Whitney AUC with tied scores counted half."""
    import numpy as np

    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ordered = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def file_metric(
    task: str, gold: list[dict[str, Any]], arms: list[dict[str, Any]], draws: list[Any]
) -> tuple[str, list[Any]]:
    import numpy as np

    series = []
    if task in CLASSES:
        classes = CLASSES[task]
        g = np.array([classes.index(r["label"]) for r in gold])
        for p in arms:
            pred = np.array(
                [
                    int(
                        np.argmax(
                            [
                                scores_of(task, p[r["id"]]["scores"]).get(c, 0.0)
                                for c in classes
                            ]
                        )
                    )
                    for r in gold
                ]
            )
            correct = pred == g
            series.append(np.array([correct[idx].mean() for idx in draws]))
        return "accuracy", series
    positive = POSITIVE[task]
    y = np.array([r["label"] == positive for r in gold])
    one_class = bool(y.all() or not y.any())
    for p in arms:
        s = np.array(
            [scores_of(task, p[r["id"]]["scores"]).get(positive, 0.0) for r in gold]
        )
        if one_class:
            hit = (s >= 0.5) == y
            series.append(np.array([hit[idx].mean() for idx in draws]))
        else:
            series.append(np.array([auc(s[idx], y[idx]) for idx in draws]))
    kind = "AUC" if not one_class else ("recall@0.5" if y.all() else "specificity@0.5")
    return kind, series


def interval(series: Any) -> list[float]:
    import numpy as np

    boot = series[1:][~np.isnan(series[1:])]
    return [
        round(float(series[0]), 4),
        round(float(np.percentile(boot, 2.5)), 4),
        round(float(np.percentile(boot, 97.5)), 4),
    ]


def predictions(root: str, task: str, stems: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for stem in stems:
        path = f"{root}/{task}/{stem}.pred.jsonl"
        if os.path.exists(path):
            out.update({row["id"]: row for row in read_jsonl(path) if "scores" in row})
    return out


def cmd_score(args: argparse.Namespace) -> None:
    import numpy as np

    report: dict[str, Any] = {
        "a": args.a_name,
        "b": args.b_name,
        "rounds": args.rounds,
        "tasks": {},
    }
    lines = [f"| signal | file | rows | metric | {args.a_name} | {args.b_name} | {args.b_name} − {args.a_name} [95% CI] |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: |"]  # fmt: skip
    for ti, task in enumerate(args.tasks.split(",")):
        golds = {
            stem: [row for path in paths for row in read_jsonl(path)]
            for stem, paths in suite_files(args.suite, task).items()
        }
        for name, parts in COMBOS.get(task, {}).items():
            if all(p in golds for p in parts):
                golds[name] = [row for p in parts for row in golds[p]]
        entries: dict[str, Any] = {}
        series: dict[str, list[Any]] = {}
        for fi, (stem, gold) in enumerate(sorted(golds.items())):
            parts = COMBOS.get(task, {}).get(stem, [stem])
            pa, pb = predictions(args.a, task, parts), predictions(args.b, task, parts)
            rows = [r for r in gold if r["id"] in pa and r["id"] in pb]
            if len(rows) < 20:
                continue
            groups: dict[str, list[int]] = collections.defaultdict(list)
            for i, r in enumerate(rows):
                groups[str(r.get("group") or r["id"])].append(i)
            clusters = [np.array(v) for v in groups.values()]
            rng = np.random.default_rng([args.seed, ti, fi])
            draws = [np.arange(len(rows))] + [
                np.concatenate(
                    [clusters[k] for k in rng.integers(0, len(clusters), len(clusters))]
                )
                for _ in range(args.rounds)
            ]
            kind, (sa, sb) = file_metric(task, rows, [pa, pb], draws)
            entries[stem] = {"rows": len(rows), "missing": len(gold) - len(rows), "metric": kind,
                             "a": interval(sa), "b": interval(sb), "b_minus_a": interval(sb - sa)}  # fmt: skip
            series[stem] = [sa, sb]
            d = entries[stem]["b_minus_a"]
            lines.append(f"| {task} | {stem} | {len(rows):,} | {kind} | {entries[stem]['a'][0]:.3f} | {entries[stem]['b'][0]:.3f} | "
                         f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")  # fmt: skip
        sets = {"held_out": [s for s in HELD_OUT.get(task, []) if s in series],
                "fresh": [s for s in series if s.startswith("fresh-")],
                "in_distribution": [s for s in series if s == "test" or s.startswith("test-")]}  # fmt: skip
        means: dict[str, Any] = {}
        for name, stems in sets.items():
            stems = [s for s in stems if entries[s]["metric"] in ("AUC", "accuracy")]
            if not stems:
                continue
            ma = np.mean(np.stack([series[s][0] for s in stems]), axis=0)
            mb = np.mean(np.stack([series[s][1] for s in stems]), axis=0)
            means[name] = {
                "files": stems,
                "a": interval(ma),
                "b": interval(mb),
                "b_minus_a": interval(mb - ma),
            }
            d = means[name]["b_minus_a"]
            lines.append(f"| **{task}** | **mean, {name.replace('_', ' ')}** ({len(stems)} files) | | {entries[stems[0]]['metric']} | "
                         f"**{means[name]['a'][0]:.3f}** | **{means[name]['b'][0]:.3f}** | **{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]** |")  # fmt: skip
        report["tasks"][task] = {"files": entries, "means": means}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    text = "\n".join(lines) + "\n"
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    print(text)


FEEDBACK_RULES = {"SAT": "satisfied", "NEED_CLARIFICATION": "need_clarification", "WRONG_ANSWER": "wrong_answer",
                  "WANT_DIFFERENT": "want_different", "NO_FEEDBACK": "no_feedback"}  # fmt: skip


def verdict(task: str, row: dict[str, Any], pred: dict[str, Any]) -> bool | None:
    """Whether the Router's own verdict (its matched rules, or a hallucination span) is the gold one."""
    label = row["label"]
    if task == "hallucination":
        return bool(pred.get("spans")) == (label == "hallucinated")
    matched = pred.get("matched") or {}
    if task == "domain":
        return matched.get("domains") == [label]
    if task == "feedback":
        return (matched.get("user_feedback") or ["no_feedback"]) == [
            FEEDBACK_RULES[label]
        ]
    if task == "modality":
        return ("AR" in (matched.get("modality") or [])) == (label == "AR")
    if task == "fact_check":
        return ("needs_fact_check" in (matched.get("fact_check") or [])) == (
            label == "FACT_CHECK_NEEDED"
        )
    key = {"jailbreak": "jailbreak", "safety": "safety", "pii": "pii"}[task]
    return bool(matched.get(key)) == (label == POSITIVE[task])


def cmd_verdicts(args: argparse.Namespace) -> None:
    """Per signal and file set, the share of rows whose Router verdict is right (balanced for the binary signals)."""
    out: dict[str, Any] = {}
    for task in args.tasks.split(","):
        golds = {
            stem: [row for path in paths for row in read_jsonl(path)]
            for stem, paths in suite_files(args.suite, task).items()
        }
        sets = {"held_out": HELD_OUT.get(task, []), "fresh": [s for s in golds if s.startswith("fresh-")],
                "in_distribution": [s for s in golds if s == "test" or s.startswith("test-")]}  # fmt: skip
        for name, stems in sets.items():
            stems = [
                s for p in stems for s in COMBOS.get(task, {}).get(p, [p]) if s in golds
            ]
            result = {}
            for arm, root in (("a", args.a), ("b", args.b)):
                preds = {
                    k: v for s in stems for k, v in predictions(root, task, [s]).items()
                }
                hits: dict[str, list[bool]] = collections.defaultdict(list)
                for row in (r for s in stems for r in golds[s]):
                    if (
                        row["id"] in preds
                        and (hit := verdict(task, row, preds[row["id"]])) is not None
                    ):
                        hits[row["label"]].append(hit)
                if hits:
                    result[arm] = round(
                        sum(sum(v) / len(v) for v in hits.values()) / len(hits), 4
                    )
            if result:
                out.setdefault(task, {})[name] = result
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("config")
    p.add_argument("--arm", required=True, choices=["vela1", "vela2"])
    p.add_argument("--set", default="full", choices=["full", "latency"])
    p.add_argument("--port", type=int, default=18899)
    p.add_argument(
        "--latency-record",
        default=str(
            Path(__file__).resolve().parent.parent
            / "docs/records/router-latency-cpu.yaml"
        ),
    )
    p = sub.add_parser("rows")
    p.add_argument("--suite", required=True, help="the suite's text/ directory")
    p.add_argument("--out", required=True)
    p = sub.add_parser("record")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--real", required=True, help="the runtime command (vllm-srun)")
    p.add_argument(
        "--append",
        default="",
        help="space-separated arguments added to the runtime's own",
    )
    p = sub.add_parser("run")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("join")
    p.add_argument("--rows", required=True)
    p.add_argument("--run", required=True, nargs="+")
    p.add_argument("--log-dir", required=True, nargs="+")
    p.add_argument("--arm", required=True, choices=["vela1", "vela2"])
    p.add_argument("--out", required=True)
    p = sub.add_parser("halu")
    p.add_argument("--side", required=True, choices=["vela1", "vela2"])
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ports", required=True)
    p = sub.add_parser("score")
    p.add_argument("--suite", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--a-name", default="vela1")
    p.add_argument("--b-name", default="vela2")
    p.add_argument("--tasks", default=",".join(TASKS + ["hallucination"]))
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--out", required=True)
    p.add_argument("--md")
    p = sub.add_parser("verdicts")
    p.add_argument("--suite", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--tasks", default=",".join(TASKS + ["hallucination"]))
    p.add_argument("--out", required=True)
    args, rest = ap.parse_known_args()
    if args.cmd == "record":
        return cmd_record(args, rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    commands = {"config": cmd_config, "rows": cmd_rows, "run": cmd_run, "join": cmd_join, "halu": cmd_halu,
                "score": cmd_score, "verdicts": cmd_verdicts}  # fmt: skip
    commands[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
