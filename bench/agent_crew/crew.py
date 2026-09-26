#!/usr/bin/env python3
"""A four-agent review crew whose every model call goes through vLLM Semantic Router.

Only the model name each agent sends changes between arms:
  all-frontier  every agent sends "frontier"
  per-agent     reviewer sends "frontier", planner/summarizer/writer send "local"
  per-call      every agent sends "MoM" and the router chooses per call
  all-local     every agent sends "local"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture"
PR_DIR = FIXTURE / "pr"

AGENTS = ("planner", "summarizer", "reviewer", "writer")
ARMS = {
    "all-frontier": dict.fromkeys(AGENTS, "frontier"),
    "per-agent": {
        "planner": "local",
        "summarizer": "local",
        "reviewer": "frontier",
        "writer": "local",
    },
    "per-call": dict.fromkeys(AGENTS, "MoM"),
    "all-local": dict.fromkeys(AGENTS, "local"),
}

SYSTEM = {
    "planner": "You plan code reviews. Look at the pull request with the tools before planning. Be brief.",
    "summarizer": "You write one-sentence summaries of changed files.",
    "reviewer": (
        "You review code in pull requests. Report real defects only, each with the line, "
        "the problem and the fix. You may call read_file to see other files in the pull request."
    ),
    "writer": "You write the final review comment for a pull request from notes prepared by other reviewers.",
}
PLAN_TASK = (
    "Plan the review of this pull request. Look at the changed files with the tools, "
    "then write one line per file saying what to check first.\n\n{pr}"
)
SUMMARY_TASK = "Summarize what this changed file does in one sentence.\n\nFile: {path}\n```\n{content}\n```"
REVIEW_TASK = (
    "Find the defects in this file from the pull request. For each defect give the line "
    "number, the problem and the fix. If there are none, say so.\n\nFile: {path}\n```\n{content}\n```"
)
WRITE_TASK = (
    "Write the final review comment for this pull request from the notes below. Group the "
    "findings by file, most serious first, and keep every concrete defect. Stay under 300 words."
    "\n\nReview plan:\n{plan}\n\nFile summaries:\n{summaries}\n\nFindings:\n{findings}"
)

TOOL_SPECS = {
    "list_changed_files": {
        "type": "function",
        "function": {
            "name": "list_changed_files",
            "description": "List the files changed in the pull request.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File name, e.g. billing.py",
                    }
                },
                "required": ["path"],
            },
        },
    },
}

HEADER_MODEL = "x-vsr-selected-model"
HEADER_DECISION = "x-vsr-selected-decision"
HEADER_ROUTING_MS = "x-vsr-routing-latency-ms"
HEADER_COST = "x-vsr-cost"
HEADER_CURRENCY = "x-vsr-cost-currency"
HEADER_REPLAY = "x-vsr-replay-id"

METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)")
METRIC_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
COLORS = {"local": "\033[32m", "frontier": "\033[35m"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--base-url", default=os.environ.get("ROUTER_URL", "http://localhost:8899/v1")
    )
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument(
        "--router-metrics-url", default=os.environ.get("ROUTER_METRICS_URL", "")
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-turns", type=int, default=6, help="model calls allowed per conversation"
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    return parser.parse_args(argv)


def changed_files() -> list[str]:
    return sorted(p.name for p in PR_DIR.iterdir() if p.is_file())


def numbered(text: str) -> str:
    return "\n".join(f"{i:>3} | {line}" for i, line in enumerate(text.splitlines(), 1))


def run_tool(name: str, arguments: str) -> str:
    if name == "list_changed_files":
        return "\n".join(changed_files())
    if name != "read_file":
        return f"error: unknown tool {name}"
    try:
        path = str(json.loads(arguments or "{}").get("path", ""))
    except (json.JSONDecodeError, AttributeError):
        return "error: arguments were not valid JSON"
    target = (PR_DIR / Path(path).name).resolve()
    if target.parent != PR_DIR.resolve() or not target.is_file():
        return f"error: {path!r} is not a file in this pull request"
    return target.read_text()


def header_float(headers: Any, name: str) -> float | None:
    try:
        return float(headers.get(name))
    except (TypeError, ValueError):
        return None


class Crew:
    def __init__(self, client: OpenAI, args: argparse.Namespace, run_id: str) -> None:
        self.client = client
        self.args = args
        self.run_id = run_id
        self.models = ARMS[args.arm]
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[dict[str, Any]] = []
        self.capped: list[str] = []

    def converse(
        self, agent: str, step: str, task: str, tools: tuple[str, ...] = ()
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM[agent]},
            {"role": "user", "content": task},
        ]
        specs = [TOOL_SPECS[t] for t in tools]
        final = ""
        for turn in range(1, self.args.max_turns + 1):
            message = self.call(agent, step, turn, messages, specs)
            if message is None:
                break
            if not message.tool_calls:
                final = message.content or ""
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            for tc in message.tool_calls:
                result = run_tool(tc.function.name, tc.function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
        else:
            self.capped.append(step)
        self.conversations.append(
            {"agent": agent, "step": step, "messages": messages, "final": final}
        )
        return final

    def call(
        self, agent: str, step: str, turn: int, messages: list, specs: list
    ) -> Any:
        requested = self.models[agent]
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "arm": self.args.arm,
            "index": len(self.calls) + 1,
            "agent": agent,
            "step": step,
            "turn": turn,
            "requested_model": requested,
            "error": "",
        }
        kwargs: dict[str, Any] = {
            "model": requested,
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
        }
        if specs:
            kwargs["tools"] = specs
        started = time.perf_counter()
        try:
            raw = self.client.chat.completions.with_raw_response.create(**kwargs)
            response = raw.parse()
        except Exception as exc:  # every failure is recorded, not raised
            record.update(
                latency_s=round(time.perf_counter() - started, 3),
                error=f"{type(exc).__name__}: {exc}"[:400],
            )
            self.calls.append(record)
            self.progress(record)
            return None
        headers = raw.headers
        usage = response.usage
        choice = response.choices[0]
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        selected = headers.get(HEADER_MODEL) or (
            requested if requested != "MoM" else ""
        )
        record.update(
            selected_model=selected,
            selected_by="header" if headers.get(HEADER_MODEL) else "request",
            served_model=response.model,
            decision=headers.get(HEADER_DECISION) or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "",
            tool_calls=len(choice.message.tool_calls or []),
            latency_s=round(time.perf_counter() - started, 3),
            routing_latency_ms=header_float(headers, HEADER_ROUTING_MS),
            cost=header_float(headers, HEADER_COST),
            cost_currency=headers.get(HEADER_CURRENCY) or "",
            replay_id=headers.get(HEADER_REPLAY) or "",
        )
        self.calls.append(record)
        self.progress(record)
        return choice.message

    def progress(self, r: dict[str, Any]) -> None:
        selected = r.get("selected_model") or "?"
        color, reset = (
            (COLORS.get(selected, ""), "\033[0m") if sys.stderr.isatty() else ("", "")
        )
        outcome = (
            r["error"][:60]
            if r["error"]
            else f"{r.get('decision') or '-':<14} {r['latency_s']:6.1f}s"
        )
        print(
            f"{r['index']:>3}  {r['agent']:<10} {r['step']:<24} {r['requested_model']:>8} -> "
            f"{color}{selected:<8}{reset} {outcome}",
            file=sys.stderr,
        )

    def run(self) -> str:
        files = changed_files()
        pr = (FIXTURE / "PR.md").read_text()
        plan = self.converse(
            "planner",
            "plan",
            PLAN_TASK.format(pr=pr),
            ("list_changed_files", "read_file"),
        )
        summaries = []
        for path in files:
            content = (PR_DIR / path).read_text()
            text = self.converse(
                "summarizer",
                f"summarize:{path}",
                SUMMARY_TASK.format(path=path, content=content),
            )
            summaries.append(f"- {path}: {text.strip()}")
        findings = []
        for path in (f for f in files if f.endswith(".py")):
            content = numbered((PR_DIR / path).read_text())
            text = self.converse(
                "reviewer",
                f"review:{path}",
                REVIEW_TASK.format(path=path, content=content),
                ("read_file",),
            )
            findings.append(f"### {path}\n{text.strip()}")
        return self.converse(
            "writer",
            "write",
            WRITE_TASK.format(
                plan=plan.strip(),
                summaries="\n".join(summaries),
                findings="\n\n".join(findings),
            ),
        )


def parse_metrics(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    samples = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if not match or line.startswith("#"):
            continue
        labels = tuple(sorted(METRIC_LABEL.findall(match.group(2) or "")))
        try:
            samples[(match.group(1), labels)] = float(match.group(3))
        except ValueError:
            continue
    return samples


def scrape_metrics(url: str) -> dict | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return parse_metrics(response.read().decode("utf-8", "replace"))
    except Exception as exc:
        print(
            f"warning: could not read router metrics from {url}: {exc}", file=sys.stderr
        )
        return None


def metrics_delta(before: dict | None, after: dict | None) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    cost: dict[str, dict[str, float]] = defaultdict(dict)
    latency_sum = latency_count = 0.0
    for key, value in after.items():
        name, labels = key
        delta = value - before.get(key, 0.0)
        if name == "llm_model_cost_total" and delta:
            label = dict(labels)
            model, currency = label.get("model", "?"), label.get("currency", "?")
            cost[model][currency] = round(cost[model].get(currency, 0.0) + delta, 10)
        elif name == "llm_model_routing_latency_seconds_sum":
            latency_sum += delta
        elif name == "llm_model_routing_latency_seconds_count":
            latency_count += delta
    totals: dict[str, float] = defaultdict(float)
    for by_currency in cost.values():
        for currency, amount in by_currency.items():
            totals[currency] = round(totals[currency] + amount, 10)
    return {
        "source": "router /metrics, before and after the run",
        "cost_by_model": dict(cost),
        "cost_total": dict(totals),
        "routing_decisions": int(latency_count),
        "routing_latency_ms_mean": (
            round(latency_sum / latency_count * 1000, 3) if latency_count else None
        ),
    }


def value_summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    return {
        "samples": len(ordered),
        "p50": pick(0.5),
        "p95": pick(0.95),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
    }


def summarize(
    crew: Crew, wall_time_s: float, router_metrics: dict | None
) -> dict[str, Any]:
    ok = [c for c in crew.calls if not c["error"]]
    by_agent: dict[str, Any] = {}
    for agent in AGENTS:
        rows = [c for c in ok if c["agent"] == agent]
        by_agent[agent] = {
            "calls": len(rows),
            "selected_models": dict(Counter(c["selected_model"] or "?" for c in rows)),
            "prompt_tokens": sum(c["prompt_tokens"] for c in rows),
            "completion_tokens": sum(c["completion_tokens"] for c in rows),
        }
    tokens_by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    for c in ok:
        tokens_by_model[c["selected_model"] or "?"]["prompt_tokens"] += c[
            "prompt_tokens"
        ]
        tokens_by_model[c["selected_model"] or "?"]["completion_tokens"] += c[
            "completion_tokens"
        ]
    header_costs = [c for c in ok if c["cost"] is not None]
    header_total: dict[str, float] = defaultdict(float)
    for c in header_costs:
        header_total[c["cost_currency"] or "?"] = round(
            header_total[c["cost_currency"] or "?"] + c["cost"], 10
        )
    model_counts = Counter(c["selected_model"] or "?" for c in ok)
    return {
        "run_id": crew.run_id,
        "arm": crew.args.arm,
        "models_by_agent": crew.models,
        "base_url": crew.args.base_url,
        "max_tokens": crew.args.max_tokens,
        "temperature": crew.args.temperature,
        "calls": len(crew.calls),
        "failed_calls": len(crew.calls) - len(ok),
        "selected_model_counts": dict(model_counts),
        "frontier_calls": model_counts.get("frontier", 0),
        "frontier_share": (
            round(model_counts.get("frontier", 0) / len(ok), 4) if ok else None
        ),
        "decision_counts": dict(Counter(c["decision"] or "-" for c in ok)),
        "by_agent": by_agent,
        "tokens_by_model": dict(tokens_by_model),
        "prompt_tokens": sum(c["prompt_tokens"] for c in ok),
        "completion_tokens": sum(c["completion_tokens"] for c in ok),
        "reasoning_tokens": sum(c["reasoning_tokens"] for c in ok),
        "truncated_calls": sum(1 for c in ok if c["finish_reason"] == "length"),
        "capped_conversations": crew.capped,
        "wall_time_s": round(wall_time_s, 2),
        "model_time_s": round(sum(c["latency_s"] for c in crew.calls), 2),
        "cost_from_headers": {
            "basis": "configured pricing (x-vsr-cost), not a provider bill",
            "total": dict(header_total),
            "priced_calls": len(header_costs),
            "unpriced_calls": len(ok) - len(header_costs),
        },
        "routing_latency_ms": value_summary(
            [c["routing_latency_ms"] for c in ok if c["routing_latency_ms"] is not None]
        ),
        "router_metrics": router_metrics,
    }


def write_run(run_dir: Path, crew: Crew, review: str, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "calls.jsonl").open("w") as handle:
        for call in crew.calls:
            handle.write(json.dumps(call) + "\n")
    (run_dir / "conversations.json").write_text(
        json.dumps(crew.conversations, indent=2)
    )
    (run_dir / "review.md").write_text(review.strip() + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def print_summary(summary: dict[str, Any], run_dir: Path) -> None:
    metrics = summary["router_metrics"] or {}
    lines = [
        f"\n{summary['arm']}  ->  {run_dir}",
        f"  calls              {summary['calls']} ({summary['failed_calls']} failed)",
        f"  selected models    {summary['selected_model_counts']}",
        f"  tokens             {summary['prompt_tokens']} prompt / {summary['completion_tokens']} completion",
        f"  truncated / capped {summary['truncated_calls']} / {len(summary['capped_conversations'])}",
        f"  wall time          {summary['wall_time_s']} s",
        f"  cost (headers)     {summary['cost_from_headers']['total'] or 'no x-vsr-cost headers'}",
        f"  cost (/metrics)    {metrics.get('cost_total', 'not scraped')}",
        f"  routing ms         {summary['routing_latency_ms'] or metrics.get('routing_latency_ms_mean', 'n/a')}",
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=6,
    )
    run_dirs = []
    for repeat in range(1, args.repeat + 1):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{repeat}"
        crew = Crew(client, args, run_id)
        before = scrape_metrics(args.router_metrics_url)
        started = time.perf_counter()
        review = crew.run()
        wall = time.perf_counter() - started
        after = scrape_metrics(args.router_metrics_url)
        summary = summarize(crew, wall, metrics_delta(before, after))
        run_dir = args.out / args.arm / run_id
        write_run(run_dir, crew, review, summary)
        print_summary(summary, run_dir)
        run_dirs.append(run_dir)
    return run_dirs


if __name__ == "__main__":
    main()
