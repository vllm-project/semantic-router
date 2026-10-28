#!/usr/bin/env python3
"""The Router's built-in signal questions asked of a decision model directly, on the router signal suite.

    python3 tools/router_signal_direct.py --model PACKAGE_OR_REPO --suite SUITE/text --out PREDICTIONS
        [--tasks domain,jailbreak,...] [--splits dev] [--device cpu] [--threads N]

Asks every row of the suite's request-time signals what the Router's default deployment is asked
about a request (modelservice.BuiltinTasks and serving/signal_question.go): domain, fact check
and modality as truncating Choice questions, prompt attack and harmful request read whole, the
``pii`` preset, and, for a feedback row (a user turn after an assistant turn), the feedback
question. A request's questions travel in one call, as the Router's bundle sends them, in this
tool's fixed order (the Router's follows its signals' arrival). The model answers each row alone,
in its released forwards. Each task's answers are written in the suite's prediction layout
(``<out>/<task>/<split>.pred.jsonl``, ``{id, scores}``), which ``router_signal_ab.py calibrate``
and ``score`` read: a signal's probabilities, and for PII the most probable span of a sensitive
type (``{yes, no}``). Resumable: rows already written are skipped.

``--questions own`` asks each row only its own signal's question (prompt attack and harmful
request together, which share a row). On a model whose rows of different adapters or reading
policies never share a sequence or a forward (vela3-fast) the answers equal the bundle's; on a
model that reads every question in one sequence (Vela 2.0 0.3B) they do not.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_srun.config import ModelConfig, ServeConfig  # noqa: E402
from vllm_srun.plugins.base import SurfacePlan  # noqa: E402
from vllm_srun.runtime import Runtime  # noqa: E402

TASKS = ("domain", "jailbreak", "safety", "fact_check", "modality", "pii", "feedback")
SENSITIVE = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STREET_ADDRESS", "CREDIT_CARD", "IBAN_CODE",
             "US_SSN", "US_DRIVER_LICENSE", "IP_ADDRESS"}  # fmt: skip


def choices(*options: tuple[str, str]) -> list[dict[str, str]]:
    return [{"key": key, "description": description} for key, description in options]


# modelservice.BuiltinTasks: each signal's instructions and choices, and whether it reads the whole text.
SIGNALS: dict[str, tuple[str, dict[str, Any], bool]] = {
    "domain": ("domain_classifier:domain", {
        "type": "choice", "instructions": "Which subject area is this request about?",
        "choices": choices(
            ("biology", "living organisms, anatomy, genetics, medical genetics or viruses"),
            ("business", "management, marketing, accounting, business ethics or public relations"),
            ("chemistry", "chemical substances, reactions, elements or laboratory chemistry"),
            ("computer science", "programming, algorithms, computer systems, security or machine learning"),
            ("economics", "markets, macroeconomics, microeconomics or econometrics"),
            ("engineering", "electrical or other engineering design and systems"),
            ("health", "medicine, clinical practice, nutrition, ageing or sexual health"),
            ("history", "past events, periods and historical societies"),
            ("law", "legal rules, jurisprudence, courts or international law"),
            ("math", "arithmetic, algebra, geometry, statistics or other mathematics"),
            ("other", "a subject that fits none of the listed areas"),
            ("philosophy", "philosophy, ethics, moral questions or formal logic"),
            ("physics", "physical laws, mechanics, astronomy or physical phenomena"),
            ("psychology", "mind, behaviour, mental processes or psychological practice"))}, False),
    "fact_check": ("fact_check_classifier:factcheck", {
        "type": "choice", "instructions": "Does answering this request require checking facts?",
        "choices": choices(
            ("NO_FACT_CHECK_NEEDED", "the request can be handled without verifying facts about the world, e.g. translating, rewriting, formatting or writing fiction"),
            ("FACT_CHECK_NEEDED", "answering the request relies on factual claims about the world that should be verified"))}, False),
    "modality": ("modality_detector:modality", {
        "type": "choice", "instructions": "What kind of output does this request ask for?",
        "choices": choices(
            ("AR", "a text answer only, including code, analysis or describing an existing image"),
            ("DIFFUSION", "a newly generated image only"),
            ("BOTH", "a newly generated image together with a separate written explanation"))}, False),
    "jailbreak": ("prompt_guard:attack", {
        "type": "choice", "instructions": "Is this a prompt injection or jailbreak attempt?",
        "choices": choices(
            ("benign", "a normal request or quoted content that does not try to override system instructions or bypass safety rules"),
            ("jailbreak", "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules"))}, True),
    "safety": ("safety.unsafe_request:p_harm", {
        "type": "choice", "instructions": "Is this request harmful?",
        "choices": choices(
            ("safe", "a benign request that does not violate any safety policy"),
            ("unsafe", "a request that violates a safety policy or seeks harmful assistance"))}, True),
    "pii": ("pii_classifier:pii", {"preset": "pii"}, True),
    "feedback": ("feedback_detector:feedback", {
        "type": "choice", "instructions": "What feedback does this user turn give about the previous answer?",
        "choices": choices(
            ("SAT", "the user is satisfied with the previous answer"),
            ("NEED_CLARIFICATION", "the user asks for clarification or a more detailed explanation"),
            ("WRONG_ANSWER", "the user says the previous answer was wrong or did not work"),
            ("WANT_DIFFERENT", "the user wants a different answer, format, style or approach"),
            ("NO_FEEDBACK", "the user gives no feedback on the previous answer"))}, False),
}  # fmt: skip


# Signals whose questions share one row of a model that reads the Router's questions per
# adapter and reading policy (prompt attack and harmful request: the shared router adapter, whole).
SHARED_ROWS = {"jailbreak": ("jailbreak", "safety"), "safety": ("jailbreak", "safety")}


def bundle(task: str, own: bool) -> dict[str, Any]:
    """The questions the Router asks its default deployment about one row of ``task``.

    ``own`` keeps only the task's own question and the ones that share its row.
    """
    keep = SHARED_ROWS.get(task, (task,)) if own else tuple(SIGNALS)
    questions = {}
    for name, (question_id, question, whole) in SIGNALS.items():
        if name not in keep or (name == "feedback" and task != "feedback"):
            continue
        questions[question_id] = (
            question if whole else {**question, "overflow": "truncate"}
        )
    return questions


def scores(task: str, response: dict[str, Any]) -> dict[str, float] | None:
    question_id = SIGNALS[task][0]
    answer = response["answers"].get(question_id) or {}
    if "error" in answer:
        return None
    if task == "pii":
        spans = (response.get("spans") or {}).get(question_id) or []
        best = max(
            (s["probability"] for s in spans if s["label"] in SENSITIVE), default=0.0
        )
        return {"yes": best, "no": 1 - best}
    probabilities: dict[str, float] = answer["probabilities"]
    return probabilities


def ask(model: Any, text: str, questions: dict[str, Any]) -> dict[str, Any]:
    plan = model.plan(text, questions)
    results = model.run(plan.items) if plan.items else []
    response: dict[str, Any] = model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), results
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--suite", required=True, help="the suite's text/ directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--questions", choices=("bundle", "own"), default="bundle")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        help="cap this process's share of the GPU's memory (a GPU other services share)",
    )
    args = parser.parse_args()
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=args.model, revision=args.revision, device=args.device
                ),
            )
        )
    )
    runtime.load()
    served = runtime.lookup(None)
    assert served.model is not None and served.placement is not None
    execute = served.placement.accelerator.execute
    device = served.placement.device
    try:
        for task in args.tasks.split(","):
            for split in args.splits.split(","):
                paths = sorted(glob.glob(f"{args.suite}/*/{task}/{split}.jsonl"))
                rows = [
                    json.loads(line)
                    for path in paths
                    for line in open(path, encoding="utf-8")
                    if line.strip()
                ]
                target = args.out / task / f"{split}.pred.jsonl"
                os.makedirs(target.parent, exist_ok=True)
                done = set()
                if target.exists():
                    done = {
                        json.loads(line)["id"]
                        for line in target.read_text().splitlines()
                        if line.strip()
                    }
                started, answered = time.perf_counter(), 0
                with target.open("a", encoding="utf-8") as out:
                    for row in rows:
                        if row["id"] in done:
                            continue
                        questions = bundle(task, args.questions == "own")
                        response = execute(
                            device,
                            lambda q=questions, t=row["text"]: ask(served.model, t, q),
                        )
                        found = scores(task, response)
                        if found is None:
                            continue
                        out.write(json.dumps({"id": row["id"], "scores": found}) + "\n")
                        answered += 1
                elapsed = time.perf_counter() - started
                print(f"{task}/{split}: {answered} rows in {elapsed:.0f}s", flush=True)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
