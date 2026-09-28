"""Native System One examples, card-code execution and scored-panel parity.

Run this file directly with an isolated interpreter (``python -I -B examples.py``)
so the package is imported from its own directory only; it needs nothing from
this repository. Subcommands:

  run      fixed Choice/Noul/Score requests through ``Decision2`` -> outputs JSON
  compare  two ``run`` outputs (cross-process / pre- vs post-download)
  card     execute the README's Python block and compare with a ``run`` output
  parity   package answers on gold-free panel prompts vs sealed scored predictions

``--site DIR`` puts an image directory of kernel packages on the path after the
package (images that expose FLA only through ``PYTHONPATH``, which ``-I``
drops); the card example then runs with exactly those directories as its
``PYTHONPATH``, like a user environment with the kernels installed.
``--require-kernels`` fails a Qwen3.5-family run unless the gated-delta and
causal-conv1d functions are bound to their kernels and a persisted Triton
autotune cache is set (the decoder track's ``v2/dec/runtime_check.py``).
Receipts hold hashes, counts and drift only, never panel text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "dev2-release-examples/1"
LONG_WORDS = ("amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "harbor")

EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "support-routing",
        "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "returns": "Refunds, replacements and damaged deliveries",
                    "billing": "Payments, invoices and charges",
                    "technical": "Product setup and faults",
                },
            },
            "receipt": {
                "type": "noul",
                "instructions": "Does the customer have a receipt?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["Routine", "Soon", "Today"],
            },
        },
    },
    {
        "id": "structured-state",
        "state": {
            "ticket": {
                "channel": "email",
                "plan": "enterprise",
                "message": "Our dashboard has been unavailable for two hours.",
            },
            "history": ["Outage reported at 09:10", "No workaround yet"],
        },
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "Rate the incident severity.",
                "criteria": ["Minor", "Moderate", "Major", "Critical", "Full outage"],
            },
            "escalate": {
                "type": "noul",
                "instructions": "Should this be escalated to the on-call engineer?",
                "criteria": {
                    "false": "No escalation is needed",
                    "true": "Escalate now",
                },
            },
        },
    },
    {
        "id": "policy-check",
        "state": "Policy: refunds are allowed within 30 days of delivery with proof of purchase. The item was delivered 45 days ago.",
        "questions": {
            "eligible": {
                "type": "noul",
                "instructions": "Is the customer eligible for a refund under this policy?",
            },
            "action": {
                "type": "choice",
                "instructions": "What should the agent do?",
                "criteria": {
                    "approve": "Approve the refund",
                    "decline": "Explain that the refund window has closed",
                    "escalate": "Escalate to a supervisor",
                },
            },
        },
    },
    {
        "id": "multilingual",
        "state": "客户说：昨天收到的耳机左边没有声音，想换一个新的。",
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which queue fits this message?",
                "criteria": {
                    "hardware": "Defective or broken devices",
                    "shipping": "Late or lost deliveries",
                    "account": "Login and account access",
                },
            }
        },
    },
    {
        "id": "null-description",
        "state": "Please cancel my subscription at the end of this month.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the user want?",
                "criteria": {"cancel": None, "upgrade": None, "pause": None},
            }
        },
    },
]


def over_budget_example(cap: int) -> dict[str, Any]:
    words = [LONG_WORDS[i % len(LONG_WORDS)] for i in range(2 * cap + 64)]
    return {
        "id": "over-budget",
        "state": " ".join(words),
        "questions": {
            "weather": {
                "type": "noul",
                "instructions": "Is this text about the weather?",
            },
            "tone": {
                "type": "score",
                "instructions": "Rate the tone.",
                "criteria": ["Negative", "Neutral", "Positive"],
            },
        },
    }


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_exclusive(path: Path, value: Any) -> str:
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(data)
    return hashlib.sha256(data).hexdigest()


def load_package(
    package: Path, device: str | None, threads: int | None, base_path: str | None
):
    sys.path.insert(0, str(package.resolve()))
    from decision2 import Decision2

    started = time.perf_counter()
    model = Decision2.from_pretrained(
        package, device=device, threads=threads, base_path=base_path
    )
    return model, time.perf_counter() - started


def runtime_versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "peft", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    try:
        import torch

        out["hip"] = torch.version.hip
        out["threads"] = torch.get_num_threads()
    except Exception:
        pass
    return out


RUNTIME_CHECK = Path(__file__).resolve().parents[1] / "dec" / "runtime_check.py"


def kernel_runtime(required: bool) -> dict[str, Any] | None:
    """Qwen3.5 kernel bindings and autotune cache; raises if required and not met."""
    if not required:
        return None
    spec = importlib.util.spec_from_file_location("dev2_runtime_check", RUNTIME_CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = module.runtime_identity()
    problems = module.violations(identity)
    if problems:
        raise RuntimeError("Kernel runtime check failed: " + "; ".join(problems))
    return {**identity, "runtime_check_sha256": sha_file(RUNTIME_CHECK)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, load_seconds = load_package(
        args.package, args.device, args.threads, args.base_path
    )
    kernels = kernel_runtime(args.require_kernels)
    examples = [*EXAMPLES, over_budget_example(model.max_input_tokens)]
    outputs = []
    for example in examples:
        started = time.perf_counter()
        response = model.system_one(
            state=example["state"], questions=example["questions"]
        )
        outputs.append(
            {
                "id": example["id"],
                "input_sha256": digest(
                    {"state": example["state"], "questions": example["questions"]}
                ),
                "response": response,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    checks = check_outputs(examples, outputs, model.max_input_tokens)
    result = {
        "schema": SCHEMA,
        "mode": "run",
        "package_manifest_sha256": sha_file(args.package / "MODEL_MANIFEST.json"),
        "model_name": model.model_name,
        "profile": model.manifest["profile"],
        "device": str(getattr(model.backend, "device", args.device)),
        "loaded_parameters": model.backend.parameter_count(),
        "load_seconds": load_seconds,
        "runtime": {**runtime_versions(), "sites": args.site, "kernels": kernels},
        "outputs": outputs,
        "answers_sha256": digest(
            [{"id": o["id"], "response": o["response"]} for o in outputs]
        ),
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
    }
    return result


def check_outputs(
    examples: list[dict[str, Any]], outputs: list[dict[str, Any]], cap: int
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for example, output in zip(examples, outputs):
        answers = output["response"]["answers"]
        problems = []
        if list(answers) != list(example["questions"]):
            problems.append("question IDs or order differ")
        for qid, question in example["questions"].items():
            answer = answers.get(qid) or {}
            if example["id"] == "over-budget":
                if "error" not in answer:
                    problems.append(f"{qid}: over-budget input was answered")
                continue
            problems += [f"{qid}: {p}" for p in answer_problems(question, answer)]
        checks[example["id"]] = {"passed": not problems, "problems": problems}
    return checks


def answer_problems(question: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    kind = question["type"]
    if answer.get("type") != kind or "error" in answer:
        return [f"invalid {kind} answer"]
    if kind == "noul":
        value = answer.get("noul")
        return (
            []
            if isinstance(value, float) and 0 <= value <= 1
            else ["noul is not a probability"]
        )
    keys = (
        list(question["criteria"])
        if kind == "choice"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or list(probs) != keys:
        return [f"{kind} probabilities do not cover the supplied candidates in order"]
    values = list(probs.values())
    if (
        not all(isinstance(v, float) and 0 <= v <= 1 for v in values)
        or abs(sum(values) - 1) > 1e-4
    ):
        return [f"{kind} probabilities do not form a distribution"]
    if kind == "choice" and answer.get("choice") not in keys:
        return ["choice is not a supplied option"]
    if kind == "score":
        score = answer.get("score")
        if not isinstance(score, float) or not 0 <= score <= len(keys) - 1:
            return ["score is outside the level range"]
    return []


def category(answer: Any) -> Any:
    if not isinstance(answer, dict) or "error" in answer:
        return (
            "invalid",
            (answer or {}).get("error") if isinstance(answer, dict) else None,
        )
    if answer.get("type") == "noul" or "noul" in answer:
        value = answer.get("noul")
        return ("noul", None if value is None or value == 0.5 else value > 0.5)
    if "choice" in answer:
        return ("choice", answer["choice"])
    probs = answer.get("probabilities") or {}
    if probs:
        best = max(probs.values())
        winners = sorted(k for k, v in probs.items() if abs(v - best) <= 1e-8)
        return ("score", winners[0] if len(winners) == 1 else None)
    return ("score", None)


def numbers(answer: Any) -> dict[str, float]:
    if not isinstance(answer, dict):
        return {}
    out = {}
    if isinstance(answer.get("noul"), (int, float)):
        out["noul"] = float(answer["noul"])
    if isinstance(answer.get("score"), (int, float)):
        out["score"] = float(answer["score"])
    for key, value in (answer.get("probabilities") or {}).items():
        if isinstance(value, (int, float)):
            out[f"p/{key}"] = float(value)
    return out


def compare_answers(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    slots = changed = missing = 0
    drift = 0.0
    for qid in sorted(set(left) | set(right)):
        slots += 1
        if qid not in left or qid not in right:
            missing += 1
            continue
        if category(left[qid]) != category(right[qid]):
            changed += 1
        a, b = numbers(left[qid]), numbers(right[qid])
        if set(a) != set(b):
            changed += 1
        for key in set(a) & set(b):
            drift = max(drift, abs(a[key] - b[key]))
    return {
        "slots": slots,
        "category_changes": changed,
        "missing": missing,
        "max_abs_drift": drift,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    per_example = {}
    totals = {"slots": 0, "category_changes": 0, "missing": 0, "max_abs_drift": 0.0}
    left_out = {o["id"]: o for o in left["outputs"]}
    right_out = {o["id"]: o for o in right["outputs"]}
    for key in sorted(set(left_out) | set(right_out)):
        a = (left_out.get(key) or {}).get("response", {}).get("answers", {})
        b = (right_out.get(key) or {}).get("response", {}).get("answers", {})
        result = compare_answers(a, b)
        per_example[key] = result
        for name in ("slots", "category_changes", "missing"):
            totals[name] += result[name]
        totals["max_abs_drift"] = max(totals["max_abs_drift"], result["max_abs_drift"])
    identical = left["answers_sha256"] == right["answers_sha256"]
    passed = (
        totals["category_changes"] == 0
        and totals["missing"] == 0
        and totals["max_abs_drift"] <= args.tolerance
        and left["loaded_parameters"] == right["loaded_parameters"]
        and left["passed"]
        and right["passed"]
    )
    return {
        "schema": SCHEMA,
        "mode": "compare",
        "left": {
            "sha256": sha_file(args.left),
            "answers_sha256": left["answers_sha256"],
            "manifest": left["package_manifest_sha256"],
        },
        "right": {
            "sha256": sha_file(args.right),
            "answers_sha256": right["answers_sha256"],
            "manifest": right["package_manifest_sha256"],
        },
        "same_package_manifest": left["package_manifest_sha256"]
        == right["package_manifest_sha256"],
        "bit_identical_answers": identical,
        "tolerance": args.tolerance,
        "totals": totals,
        "per_example": per_example,
        "passed": passed,
    }


def card(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the README's Python example exactly as a user would, in a fresh process."""
    readme = (args.package / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    if len(blocks) != 1:
        raise ValueError("The card must contain exactly one Python example")
    code = blocks[0]
    name = args.package.resolve().name
    if f'"{name}"' not in code:
        raise ValueError("The card example does not load this package directory")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = next(o for o in reference["outputs"] if o["id"] == EXAMPLES[0]["id"])
    with tempfile.TemporaryDirectory() as scratch:
        script = Path(scratch) / "card_example.py"
        script.write_text(code, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        flags = ["-I", "-B"]
        if args.site:
            # Only the named kernel directories, as if installed in the user's environment.
            env["PYTHONPATH"] = os.pathsep.join(args.site)
            flags = ["-s", "-B"]
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, *flags, str(script)],
            cwd=args.package.resolve().parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    printed = None
    if completed.returncode == 0:
        try:
            printed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            printed = None
    comparison = compare_answers(printed or {}, expected["response"]["answers"])
    return {
        "schema": SCHEMA,
        "mode": "card",
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "readme_sha256": sha_file(args.package / "README.md"),
        "interpreter_flags": flags,
        "sites": args.site,
        "exit_code": completed.returncode,
        "stderr_tail": completed.stderr[-2000:] if completed.returncode else "",
        "seconds": time.perf_counter() - started,
        "comparison": comparison,
        "passed": completed.returncode == 0
        and printed is not None
        and comparison["category_changes"] == 0
        and comparison["missing"] == 0
        and comparison["max_abs_drift"] <= args.tolerance,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parity(args: argparse.Namespace) -> dict[str, Any]:
    """Answers of the package vs the sealed same-panel predictions on the first N prompts."""
    model, load_seconds = load_package(
        args.package, args.device, args.threads, args.base_path
    )
    kernels = kernel_runtime(args.require_kernels)
    panels = {}
    for spec in args.panel:
        name, prompts_path, predictions_path, count = spec.split(":")
        prompts = read_jsonl(Path(prompts_path))[: int(count)]
        sealed = {row["id"]: row for row in read_jsonl(Path(predictions_path))}
        totals = {
            "prompts": 0,
            "slots": 0,
            "category_changes": 0,
            "missing": 0,
            "max_abs_drift": 0.0,
            "input_mismatch": 0,
        }
        started = time.perf_counter()
        for prompt in prompts:
            payload = {"state": prompt["state"], "questions": prompt["questions"]}
            reference = sealed.get(prompt["id"]) or {}
            if (
                reference.get("source_input_sha256")
                != hashlib.sha256(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            ):
                totals["input_mismatch"] += 1
            response = model.system_one(
                state=prompt["state"], questions=prompt["questions"]
            )
            result = compare_answers(
                response["answers"], reference.get("answers") or {}
            )
            totals["prompts"] += 1
            for key in ("slots", "category_changes", "missing"):
                totals[key] += result[key]
            totals["max_abs_drift"] = max(
                totals["max_abs_drift"], result["max_abs_drift"]
            )
        panels[name] = {
            **totals,
            "prompts_sha256": sha_file(Path(prompts_path)),
            "predictions_sha256": sha_file(Path(predictions_path)),
            "ids_sha256": digest([p["id"] for p in prompts]),
            "seconds": time.perf_counter() - started,
        }
    passed = all(
        p["category_changes"] == 0
        and p["missing"] == 0
        and p["input_mismatch"] == 0
        and p["max_abs_drift"] <= args.tolerance
        for p in panels.values()
    )
    return {
        "schema": SCHEMA,
        "mode": "parity",
        "package_manifest_sha256": sha_file(args.package / "MODEL_MANIFEST.json"),
        "device": str(getattr(model.backend, "device", args.device)),
        "runtime": {**runtime_versions(), "sites": args.site, "kernels": kernels},
        "load_seconds": load_seconds,
        "tolerance": args.tolerance,
        "panels": panels,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "card", "parity"):
        p = sub.add_parser(name)
        p.add_argument("--package", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--site",
            action="append",
            default=[],
            help="image directory of kernel packages to import after the package",
        )
        if name != "card":
            p.add_argument("--device")
            p.add_argument("--threads", type=int)
            p.add_argument("--base-path")
            p.add_argument("--require-kernels", action="store_true")
    sub.choices["card"].add_argument("--reference", type=Path, required=True)
    sub.choices["card"].add_argument("--tolerance", type=float, default=0.0)
    sub.choices["parity"].add_argument(
        "--panel",
        action="append",
        required=True,
        help="NAME:PROMPTS_JSONL:SEALED_PREDICTIONS_JSONL:COUNT",
    )
    sub.choices["parity"].add_argument("--tolerance", type=float, default=1e-4)
    cmp = sub.add_parser("compare")
    cmp.add_argument("left", type=Path)
    cmp.add_argument("right", type=Path)
    cmp.add_argument("--output", type=Path, required=True)
    cmp.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    handler = {"run": run, "compare": compare, "card": card, "parity": parity}[
        args.command
    ]
    for site in reversed(getattr(args, "site", [])):
        if not Path(site).is_absolute() or not Path(site).is_dir():
            raise ValueError(f"--site needs an existing absolute directory: {site}")
        # load_package later inserts the package itself in front of these.
        sys.path.insert(0, site)
    result = handler(args)
    write_exclusive(args.output, result)
    print(json.dumps({"mode": args.command, "passed": result["passed"]}))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
