"""Native System One examples, card-code execution and scored-panel parity.

Run this file directly with an isolated interpreter (``python -I -B examples.py``)
so the package is imported from its own directory only; it needs nothing from
this repository. Subcommands:

  run      fixed Choice/Noul/Score requests through ``Decision2`` -> outputs JSON
  compare  two ``run`` outputs (cross-process / pre- vs post-download)
  card     execute the README's native Python block and compare with a ``run`` output
  parity   package answers on gold-free panel prompts vs sealed scored predictions
  automap         AutoConfig / AutoTokenizer / AutoModel / pipeline("decision") with
                  trust_remote_code vs a native ``run`` output (bit-identical answers)
  automap-card    the README's Transformers block (package path, or ``--hub`` as written)
  automap-parity  ``parity`` through AutoModel with trust_remote_code
  compare-answers per-prompt answers of two ``--answers`` runs: changes and drift

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
    package: Path,
    device: str | None,
    threads: int | None,
    base_path: str | None,
    fp32_master: bool = False,
):
    sys.path.insert(0, str(package.resolve()))
    from decision2 import Decision2

    # Runtimes before the BF16-resident one take no such keyword.
    options = {"bf16_resident": False} if fp32_master else {}
    started = time.perf_counter()
    model = Decision2.from_pretrained(
        package, device=device, threads=threads, base_path=base_path, **options
    )
    return model, time.perf_counter() - started


def parameter_dtypes(backend: Any) -> dict[str, int] | None:
    """Loaded elements by dtype, for torch-module backends."""
    module = getattr(backend, "model", None)
    if not callable(getattr(module, "parameters", None)):
        return None
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        key = str(parameter.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + parameter.numel()
    return dict(sorted(counts.items()))


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
        args.package, args.device, args.threads, args.base_path, args.fp32_master
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
        "runtime": {
            **runtime_versions(),
            "sites": args.site,
            "kernels": kernels,
            "residency": getattr(model.backend, "residency", None),
            "parameter_dtypes": parameter_dtypes(model.backend),
            "fp32_master": args.fp32_master,
        },
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


TRANSFORMERS_BLOCK = "trust_remote_code=True"


def card_block(readme: str, transformers: bool) -> str:
    """The card's one native Python example, or its one Transformers (trust_remote_code) example."""
    blocks = [
        b
        for b in re.findall(r"```python\n(.*?)```", readme, flags=re.S)
        if (TRANSFORMERS_BLOCK in b) == transformers
    ]
    if len(blocks) != 1:
        kind = "Transformers" if transformers else "native"
        raise ValueError(f"The card must contain exactly one {kind} Python example")
    return blocks[0]


def run_card_code(
    code: str, cwd: Path, sites: list[str], env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, list[str], float]:
    """Run card code in a fresh interpreter; ``sites`` only, as if installed in the user's environment."""
    with tempfile.TemporaryDirectory() as scratch:
        script = Path(scratch) / "card_example.py"
        script.write_text(code, encoding="utf-8")
        env = env or {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        flags = ["-I", "-B"]
        if sites:
            env["PYTHONPATH"] = os.pathsep.join(sites)
            flags = ["-s", "-B"]
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, *flags, str(script)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    return completed, flags, started


def card(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the README's native Python example exactly as a user would, in a fresh process."""
    readme = (args.package / "README.md").read_text(encoding="utf-8")
    code = card_block(readme, transformers=False)
    name = args.package.resolve().name
    if f'"{name}"' not in code:
        raise ValueError("The card example does not load this package directory")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = next(o for o in reference["outputs"] if o["id"] == EXAMPLES[0]["id"])
    completed, flags, started = run_card_code(
        code, args.package.resolve().parent, args.site
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


def panel_parity(
    system_one: Any, specs: list[str], answers: Path | None = None
) -> dict[str, Any]:
    """``system_one`` answers vs the sealed predictions of each NAME:PROMPTS:PREDICTIONS:COUNT panel.

    ``answers`` (optional) receives one JSON line per prompt, ``{"panel", "id", "answers"}``, for a
    direct comparison of two runtimes (``compare-answers``); it stays on the node like the predictions.
    """
    panels = {}
    out = answers.open("x", encoding="utf-8") if answers else None
    try:
        for spec in specs:
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
                response = system_one(
                    state=prompt["state"], questions=prompt["questions"]
                )
                if out is not None:
                    out.write(
                        canonical(
                            {
                                "panel": name,
                                "id": prompt["id"],
                                "answers": response["answers"],
                            }
                        )
                        + "\n"
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
    finally:
        if out is not None:
            out.close()
    return panels


def parity(args: argparse.Namespace) -> dict[str, Any]:
    """Answers of the package vs the sealed same-panel predictions on the first N prompts."""
    model, load_seconds = load_package(
        args.package, args.device, args.threads, args.base_path
    )
    kernels = kernel_runtime(args.require_kernels)
    panels = panel_parity(model.system_one, args.panel, args.answers)
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
        "runtime": {
            **runtime_versions(),
            "sites": args.site,
            "kernels": kernels,
            "residency": getattr(model.backend, "residency", None),
            "parameter_dtypes": parameter_dtypes(model.backend),
        },
        "load_seconds": load_seconds,
        "tolerance": args.tolerance,
        "panels": panels,
        "passed": passed,
    }


def automap_load(
    package: str | Path, device: str | None, threads: int | None, base_path: str | None
):
    """The package through stock Transformers: AutoModel with trust_remote_code, nothing on sys.path."""
    from transformers import AutoModel

    options = {"device": device, "threads": threads, "base_path": base_path}
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(package),
        trust_remote_code=True,
        **{k: v for k, v in options.items() if v is not None},
    )
    return model, time.perf_counter() - started


def free(model: Any) -> None:
    import gc

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def automap(args: argparse.Namespace) -> dict[str, Any]:
    """AutoConfig, AutoTokenizer, AutoModel and pipeline("decision") with trust_remote_code on the
    package, each answer compared with the native ``run`` receipt (bit-identical required).
    """
    from transformers import AutoConfig, AutoTokenizer, pipeline
    from transformers.utils import find_adapter_config_file, is_peft_available

    package = str(args.package.resolve())
    manifest = json.loads(
        (args.package / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = {o["id"]: o["response"] for o in reference["outputs"]}
    checks: dict[str, Any] = {}
    config = AutoConfig.from_pretrained(package, trust_remote_code=True)
    checks["config"] = {
        "class": type(config).__name__,
        "model_type": config.model_type,
        "passed": type(config).__name__ == "Decision2Config"
        and config.model_type == "decision2"
        and getattr(config, "model_name", None) == manifest["model_name"],
    }
    tokenizer = AutoTokenizer.from_pretrained(package, trust_remote_code=True)
    # An adapter_config.json at the root would make AutoModel load the adapter's base instead.
    root_adapter = find_adapter_config_file(package)
    checks["peft_detection"] = {
        "peft_available": is_peft_available(),
        "root_adapter_config": root_adapter,
        "passed": root_adapter is None,
    }
    model, load_seconds = automap_load(
        package, args.device, args.threads, args.base_path
    )
    kernels = kernel_runtime(args.require_kernels)
    runtime_tokenizer = model.runtime.backend.tokenizer
    texts = [
        e["state"] if isinstance(e["state"], str) else canonical(e["state"])
        for e in EXAMPLES
    ]
    same_ids = all(
        tokenizer.encode(t, add_special_tokens=False)
        == runtime_tokenizer.encode(t, add_special_tokens=False)
        for t in texts
    )
    checks["tokenizer"] = {"class": type(tokenizer).__name__, "passed": same_ids}
    checks["model"] = {
        "class": type(model).__name__,
        "module": type(model).__module__.split(".")[0],
        "runtime_module": type(model.runtime).__module__.rsplit(".", 2)[-2],
        "num_parameters": model.num_parameters(),
        "cpu_reference_layers": model.cpu_reference_layers,
        "passed": type(model).__name__ == "Decision2Model"
        and type(model).__module__.startswith("transformers_modules.")
        and model.num_parameters() == manifest["parameters"]["loaded"]
        and model.model_name == manifest["model_name"],
    }
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
                "response": response,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    first = EXAMPLES[0]
    checks["forward"] = {
        "passed": model(state=first["state"], questions=first["questions"])
        == outputs[0]["response"]
    }
    device = str(model.device)
    free(model)
    decide = pipeline(
        "decision",
        model=package,
        trust_remote_code=True,
        device_map=args.device,
        model_kwargs={
            k: v
            for k, v in (("threads", args.threads), ("base_path", args.base_path))
            if v is not None
        },
    )
    requests = [
        {"state": e["state"], "questions": e["questions"]} for e in EXAMPLES[:2]
    ]
    single = decide(requests[0])
    batch = decide(requests)
    keywords = decide(state=first["state"], questions=first["questions"])
    checks["pipeline"] = {
        "class": type(decide).__name__,
        "passed": type(decide).__name__ == "Decision2Pipeline"
        and single == keywords == expected[first["id"]]
        and batch == [expected[e["id"]] for e in EXAMPLES[:2]],
    }
    free(decide)
    comparison = {
        o["id"]: compare_answers(o["response"]["answers"], expected[o["id"]]["answers"])
        for o in outputs
    }
    answers = digest([{"id": o["id"], "response": o["response"]} for o in outputs])
    identical = answers == reference["answers_sha256"]
    checks["answers"] = {"bit_identical_to_native": identical, "passed": identical}
    return {
        "schema": SCHEMA,
        "mode": "automap",
        "package_manifest_sha256": sha_file(args.package / "MODEL_MANIFEST.json"),
        "reference_sha256": sha_file(args.reference),
        "device": device,
        "load_seconds": load_seconds,
        "runtime": {**runtime_versions(), "sites": args.site, "kernels": kernels},
        "checks": checks,
        "outputs": outputs,
        "answers_sha256": answers,
        "comparison": comparison,
        "passed": all(c["passed"] for c in checks.values()),
    }


def automap_card(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the README's Transformers example in a fresh interpreter and compare with the native run.

    Without ``--hub`` the repository ID is replaced by the package path (no network); with ``--hub`` the
    code runs exactly as written against the Hub and ``--expect-revision`` must be the downloaded commit.
    """
    readme = (args.package / "README.md").read_text(encoding="utf-8")
    code = card_block(readme, transformers=True)
    repo = json.loads(
        (args.package / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )["repo_id"]
    if f'"{repo}"' not in code:
        raise ValueError("The Transformers example does not load this repository")
    if not args.hub:
        code = code.replace(f'"{repo}"', json.dumps(str(args.package.resolve())))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = next(o for o in reference["outputs"] if o["id"] == EXAMPLES[0]["id"])
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    with tempfile.TemporaryDirectory() as scratch:
        completed, flags, started = run_card_code(code, Path(scratch), args.site, env)
    printed = None
    if completed.returncode == 0:
        try:
            printed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            printed = None
    comparison = compare_answers(printed or {}, expected["response"]["answers"])
    downloaded = None
    if args.hub:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound

        try:
            repos = scan_cache_dir().repos
        except CacheNotFound:
            repos = []
        downloaded = sorted(
            revision.commit_hash
            for cached in repos
            if cached.repo_id == repo
            for revision in cached.revisions
        )
    return {
        "schema": SCHEMA,
        "mode": "automap-card",
        "hub": args.hub,
        "code_sha256": hashlib.sha256(
            card_block(readme, transformers=True).encode("utf-8")
        ).hexdigest(),
        "readme_sha256": sha_file(args.package / "README.md"),
        "interpreter_flags": flags,
        "sites": args.site,
        "exit_code": completed.returncode,
        "stderr_tail": (
            completed.stderr[-2000:] if completed.returncode or printed is None else ""
        ),
        "seconds": time.perf_counter() - started,
        "downloaded_revisions": downloaded,
        "runtime": runtime_versions(),
        "comparison": comparison,
        "passed": completed.returncode == 0
        and printed is not None
        and comparison["category_changes"] == 0
        and comparison["missing"] == 0
        and comparison["max_abs_drift"] <= args.tolerance
        and (not args.hub or downloaded == [args.expect_revision]),
    }


def automap_parity(args: argparse.Namespace) -> dict[str, Any]:
    """``parity`` through AutoModel with trust_remote_code instead of the native import."""
    model, load_seconds = automap_load(
        args.package, args.device, args.threads, args.base_path
    )
    kernels = kernel_runtime(args.require_kernels)
    panels = panel_parity(model.system_one, args.panel, args.answers)
    passed = all(
        p["category_changes"] == 0
        and p["missing"] == 0
        and p["input_mismatch"] == 0
        and p["max_abs_drift"] <= args.tolerance
        for p in panels.values()
    )
    return {
        "schema": SCHEMA,
        "mode": "automap-parity",
        "package_manifest_sha256": sha_file(args.package / "MODEL_MANIFEST.json"),
        "device": str(model.device),
        "runtime": {
            **runtime_versions(),
            "sites": args.site,
            "kernels": kernels,
            "cpu_reference_layers": model.cpu_reference_layers,
        },
        "load_seconds": load_seconds,
        "tolerance": args.tolerance,
        "panels": panels,
        "passed": passed,
    }


def compare_answer_files(args: argparse.Namespace) -> dict[str, Any]:
    """Per-prompt answers of two runs (``--answers`` files) on the same panels: changes and drift."""
    left, right = (
        {(r["panel"], r["id"]): r["answers"] for r in read_jsonl(p)}
        for p in (args.left, args.right)
    )
    panels: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left) | set(right)):
        totals = panels.setdefault(
            key[0],
            {
                "prompts": 0,
                "identical_prompts": 0,
                "slots": 0,
                "category_changes": 0,
                "missing": 0,
                "max_abs_drift": 0.0,
            },
        )
        totals["prompts"] += 1
        if key not in left or key not in right:
            totals["missing"] += 1
            continue
        totals["identical_prompts"] += canonical(left[key]) == canonical(right[key])
        result = compare_answers(left[key], right[key])
        for name in ("slots", "category_changes", "missing"):
            totals[name] += result[name]
        totals["max_abs_drift"] = max(totals["max_abs_drift"], result["max_abs_drift"])
    passed = bool(panels) and all(
        p["category_changes"] == 0
        and p["missing"] == 0
        and p["max_abs_drift"] <= args.tolerance
        for p in panels.values()
    )
    return {
        "schema": SCHEMA,
        "mode": "compare-answers",
        "left_sha256": sha_file(args.left),
        "right_sha256": sha_file(args.right),
        "tolerance": args.tolerance,
        "panels": panels,
        "max_abs_drift": max(
            (p["max_abs_drift"] for p in panels.values()), default=0.0
        ),
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
    sub.choices["run"].add_argument(
        "--fp32-master",
        action="store_true",
        help="load with bf16_resident=False: FP32 Linear weights cast on every call",
    )
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
    for name in ("automap", "automap-card", "automap-parity"):
        p = sub.add_parser(name)
        p.add_argument("--package", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--site", action="append", default=[])
        if name != "automap-card":
            p.add_argument("--device")
            p.add_argument("--threads", type=int)
            p.add_argument("--base-path")
            p.add_argument("--require-kernels", action="store_true")
        if name != "automap-parity":
            p.add_argument("--reference", type=Path, required=True)
    sub.choices["automap-card"].add_argument("--tolerance", type=float, default=0.0)
    sub.choices["automap-card"].add_argument("--hub", action="store_true")
    sub.choices["automap-card"].add_argument("--expect-revision")
    sub.choices["automap-parity"].add_argument(
        "--panel",
        action="append",
        required=True,
        help="NAME:PROMPTS_JSONL:SEALED_PREDICTIONS_JSONL:COUNT",
    )
    sub.choices["automap-parity"].add_argument("--tolerance", type=float, default=1e-4)
    for name in ("parity", "automap-parity"):
        sub.choices[name].add_argument(
            "--answers",
            type=Path,
            help="write each prompt's answers here (JSON lines, kept on the node)",
        )
    answers = sub.add_parser("compare-answers")
    answers.add_argument("left", type=Path)
    answers.add_argument("right", type=Path)
    answers.add_argument("--output", type=Path, required=True)
    answers.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    handler = {
        "run": run,
        "compare": compare,
        "card": card,
        "parity": parity,
        "automap": automap,
        "automap-card": automap_card,
        "automap-parity": automap_parity,
        "compare-answers": compare_answer_files,
    }[args.command]
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
