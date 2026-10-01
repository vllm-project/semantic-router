"""Fresh-environment Hub smoke test of a Decision 1.0 repository with stock Transformers.

Run in a new virtual environment with an empty Hugging Face cache. Loads the
repository at ``--revision`` with ``trust_remote_code=True`` through
``AutoConfig``, ``AutoTokenizer``, ``AutoModel`` and ``pipeline("decision")``,
answers the card's example, an over-length request and a malformed question,
and with ``--card`` also runs the card's Python block verbatim in a separate
process (it loads ``main``). Prints one JSON receipt; answers only, no panels.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

FALLBACK_REQUEST = {
    "state": "The parcel arrived damaged. Please send a replacement today.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "delivery": "Damaged or missing parcels",
                "billing": "Payments and invoices",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer request action today?",
        },
    },
}


def versions() -> dict[str, str | None]:
    found = {"python": platform.python_version()}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
    ):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def card_block(readme: str) -> str:
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    transformers_blocks = [b for b in blocks if "trust_remote_code=True" in b]
    if len(transformers_blocks) != 1:
        raise ValueError("The card must have exactly one Transformers Python example")
    return transformers_blocks[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--card", action="store_true", help="also run the card's Python block"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer, pipeline

    hub = {"revision": args.revision} if args.revision else {}
    receipt: dict = {
        "repo": args.repo,
        "revision": args.revision,
        "versions": versions(),
    }
    config = AutoConfig.from_pretrained(args.repo, trust_remote_code=True, **hub)
    receipt["config"] = {
        "class": type(config).__name__,
        "model_type": config.model_type,
        "model_name": config.model_name,
        "commit": getattr(config, "_commit_hash", None),
    }
    encoder = config.runtime_family == "vela-encoder"
    tokenizer_options = (
        {"subfolder": "native/tokenizer"} if encoder else {"trust_remote_code": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.repo, **tokenizer_options, **hub)
    receipt["tokenizer"] = {"class": type(tokenizer).__name__, "vocab": len(tokenizer)}

    readme = Path(hf_hub_download(args.repo, "README.md", **hub)).read_text(
        encoding="utf-8"
    )
    example = card_block(readme)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.repo, trust_remote_code=True, device=args.device, **hub
    )
    receipt["model"] = {
        "class": type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(next(model.parameters()).device),
        "load_seconds": round(time.perf_counter() - started, 1),
    }
    call = next(
        node
        for node in ast.walk(ast.parse(example))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "system_one"
    )
    try:
        request = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
    except (
        ValueError
    ):  # the card builds its request at run time; the card block itself runs with --card
        request = FALLBACK_REQUEST
    response = model.system_one(**request)
    receipt["system_one"] = response
    decide = pipeline(
        "decision", model=args.repo, trust_remote_code=True, device=args.device, **hub
    )
    receipt["pipeline_equal"] = decide(request) == response
    long_state = " ".join(["word"] * (2 * model.max_input_tokens))
    question = next(iter(request["questions"].values()))
    over = model.system_one(state=long_state, questions={"q": question, "r": question})
    receipt["over_length"] = sorted({a.get("error") for a in over["answers"].values()})
    bad = model.system_one(
        state="text",
        questions={
            "good": question,
            "bad": {"type": "choice", "instructions": "x", "criteria": {"only": None}},
        },
    )
    receipt["invalid_question"] = bad["answers"]["bad"]
    receipt["valid_beside_invalid"] = "error" not in bad["answers"]["good"]
    if args.card:
        with tempfile.TemporaryDirectory() as scratch:
            script = Path(scratch) / "card_example.py"
            script.write_text(example, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-B", str(script)],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        receipt["card"] = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip()[-500:],
            "stderr_tail": completed.stderr[-1500:] if completed.returncode else "",
        }
    receipt["passed"] = (
        receipt["config"]["model_type"] in ("decision1", "decision2")
        and receipt["pipeline_equal"]
        and receipt["over_length"] == ["max_length_exceeded"]
        and receipt["invalid_question"].get("error") == "invalid_question"
        and receipt["valid_beside_invalid"]
        and all("error" not in a for a in response["answers"].values())
        and (not args.card or receipt["card"]["exit_code"] == 0)
    )
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "repo": args.repo,
                "passed": receipt["passed"],
                "answers": response["answers"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
