"""Parity of a state-once Decision model's branched forward with each question run alone as a row.

    python3 tools/state_once_parity.py --package DIR --output OUT.json [--requests REQUESTS.jsonl]
        [--attention-mode prefix_bidir|prefix_causal|causal] [--device rocm:0] [--limit N]
        [--answers ANSWERS.jsonl]

The package is verified and loaded through the decision3 family and the native engine; no package code runs.
``--attention-mode`` first writes a copy of a Decision 3.0 package that declares that layout next to it
(``DIR-MODE``: weights hard-linked, ``decision_config.json`` and the manifest's identity rewritten), so released
weights can stand in for a state-once checkpoint: the two paths must agree on any weights. Every request (``{"id", "state",
"questions"[, "images", "videos"]}`` per line; without ``--requests`` the public many-question ticket at a
~400-token and a ~2,000-token state with 1, 4, 16 and 64 questions) is answered branched (``model.run``: the
state once, every question from it) and per row (``model.run_rows``: each question as its own row under the same
mask). Reported: questions compared, decisions changed (and each change's margin, the per-row top-1 minus top-2
probability), the largest, mean and median absolute probability difference, and with gold labels (``expected``:
a Choice key or a Noul boolean per question) both paths' accuracy and McNemar's exact p-value. ``--floor`` also
runs each question alone as a row, so the same figures compare branched and per-row answers with the noise of
batching the per-row path itself. ``--answers`` writes every question's probability vectors. ``--row-tokens N``
runs a state's per-row questions in consecutive groups of at most N padded tokens (rows times the group's longest
row), one forward per group, so that a large model in FP32 fits one GPU on many questions over a long state; each
question is still its own row under the same mask.

``--golden`` checks the state split against a reference: a golden fixture holds, per question of the same
requests, the token IDs and the state length the reference implementation's own split gives (a header line, then one
``{"id", "question", "ids", "s"}`` line per question). Every planned question must have the fixture's token IDs
and state length; ``--boundary-only`` plans the requests without running the model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import many_questions  # noqa: E402
from vllm_srun import dtypes  # noqa: E402
from vllm_srun.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_srun.accel.cuda import CUDAAccelerator  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.engines.native.engine import NativeEngine  # noqa: E402
from vllm_srun.families.decision3 import package as pkg  # noqa: E402
from vllm_srun.families.decision3.family import Decision3Family  # noqa: E402
from vllm_srun.families.decision3.state_once import StateOnceModel  # noqa: E402
from vllm_srun.plugins.base import (  # noqa: E402
    EngineOptions,
    PackageRef,
    RegistryOptions,
)

ACCELERATORS = {"cpu": CPUAccelerator, "cuda": CUDAAccelerator, "rocm": ROCmAccelerator}
# A second, longer state: the ticket followed by the store's earlier replies (about 2,000 tokens in all).
HISTORY = (
    "Earlier messages in this thread, oldest first. Agent, Monday: Thank you for contacting us about order "
    "#48213. We are sorry the blender arrived damaged. Please reply with photos of the box and the jug and "
    "we will arrange a replacement. Customer, Monday: Photos attached; the jug is cracked along the handle "
    "and the base is scratched. Agent, Tuesday: Thanks, the photos are clear. A replacement has been "
    "requested from the warehouse; it usually ships within one business day. You will receive a tracking "
    "number by email. Customer, Wednesday: No tracking number yet. Also, my card shows two express "
    "shipping charges of 14.99 each. Agent, Wednesday: I can see the duplicate charge and have asked the "
    "billing team to review it; refunds take three to five business days to appear. The replacement is "
    "still marked as pending at the warehouse. Customer, Thursday: Still nothing. I need the blender by "
    "Saturday. "
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(source: Path, target: Path, attention_mode: str) -> Path:
    """A copy of a Decision 3.0 package that declares ``attention_mode``: every other file hard-linked
    (copied across file systems; packages refuse symbolic links), the decision config and the manifest
    (file digests, scored identity) rewritten. An existing ``target`` is replaced."""
    if attention_mode not in pkg.STATE_ONCE_MODES:
        raise ValueError(f"attention mode must be one of {pkg.STATE_ONCE_MODES}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in (
            "decision_config.json",
            pkg.MANIFEST_NAME,
        ):
            try:
                os.link(path.resolve(), target / path.name)
            except OSError:
                shutil.copy2(path.resolve(), target / path.name)
    decision = json.loads((source / "decision_config.json").read_text(encoding="utf-8"))
    decision["attention_mode"] = attention_mode
    (target / "decision_config.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    manifest = json.loads((source / pkg.MANIFEST_NAME).read_text(encoding="utf-8"))
    files = manifest["files_sha256"]
    files["decision_config.json"] = sha256(target / "decision_config.json")
    if isinstance(manifest.get("files_bytes"), dict):
        manifest["files_bytes"]["decision_config.json"] = (
            (target / "decision_config.json").stat().st_size
        )
    identity_files = {
        name: files[name]
        for name in (
            *[w for w in pkg.weight_files(source) if w.endswith(".safetensors")],
            *[n for n in pkg.IDENTITY_FILES if (source / n).is_file()],
        )
    }
    identity = manifest.setdefault("identity", {})
    identity["model_sha256"] = pkg.model_identity(identity_files, decision)
    identity["decision"] = {
        key: decision[key] for key in pkg.INFERENCE_FIELDS if key in decision
    }
    manifest["attention_mode"] = attention_mode
    (target / pkg.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    return target


def ieee_triton() -> None:
    """IEEE FP32 dots in every Triton kernel: FLA's chunked delta rule otherwise takes TF32 where the GPU has
    it, so an FP32 reference run would round to TF32 there."""
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    try:
        import triton.language as tl
        from fla.ops.gated_delta_rule import chunk_fwd
    except ImportError:
        return
    chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


def load(package: Path, device_name: str, dtype: str = "auto") -> StateOnceModel:
    """The package's model through the family and the native engine; ``dtype`` ``float32`` holds every
    weight in FP32 with exact kernels only (the reference precision)."""
    family = Decision3Family(RegistryOptions(offline=True))
    verified = family.verify(PackageRef(root=package))
    spec = dtypes.apply(family.describe(verified), dtype, True)
    kind, _, index = device_name.partition(":")
    accelerator = ACCELERATORS[kind]()
    devices = accelerator.devices()
    device = devices[int(index or 0)] if kind != "cpu" else devices[0]
    engine_model = NativeEngine().load(
        spec, accelerator, device, EngineOptions(exact_kernels_only=dtype != "auto")
    )
    model = family.load(verified, spec, engine_model)
    if not isinstance(model, StateOnceModel):
        raise SystemExit(f"{package} does not declare a state-once attention mode")
    return model


def default_requests() -> list[dict[str, Any]]:
    """The public many-question ticket at two state lengths, 1 / 4 / 16 / 64 questions each."""
    states = {
        "ticket": many_questions.TICKET,
        "thread": many_questions.TICKET + " " + HISTORY * 4,
    }
    return [
        {
            "id": f"{name}-{count}",
            "state": state,
            "questions": many_questions.questions(count),
        }
        for name, state in states.items()
        for count in (1, 4, 16, 64)
    ]


def margin(values: list[float]) -> float:
    top = sorted(values, reverse=True)
    return top[0] - top[1] if len(top) > 1 else 1.0


def correct(item: Any, values: list[float], expected: Any) -> bool | None:
    """Whether the arg-max answer is the gold label (a Choice key or a Noul boolean); None without one."""
    best = max(range(len(values)), key=values.__getitem__)
    if item.task_type == "noul" and isinstance(expected, bool):
        return (best == 1) == expected
    if item.task_type == "choice" and isinstance(expected, str):
        return item.keys[best] == expected
    return None


def mcnemar(only_left: int, only_right: int) -> float:
    """Two-sided exact McNemar p-value of the discordant pairs."""
    n = only_left + only_right
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_left, only_right) + 1)) / 2**n
    return min(1.0, 2 * tail)


def read_requests(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


class Boundary:
    """The runtime's state split of each question against a golden fixture of the reference split."""

    TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

    def __init__(self, path: Path, package: Path) -> None:
        rows = read_requests(path)
        self.header = rows[0]
        self.records = {(row["id"], row["question"]): row for row in rows[1:]}
        expected = self.header.get("tokenizer_sha256") or {}
        self.tokenizer_matches = bool(expected) and all(
            sha256(package / name) == digest for name, digest in expected.items()
        )
        self.compared = self.ids_differ = self.state_differ = 0
        self.missing = self.refused = 0
        self.examples: list[dict[str, Any]] = []

    def check(self, request: dict[str, Any], plan: Any) -> None:
        key = request.get("id")
        for item in plan.items:
            record = self.records.get((key, item.question_id))
            if record is None or "ids" not in record:
                self.missing += 1
                continue
            self.compared += 1
            state = len(item.state.ids)
            same_ids = list(item.ids) == record["ids"]
            same_state = state == record["s"]
            self.ids_differ += not same_ids
            self.state_differ += not same_state
            if not (same_ids and same_state) and len(self.examples) < 10:
                self.examples.append(
                    {
                        "id": key,
                        "question": item.question_id,
                        "state": state,
                        "golden_state": record["s"],
                        "tokens": len(item.ids),
                        "golden_tokens": record["n"],
                    }
                )
        self.refused += sum(
            1
            for question in plan.errors
            if "ids" in self.records.get((key, question), {})
        )

    def refuse(self, request: dict[str, Any]) -> None:
        """A request the runtime refused as a whole: its golden questions count as refused."""
        key = request.get("id")
        self.refused += sum(
            1
            for question in request["questions"]
            if "ids" in self.records.get((key, question), {})
        )

    def report(self) -> dict[str, Any]:
        return {
            "golden": self.header.get("golden"),
            "decision_format_sha256": self.header.get("decision_format_sha256"),
            "tokenizer_matches": self.tokenizer_matches,
            "questions": self.compared,
            "ids_differ": self.ids_differ,
            "state_differ": self.state_differ,
            "missing_from_golden": self.missing,
            "refused_by_the_runtime": self.refused,
            "examples": self.examples,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--attention-mode", choices=pkg.STATE_ONCE_MODES)
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--dtype", choices=("auto", "float32"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--shard", default="0/1", help="I/N: every N-th request from the I-th"
    )
    parser.add_argument("--answers", type=Path)
    parser.add_argument(
        "--floor",
        action="store_true",
        help="also run each question alone as a row: the batching noise of the per-row path",
    )
    parser.add_argument(
        "--row-tokens",
        type=int,
        help="the per-row path runs a state's questions in groups of at most this many padded tokens",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        help="the reference split of the same requests: token IDs and state length per question",
    )
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="with --golden: plan the requests and check the split, without forwards",
    )
    args = parser.parse_args()
    package = args.package
    if args.attention_mode:
        target = package.parent / f"{package.name}-{args.attention_mode}"
        if not (target / pkg.MANIFEST_NAME).is_file():
            convert(package, target, args.attention_mode)
        package = target
    if args.dtype == "float32":
        ieee_triton()
    model = load(package, args.device, args.dtype)
    requests = read_requests(args.requests) if args.requests else default_requests()
    requests = requests[: args.limit] if args.limit else requests
    index, count = (int(part) for part in args.shard.split("/"))
    requests = requests[index::count]
    paths = ["branched", "rows"] + (["alone"] if args.floor else [])
    pairs = {
        name: Pair()
        for name in (
            ["branched/rows", "branched/alone", "alone/rows"]
            if args.floor
            else ["branched/rows"]
        )
    }
    boundary = Boundary(args.golden, package) if args.golden else None
    out = args.answers.open("w", encoding="utf-8") if args.answers else None
    skipped = 0
    refused: list[dict[str, Any]] = []
    for request in requests:
        try:
            if request.get("images") or request.get("videos"):
                plan = model.plan_images(
                    request["state"],
                    request["questions"],
                    request.get("images") or [],
                    request.get("videos"),
                )
            else:
                plan = model.plan(request["state"], request["questions"])
        except ValueError as exc:
            # The API answers such a request with an error as a whole (e.g. a null state).
            refused.append({"id": request.get("id"), "error": str(exc)})
            if boundary is not None:
                boundary.refuse(request)
            continue
        if boundary is not None and not (
            request.get("images") or request.get("videos")
        ):
            boundary.check(request, plan)
        items = list(plan.items)
        skipped += len(plan.errors)
        if not items or args.boundary_only:
            continue
        answers: dict[str, list[Any]] = {
            "branched": model.run(items),
            "rows": run_rows(model, items, args.row_tokens),
        }
        if args.floor:
            answers["alone"] = [model.run_rows([item])[0] for item in items]
        expected = request.get("expected") or {}
        for index, item in enumerate(items):
            for name, pair in pairs.items():
                left, right = name.split("/")
                pair.add(
                    request.get("id"),
                    item,
                    answers[left][index],
                    answers[right][index],
                    expected.get(item.question_id),
                )
            if out is not None:
                record = {"id": request.get("id"), "question": item.question_id}
                record |= {path: answers[path][index] for path in paths}
                out.write(json.dumps(record) + "\n")
    if out is not None:
        out.close()
    decision = json.loads(
        (package / "decision_config.json").read_text(encoding="utf-8")
    )
    report = {
        "package": str(args.package),
        "attention_mode": decision["attention_mode"],
        "device": args.device,
        "dtype": args.dtype,
        "row_tokens": args.row_tokens,
        "requests": len(requests),
        "questions_refused": skipped,
        "requests_refused": refused,
        "boundary": boundary.report() if boundary is not None else None,
        **pairs["branched/rows"].report(),
        "floor": {
            name: pair.report()
            for name, pair in pairs.items()
            if name != "branched/rows"
        },
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "changed"}
    summary["floor"] = {
        name: {k: v for k, v in floor.items() if k != "changed"}
        for name, floor in report["floor"].items()
    }
    print(json.dumps(summary))
    return 0


def row_groups(items: list[Any], budget: int | None) -> list[list[Any]]:
    """Consecutive groups of ``items`` for the per-row path: a new group starts at another state, or where the
    group's padded size (its rows times its longest row) would pass ``budget``; a row longer than ``budget`` is a
    group of its own. No budget: one group per run of the same state, as ``run_rows`` forms them itself.
    """
    groups: list[list[Any]] = []
    longest = 0
    for item in items:
        group = groups[-1] if groups else None
        size = len(item.ids)
        if (
            group is None
            or item.state is not group[0].state
            or (budget and (len(group) + 1) * max(longest, size) > budget)
        ):
            groups.append([item])
            longest = size
        else:
            group.append(item)
            longest = max(longest, size)
    return groups


def run_rows(model: Any, items: list[Any], budget: int | None) -> list[Any]:
    """``model.run_rows`` group by group (``row_groups``), the answers in the items' order."""
    if not budget:
        return model.run_rows(items)
    out: list[Any] = []
    for group in row_groups(items, budget):
        out.extend(model.run_rows(group))
    return out


class Pair:
    """Agreement of two paths' answers: decisions, probability gaps, gold accuracy and McNemar."""

    def __init__(self) -> None:
        self.gaps: list[float] = []
        self.changed: list[dict[str, Any]] = []
        self.gold = {
            "questions": 0,
            "left": 0,
            "right": 0,
            "only_left": 0,
            "only_right": 0,
        }

    def add(
        self,
        request: Any,
        item: Any,
        left: list[float],
        right: list[float],
        expected: Any,
    ) -> None:
        self.gaps.append(max(abs(a - b) for a, b in zip(left, right, strict=True)))
        if max(range(len(left)), key=left.__getitem__) != max(
            range(len(right)), key=right.__getitem__
        ):
            self.changed.append(
                {"id": request, "question": item.question_id, "margin": margin(right)}
            )
        hits = correct(item, left, expected), correct(item, right, expected)
        if hits[0] is not None and hits[1] is not None:
            self.gold["questions"] += 1
            self.gold["left"] += hits[0]
            self.gold["right"] += hits[1]
            self.gold["only_left"] += hits[0] and not hits[1]
            self.gold["only_right"] += hits[1] and not hits[0]

    def report(self) -> dict[str, Any]:
        gaps = self.gaps
        return {
            "questions": len(gaps),
            "decisions_changed": len(self.changed),
            "agreement": 1 - len(self.changed) / len(gaps) if gaps else None,
            "max_abs_dp": max(gaps) if gaps else None,
            "mean_abs_dp": statistics.fmean(gaps) if gaps else None,
            "median_abs_dp": statistics.median(gaps) if gaps else None,
            "max_changed_margin": max(
                (change["margin"] for change in self.changed), default=None
            ),
            "gold": {
                **self.gold,
                "mcnemar_p": mcnemar(self.gold["only_left"], self.gold["only_right"]),
            },
            "changed": self.changed,
        }


if __name__ == "__main__":
    raise SystemExit(main())
