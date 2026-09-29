"""Kernel-path CAL fit and SELECT / CAL probabilities for ~27B development readouts.

``fit`` (GPU, container): per-type CAL temperatures for one frozen checkpoint
(an arm's BEST or a soup) at an explicit inference limit, on the kernel path
(``typed_collect_kernel.kernel_runtime`` first). The logits come from the pinned
``training.model.calibrate.collect_logits`` (the code Milestone 2's
``calibrate_context`` refits ran: FP32 parameters, BF16 backbone compute, FP32
head, no truncation) and the temperatures from ``fit_report``. The report
follows ``v2.dec.calibrate_ckpt``'s ``frozen_checkpoint`` contract (CAL698 is
not the CAL a run audited, so ``training.model.calibrate`` cannot bind it), so
``training.model.infer --calibration`` accepts it. It also writes the CAL
logits and calibrated / raw CAL probabilities, and with ``--select`` raw SELECT
probabilities for a checkpoint without trainer SELECT output (a soup).

``select-from-trainer`` (host): the trainer's own SELECT700 probabilities at a
completed run's BEST, in ``v2.eval.dev_readout`` rows aligned with the options.

``cal-summary`` (host): ``dev_readout``'s SELECT/CAL metrics on CAL698 rows
(``dev_readout --cal`` scores the 700-row CAL of the panel root).

``exec`` (GPU, container): run a module (``v2.27b.aho_eval``) on the verified
kernel path and write the runtime identity.

``t1-calibration`` (host): the T = 1 package calibration of a checkpoint whose
CAL698 fit was rejected. The kernel adapter always passes ``--calibration``;
this report keeps the rejected fit's binding (model, checkpoint and CAL hashes,
inference limit) with every temperature 1.0, so collected answers equal those of
a package without ``calibration.json`` (softmax(z / 1.0) is softmax(z) exactly).
It records the rejected temperatures and the adoption receipt.

``argcheck`` (dry runs, CPU): run a module up to its ``parse_args`` and stop,
so a script's command line is checked against the target's argparse. It follows
``kernel_readout exec`` into its module, and ``same_panel collect`` into the
adapter command (the kernel adapter's options are ``training.model.infer``'s).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import runpy
import sys
import warnings
from pathlib import Path
from typing import Any

from training.model.calibration import (
    CALIBRATION_VERSION,
    fit_report,
    load_calibration,
    probabilities,
)
from training.model.data import canonical, file_sha256, load_partition

kernel = importlib.import_module("v2.27b.typed_collect_kernel")
MODEL_DIR = Path(importlib.import_module("training.model.data").__file__).parent
BINDING_KEYS = (
    "calibration_version",
    "fit_split",
    "selection_policy",
    "model_sha256",
    "checkpoint_sha256",
    "cal_sha256",
    "best_sha256",
    "complete_sha256",
    "provenance_sha256",
    "logits_sha256",
    "inference",
)
FORWARD = {"v2.27b.typed_collect_kernel": "training.model.infer"}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def probability_rows(
    records: list[dict[str, Any]], temperature_by_type: dict[str, float] | None
) -> list[dict[str, Any]]:
    return [
        {
            "id": record["id"],
            "probabilities": probabilities(
                record["logits"],
                (temperature_by_type or {}).get(record["task_type"], 1.0),
            ),
        }
        for record in records
    ]


def build_report(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    cal_sha256: str,
    inference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "calibration_version": CALIBRATION_VERSION,
        "fit_split": "cal",
        "selection_policy": "frozen_checkpoint",
        "model_sha256": identity["model_sha256"],
        "checkpoint_sha256": _digest(
            {
                name.removeprefix("checkpoint/"): sha
                for name, sha in identity["files_sha256"].items()
                if not name.startswith("source/")
            }
        ),
        "cal_sha256": cal_sha256,
        "logits_sha256": _digest(records),
        "code_sha256": {
            "kernel_readout.py": file_sha256(Path(__file__)),
            "typed_collect_kernel.py": file_sha256(Path(kernel.__file__)),
            **{
                name: file_sha256(MODEL_DIR / name)
                for name in ("calibrate.py", "calibration.py", "decision_model.py")
            },
        },
        "inference": inference,
        **fit_report(records),
    }


def fit(args: argparse.Namespace) -> None:
    out: Path = args.out_dir
    names = ["calibration.json", "cal.logits.jsonl", "cal.probs.jsonl"]
    names += ["cal.raw.probs.jsonl", "runtime.json"]
    names += ["select.probs.jsonl"] if args.select else []
    if any((out / name).exists() for name in names):
        raise FileExistsError(f"{out} already holds calibration outputs")
    if file_sha256(args.cal) != args.cal_sha256:
        raise SystemExit("CAL file differs from its pinned SHA-256")
    runtime = kernel.kernel_runtime()
    from training.model.calibrate import collect_logits
    from training.model.infer import checkpoint_fingerprint

    rows = load_partition(args.cal, "cal")
    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    options = dict(
        source_path=args.source_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device_name="cuda:0",
    )
    records = collect_logits(args.checkpoint, rows, **options)
    if [r["id"] for r in records] != [r["id"] for r in rows]:
        raise SystemExit("CAL logit collection dropped or reordered rows")
    report = build_report(
        records,
        identity,
        args.cal_sha256,
        {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "device": "cuda:0",
            "precision": "FP32 parameters; BF16 backbone compute and FP32 head on CUDA",
            "runtime": runtime,
        },
    )
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cal.logits.jsonl", records)
    write_jsonl(
        out / "cal.probs.jsonl",
        probability_rows(records, report["temperature_by_type"]),
    )
    write_jsonl(out / "cal.raw.probs.jsonl", probability_rows(records, None))
    if args.select:
        select_rows = load_partition(args.select, "select")
        select = collect_logits(args.checkpoint, select_rows, **options)
        write_jsonl(out / "select.probs.jsonl", probability_rows(select, None))
    write_json(out / "calibration.json", report)
    write_json(
        out / "runtime.json",
        {**runtime, "versions": kernel.versions(), "memory": kernel.memory()},
    )
    print(
        json.dumps(
            {
                "output": str(out / "calibration.json"),
                "model_sha256": report["model_sha256"],
                "temperature_by_type": report["temperature_by_type"],
            },
            sort_keys=True,
        )
    )


def trainer_select_rows(
    run_dir: Path, rows: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    best = json.loads((run_dir / "BEST.json").read_text(encoding="utf-8"))["checkpoint"]
    complete = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("best") != best:
        raise SystemExit("run is not complete with a frozen BEST")
    step = re.fullmatch(r"checkpoint-([0-9]{7})", best)
    if step is None:
        raise SystemExit(f"unexpected BEST checkpoint {best}")
    path = run_dir / f"select-step-{step.group(1)}-predictions.jsonl"
    predictions = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    out = []
    for row in rows:
        record = predictions.pop(row["id"])
        keys = [option["key"] for option in row["options"]]
        answer = record["answer"]
        if answer["type"] == "noul":
            by_key = {"true": answer["noul"], "false": 1.0 - answer["noul"]}
        else:
            by_key = answer["probabilities"]
        if set(by_key) != set(keys):
            raise SystemExit(f"{row['id']}: prediction keys differ from the options")
        out.append({"id": row["id"], "probabilities": [by_key[k] for k in keys]})
    if predictions:
        raise SystemExit(f"{len(predictions)} SELECT predictions are not in the rows")
    return best, out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_from_trainer(args: argparse.Namespace) -> None:
    best, rows = trainer_select_rows(args.run_dir, read_jsonl(args.rows))
    write_jsonl(args.output, rows)
    print(json.dumps({"best": best, "rows": len(rows), "output": str(args.output)}))


def cal_summary(args: argparse.Namespace) -> None:
    from v2.eval.dev_readout import select_summary

    if file_sha256(args.rows) != args.cal_sha256:
        raise SystemExit("CAL file differs from its pinned SHA-256")
    summary = select_summary(read_jsonl(args.rows), args.probabilities)
    write_json(
        args.output,
        {
            "schema": "decision2-27b-cal-summary/1",
            "cal_sha256": args.cal_sha256,
            "probabilities": str(args.probabilities),
            **summary,
        },
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("n", "correct", "family_macro_brier")},
            sort_keys=True,
        )
    )


def run_module(args: argparse.Namespace) -> None:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("exec needs a module after --")
    if args.runtime.exists():
        raise FileExistsError(args.runtime)
    runtime = kernel.kernel_runtime()
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = list(command)
    runpy.run_module(command[0], run_name="__main__", alter_sys=True)
    write_json(
        args.runtime,
        {
            **runtime,
            "module": command[0],
            "versions": kernel.versions(),
            "memory": kernel.memory(),
        },
    )


def t1_report(
    rejected: dict[str, Any],
    rejected_sha256: str,
    adoption: dict[str, Any],
    adoption_sha256: str,
) -> dict[str, Any]:
    if adoption.get("adopt") is not False:
        raise SystemExit("the adoption receipt did not reject the CAL698 fit")
    if adoption.get("candidate_sha256") != rejected_sha256:
        raise SystemExit("the adoption receipt judged a different calibration file")
    return {
        **{key: rejected[key] for key in BINDING_KEYS if key in rejected},
        "temperature_by_type": {k: 1.0 for k in rejected["temperature_by_type"]},
        "package_temperature": "T = 1 (CAL698 fit rejected on the development panels)",
        "rejected_calibration_sha256": rejected_sha256,
        "rejected_temperature_by_type": rejected["temperature_by_type"],
        "adoption_receipt_sha256": adoption_sha256,
        "adoption_worsened": adoption.get("worsened"),
    }


def t1_calibration(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    report = t1_report(
        json.loads(args.rejected.read_text(encoding="utf-8")),
        file_sha256(args.rejected),
        json.loads(args.adoption.read_text(encoding="utf-8")),
        file_sha256(args.adoption),
    )
    write_json(args.output, report)
    temperatures, _ = load_calibration(args.output, report["model_sha256"])
    print(json.dumps({"output": str(args.output), "temperature_by_type": temperatures}))


class _Parsed(Exception):
    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__(namespace)
        self.namespace = namespace


def parsed_args(command: list[str]) -> argparse.Namespace:
    """The namespace ``command``'s module parses; nothing after parse_args runs."""
    original, argv = argparse.ArgumentParser.parse_args, sys.argv

    def stop(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    argparse.ArgumentParser.parse_args = stop
    try:
        sys.argv = list(command)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module(command[0], run_name="__main__", alter_sys=True)
    except _Parsed as parsed:
        return parsed.namespace
    finally:
        argparse.ArgumentParser.parse_args, sys.argv = original, argv
    raise SystemExit(f"{command[0]} returned before parsing its arguments")


def check_command(command: list[str]) -> list[str]:
    namespace = parsed_args(command)
    checked = [command[0]]
    if command[0] == "v2.27b.kernel_readout" and namespace.mode == "exec":
        inner = namespace.command
        checked += check_command(inner[1:] if inner[:1] == ["--"] else inner)
    if command[0] == "v2.eval.same_panel" and namespace.command == "collect":
        adapters = importlib.import_module("v2.eval.adapters")
        adapter = adapters.load(namespace.adapter, namespace.adapter_spec)
        values = {
            "model": str(namespace.model_path),
            "revision": namespace.revision,
            "device": namespace.device,
            "model_id": namespace.model_id or adapter.model_id or "",
            **dict(entry.partition("=")[::2] for entry in namespace.extra),
        }
        argv = adapter.command({**values, "input": "in.jsonl", "output": "out.jsonl"})
        checked += check_command([FORWARD.get(argv[2], argv[2]), *argv[3:]])
    return checked


def argcheck(args: argparse.Namespace) -> None:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("argcheck needs a module after --")
    print(json.dumps({"argcheck": check_command(command), "ok": True}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-path", type=Path)
    p.add_argument("--cal", type=Path, required=True)
    p.add_argument("--cal-sha256", required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--select", type=Path, help="SELECT rows for raw probabilities")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("select-from-trainer")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True, help="SELECT700 rows")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("cal-summary")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--cal-sha256", required=True)
    p.add_argument("--probabilities", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("exec")
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("t1-calibration")
    p.add_argument("--rejected", type=Path, required=True, help="rejected CAL698 fit")
    p.add_argument(
        "--adoption", type=Path, required=True, help="dev_calibration receipt"
    )
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("argcheck")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    {
        "fit": fit,
        "select-from-trainer": select_from_trainer,
        "cal-summary": cal_summary,
        "exec": run_module,
        "t1-calibration": t1_calibration,
        "argcheck": argcheck,
    }[args.mode](args)


if __name__ == "__main__":
    main()
