"""Convert native teacher output on RP-v2 prompts into compact target files.

    python3 -m v2.data.m2.targets --rows rp-v2.rows.jsonl --teacher-output lux.wave1.jsonl \\
        --teacher lux --model-id llm-semantic-router/Decision-1.0-Lux-9B --revision bd45a30a... \\
        --out lux.wave1.targets.jsonl --report lux.wave1.report.json \\
        [--prompts wave.prompts.jsonl --attestation wave.attestation.jsonl \\
         --provenance provenance.json --wave NAME]

Every receipt is checked by ``v2.data.replay_targets.convert`` (model identity,
attested revision, validated runtime, prompt digest bound to the training row).
Rows the teacher could not answer natively get no target and are counted. The
output keeps ``{id, input_sha256, teacher_probs}`` sorted by id; join by id to
``m2/arms/<ARM>/train.jsonl`` (or v1 arm files) to build replay rows.

With ``--prompts`` the receipts must answer exactly the prompt file that was sent (same
ids; each prompt equal to its row's native prompt; digests checked on the file as sent)
and the report names every prompt without a target and why. ``--attestation`` writes one
line per prompt, sorted by id: the row's ``input_sha256``, the receipt's identity,
attested revision, adapter, config hash, validated-runtime flag and prompt digest, whether
it got a target, and the ``per_row`` fields of ``--provenance`` (a JSON object on where and
how the teacher ran, copied to the report).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl
from v2.data.replay_targets import convert

ATTESTED = (
    "model_id",
    "model_revision",
    "revision_attested",
    "adapter_version",
    "backend",
    "model_config_sha256",
    "runtime_matches_validated",
    "source_input_sha256",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def no_target_reason(receipt: dict[str, Any]) -> str:
    answer = receipt["answers"].get("decision") or {}
    native = receipt.get("native_error") or {}
    return str(native.get("kind") or answer.get("error") or "no_native_answer")


def attestation(
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    no_target: dict[str, str],
    per_row: dict[str, Any],
) -> bytes:
    input_sha256 = {row["id"]: row["input_sha256"] for row in rows}
    lines = []
    for receipt in sorted(receipts, key=lambda r: r["id"]):
        line = {
            "id": receipt["id"],
            "input_sha256": input_sha256[receipt["id"]],
            **{key: receipt.get(key) for key in ATTESTED},
            "target": receipt["id"] not in no_target,
        }
        if receipt["id"] in no_target:
            line["no_target_reason"] = no_target[receipt["id"]]
            if receipt.get("native_error"):
                line["native_error"] = receipt["native_error"]
        if set(per_row) & set(line):
            raise ValueError("provenance per_row fields shadow receipt fields")
        line.update(per_row)
        lines.append(canonical(line) + "\n")
    return "".join(lines).encode("utf-8")


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--wave")
    args = parser.parse_args(argv)
    for path in (args.out, args.report, args.attestation):
        if path is not None and path.exists():
            raise FileExistsError(path)
    provenance = (
        json.loads(args.provenance.read_text(encoding="utf-8"))
        if args.provenance
        else None
    )
    receipts = list(read_jsonl(args.teacher_output))
    prompts = list(read_jsonl(args.prompts)) if args.prompts else None
    if prompts is None:
        wanted = {r["id"] for r in receipts}
    else:
        wanted = {p["id"] for p in prompts}
        if (
            len(wanted) != len(prompts)
            or len(receipts) != len(wanted)
            or {r["id"] for r in receipts} != wanted
        ):
            raise ValueError("teacher output does not answer exactly the prompt file")
    rows = [row for row in read_jsonl(args.rows) if row["id"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError("teacher output names ids outside the RP-v2 rows")
    replay, report = convert(
        rows,
        prompts if prompts is not None else [native_prompt(row) for row in rows],
        receipts,
        teacher=args.teacher,
        model_id=args.model_id,
        revision=args.revision,
    )
    data = "".join(
        canonical(
            {
                "id": r["id"],
                "input_sha256": r["input_sha256"],
                "teacher_probs": r["teacher_probs"],
            }
        )
        + "\n"
        for r in sorted(replay, key=lambda r: r["id"])
    ).encode("utf-8")
    targets = {r["id"] for r in replay}
    no_target = {
        r["id"]: no_target_reason(r)
        for r in sorted(receipts, key=lambda r: r["id"])
        if r["id"] not in targets
    }
    attest = None
    if args.attestation:
        per_row = (provenance or {}).get("per_row", {})
        attest = attestation(rows, receipts, no_target, per_row)
    _write_new(args.out, data)
    report.update(
        rows=len(replay),
        prompts=len(rows),
        content_sha256=hashlib.sha256(data).hexdigest(),
        teacher_output_sha256=hashlib.sha256(
            args.teacher_output.read_bytes()
        ).hexdigest(),
    )
    if args.wave:
        report["wave"] = args.wave
    if prompts is not None:
        report.update(
            prompts_sha256=file_sha256(args.prompts),
            rows_file_sha256=file_sha256(args.rows),
            teacher_output_rows=len(receipts),
            no_target=no_target,
        )
    if attest is not None:
        _write_new(args.attestation, attest)
        report.update(
            attestation_sha256=hashlib.sha256(attest).hexdigest(),
            attestation_fields=[
                "id",
                "input_sha256",
                *ATTESTED,
                "target",
                "no_target_reason (no target only)",
                "native_error (no target only)",
                *sorted((provenance or {}).get("per_row", {})),
            ],
        )
    if provenance is not None:
        report["provenance"] = provenance
    _write_new(
        args.report,
        json.dumps(report, indent=1, sort_keys=True).encode("utf-8"),
    )
    keys = ("wave", "rows", "prompts", "content_sha256", "attestation_sha256")
    print(json.dumps({k: report[k] for k in keys if k in report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
