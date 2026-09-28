"""Convert qualified AutoJev-27B receipts into compact teacher targets (M3a).

    python3 -m v2.data.m3.teacher_targets --wave aj-m --rows rows.jsonl \\
        --prompts aj-m.prompts.jsonl --qualification qualification.json \\
        --shard S0.jsonl=S0.jsonl.manifest.json ... --guard guard.json \\
        --out aj-m.targets.jsonl --report aj-m.report.json

The pinned collector marks receipts ``pytorch_bf16_rocm_pending_repeatability``. They are
accepted only when the qualification report passed, the target guard passed on exactly
this prompt file, and every shard ran the qualified runtime: pinned image, FLA path, the
frozen autotune entries unchanged, the qualified model and source hashes, exit code 0,
output bytes as recorded. ``v2.data.replay_targets.convert`` then checks, per row, the
model identity, attested revision and the prompt digest bound to the training row. Output
rows are ``{id, input_sha256, teacher_probs}`` sorted by id (the own-Lux wave format); the
report carries the per-shard receipts and the teacher provenance caveat.
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
from v2.data.m3.qualify import IDENTITY, IMAGE_ID, PENDING, RUN_CONSTANT
from v2.data.replay_targets import convert

TEACHER = "autojev27"
PROVENANCE_CAVEAT = (
    "Teacher: denis-pplx/autojev-27b@6f5b557e (Apache-2.0 weights; third-party decision "
    "model, never a Decision 2.0 weight origin). Its public training pipeline reportedly "
    "used SFT data generated with a closed OpenAI model, so these targets carry that "
    "provenance: every AutoJev-distilled candidate discloses it on its card and in its "
    "records, and needs a matched own-Lux-target control. Own-Lux targets remain the clean "
    "default. Allowed for release candidates by coordinator decision (2026-09-28 18:45 "
    "UTC+8, re-qualification ordered 20:15) after the node-A ROCm runtime passed the M3a "
    "qualification v2 (bitwise repeat determinism with a shared, fully warmed autotune cache)."
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check_shard(
    manifest: dict[str, Any], output: Path, qualification: dict[str, Any]
) -> list[str]:
    problems = []
    frozen = qualification["autotune_frozen"]
    if manifest.get("exit_code") != 0:
        problems.append("exit_code")
    if manifest.get("image_id") != IMAGE_ID:
        problems.append("image_id")
    if manifest.get("fla_reference_fallback") is not False:
        problems.append("fla_path")
    if (
        manifest.get("autotune_before") != frozen
        or manifest.get("autotune_after") != frozen
    ):
        problems.append("autotune_changed")
    if manifest.get("output_sha256") != file_sha256(output):
        problems.append("output_bytes")
    if (manifest.get("collector") or {}).get("loaded_parameters") is None:
        problems.append("collector_summary")
    return problems


def qualified_receipts(
    shards: list[tuple[Path, Path]], qualification: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if qualification.get("pass") is not True:
        raise ValueError("AutoJev runtime qualification did not pass")
    package = {key: qualification["package_hashes"][key] for key in RUN_CONSTANT}
    if any(len(v) != 1 for v in package.values()):
        raise ValueError("qualification has no single package identity")
    receipts: list[dict[str, Any]] = []
    shard_report = []
    for output, manifest_path in shards:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = check_shard(manifest, output, qualification)
        if problems:
            raise ValueError(
                f"{output.name}: shard not on the qualified runtime: {problems}"
            )
        for record in read_jsonl(output):
            if record.get("runtime_qualification") != PENDING:
                raise ValueError(
                    f"{record.get('id')}: unexpected runtime qualification"
                )
            for key, value in IDENTITY.items():
                if record.get(key) != value:
                    raise ValueError(
                        f"{record.get('id')}: {key} differs from the qualified identity"
                    )
            for key, (value,) in package.items():
                if record.get(key) != value:
                    raise ValueError(
                        f"{record.get('id')}: {key} differs from the qualified package"
                    )
            receipts.append(
                dict(
                    record,
                    runtime_matches_validated=True,
                    _shard=manifest["label"],
                    _gpu=manifest["gpu"],
                )
            )
        shard_report.append(
            {
                "label": manifest["label"],
                "gpu": manifest["gpu"],
                "rows": manifest["output_rows"],
                "output_sha256": manifest["output_sha256"],
                "input_sha256": manifest["input_sha256"],
                "mirror_commit": manifest["mirror"]["commit"],
                "start_utc": manifest["start_utc"],
                "end_utc": manifest["end_utc"],
                "gpu_hours": round(manifest["gpu_hours"], 4),
            }
        )
    return receipts, {
        "shards": shard_report,
        "package": {k: v[0] for k, v in package.items()},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    guard = json.loads(args.guard.read_text(encoding="utf-8"))
    if guard.get("pass") is not True or guard.get("prompts_sha256") != file_sha256(
        args.prompts
    ):
        raise ValueError("target guard did not pass on this prompt file")
    shards = []
    for spec in args.shard:
        output, _, manifest = spec.partition("=")
        shards.append((Path(output), Path(manifest)))
    receipts, runtime = qualified_receipts(shards, qualification)
    prompts = list(read_jsonl(args.prompts))
    wanted = {p["id"] for p in prompts}
    if {r["id"] for r in receipts} != wanted or len(receipts) != len(wanted):
        raise ValueError("shard receipts do not cover exactly the wave prompts")
    rows = [row for row in read_jsonl(args.rows) if row["id"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError("wave prompts name ids outside the rows file")
    by_id = {p["id"]: p for p in prompts}
    for row in rows:
        if canonical(by_id[row["id"]]) != canonical(native_prompt(row)):
            raise ValueError(f"{row['id']}: wave prompt differs from its training row")
    shard_of = {r["id"]: (r.pop("_shard"), r.pop("_gpu")) for r in receipts}
    attest = {r["id"]: r for r in receipts}
    replay, report = convert(
        rows,
        [native_prompt(row) for row in rows],
        receipts,
        teacher=TEACHER,
        model_id=IDENTITY["model_id"],
        revision=IDENTITY["model_revision"],
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
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    fields = ("source_input_sha256", *IDENTITY, *RUN_CONSTANT, "runtime_qualification")
    attestation = "".join(
        canonical(
            {
                "id": r["id"],
                "input_sha256": r["input_sha256"],
                **{k: attest[r["id"]][k] for k in fields},
                "shard": shard_of[r["id"]][0],
                "gpu": shard_of[r["id"]][1],
                "node": "node A",
                "image_id": IMAGE_ID,
                "runtime_qualified_by": "M3a qualification v2",
            }
        )
        + "\n"
        for r in sorted(replay, key=lambda r: r["id"])
    ).encode("utf-8")
    fd = os.open(args.attestation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(attestation)
    report.update(
        schema="decision2-m3a-autojev-targets/1",
        wave=args.wave,
        rows=len(replay),
        prompts=len(rows),
        content_sha256=hashlib.sha256(data).hexdigest(),
        attestation_sha256=hashlib.sha256(attestation).hexdigest(),
        prompts_sha256=file_sha256(args.prompts),
        rows_file_sha256=file_sha256(args.rows),
        qualification_sha256=file_sha256(args.qualification),
        guard_sha256=file_sha256(args.guard),
        runtime={
            "node": "node A",
            "image_id": IMAGE_ID,
            "collector": "inference.autojev27 (package-native DecisionModel, no chat API)",
            "autotune_frozen": qualification["autotune_frozen"],
            "fla": "FLA 0.5.2 gated-delta kernels (no reference fallback)",
            **runtime,
        },
        attestation_per_row=[
            "model_id",
            "model_revision (attested)",
            "adapter_version",
            "backend",
            "model_config_sha256",
            "native_model_sha256",
            "runtime_source_sha256",
            "qualified runtime (shard manifest)",
            "source_input_sha256 = digest of the training row's native prompt",
        ],
        provenance_caveat=PROVENANCE_CAVEAT,
    )
    fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    report = build(parser.parse_args(argv))
    print(
        json.dumps(
            {k: report[k] for k in ("wave", "rows", "prompts", "content_sha256")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
