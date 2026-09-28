"""Release gate: bind the coordinator's decision to one verified package revision (stdlib).

The per-size gate (COORDINATION "Full-autonomy mandate") has a judgement part and
a mechanical part. The coordinator records the judgement in a decision file
(``dev2-release-decision/1``) naming the candidate identity, its same-panel report
and the paired comparison with the tier's own Decision 1.0 model. ``check`` runs
at build time; ``seal`` runs after the verification chain and writes the
``dev2-release-gate/1`` receipt that ``hub collect`` requires.

A decision is ``status: final`` (named ``decided_by``; the default when absent)
or ``status: draft`` (``prepared_by`` release engineering, no decider yet, e.g.
while an independent confirmation is pending). A draft binds the same identity,
report and paired comparison, so it can drive a private build, upload and
verification, but ``seal`` refuses it: nothing enters the collection on a draft.

  evaluate  print the six gate items with evidence from a work directory
  seal      write <work>/receipts/gate.json if every item passes
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from v2.release.layout import sha_file, write_json

DECISION_SCHEMA = "dev2-release-decision/1"
GATE_SCHEMA = "dev2-release-gate/1"
VERIFY_STEPS = (
    "repeat-pre",
    "card-pre",
    "upload",
    "download",
    "tree",
    "post",
    "repeat-post",
    "card-post",
    "readback",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(
    spec: dict[str, Any], decision_path: Path, *, final: bool = False
) -> dict[str, Any]:
    """The decision must approve exactly this candidate, report and comparison."""
    decision = _json(decision_path)
    scored = spec.get("scored") or {}
    problems = []
    status = decision.get("status", "final")
    if status not in ("draft", "final"):
        problems.append("status is draft or final")
    elif status == "draft" and (
        not decision.get("prepared_by") or decision.get("decided_by")
    ):
        problems.append("a draft names prepared_by and no decided_by")
    elif final and status != "final":
        problems.append("only a final decision can be sealed")
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("decision") != "release"
    ):
        problems.append("not a release decision")
    if (
        decision.get("model_name") != spec["model_name"]
        or decision.get("repo_id") != spec["repo_id"]
    ):
        problems.append("decision names a different model or repository")
    if decision.get("identity") != spec["expected_identity"]:
        problems.append("decision names a different checkpoint identity")
    if decision.get("report_sha256") != scored.get("report_sha256"):
        problems.append("decision names a different same-panel report")
    paired = spec["card"].get("paired")
    if not paired or decision.get("paired_sha256") != sha_file(Path(paired)):
        problems.append("decision names a different paired comparison")
    if not decision.get("rationale") or (
        status == "final" and not decision.get("decided_by")
    ):
        problems.append("decision needs a rationale and, when final, decided_by")
    if problems:
        raise ValueError(
            "Release decision does not approve this spec: " + "; ".join(problems)
        )
    return decision


def evaluate(work: Path) -> dict[str, Any]:
    receipts = work / "receipts"
    spec = _json(receipts / "spec.json")
    steps = {
        name: _json(receipts / f"{name}.json")
        for name in VERIFY_STEPS
        if (receipts / f"{name}.json").is_file()
    }
    build = _json(receipts / "build.json")
    paired = _json(Path(spec["card"]["paired"]))
    ci = paired["ci95"]
    low = ci["low"] if isinstance(ci, dict) else ci[0]
    readback = steps.get("readback", {})
    download = steps.get("download", {})
    # Download receipts written before they carried "passed" count only if they name the upload.
    downloaded = download.get(
        "passed",
        bool(download)
        and download.get("revision") == steps.get("upload", {}).get("revision"),
    )
    items = {
        "1_beats_own_1_0": {
            "passed": low > 0,
            "evidence": f"paired v3 95% interval low {low:+.3f} (sha {sha_file(Path(spec['card']['paired']))[:12]})",
        },
        "2_regressions_disclosed": {
            "passed": readback.get("passed", False)
            and not readback.get("card_problems"),
            "evidence": f"{len(build['card']['tradeoffs'])} results below own 1.0 listed in the card tradeoffs table",
        },
        "3_download_hash_parameters": {
            "passed": bool(downloaded)
            and all(steps.get(s, {}).get("passed") for s in ("tree", "post"))
            and steps.get("post", {}).get("loaded_parameters")
            == build["parameters"]["loaded"],
            "evidence": f"tree {steps.get('tree', {}).get('files')} files re-hashed; loaded {steps.get('post', {}).get('loaded_parameters')}",
        },
        "4_native_examples": {
            "passed": steps.get("post", {}).get("passed", False)
            and steps.get("card-post", {}).get("passed", False),
            "evidence": "examples and the card's Python block reproduced from the download",
        },
        "5_output_consistency": {
            "passed": all(
                steps.get(s, {}).get("passed") for s in ("repeat-pre", "repeat-post")
            )
            and all(
                _json(receipts / f"{s}.json")["passed"]
                for s in ("parity-pre", "parity-post")
                if (receipts / f"{s}.json").is_file()
            ),
            "evidence": "cross-process and pre/post-download answers identical; scored-panel parity where run",
        },
        "6_card_design": {
            "passed": readback.get("passed", False),
            "evidence": f"Hub card {readback.get('card_data', {}).get('license')} / problems {readback.get('card_problems')}",
        },
    }
    decision_path = Path(spec["gate_receipt"]) if spec.get("gate_receipt") else None
    return {
        "items": items,
        "passed": all(item["passed"] for item in items.values()),
        "decision": decision_path
        and {
            "sha256": sha_file(decision_path),
            "status": _json(decision_path).get("status", "final"),
        },
        "spec": spec,
        "steps": steps,
        "build": build,
    }


def seal(work: Path) -> dict[str, Any]:
    result = evaluate(work)
    spec = result["spec"]
    if spec["kind"] != "release" or not spec.get("gate_receipt"):
        raise ValueError("Only release specs with a coordinator decision can be sealed")
    decision = check(spec, Path(spec["gate_receipt"]), final=True)
    if not result["passed"]:
        failing = [name for name, item in result["items"].items() if not item["passed"]]
        raise ValueError(f"Gate items failed: {failing}")
    upload = result["steps"]["upload"]
    gate = {
        "schema": GATE_SCHEMA,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": "release",
        "repo_id": spec["repo_id"],
        "model_name": spec["model_name"],
        "revision": upload["revision"],
        "manifest_sha256": upload["manifest_sha256"],
        "decision_sha256": sha_file(Path(spec["gate_receipt"])),
        "decided_by": decision["decided_by"],
        "items": result["items"],
        "receipts_sha256": {
            name: sha_file(work / "receipts" / f"{name}.json")
            for name in result["steps"]
        },
    }
    write_json(work / "receipts" / "gate.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("evaluate", "seal"))
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate(args.work)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "decision": result["decision"],
                    "items": result["items"],
                },
                indent=2,
            )
        )
        sys.exit(0 if result["passed"] else 1)
    gate = seal(args.work)
    print(json.dumps({"gate": "sealed", "revision": gate["revision"]}))


if __name__ == "__main__":
    main()
