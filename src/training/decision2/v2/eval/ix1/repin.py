"""Move a published Decision Index run's pin to a runtime-only successor revision with the same weights.

    python3 -m v2.eval.ix1.repin weights --current WEIGHTS_JSON --package DIR --revision SHA --out FILE
    python3 -m v2.eval.ix1.repin spotcheck --report SPOTCHECK_JSON --stored RESULTS --check RESULTS [...] --out FILE

``weights``: the run's ``harness/weights-vs-release.json`` with its ``release`` block read from the successor's
downloaded package (repository, revision, weights identity, loaded parameters, profile, input limit, base) and its
checks recomputed against the unchanged ``scored_package`` block; ``match`` only when every check holds. Exits 1
otherwise.

``spotcheck``: ``v2.eval.ix1.submission spotcheck``'s report with each flipped question's choices and top-two margins
in both runs, and ``all_flips_near_ties`` under the published spot checks' rule. Exits 1 when a status differs or a
flip is not a near tie.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from v2.eval.ix1.merge import final_records
from v2.eval.ix1.submission import _read_lines, _records

NEAR_TIE = 0.025
NEAR_TIE_RULE = "a flip is a near tie when the top two options are within 0.025 of each other in at least one of the two runs"


def release_block(package: Path, revision: str) -> dict[str, Any]:
    m = json.loads((package / "MODEL_MANIFEST.json").read_text())
    base = m.get("base")
    return {
        "base": (
            {"repo_id": base["repo_id"], "revision": base["revision"]} if base else None
        ),
        "identity": {
            "fingerprint_files": m["identity"]["fingerprint_files"],
            "model_sha256": m["identity"]["model_sha256"],
        },
        "max_input_tokens": m["max_input_tokens"],
        "model_name": m["model_name"],
        "parameters_loaded": m["parameters"]["loaded"],
        "profile": m["profile"],
        "repo_id": m["repo_id"],
        "revision": revision,
    }


def weights(current: dict[str, Any], package: Path, revision: str) -> dict[str, Any]:
    release = release_block(package, revision)
    scored = current["scored_package"]
    checks = {
        "base": release["base"] == scored["base"],
        "max_input_tokens": release["max_input_tokens"] == scored["max_input_tokens"],
        "model_sha256": release["identity"]["model_sha256"]
        == scored["identity"]["model_sha256"],
        "parameters_loaded": release["parameters_loaded"]
        == scored["parameters_loaded"],
        "profile": release["profile"] == scored["profile"],
        "release_fingerprint_files_vs_scored_package_files": release["identity"][
            "fingerprint_files"
        ]
        == scored["files_sha256_recomputed"],
    }
    return {
        **current,
        "checks": checks,
        "match": all(checks.values()),
        "release": release,
    }


def margin(answer: dict[str, Any]) -> float | None:
    if answer.get("type") == "noul":
        return None
    top = sorted(answer["probabilities"].values(), reverse=True)
    return round(top[0] - top[1], 6) if len(top) > 1 else None


def near_ties(
    report: dict[str, Any], stored_path: Path, check_paths: list[Path]
) -> dict[str, Any]:
    stored = _records(_read_lines(stored_path))
    check: dict[str, dict[str, Any]] = {}
    for path in check_paths:
        records, _ = final_records(_read_lines(path))
        check.update(records)
    flipped = []
    for run_id in report["mismatched_run_ids"]:
        a, b = stored[run_id], check[run_id]
        if a["status"] != b["status"] or a["status"] != "ok":
            flipped.append(
                {
                    "run_id": run_id,
                    "stored_status": a["status"],
                    "rerun_status": b["status"],
                }
            )
            continue
        left, right = a["response"]["answers"], b["response"]["answers"]
        for question in sorted(left):
            x, y = left[question], right.get(question, {})
            if x.get("choice") != y.get("choice"):
                flipped.append(
                    {
                        "question": question,
                        "rerun_choice": y.get("choice"),
                        "rerun_top2_margin": margin(y) if y else None,
                        "run_id": run_id,
                        "stored_choice": x.get("choice"),
                        "stored_top2_margin": margin(x),
                    }
                )
    near = all(
        "question" in f
        and any(
            m is not None and m <= NEAR_TIE
            for m in (f["stored_top2_margin"], f["rerun_top2_margin"])
        )
        for f in flipped
    )
    return {
        **report,
        "all_flips_near_ties": near,
        "flipped_questions": flipped,
        "near_tie_rule": NEAR_TIE_RULE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("weights")
    w.add_argument("--current", type=Path, required=True)
    w.add_argument("--package", type=Path, required=True)
    w.add_argument("--revision", required=True)
    w.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("spotcheck")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--stored", type=Path, required=True)
    s.add_argument("--check", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "weights":
        out = weights(json.loads(args.current.read_text()), args.package, args.revision)
        ok = out["match"]
        summary = {"match": ok, "checks": out["checks"], "revision": args.revision}
    else:
        out = near_ties(json.loads(args.report.read_text()), args.stored, args.check)
        ok = out["all_flips_near_ties"] and set(out["statuses"]) <= {
            "ok",
            "unsupported",
        }
        summary = {
            k: out[k]
            for k in (
                "requests",
                "statuses",
                "status_or_choice_mismatches",
                "max_abs_dp",
                "all_flips_near_ties",
            )
        }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
