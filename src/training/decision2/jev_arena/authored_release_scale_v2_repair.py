"""Version an impossible mass witness without rewriting a frozen casebook.

This narrow candidate repair changes only left-side necessity completions
where gross mass is below the fixed tare. It chooses a physically possible
net mass one unit below the case's minimum and rechecks every oracle proof.
Private facts and snapshots must remain on the authorized remote host.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import file_sha, inspect, write_private
from jev_arena.authored_release_scale_v2 import _domain_witness_issues

VERSION = "jevarena-authored-release-scale-v2-mass-witness-repair/2"


def _write_preserving_order(path: Path, source: dict[str, Any]) -> None:
    """Keep the authored structured-field order used by the prompt renderer."""
    path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)


def repair_case(case: dict[str, Any]) -> tuple[dict[str, Any], int]:
    result = copy.deepcopy(case)
    if result["operation"] != "net_range":
        return result, 0
    minimum = result["params"]["minimum"]
    if type(minimum) is not int or minimum <= 1:
        raise ValueError("Mass witness repair needs an integer minimum above one")
    fixed_right = result["sources"][1]["data"]
    variant = result["variant"]
    changed = 0
    for phase, group in (
        ("original", result["witnesses"]),
        ("variant", result["variant_witnesses"]),
    ):
        right = (
            variant["data"]
            if phase == "variant" and variant["side"] == "right"
            else fixed_right
        )
        tare = right["tare"]
        for witness in group["left"]:
            if witness["gross"] < tare:
                witness["gross"] = tare + minimum - 1
                changed += 1
    if _domain_witness_issues(result):
        raise ValueError("Mass witness repair left invalid alternatives")
    inspect(result)
    return result, changed


def repair(casebook: Path, output: Path, receipt: Path) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise FileExistsError("Repaired candidate version cannot be overwritten")
    source = json.loads(casebook.read_text())
    before = sum(len(_domain_witness_issues(case)) for case in source["cases"])
    if before != 2:
        raise ValueError("Expected exactly two identified mass-witness defects")
    changed = 0
    new_cases = []
    for case in source["cases"]:
        repaired, count = repair_case(case)
        changed += count
        new_cases.append(repaired)
    if changed != before:
        raise ValueError("Unrepaired or unrelated witness change")
    source["cases"] = new_cases
    source["version"] = "authored-release-scale-v2-feasibility-r2"
    _write_preserving_order(output, source)
    record = {
        "version": VERSION,
        "input_casebook_sha256": file_sha(casebook),
        "output_casebook_sha256": file_sha(output),
        "changed_witnesses": changed,
        "originals": len(new_cases),
        "status": "REVISED_CANDIDATE_PREFLIGHT_PENDING",
        "release_qualified": False,
    }
    write_private(receipt, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    record = repair(args.casebook, args.output, args.receipt)
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "status",
                    "originals",
                    "changed_witnesses",
                    "input_casebook_sha256",
                    "output_casebook_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
