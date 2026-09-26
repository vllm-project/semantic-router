"""Private domain-validity screen for v2 necessity witnesses.

The oracle can produce different answers from an impossible hypothetical
source. This screen opens only the private casebook and reports aggregate
counts; no source fact, answer, prompt, or candidate identifier is printed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import file_sha, write_private
from jev_arena.authored_release_scale_v2 import _domain_witness_issues

VERSION = "jevarena-authored-release-scale-v2-witness-audit/1"


def audit(casebook: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Private witness audit is immutable")
    source = json.loads(casebook.read_text())
    cases = source["cases"]
    issues = [(case["operation"], _domain_witness_issues(case)) for case in cases]
    affected = Counter(operation for operation, found in issues if found)
    report = {
        "version": VERSION,
        "casebook_sha256": file_sha(casebook),
        "candidate_originals": len(cases),
        "domain_invalid_witnesses": sum(len(found) for _, found in issues),
        "affected_originals": sum(bool(found) for _, found in issues),
        "affected_mechanisms": dict(sorted(affected.items())),
        "status": "HOLD_BEFORE_BLIND_PACKET" if affected else "DOMAIN_SCREEN_PASS",
        "reviewer_packet_created": False,
        "release_qualified": False,
    }
    write_private(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.casebook, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "candidate_originals",
                    "domain_invalid_witnesses",
                    "affected_originals",
                    "affected_mechanisms",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
