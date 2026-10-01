"""27B MoE milestone rule verdicts (host CPU, stdlib; reads ``moe-tail.sh gates`` outputs).

Per sealed finalist (preregistration ``records/moe-prereg-2026-10-01.md``, "Formal runs and verdicts"), the M5
verdict code (``v2.27b.m5.m5_verdicts``) on this milestone's gates directory:

* successor items 1-7 vs DEV2.0-27B (A20r's scored run): 1 v3 paired lower bound > 0; 2 human transfer not
  significantly below; 3 every ``gates types`` verdict OK; 4 mlx-diag card-eligible Choice + Noul not significantly
  below (node A ``mlx-paired`` R4); 5 v3 >= 64.92 and H not significantly below AutoJev-27B, Eikos-27B and
  Jebadiah-27B; 6 no overlap exposure of ``a20`` (``gates/overlap/exposure-moe-a20.json``); 7 ``gates public231`` not
  REGRESSION; item 8 (C1 post-key) is the eval custodian's and stays PENDING;
* beats AutoJev-27B: v3 > 72.133, paired lower bound vs AutoJev-27B > 0, H not significantly below it;
* tie-break among passers: highest paired v3 lower bound vs AutoJev-27B, then vs A20r, then fewer active parameters
  (``PACKAGE.json``).

    python3 -m v2.27b.moe.moe_verdicts --gates DIR --finalist NAME=FORMAL_RUN=PACKAGE.json [...] --output OUT.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

m5 = importlib.import_module("v2.27b.m5.m5_verdicts")

SCHEMA = "decision2-27b-moe-verdicts/1"
TRAIN = "a20"


def finalist(
    name: str, run: Path, package: Path, gates: Path, files: dict[str, str]
) -> dict[str, Any]:
    out = m5.finalist(name, run, gates, files)
    exposure = m5.read(gates / "overlap" / f"exposure-moe-{TRAIN}.json", files)
    items = out["successor_items"]
    items["6_no_overlap_exposure"] = {
        "exposure": {
            TRAIN: {
                "groups": len(exposure["groups"]),
                "methods_agree": exposure["methods_agree"],
            }
        },
        "pass": len(exposure["groups"]) == 0,
    }
    decided = [i["pass"] for k, i in items.items() if not k.startswith("8_")]
    out["successor_items_1_7"] = None if None in decided else all(decided)
    pkg = m5.read(package, files)
    out["package"] = {
        "path": str(package),
        "model_sha256": pkg["model_sha256"],
        "decision": pkg["decision"],
        "loaded_parameters": pkg["loaded_parameters"],
        "active_parameters": pkg["active_parameters"],
    }
    return out


def rank(out: dict[str, dict[str, Any]], names: list[str]) -> list[str]:
    return sorted(
        names,
        key=lambda n: (
            -out[n]["paired"]["autojev27"]["ci95"][0],
            -out[n]["paired"]["A20r"]["ci95"][0],
            out[n]["package"]["active_parameters"],
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument(
        "--finalist",
        action="append",
        required=True,
        help="NAME=FORMAL_RUN=PACKAGE.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    files: dict[str, str] = {}
    out = {}
    for spec in args.finalist:
        name, run, package = spec.split("=", 2)
        out[name] = finalist(name, Path(run), Path(package), args.gates, files)
    beating = rank(
        out,
        [
            n
            for n, v in out.items()
            if v["beats_autojev27"]["pass"] and v["successor_items_1_7"]
        ],
    )
    passing = rank(out, [n for n, v in out.items() if v["successor_items_1_7"]])
    result = {
        "schema": SCHEMA,
        "label": "post-key same-panel rule verdicts (node B); item 8 is the eval custodian's",
        "finalists": out,
        "beats_autojev_and_successor": beating,
        "successor_items_1_7": passing,
        "choice": beating[0] if beating else (passing[0] if passing else None),
        "inputs_sha256": dict(sorted(files.items())),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                n: {
                    "v3": round(v["v3"], 3),
                    "items_1_7": v["successor_items_1_7"],
                    "beats_autojev27": v["beats_autojev27"]["pass"],
                    "lb_vs_autojev27": round(v["paired"]["autojev27"]["ci95"][0], 3),
                    "lb_vs_A20r": round(v["paired"]["A20r"]["ci95"][0], 3),
                }
                for n, v in out.items()
            }
            | {"choice": result["choice"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
