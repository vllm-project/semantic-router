"""Source-level C1-independence check for the ~27B M4 A7 slice (host side, stdlib, CPU).

Reads one frozen M4 training file and M3-A's frozen file, keeps the rows that are new in M4 (not
byte-identical to an M3-A row, which the event-3 recheck already covered by containment), and reports:

- the distinct ``source`` / ``family`` / ``task_type`` / ``language`` values of those rows, with row
  counts;
- for each of the 25 datasets of the recorded C1 source registry (``sealed-c1-source-registry-2026-09-28``;
  C1 v1.1 draws its 8 sources from it), the number of rows whose metadata or text names it.

It never opens the sealed directory. Only counts and field values are printed, never row text.
Usage: python3 m4_c1_sources.py --train FILE --m3a FILE --output JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REGISTRY = {
    "Bekhouche/HalluTruthQA-4K": ["hallutruthqa"],
    "Sami2305341176/JudgmentBench": ["judgmentbench"],
    "dreadnode/scopejudge": ["scopejudge"],
    "SpaceHunterInf/DeliChess": ["delichess"],
    "jaelly/MCJudgeBench": ["mcjudgebench"],
    "Flaglab/esnlir-al-annotated-test": ["esnlir", "esnli-r"],
    "BRi002/SentiTaglishProductsAndServices": ["sentitaglish"],
    "Hplss/wb-review-dataset": ["wb-review"],
    "teplitsa-soc-tech/factbutcher-benchmark": ["factbutcher"],
    "CLS-Lab/narrative-gold-annotations": ["narrative-gold-annotations"],
    "allenai/tutormoments-preview": ["tutormoments"],
    "lbrenap1/mining-legal-arguments-us-corporate-case-law": ["mining-legal-arguments"],
    "McGill-NLP/ImplicatureX": ["implicaturex"],
    "laallein/ClimateCause": ["climatecause"],
    "NordosoftOy/innoduel-rlhf-real-world-human-preferences-sample": ["innoduel"],
    "Nandan007/NepFakeV2": ["nepfake"],
    "alisa-yingjia-wan/gapa": ["alisa-yingjia-wan"],
    "HiTZ/EusExams-v2": ["eusexams"],
    "LunaTsai/tiktok-political-stance-dataset-taiwan-2025": ["tiktok-political-stance"],
    "NUAA-MMMI/VARM-Bench": ["varm-bench"],
    "giuseppe-aiello/stance-detection-it-dataset": ["stance-detection-it"],
    "lab-flair/constructcie": ["constructcie"],
    "HiTZ/safety-GuardEUS": ["guardeus"],
    "iamjayeshc/reddit-self-medication-claim-dataset": ["self-medication-claim"],
    "ministere-culture/comparia-fr-arena": ["comparia"],
}
FIELDS = ("source", "family", "task_type", "language")
TEXT_SKIP = {"id", "group_id", "input_sha256"}


def strings(value, key=""):
    if isinstance(value, str):
        if key not in TEXT_SKIP:
            yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from strings(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--m3a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    known = set(args.m3a.read_bytes().splitlines())
    counts = {field: Counter() for field in FIELDS}
    keys = Counter()
    hits = {name: {"metadata_rows": 0, "text_rows": 0} for name in REGISTRY}
    new_rows = 0
    for line in args.train.read_bytes().splitlines():
        if line in known:
            continue
        new_rows += 1
        row = json.loads(line)
        keys.update(row.keys())
        for field in FIELDS:
            counts[field][str(row.get(field))] += 1
        meta = " ".join(str(row.get(f, "")) for f in FIELDS).lower()
        text = " ".join(v for _, v in strings(row)).lower()
        for name, needles in REGISTRY.items():
            probes = [name.lower(), *needles]
            if any(p in meta for p in probes):
                hits[name]["metadata_rows"] += 1
            if any(p in text for p in probes):
                hits[name]["text_rows"] += 1
    report = {
        "schema": "decision2-27b-m4-c1-sources/1",
        "train": str(args.train),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "m3a_sha256": hashlib.sha256(args.m3a.read_bytes()).hexdigest(),
        "rows_new_vs_m3a": new_rows,
        "row_keys": dict(sorted(keys.items())),
        "values": {f: dict(sorted(c.items())) for f, c in counts.items()},
        "registry_datasets": len(REGISTRY),
        "registry_hits": hits,
        "clean": all(
            v["metadata_rows"] == 0 and v["text_rows"] == 0 for v in hits.values()
        ),
    }
    args.output.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {k: report[k] for k in ("rows_new_vs_m3a", "registry_datasets", "clean")}
    summary["sources"] = report["values"]["source"]
    summary["hits"] = {
        k: v for k, v in hits.items() if v["metadata_rows"] or v["text_rows"]
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
