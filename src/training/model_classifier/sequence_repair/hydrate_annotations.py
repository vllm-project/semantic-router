"""Reconstruct reviewed requests from pinned source data and text-free label sidecars."""

# ruff: noqa: PLC0415
import argparse
import hashlib
import json
from pathlib import Path


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hydrate_records(sidecar, sources):
    repository = sidecar["source"]["repository"]
    result = []
    if repository == "CohereLabs/aya_dataset":
        lookup = {}
        for row in sources:
            if not row.get("user_id"):
                continue
            key = (
                sha256_text(row["inputs"]),
                row["language_code"],
                "aya-author-" + sha256_text(row["user_id"]),
            )
            lookup[key] = row["inputs"]
        for row in sidecar["records"]:
            key = (row["source_text_sha256"], row["language"], row["group_id"])
            if key not in lookup:
                raise ValueError(
                    "Reviewed Aya request or author differs from the pinned source"
                )
            result.append({**row, "text": lookup[key]})
    elif repository == "databricks/databricks-dolly-15k":
        for row in sidecar["records"]:
            source = sources[row["source_row"]]
            instruction, context = (
                source["instruction"].strip(),
                source["context"].strip(),
            )
            text = instruction + ("\n\n" + context if context else "")
            if sha256_text(text) != row["source_text_sha256"]:
                raise ValueError(
                    "Reviewed Dolly request differs from the pinned source"
                )
            result.append(
                {**row, "text": text, "instruction": instruction, "context": context}
            )
    else:
        raise ValueError("Unknown reviewed source")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite reviewed annotations")
    sidecar = json.loads(args.sidecar.read_text())
    if (
        hashlib.sha256(args.source.read_bytes()).hexdigest()
        != sidecar["source"]["file_sha256"]
    ):
        raise ValueError("Source file hash differs from the reviewed revision")
    if sidecar["source"]["repository"] == "CohereLabs/aya_dataset":
        from pyarrow import parquet

        sources = parquet.read_table(
            args.source, columns=["inputs", "language_code", "user_id"]
        ).to_pylist()
    else:
        sources = [
            json.loads(line)
            for line in args.source.read_text().split("\n")
            if line.strip()
        ]
    rows = hydrate_records(sidecar, sources)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(
        json.dumps(
            {"rows": len(rows), "sha256": hashlib.sha256(content.encode()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
