"""Back up the gold-free files of one C1 post-key job to the private eval-artifacts dataset (hf-cli python, node A).

The registry entry of a tier names a backup of its C1 run (as the 0.6B post-key run and the event runs have). This
uploads the job's plan, verification, smoke parity, sealed predictions (no prompts, gold, salt or key), seal,
report, gate, summary and log with a SHA256SUMS file to c1-postkey/<tier>/<job>/, then downloads every file at the
new commit into a fresh cache and re-hashes it.

    <hf-cli python> backup_c1.py --job /data/dev2/runs/eval/c1-postkey/<tier>/<job> --receipt OUT.json
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = "llm-semantic-router/decision-2.0-eval-artifacts"
FILES = (
    "GATE-C1-vs-baseline.json",
    "PLAN.json",
    "POSTKEY.log",
    "SUMMARY.json",
    "VERIFY.json",
    "cand/COLLECT.json",
    "cand/GPU-TIME.json",
    "cand/REPORT-C1.json",
    "cand/SEAL-C1.json",
    "cand/output/sealed-c1.predictions.jsonl",
    "cand/output/sealed-c1.predictions.jsonl.manifest.json",
    "smoke/PARITY.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise SystemExit(f"{args.receipt} exists")
    job = args.job
    tier = job.parent.name
    prefix = f"c1-postkey/{tier}/{job.name}"
    api = HfApi()
    if api.dataset_info(REPO).private is not True:
        raise SystemExit(f"{REPO} is not private")
    if any(
        f.startswith(prefix + "/")
        for f in api.list_repo_files(REPO, repo_type="dataset")
    ):
        raise SystemExit(f"{prefix}/ already exists")
    hashes = {name: sha(job / name) for name in FILES}
    sums = "".join(f"{hashes[n]}  {n}\n" for n in FILES).encode()
    ops = [CommitOperationAdd(f"{prefix}/{n}", str(job / n)) for n in FILES]
    ops.append(CommitOperationAdd(f"{prefix}/SHA256SUMS", sums))
    commit = api.create_commit(
        REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"C1 post-key job {tier}/{job.name}: gold-free files and SHA256SUMS",
    )
    readback = {}
    with tempfile.TemporaryDirectory(dir="/data/dev2/tmp") as cache:
        for name in (*FILES, "SHA256SUMS"):
            path = hf_hub_download(
                REPO,
                f"{prefix}/{name}",
                repo_type="dataset",
                revision=commit.oid,
                cache_dir=cache,
            )
            readback[name] = sha(Path(path))
    expected = {**hashes, "SHA256SUMS": hashlib.sha256(sums).hexdigest()}
    receipt = {
        "schema": "dev2-c1-postkey-backup/1",
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": REPO,
        "private": True,
        "revision": commit.oid,
        "prefix": prefix,
        "files_sha256": expected,
        "readback_equal": readback == expected,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "revision": commit.oid,
                "files": len(expected),
                "readback_equal": receipt["readback_equal"],
            }
        )
    )
    return 0 if receipt["readback_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
