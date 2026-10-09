"""Build, verify, publish and fetch the Decision Index 0.3 public suite.

Edition 0.3 reads the 0.2.1 files unchanged plus ``gsm8k-rows.jsonl.gz`` (2,638 rebuilt GSM8K
requests). Every step goes through the official kit (``decision_index``) so the hash checks are the
kit's own: ``gsm8k_v3.build/write`` builds the GSM8K rows from the pinned GSM8K test file and the
release-v2 rows, ``download.from_local`` stages the edition with the kit's ``hub/0.3`` manifest and
exclusion list, and ``Suite.verify(strict=True)`` refuses any mismatch.

    python -m d25.vega.eval.suite03 build --kit KIT --prior SUITE_0.2_DIR --work WORK --out SUITE_0.3_DIR
    python -m d25.vega.eval.suite03 verify --kit KIT --dir SUITE_0.3_DIR
    python -m d25.vega.eval.suite03 upload --dir SUITE_0.3_DIR --repo vllm-sr/d25-index-suite-0.3
    python -m d25.vega.eval.suite03 fetch --kit KIT --repo vllm-sr/d25-index-suite-0.3 --out SUITE_0.3_DIR

The suite stays private (upstream source terms): the Hub copy is a private dataset.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILE = "main/test-00000-of-00001.parquet"
EDITION = "0.3"
SUM_FILE = "SHA256SUMS"
BUILD_FILE = "BUILD.json"


def use_kit(kit: str | Path) -> None:
    kit = str(Path(kit).resolve())
    if kit not in sys.path:
        sys.path.insert(0, kit)


def sha256(path: Path, gunzip: bool = False) -> str:
    opener = gzip.open if gunzip else open
    with opener(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def counts(directory: Path) -> dict:
    from decision_index import editions
    from decision_index.scoring import index02
    from decision_index.suite.io import Suite

    suite = Suite(directory, EDITION)
    added = {int(n) for n in index02.spec(EDITION)["added"]}
    by_bench: collections.Counter = collections.Counter()
    base = extra = questions = 0
    for row in suite.rows(apply_exclusions=True):
        n = row["_evaluation"]["catalog_id"]
        by_bench[n] += 1
        questions += len(row["questions"])
        if n in added:
            extra += 1
        else:
            base += 1
    e = editions.get(EDITION)
    return {
        "scoreable_base": base,
        "scoreable_added": extra,
        "scoreable_total": base + extra,
        "expected_base": e["scoreable"],
        "expected_added": e["added_requests"],
        "questions": questions,
        "benchmarks": dict(sorted(by_bench.items())),
        "ok": base == e["scoreable"] and extra == e["added_requests"],
    }


def write_sums(directory: Path) -> dict:
    from decision_index import editions

    names = [n for n in editions.files(EDITION)]
    sums = {}
    for name in names:
        path = directory / name
        entry = {"sha256": sha256(path)}
        if name.endswith(".gz"):
            entry["uncompressed_sha256"] = sha256(path, gunzip=True)
        sums[name] = entry
    lines = [f"{v['sha256']}  {k}" for k, v in sums.items()]
    (directory / SUM_FILE).write_text("\n".join(lines) + "\n")
    return sums


def verify(directory: Path) -> dict:
    from decision_index.suite.io import Suite

    report = Suite(directory, EDITION).verify(strict=True)
    report["counts"] = counts(directory)
    if not report["counts"]["ok"]:
        raise ValueError(
            f"scoreable counts differ from the edition: {report['counts']}"
        )
    return report


def cmd_build(args) -> dict:
    use_kit(args.kit)
    from decision_index import editions
    from decision_index.suite.build import gsm8k_v3
    from decision_index.suite.download import from_local
    from huggingface_hub import hf_hub_download

    prior, work, out = Path(args.prior), Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    test = Path(
        hf_hub_download(
            GSM8K_REPO,
            GSM8K_FILE,
            repo_type="dataset",
            revision=GSM8K_REVISION,
            cache_dir=str(work / "hf-cache"),
        )
    )
    rows = gsm8k_v3.build(test, prior / editions.ROWS_FILE)
    gsm = gsm8k_v3.write(rows, work / "release-v3-rebuilt")
    expected = editions.get(EDITION)["gsm8k_sha256"]
    if gsm["gsm8k_sha256"] != expected:
        raise ValueError(f"rebuilt GSM8K rows {gsm['gsm8k_sha256']} != {expected}")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    imported = from_local(
        tmp,
        prior / editions.ROWS_FILE,
        verify=True,
        edition=EDITION,
        added=prior / editions.ADDED_FILE,
        gsm8k=Path(gsm["out"]),
    )
    report = verify(tmp)
    sums = write_sums(tmp)
    kit_commit = os.environ.get("KIT_COMMIT", "")
    build = {
        "edition": EDITION,
        "kit_commit": kit_commit,
        "prior_suite": str(prior),
        "prior_sha256": {
            n: sha256(prior / n) for n in (editions.ROWS_FILE, editions.ADDED_FILE)
        },
        "gsm8k_source": {
            "repo": GSM8K_REPO,
            "revision": GSM8K_REVISION,
            "file": GSM8K_FILE,
            "sha256": sha256(test),
        },
        "gsm8k_build": gsm,
        "import": imported,
        "verify": report,
        "files": sums,
    }
    (tmp / BUILD_FILE).write_text(json.dumps(build, indent=2) + "\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return build


def cmd_verify(args) -> dict:
    use_kit(args.kit)
    directory = Path(args.dir)
    report = verify(directory)
    if (directory / SUM_FILE).exists():
        listed = dict(
            reversed(line.split("  ", 1))
            for line in (directory / SUM_FILE).read_text().splitlines()
            if line
        )
        report["sha256sums_match"] = all(
            sha256(directory / name) == digest for name, digest in listed.items()
        )
        if not report["sha256sums_match"]:
            raise ValueError("SHA256SUMS mismatch")
    return report


def cmd_upload(args) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise RuntimeError(f"{args.repo} is not private; refusing to upload the suite")
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.dir,
        path_in_repo=".",
        commit_message="Decision Index 0.3 public suite (kit-verified)",
    )
    return {
        "repo": args.repo,
        "private": True,
        "commit": getattr(commit, "oid", str(commit)),
    }


def cmd_fetch(args) -> dict:
    use_kit(args.kit)
    from decision_index import editions
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name in list(editions.files(EDITION)) + [SUM_FILE, BUILD_FILE]:
        local = hf_hub_download(
            args.repo, name, repo_type="dataset", revision=args.revision
        )
        shutil.copyfile(local, tmp / name)
    args.dir = str(tmp)
    report = cmd_verify(args)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    report["dir"] = str(out)
    return report


def main(argv=None) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--kit", required=True)
    b.add_argument(
        "--prior", required=True, help="0.2 suite dir with selected-rows/added-rows"
    )
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--kit", required=True)
    v.add_argument("--dir", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--dir", required=True)
    u.add_argument("--repo", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--kit", required=True)
    f.add_argument("--repo", required=True)
    f.add_argument("--revision")
    f.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = {
        "build": cmd_build,
        "verify": cmd_verify,
        "upload": cmd_upload,
        "fetch": cmd_fetch,
    }[args.cmd](args)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
