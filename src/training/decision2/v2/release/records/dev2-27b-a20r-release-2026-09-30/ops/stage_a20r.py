"""Stage the 27B M4-A20r soup on node A through the private staging repo (hf-cli python, node token).

Coordinator note 2026-09-30 07:20: A20r is THE 27B successor. Its checkpoint (rank-64 adapter and head) moves
from node B to node A through llm-semantic-router/dev2-27b-staging, is re-hashed on node A, and the staging
copy is deleted afterwards with rewrite_history=False (node copies stay the durable store).

    <hf-cli python> stage_a20r.py upload --root M4-A20r-soup --receipt OUT.json            (node B)
    <hf-cli python> stage_a20r.py verify --download DIR --upload-receipt R.json --base DIR [--base DIR]
                                         --receipt OUT.json                                (node A, CPU)
    <hf-cli python> stage_a20r.py purge plan|apply --upload-receipt R.json --node-copy DIR --receipt OUT.json

upload refuses a public repository or an existing prefix. verify re-hashes every staged file against the upload
receipt and checks the adapter's base binding: every file of decision_config.json lora.source_fingerprint
re-hashes equal under each --base. purge deletes exactly the LFS objects of the prefix whose bytes a node copy
re-hashes to; older paths naming the same object (whose objects earlier cleanups already deleted) are listed.
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "llm-semantic-router/dev2-27b-staging"
PREFIX = "m4/M4-A20r-soup"
MODEL_SHA = "2e07451107a2ca9735064cc01f7e43e8522b4bae48f2d77bdf29f7ea2fe6362c"
LOADED = 26096775168
SEAL_PREFIX = "9d60e611"
api = HfApi()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_many(paths: list[Path]) -> list[str]:
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(sha, paths))


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write_new(path: str, value: dict) -> None:
    out = Path(path)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mapping(root: Path) -> dict[str, Path]:
    files = {
        f"checkpoint/{p.relative_to(root / 'soup/checkpoint')}": p
        for p in sorted((root / "soup/checkpoint").rglob("*"))
        if p.is_file()
    }
    files["package/PACKAGE.json"] = root / "package/PACKAGE.json"
    files["package/calibration.json"] = root / "package/calibration.json"
    files["package/ADOPTION.json"] = root / "ADOPTION.json"
    files["package/dev-calibration.json"] = root / "adopt/dev-calibration.json"
    return files


def upload(args: argparse.Namespace) -> int:
    root = Path(args.root)
    files = mapping(root)
    hashes = dict(zip(files, hash_many(list(files.values()))))
    package = json.loads((root / "package/PACKAGE.json").read_text())
    adoption = json.loads((root / "ADOPTION.json").read_text())
    soup = json.loads((root / "soup/checkpoint/soup_manifest.json").read_text())
    seal = sha(root / "formal/SEAL.json")
    problems = []
    if not (package["model_sha256"] == package["checkpoint_sha256"] == MODEL_SHA):
        problems.append("PACKAGE.json model identity differs")
    if package["loaded_parameters"] != LOADED:
        problems.append("loaded parameter count differs")
    if package["calibration"]["sha256"] != hashes["package/calibration.json"]:
        problems.append("calibration.json differs from PACKAGE.json")
    if set(package["calibration"]["temperature_by_type"].values()) != {1.0}:
        problems.append("the package is not T = 1")
    if (
        adoption["adopt"] is not False
        or adoption["receipt"]["sha256"] != hashes["package/dev-calibration.json"]
    ):
        problems.append("ADOPTION.json is not the T = 1 decision bound to its receipt")
    if soup["output"]["model_sha256"] != MODEL_SHA:
        problems.append("soup manifest identity differs")
    soup_files = soup["output"].get("files_sha256") or {}
    soup_bad = [n for n, h in soup_files.items() if hashes.get(f"checkpoint/{n}") != h]
    if not soup_files or soup_bad:
        problems.append(f"soup manifest files differ: {soup_bad}")
    if not seal.startswith(SEAL_PREFIX):
        problems.append("formal SEAL.json differs")
    info = api.model_info(REPO)
    if info.private is not True:
        problems.append("the staging repository is not private")
    existing = [f for f in api.list_repo_files(REPO) if f.startswith(PREFIX + "/")]
    if existing:
        problems.append(f"{PREFIX}/ already exists in {REPO}")
    lfs_before = sorted(f.file_oid for f in api.list_lfs_files(REPO))
    if problems:
        print("upload refused: " + "; ".join(problems), file=sys.stderr)
        return 1
    staging = {
        "schema": "dev2-27b-staging/1",
        "candidate": "M4-A20r-soup",
        "milestone": "27B Milestone 4 successor (coordinator 2026-09-30 07:20)",
        "created_utc": utc(),
        "hf_policy": "private; soup only (no individual seeds); base not uploaded; deleted with rewrite_history=False after the release",
        "profile": "qwen-adapter",
        "max_input_tokens": package["max_input_tokens"],
        "package_decision": package["decision"],
        "checkpoint": {
            "model_sha256": MODEL_SHA,
            "loaded_parameters": LOADED,
            "parameter_source": package["parameter_source"],
            "lora": soup["lora"],
            "soup_manifest_files_match": True,
            "base_source_files_sha256_digest": soup["source_files_sha256"],
        },
        "formal": {
            "run_dir_node_b": str(root / "formal"),
            "seal_sha256": seal,
            "report_sha256": sha(root / "formal/REPORT.json"),
            "collect_sha256": sha(root / "formal/COLLECT.json"),
        },
        "node_b_source_paths": {k: str(v) for k, v in files.items()},
        "files": {
            f"{PREFIX}/{k}": {"bytes": files[k].stat().st_size, "sha256": hashes[k]}
            for k in files
        },
    }
    manifest = Path(args.receipt).with_suffix(".STAGING.json")
    write_new(str(manifest), staging)
    ops = [CommitOperationAdd(f"{PREFIX}/{k}", str(p)) for k, p in files.items()]
    ops.append(CommitOperationAdd(f"{PREFIX}/STAGING.json", str(manifest)))
    started = time.time()
    commit = api.create_commit(
        REPO,
        operations=ops,
        commit_message="Stage 27B M4-A20r soup (DEV2.0-27B successor) for node A",
    )
    revision = commit.oid
    remote = {
        p.path: p
        for p in api.get_paths_info(
            REPO, [f"{PREFIX}/{k}" for k in files], revision=revision, expand=True
        )
    }
    bad = []
    for k, p in files.items():
        r = remote.get(f"{PREFIX}/{k}")
        if r is None:
            bad.append(k)
        elif r.lfs is not None:
            if r.lfs.sha256 != hashes[k]:
                bad.append(k)
        elif r.blob_id != git_blob(p):
            bad.append(k)
    receipt = {
        "schema": "dev2-27b-a20r-staging-upload/1",
        "repo": REPO,
        "private": True,
        "revision": revision,
        "prefix": PREFIX,
        "lfs_objects_before": lfs_before,
        "utc": utc(),
        "upload_seconds": round(time.time() - started, 1),
        "files": staging["files"],
        "staging_json_sha256": sha(manifest),
        "lfs": sorted(k for k in files if remote[f"{PREFIX}/{k}"].lfs is not None),
        "remote_mismatch": bad,
        "passed": not bad,
    }
    write_new(args.receipt, receipt)
    print(json.dumps({"revision": revision, "files": len(files), "passed": not bad}))
    return 0 if not bad else 1


def verify(args: argparse.Namespace) -> int:
    up = json.loads(Path(args.upload_receipt).read_text())
    root = Path(args.download)
    names = sorted(up["files"])
    present = [n for n in names if (root / n).is_file()]
    actual = dict(zip(present, hash_many([root / n for n in present])))
    bad = [n for n in names if actual.get(n) != up["files"][n]["sha256"]]
    extra = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and ".cache" not in p.parts
        and str(p.relative_to(root)) not in up["files"]
        and p.name != "STAGING.json"
    )
    config = json.loads((root / PREFIX / "checkpoint/decision_config.json").read_text())
    pinned = config["lora"]["source_fingerprint"]["files_sha256"]
    bases = {}
    for base in args.base:
        broot = Path(base)
        bnames = sorted(pinned)
        got = dict(zip(bnames, hash_many([broot / n for n in bnames])))
        bbad = [n for n in bnames if got[n] != pinned[n]]
        bases[base] = {"files": len(bnames), "mismatched": bbad, "bound": not bbad}
    result = {
        "schema": "dev2-27b-a20r-staging-verify/1",
        "utc": utc(),
        "repo": up["repo"],
        "revision": up["revision"],
        "download": str(root),
        "files": len(names),
        "mismatched": bad,
        "extra": extra,
        "staging_json_sha256": sha(root / PREFIX / "STAGING.json"),
        "base_binding": {
            "source": config["lora"]["source_fingerprint"].get("source_name"),
            "base_revision": config["lora"].get("base_revision"),
            "bases": bases,
        },
    }
    result["passed"] = (
        not bad
        and not extra
        and result["staging_json_sha256"] == up["staging_json_sha256"]
        and all(b["bound"] for b in bases.values())
    )
    write_new(args.receipt, result)
    print(
        json.dumps(
            {
                "files": len(names),
                "mismatched": len(bad),
                "extra": len(extra),
                "bases_bound": {k: v["bound"] for k, v in bases.items()},
                "passed": result["passed"],
            }
        )
    )
    return 0 if result["passed"] else 1


def purge(args: argparse.Namespace) -> int:
    up = json.loads(Path(args.upload_receipt).read_text())
    copy = Path(args.node_copy)
    refs = api.list_repo_refs(REPO)
    heads = [b.target_commit for b in refs.branches] + [
        t.target_commit for t in refs.tags
    ]
    before_commits = [c.commit_id for c in api.list_repo_commits(REPO)]
    by_sha = {f.file_oid: f for f in api.list_lfs_files(REPO)}
    wanted = {up["files"][f"{PREFIX}/{n}"]["sha256"]: n for n in up["lfs"]}
    others: dict[str, list[str]] = {}
    for head in heads:
        for item in api.list_repo_tree(REPO, revision=head, recursive=True):
            lf = getattr(item, "lfs", None)
            if lf is not None and not item.path.startswith(PREFIX + "/"):
                others.setdefault(lf.sha256, []).append(item.path)
    targets, kept = {}, {}
    for digest, name in wanted.items():
        node = copy / name
        if digest not in by_sha:
            kept[name] = "no LFS object with this SHA-256 (already deleted)"
        elif not node.is_file() or sha(node) != digest:
            kept[name] = "no node copy re-hashes to it"
        else:
            targets[name] = digest
    receipt = {
        "schema": "dev2-27b-a20r-staging-purge/1",
        "utc": utc(),
        "repo": REPO,
        "prefix": PREFIX,
        "mode": args.mode,
        "rewrite_history": False,
        "node_copy": str(copy),
        "targets": targets,
        "kept": kept,
        "also_named_by_other_paths": {
            d: others[d] for d in targets.values() if d in others
        },
        "lfs_objects_before": len(by_sha),
        "commits_before": len(before_commits),
        "head_before": heads,
    }
    if args.mode == "apply" and targets:
        api.permanently_delete_lfs_files(
            REPO, [by_sha[d] for d in targets.values()], rewrite_history=False
        )
        time.sleep(10)
        after_lfs = {f.file_oid for f in api.list_lfs_files(REPO)}
        refs = api.list_repo_refs(REPO)
        receipt.update(
            deleted=sorted(targets.values()),
            gone=all(d not in after_lfs for d in targets.values()),
            head_after=[b.target_commit for b in refs.branches]
            + [t.target_commit for t in refs.tags],
            commits_after=len(api.list_repo_commits(REPO)),
        )
        receipt["history_unchanged"] = receipt["head_after"] == heads and receipt[
            "commits_after"
        ] == len(before_commits)
        receipt["passed"] = receipt["gone"] and receipt["history_unchanged"]
    write_new(args.receipt, receipt)
    print(
        json.dumps(
            {
                k: receipt.get(k)
                for k in ("mode", "targets", "kept", "gone", "history_unchanged")
            }
        )
    )
    return 0 if args.mode == "plan" or receipt.get("passed") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("upload")
    p.add_argument("--root", required=True)
    p.add_argument("--receipt", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--download", required=True)
    p.add_argument("--upload-receipt", required=True)
    p.add_argument("--base", action="append", required=True)
    p.add_argument("--receipt", required=True)
    p = sub.add_parser("purge")
    p.add_argument("mode", choices=("plan", "apply"))
    p.add_argument("--upload-receipt", required=True)
    p.add_argument("--node-copy", required=True)
    p.add_argument("--receipt", required=True)
    args = parser.parse_args()
    return {"upload": upload, "verify": verify, "purge": purge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
