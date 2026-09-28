"""Private staging of an M6 soup package (dry run by default; stdlib only).

    python3 -m v2.06b.m6_stage NAME --formal-run DIR --numbers NUMBERS.json --source-commit SHA \
        [--full DIR] [--dest DIR] [--upload [--allow-non-successor]] [--hf hf] [--hf-python PY]

Builds DEST (default /data/dev2/runs/06b/m6/staging/NAME) from the soup run's `full/`
directory (default /data/dev2/runs/06b/m1/arms/NAME/full): the `best-export` contents,
`dev2-staging/best-export.MANIFEST.json`, `dev2-staging/SOUP.json`, `dev2-staging/STAGING.json`
(source run, state and export-manifest hashes, formal run dir and its M6-SUMMARY.json hash) and
a short README.md (private staging checkpoint, not a release). NUMBERS.json holds only scalar
development and post-key same-panel numbers: {"development": {...}, "post_key_same_panel": {...}}.
The export is verified against its manifest before copying and DEST is validated after.

Only with --upload (and only for a successor per M6-SUMMARY.json unless
--allow-non-successor): `hf repos create llm-semantic-router/dev2-staging-06bm6-NAME --type
model --private`, `hf upload REPO DEST . --repo-type model --commit-message ...`, then every file
is checked against HfApi().model_info(REPO, files_metadata=True) (size, and LFS sha256 or git
blob id) and the revision is printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ARMS = Path("/data/dev2/runs/06b/m1/arms")
STAGING_ROOT = Path("/data/dev2/runs/06b/m6/staging")
ORG = "llm-semantic-router"
PREFIX = "dev2-staging-06bm6-"
HF_PYTHON = "/data/dev2/tools/hf-cli/bin/python"
EXTRA = (
    "README.md",
    "dev2-staging/STAGING.json",
    "dev2-staging/SOUP.json",
    "dev2-staging/best-export.MANIFEST.json",
)
ADDRESS = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b|@[A-Za-z0-9.-]+:")


def sha_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def local_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def verify_export(full: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    soup = json.loads((full / "SOUP.json").read_text(encoding="utf-8"))
    manifest_path = full / "best-export.MANIFEST.json"
    manifest_sha = sha_file(manifest_path)
    if soup.get("best_export_manifest_sha256") != manifest_sha:
        raise ValueError(
            "SOUP.json export manifest hash differs from best-export.MANIFEST.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    export = full / "best-export"
    if local_files(export) != sorted(manifest):
        raise ValueError("best-export file list differs from its manifest")
    for rel, meta in manifest.items():
        path = export / rel
        if path.stat().st_size != meta["bytes"] or sha_file(path) != meta["sha256"]:
            raise ValueError(f"best-export/{rel} differs from its manifest")
    return soup, manifest, manifest_sha


def scalars(section: Any, where: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError(f"numbers: {where} must be an object")
    for key, value in section.items():
        if not isinstance(value, (int, float, str, bool)) or (
            isinstance(value, str) and len(value) > 80
        ):
            raise ValueError(f"numbers: {where}.{key} must be a short scalar")
    return section


def render_numbers(section: dict[str, Any]) -> str:
    def show(value: Any) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    return "\n".join(f"- {key}: {show(value)}" for key, value in section.items())


def readme(name: str, repo: str, soup: dict[str, Any], numbers: dict[str, Any]) -> str:
    arms = ", ".join(i["arm"] for i in soup.get("ingredients", []))
    text = f"""---
license: other
private: true
---
# DEV2.0-0.6B M6 staging checkpoint `{name}` (private; not a release)

Private staging checkpoint `{repo}` of the Decision 2.0 0.6B track. It is **not a release** and
does not replace the released DEV2.0-0.6B unless the coordinator decides so.

- Weight origin: official `Qwen/Qwen3-0.6B-Base`, fine-tuned by the Decision 2.0 0.6B track; this
  package is a uniform same-init weight soup of: {arms}.
- Packaging profile `qwen-full`; native Choice / Noul / Score readout through
  `training.model.infer` at the 8,192-token cap (raw probabilities, no calibration temperature).
- Provenance: `dev2-staging/STAGING.json`, `dev2-staging/SOUP.json`,
  `dev2-staging/best-export.MANIFEST.json`.

## Development readouts (never release scores)

{render_numbers(numbers["development"])}

## Post-key same-panel numbers (JevArena v3 keys were opened earlier; JevBench public 231 is a public-subset rerun)

{render_numbers(numbers["post_key_same_panel"])}
"""
    if ADDRESS.search(text):
        raise ValueError("README would contain an address")
    return text


def build(
    full: Path,
    dest: Path,
    formal: Path,
    numbers_path: Path,
    name: str,
    source_commit: str,
) -> dict[str, Any]:
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    soup, _, manifest_sha = verify_export(full)
    numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers = {
        k: scalars(numbers.get(k), k) for k in ("development", "post_key_same_panel")
    }
    summary = formal / "M6-SUMMARY.json"
    repo = f"{ORG}/{PREFIX}{name}"
    shutil.copytree(full / "best-export", dest)
    meta = dest / "dev2-staging"
    meta.mkdir()
    shutil.copy2(full / "best-export.MANIFEST.json", meta / "best-export.MANIFEST.json")
    shutil.copy2(full / "SOUP.json", meta / "SOUP.json")
    staging = {
        "schema": "dev2-06b-m6-staging/1",
        "candidate": name,
        "repo": repo,
        "status": "private staging checkpoint (not a release)",
        "packaging_profile": "qwen-full",
        "weight_origin": "official Qwen/Qwen3-0.6B-Base fine-tuned by the Decision 2.0 0.6B track; "
        "uniform same-init weight soup",
        "runtime": "training.model.infer, 8,192-token native cap",
        "source_run": str(full),
        "state_sha256": soup.get("state_sha256"),
        "export_manifest_sha256": manifest_sha,
        "ingredients": soup.get("ingredients", []),
        "formal_run_dir": str(formal),
        "formal_summary_sha256": sha_file(summary) if summary.is_file() else None,
        "numbers_sha256": sha_file(numbers_path),
        "source_commit": source_commit,
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (meta / "STAGING.json").write_text(
        json.dumps(staging, indent=2, sort_keys=True) + "\n"
    )
    (dest / "README.md").write_text(readme(name, repo, soup, numbers), encoding="utf-8")
    return staging


def validate(dest: Path, full: Path) -> dict[str, Any]:
    problems = []
    manifest = json.loads(
        (full / "best-export.MANIFEST.json").read_text(encoding="utf-8")
    )["files"]
    expected = sorted([*manifest, *EXTRA])
    found = local_files(dest)
    if found != expected:
        problems.append(
            {
                "extra": sorted(set(found) - set(expected)),
                "missing": sorted(set(expected) - set(found)),
            }
        )
    for rel, meta in manifest.items():
        path = dest / rel
        if path.is_file() and (
            path.stat().st_size != meta["bytes"] or sha_file(path) != meta["sha256"]
        ):
            problems.append({"hash_mismatch": rel})
    for rel in ("best-export.MANIFEST.json", "SOUP.json"):
        if sha_file(dest / "dev2-staging" / rel) != sha_file(full / rel):
            problems.append({"copy_mismatch": rel})
    staging = json.loads(
        (dest / "dev2-staging" / "STAGING.json").read_text(encoding="utf-8")
    )
    if staging["export_manifest_sha256"] != sha_file(
        dest / "dev2-staging" / "best-export.MANIFEST.json"
    ):
        problems.append({"staging_manifest_hash": staging["export_manifest_sha256"]})
    for rel in ("README.md", "dev2-staging/STAGING.json"):
        if ADDRESS.search((dest / rel).read_text(encoding="utf-8")):
            problems.append({"address_in": rel})
    return {"ok": not problems, "files": len(found), "problems": problems}


VERIFY = r"""
import hashlib, json, os, sys
from huggingface_hub import HfApi
repo, root = sys.argv[1], sys.argv[2]
info = HfApi().model_info(repo, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
bad, checked = [], 0
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f); rel = os.path.relpath(p, root)
        data_len = os.path.getsize(p)
        s = remote.pop(rel, None)
        if s is None or s.size != data_len:
            bad.append(rel); continue
        if s.lfs is not None:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(8 << 20), b""):
                    h.update(b)
            ok = s.lfs.sha256 == h.hexdigest()
        else:
            with open(p, "rb") as fh:
                ok = s.blob_id == hashlib.sha1(b"blob %d\0" % data_len + fh.read()).hexdigest()
        bad += [] if ok else [rel]
        checked += 1
extra = sorted(k for k in remote if k != ".gitattributes")
print(json.dumps({"repo": repo, "revision": info.sha, "private": info.private, "checked": checked,
                  "mismatched": bad, "remote_only": extra, "ok": not bad and not extra}))
"""


def upload(
    dest: Path, repo: str, hf: str, hf_python: str, message: str
) -> dict[str, Any]:
    subprocess.run(
        [hf, "repos", "create", repo, "--type", "model", "--private"], check=True
    )
    subprocess.run(
        [
            hf,
            "upload",
            repo,
            str(dest),
            ".",
            "--repo-type",
            "model",
            "--commit-message",
            message,
        ],
        check=True,
    )
    out = subprocess.run(
        [hf_python, "-c", VERIFY, repo, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--numbers", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--allow-non-successor", action="store_true")
    parser.add_argument("--hf", default="hf")
    parser.add_argument("--hf-python", default=HF_PYTHON)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("name must be lowercase letters, digits and dashes")
    full = args.full or ARMS / args.name / "full"
    dest = args.dest or STAGING_ROOT / args.name
    info = build(
        full, dest, args.formal_run, args.numbers, args.name, args.source_commit
    )
    report = validate(dest, full)
    print(
        json.dumps(
            {
                "dest": str(dest),
                "repo": info["repo"],
                **report,
                "export_manifest_sha256": info["export_manifest_sha256"],
                "state_sha256": info["state_sha256"],
                "formal_summary_sha256": info["formal_summary_sha256"],
            },
            sort_keys=True,
        )
    )
    if not report["ok"]:
        return 1
    if not args.upload:
        print("dry run: nothing uploaded")
        return 0
    summary_path = args.formal_run / "M6-SUMMARY.json"
    successor = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        .get("successor", {})
        .get("verdict")
        if summary_path.is_file()
        else None
    )
    if successor is not True and not args.allow_non_successor:
        print(
            f"refusing upload: M6-SUMMARY successor verdict is {successor!r}",
            file=sys.stderr,
        )
        return 2
    result = upload(
        dest,
        info["repo"],
        args.hf,
        args.hf_python,
        f"DEV2.0-0.6B M6 private staging checkpoint {args.name} (not a release)",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
