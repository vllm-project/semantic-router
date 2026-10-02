"""Checks of the organization card-only revisions (2026-10-02); hf-cli python on the node, token in its default file.

compare      the fresh build against the released revision: only README.md changes, it is the released README
             with the organization renamed, MODEL_MANIFEST.json differs only in its Hub references, the README
             digests and the builder provenance, and no package text names the former organization
precheck     before the upload: the new ID resolves to itself at the current main, privately, and the former ID
             redirects to it
remote-load  (inside the release image, host network, fresh HF_HOME) AutoConfig and AutoTokenizer with
             trust_remote_code from the Hub under the new ID at the current main, as the card's snippet does
verify       after the upload: main is the new revision, private, the former ID redirects to it, exactly README.md
             and MODEL_MANIFEST.json changed against the superseded revision, and no text file of the repository
             names the former organization
scan         every text file of a repository revision (README, MODEL_MANIFEST.json, configs, code): occurrences of
             the former organization

<hf-cli python> org_check.py compare --repo R --revision OLD --package PKG --output OUT
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

FORMER, ORG = "llm-semantic-router", "vllm-sr"
WEIGHTS = (".safetensors", ".bf16z", ".bin", ".pt", ".png", ".jpg", ".model")
# Manifest fields that a card-only rebuild may change besides the Hub references.
REBUILT = ("builder", "files_sha256.README.md", "card.readme_sha256")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def text_file(name: str) -> bool:
    return not name.endswith(WEIGHTS)


def diff(old, new, path: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        return [
            p
            for k in sorted(set(old) | set(new))
            for p in diff(old.get(k), new.get(k), f"{path}.{k}" if path else k)
        ]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            p
            for i, (a, b) in enumerate(zip(old, new))
            for p in diff(a, b, f"{path}[{i}]")
        ]
    return [] if old == new else [path]


def allowed(path: str) -> bool:
    return any(path == r or path.startswith(r + ".") for r in REBUILT)


def compare(args) -> dict:
    from huggingface_hub import hf_hub_download

    from v2.release import layout

    package = args.package
    new_manifest = json.loads(
        (package / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as cache:
        fetch = lambda name: Path(  # noqa: E731
            hf_hub_download(args.repo, name, revision=args.revision, cache_dir=cache)
        ).read_text(encoding="utf-8")
        old_manifest, old_readme = json.loads(fetch("MODEL_MANIFEST.json")), fetch(
            "README.md"
        )
    old_files, new_files = old_manifest["files_sha256"], new_manifest["files_sha256"]
    changed = sorted(
        n
        for n in set(old_files) | set(new_files)
        if old_files.get(n) != new_files.get(n)
    )
    readme = (package / "README.md").read_text(encoding="utf-8")
    expected_readme = old_readme.replace(f"{FORMER}/", f"{ORG}/")
    manifest_paths = diff(layout.current_ids(old_manifest), new_manifest)
    leftovers = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and text_file(path.name):
            count = path.read_bytes().count(FORMER.encode())
            if count:
                leftovers[path.relative_to(package).as_posix()] = count
    problems = []
    if changed != ["README.md"]:
        problems.append(f"files that change: {changed} (only README.md may)")
    if readme != expected_readme:
        problems.append(
            "README.md is not the released README with the organization renamed"
        )
    if [p for p in manifest_paths if not allowed(p)]:
        problems.append(
            f"manifest fields that change: {[p for p in manifest_paths if not allowed(p)]}"
        )
    if leftovers:
        problems.append(f"package text still names {FORMER}: {leftovers}")
    return {
        "schema": "dev2-org-compare/1",
        "utc": now(),
        "repo": args.repo,
        "released_revision": args.revision,
        "files": [len(old_files), len(new_files)],
        "changed_files": changed,
        "readme_former_org_occurrences": [
            old_readme.count(FORMER),
            readme.count(FORMER),
        ],
        "readme_is_renamed_release_readme": readme == expected_readme,
        "manifest_changed_fields": manifest_paths,
        "package_former_org_occurrences": leftovers,
        "problems": problems,
        "passed": not problems,
    }


def _former(repo: str) -> str:
    return f"{FORMER}/{repo.split('/', 1)[1]}"


def precheck(args) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(args.repo)
    former = api.model_info(_former(args.repo))
    checks = {
        "new_id_resolves_to_itself": info.id == args.repo,
        "main_is_the_superseded_revision": info.sha == args.revision,
        "private": info.private is True,
        "former_id_redirects": former.id == args.repo and former.sha == info.sha,
    }
    return {
        "schema": "dev2-org-precheck/1",
        "utc": now(),
        "repo": args.repo,
        "main": info.sha,
        "checks": checks,
        "passed": all(checks.values()),
    }


def remote_load(args) -> dict:
    from huggingface_hub import scan_cache_dir
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        args.repo, revision=args.revision, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo, revision=args.revision, trust_remote_code=True
    )
    revisions = sorted(
        r.commit_hash
        for c in scan_cache_dir().repos
        if c.repo_id == args.repo
        for r in c.revisions
    )
    checks = {
        "remote_config": type(config).__name__ == "Decision2Config",
        "tokenizer": len(tokenizer) > 0,
        "downloaded_revision": revisions == [args.revision],
    }
    return {
        "schema": "dev2-org-remote-load/1",
        "utc": now(),
        "repo": args.repo,
        "revision": args.revision,
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "downloaded_revisions": revisions,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _files(api, repo: str, revision: str) -> dict[str, str]:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    out = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        digest = (
            (
                lfs.get("sha256")
                if isinstance(lfs, dict)
                else getattr(lfs, "sha256", None)
            )
            if lfs
            else None
        )
        out[sibling.rfilename] = digest or sibling.blob_id
    return out


def _scan(api, repo: str, revision: str) -> dict[str, int]:
    from huggingface_hub import hf_hub_download

    hits = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as cache:
        for name in sorted(_files(api, repo, revision)):
            if text_file(name):
                data = Path(
                    hf_hub_download(repo, name, revision=revision, cache_dir=cache)
                ).read_bytes()
                hits[name] = data.count(FORMER.encode())
    return hits


def verify(args) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(args.repo)
    former = api.model_info(_former(args.repo))
    old, new = _files(api, args.repo, args.superseded), _files(
        api, args.repo, args.revision
    )
    changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    hits = _scan(api, args.repo, args.revision)
    checks = {
        "main_is_the_revision": info.sha == args.revision,
        "private": info.private is True,
        "former_id_redirects": former.id == args.repo and former.sha == args.revision,
        "only_card_and_manifest_changed": changed
        == ["MODEL_MANIFEST.json", "README.md"],
        "no_former_org_text": sum(hits.values()) == 0,
    }
    return {
        "schema": "dev2-org-verify/1",
        "utc": now(),
        "repo": args.repo,
        "revision": args.revision,
        "superseded": args.superseded,
        "files": [len(old), len(new)],
        "changed_files": changed,
        "text_files_scanned": len(hits),
        "former_org_occurrences": {k: v for k, v in hits.items() if v},
        "checks": checks,
        "passed": all(checks.values()),
    }


def scan(args) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    revision = args.revision or api.model_info(args.repo).sha
    hits = _scan(api, args.repo, revision)
    return {
        "schema": "dev2-org-scan/1",
        "utc": now(),
        "repo": args.repo,
        "revision": revision,
        "text_files_scanned": len(hits),
        "former_org_occurrences": {k: v for k, v in hits.items() if v},
        "passed": sum(hits.values()) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--package", type=Path, required=True)
    for name in ("precheck", "remote-load"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        p.add_argument("--revision", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--superseded", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision")
    for name in sub.choices:
        sub.choices[name].add_argument("--output", type=Path)
    args = parser.parse_args()
    handler = {
        "compare": compare,
        "precheck": precheck,
        "remote-load": remote_load,
        "verify": verify,
        "scan": scan,
    }[args.command]
    result = handler(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("schema", "repo", "passed")}
            | {"sha256": hashlib.sha256(text.encode()).hexdigest()[:12]}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
