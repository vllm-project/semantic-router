"""Asset-only commits on main of the public family repos: the v2 banner and a MODEL_MANIFEST.json whose
files_sha256 / files_bytes entries for assets/banner.png match it (the runtime verifies every listed file on load).

    python -m d25.family.banner_fix <out.json> d3-flash:manifest d3-lite:manifest d3-nano:banner d3-mini:banner

"manifest": the v2 banner is already on main, only the manifest is refreshed; "banner": both files in one commit.
After each commit: anonymous snapshot of main, verify_package(fast) with the repo's own d3_runtime, and the tags.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
    snapshot_download,
)

BANNERS = Path("/data/d25/omni/family/release/banners-v2")
BANNER = "assets/banner.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fix(api: HfApi, name: str, mode: str) -> dict:
    repo = f"vllm-sr/{name}"
    banner = BANNERS / f"decision-3.0-{name}.png"
    info = api.model_info(repo)
    assert info.private is False, repo
    parent = info.sha
    with tempfile.TemporaryDirectory() as tmp:
        path = hf_hub_download(
            repo, "MODEL_MANIFEST.json", revision=parent, cache_dir=tmp
        )
        manifest = json.loads(Path(path).read_text())
    want = (sha256(banner), banner.stat().st_size)
    have = (manifest["files_sha256"].get(BANNER), manifest["files_bytes"].get(BANNER))
    ops = []
    if mode == "banner":
        ops.append(CommitOperationAdd(BANNER, str(banner)))
    else:
        tree = {
            e.path: e
            for e in api.list_repo_tree(repo, revision=parent, path_in_repo="assets")
            if hasattr(e, "lfs")
        }
        remote = tree[BANNER].lfs.sha256 if tree[BANNER].lfs else None
        assert remote == want[0], (name, remote, want[0])
    if have != want:
        manifest["files_sha256"][BANNER], manifest["files_bytes"][BANNER] = want
        ops.append(
            CommitOperationAdd(
                "MODEL_MANIFEST.json", (json.dumps(manifest, indent=1) + "\n").encode()
            )
        )
    commit = None
    if ops:
        message = (
            "Update banner"
            if mode == "banner"
            else "MODEL_MANIFEST.json: hash of the new banner"
        )
        commit = api.create_commit(
            repo, operations=ops, commit_message=message, parent_commit=parent
        ).oid
    anon = HfApi(token=False)
    for _ in range(30):
        main = anon.model_info(repo).sha
        if commit is None or main == commit:
            break
        time.sleep(10)
    # The runtime's load check, against main as anyone downloads it: every manifest file exists with the listed
    # size and SHA-256 (weights from the Hub's LFS records, everything else downloaded and hashed).
    tree = {
        e.path: e
        for e in anon.list_repo_tree(repo, revision=main, recursive=True)
        if hasattr(e, "size")
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(
            snapshot_download(
                repo,
                revision=main,
                cache_dir=tmp,
                token=False,
                ignore_patterns=["*.safetensors"],
            )
        )
        live = json.loads((root / "MODEL_MANIFEST.json").read_text())
        problems = []
        for name, digest in sorted(live["files_sha256"].items()):
            entry = tree.get(name)
            if entry is None:
                problems.append(f"missing {name}")
            elif entry.size != live["files_bytes"][name]:
                problems.append(
                    f"{name}: size {entry.size} vs {live['files_bytes'][name]}"
                )
            elif (
                entry.lfs.sha256 if getattr(entry, "lfs", None) else sha256(root / name)
            ) != digest:
                problems.append(f"{name}: sha256 differs")
        banner_ok = sha256(root / BANNER) == want[0]
    tags = {t.name: t.target_commit for t in api.list_repo_refs(repo).tags}
    return {
        "repo": repo,
        "parent": parent,
        "commit": commit,
        "main": main,
        "changed": [o.path_in_repo for o in ops],
        "manifest_problems": problems,
        "banner_v2": banner_ok,
        "tags": tags,
        "ok": not problems and banner_ok,
    }


def main() -> None:
    out, specs = Path(sys.argv[1]), sys.argv[2:]
    api = HfApi()
    report = []
    for spec in specs:
        name, mode = spec.split(":")
        try:
            report.append(fix(api, name, mode))
        except Exception as exc:  # noqa: BLE001 - report every repo
            report.append(
                {
                    "repo": f"vllm-sr/{name}",
                    "error": f"{type(exc).__name__}: {exc}",
                    "ok": False,
                }
            )
        print(json.dumps(report[-1]), flush=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(0 if all(r["ok"] for r in report) else 1)


if __name__ == "__main__":
    main()
