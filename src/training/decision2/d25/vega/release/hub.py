"""Hugging Face steps of a Decision 2.5 release: storage headroom, private repo, upload, fresh download, readback.

Run where the package is (a pod or node; weights never move through the local machine). ``upload`` and
``download`` call the real ``hf`` CLI; the token comes from ``HF_TOKEN`` in the environment.

    python -m d25.vega.release.hub headroom --org vllm-sr --need-gb 60
    python -m d25.vega.release.hub ensure --repo vllm-sr/d25-vega-staging            # creates it PRIVATE
    python -m d25.vega.release.hub upload --repo vllm-sr/d25-vega-staging --package <dir> --receipt up.json
    python -m d25.vega.release.hub download --repo vllm-sr/d25-vega-staging --revision <sha> --out <fresh dir>
    python -m d25.vega.release.hub readback --repo vllm-sr/d25-vega-staging --revision <sha> \
        --package <dir> --download <fresh dir> --receipt readback.json

``ensure`` and ``upload`` refuse a public repository (``--allow-public`` exists for the lead's public flip
only) and a repository id the Hub resolves to another name. ``readback`` checks the private flag, the exact
revision, that the remote file list is the package's (plus ``.gitattributes``), every remote LFS SHA-256 or
git blob id against ``MODEL_MANIFEST.json``, a full re-hash of the fresh download, and the card metadata
the Hub parsed from ``README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MANIFEST = "MODEL_MANIFEST.json"
HUB_ADDED = (".gitattributes",)


def api():
    from huggingface_hub import HfApi

    return HfApi()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def headroom(org: str, need_gb: float, cap_gb: float) -> dict[str, Any]:
    """Private storage the organization uses (sum of the Hub's usedStorage over its private repositories)."""
    from concurrent.futures import ThreadPoolExecutor

    hub = api()
    private = []
    for kind, lister in (
        ("model", hub.list_models),
        ("dataset", hub.list_datasets),
        ("space", hub.list_spaces),
    ):
        private += [
            (kind, info.id)
            for info in lister(author=org, expand=["private"])
            if info.private
        ]

    def used(item):
        kind, repo = item
        getter = {
            "model": hub.model_info,
            "dataset": hub.dataset_info,
            "space": hub.space_info,
        }[kind]
        info = getter(repo, expand=["usedStorage"])
        return (
            kind,
            repo,
            int(
                getattr(info, "used_storage", None)
                or getattr(info, "usedStorage", None)
                or 0
            ),
        )

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(used, private))
    total = sum(r[2] for r in rows) / 1e9
    return {
        "org": org,
        "private_repos": len(rows),
        "private_used_gb": round(total, 2),
        "cap_gb": cap_gb,
        "free_gb": round(cap_gb - total, 2),
        "need_gb": need_gb,
        "fits": cap_gb - total >= need_gb,
        "largest": [
            {"kind": k, "repo": r, "gb": round(u / 1e9, 2)}
            for k, r, u in sorted(rows, key=lambda x: -x[2])[:8]
        ],
    }


def ensure(
    repo: str, *, allow_public: bool = False, kind: str = "model"
) -> dict[str, Any]:
    hub = api()
    hub.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
    info = hub.repo_info(repo, repo_type=kind)
    if info.id != repo:
        raise SystemExit(
            f"{repo} resolves to {info.id}; refusing to write to a renamed repository"
        )
    if info.private is False and not allow_public:
        raise SystemExit(f"{repo} is public; refusing (only the lead flips visibility)")
    return {"repo": repo, "private": info.private, "sha": info.sha}


def run(cmd: list[str], env: dict[str, str] | None = None) -> float:
    started = time.time()
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})
    return time.time() - started


def upload(
    repo: str,
    package: Path,
    message: str,
    *,
    large: bool = False,
    allow_public: bool = False,
) -> dict[str, Any]:
    state = ensure(repo, allow_public=allow_public)
    manifest = json.loads((package / MANIFEST).read_text())
    if large:
        seconds = run(
            [
                "hf",
                "upload-large-folder",
                repo,
                str(package),
                "--repo-type",
                "model",
                "--exclude",
                ".cache/**",
                "--exclude",
                "**/__pycache__/**",
            ]
        )
    else:
        seconds = run(
            [
                "hf",
                "upload",
                repo,
                str(package),
                ".",
                "--repo-type",
                "model",
                "--commit-message",
                message,
                "--exclude",
                ".cache/**",
                "--exclude",
                "**/__pycache__/**",
            ]
        )
    hub = api()
    stale = sorted(
        set(hub.list_repo_files(repo)) - set(local_files(package)) - set(HUB_ADDED)
    )
    if stale:
        hub.delete_files(
            repo,
            delete_patterns=stale,
            commit_message=f"{message}: remove files of the previous package",
        )
    info = hub.repo_info(repo)
    size = sum(manifest["files_bytes"].values()) + (package / MANIFEST).stat().st_size
    return {
        "repo": repo,
        "revision": info.sha,
        "previous_revision": state["sha"],
        "private": info.private,
        "removed": stale,
        "seconds": round(seconds, 1),
        "bytes": size,
        "mb_per_s": round(size / 1e6 / max(seconds, 1e-9), 1),
        "model_sha256": manifest["identity"]["model_sha256"],
        "manifest_sha256": sha256_file(package / MANIFEST),
    }


def download(repo: str, revision: str, out: Path) -> dict[str, Any]:
    """The real ``hf download`` of one revision into a new directory with an empty cache."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; downloads go to a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="hf-home-", dir=out.parent))
    try:
        seconds = run(
            ["hf", "download", repo, "--revision", revision, "--local-dir", str(out)],
            env={"HF_HOME": str(home), "HF_HUB_CACHE": str(home / "hub")},
        )
    finally:
        shutil.rmtree(home, ignore_errors=True)
    size = sum(
        p.stat().st_size
        for p in out.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    )
    return {
        "repo": repo,
        "revision": revision,
        "out": str(out),
        "seconds": round(seconds, 1),
        "bytes": size,
        "mb_per_s": round(size / 1e6 / max(seconds, 1e-9), 1),
    }


def fetch(
    repo: str, revision: str, out: Path, workers: int = 8, attempts: int = 6
) -> dict[str, Any]:
    """Download one revision file by file from the resolve URLs and verify every file's LFS SHA-256 or git blob id.

    A fallback for ``hf download``, which in Hugging Face Jobs refused just-uploaded files ("missing the
    X-Repo-Commit header"). The token goes only to the Hub, not to the storage host it redirects to.
    """
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import constants, get_token

    info = api().repo_info(repo, revision=revision, files_metadata=True)
    token = os.environ.get("HF_TOKEN") or get_token()
    out.mkdir(parents=True, exist_ok=True)

    def one(sibling) -> int:
        target = out / sibling.rfilename
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{constants.ENDPOINT}/{repo}/resolve/{info.sha}/{urllib.parse.quote(sibling.rfilename)}"
        for attempt in range(attempts):
            try:
                request = urllib.request.Request(url)
                if token:
                    request.add_unredirected_header("Authorization", f"Bearer {token}")
                partial = target.with_name(target.name + ".part")
                with urllib.request.urlopen(request, timeout=600) as response, open(
                    partial, "wb"
                ) as stream:
                    shutil.copyfileobj(response, stream, 16 << 20)
                ok = (
                    sha256_file(partial) == sibling.lfs.sha256
                    if sibling.lfs is not None
                    else git_blob_id(partial) == sibling.blob_id
                )
                if not ok:
                    raise ValueError(f"{sibling.rfilename}: hash differs")
                partial.replace(target)
                return target.stat().st_size
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"{sibling.rfilename}: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(30)
        return 0

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        size = sum(pool.map(one, info.siblings))
    seconds = time.time() - started
    return {
        "repo": repo,
        "revision": info.sha,
        "files": len(info.siblings),
        "bytes": size,
        "verified": True,
        "seconds": round(seconds, 1),
        "mb_per_s": round(size / 1e6 / max(seconds, 1e-9), 1),
    }


def local_files(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and ".cache" not in p.relative_to(root).parts
        and "__pycache__" not in p.parts
    }


def readback(
    repo: str,
    revision: str,
    package: Path,
    downloaded: Path | None,
    *,
    expect_private: bool = True,
) -> dict[str, Any]:
    hub = api()
    info = hub.repo_info(repo, revision=revision, files_metadata=True)
    manifest = json.loads((package / MANIFEST).read_text())
    expected = {**manifest["files_sha256"], MANIFEST: sha256_file(package / MANIFEST)}
    problems: list[str] = []
    if info.sha != revision:
        problems.append(f"revision {revision} resolves to {info.sha}")
    if bool(info.private) != expect_private:
        problems.append(f"private flag is {info.private}")
    remote = {s.rfilename: s for s in info.siblings}
    extra = sorted(set(remote) - set(expected) - set(HUB_ADDED))
    missing = sorted(set(expected) - set(remote))
    if extra or missing:
        problems.append(f"remote files differ: extra {extra[:5]} missing {missing[:5]}")
    checked = {"lfs_sha256": 0, "git_blob": 0}
    for name, digest in expected.items():
        sibling = remote.get(name)
        if sibling is None:
            continue
        if sibling.lfs is not None:
            checked["lfs_sha256"] += 1
            if sibling.lfs.sha256 != digest:
                problems.append(f"{name}: remote LFS SHA-256 differs")
        else:
            checked["git_blob"] += 1
            if sibling.blob_id != git_blob_id(package / name):
                problems.append(f"{name}: remote git blob differs")
    rehashed = 0
    if downloaded is not None:
        files = local_files(downloaded)
        if set(files) - set(HUB_ADDED) != set(expected):
            problems.append(
                f"download file list differs: {sorted(set(files) ^ set(expected))[:5]}"
            )
        for name, digest in expected.items():
            if name in files:
                rehashed += 1
                if sha256_file(files[name]) != digest:
                    problems.append(f"{name}: downloaded bytes differ")
    card = {}
    if "README.md" in remote:
        data = info.card_data.to_dict() if info.card_data is not None else {}
        card = {
            k: data.get(k)
            for k in (
                "license",
                "pipeline_tag",
                "base_model",
                "base_model_relation",
                "library_name",
                "tags",
            )
        }
        if (
            card.get("license") != "apache-2.0"
            or card.get("pipeline_tag") != "zero-shot-classification"
        ):
            problems.append(f"card metadata as parsed by the Hub: {card}")
    return {
        "repo": repo,
        "revision": revision,
        "private": info.private,
        "files": len(expected),
        "checked": checked,
        "rehashed": rehashed,
        "card": card,
        "problems": problems,
        "ok": not problems,
        "model_sha256": manifest["identity"]["model_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("headroom")
    h.add_argument("--org", default="vllm-sr")
    h.add_argument("--need-gb", type=float, default=0.0)
    h.add_argument(
        "--cap-gb",
        type=float,
        default=1000.0,
        help="private storage cap; a Team org includes 1 TB per seat",
    )
    e = sub.add_parser("ensure")
    e.add_argument("--repo", required=True)
    e.add_argument("--allow-public", action="store_true")
    u = sub.add_parser("upload")
    u.add_argument("--repo", required=True)
    u.add_argument("--package", required=True, type=Path)
    u.add_argument("--message", default="Decision 2.5 package")
    u.add_argument(
        "--large",
        action="store_true",
        help="hf upload-large-folder (resumable, many workers)",
    )
    u.add_argument("--allow-public", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--repo", required=True)
    d.add_argument("--revision", required=True)
    d.add_argument("--out", required=True, type=Path)
    f = sub.add_parser("fetch")
    f.add_argument("--repo", required=True)
    f.add_argument("--revision", required=True)
    f.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("readback")
    r.add_argument("--repo", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--package", required=True, type=Path)
    r.add_argument("--download", type=Path)
    r.add_argument("--expect-public", action="store_true")
    for p in (h, e, u, d, f, r):
        p.add_argument("--receipt", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "headroom":
        out = headroom(a.org, a.need_gb, a.cap_gb)
    elif a.cmd == "ensure":
        out = ensure(a.repo, allow_public=a.allow_public)
    elif a.cmd == "upload":
        out = upload(
            a.repo, a.package, a.message, large=a.large, allow_public=a.allow_public
        )
    elif a.cmd == "download":
        out = download(a.repo, a.revision, a.out)
    elif a.cmd == "fetch":
        out = fetch(a.repo, a.revision, a.out)
    else:
        out = readback(
            a.repo,
            a.revision,
            a.package,
            a.download,
            expect_private=not a.expect_public,
        )
    text = json.dumps(out, indent=1)
    if a.receipt:
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        a.receipt.write_text(text + "\n")
    print(text)
    if a.cmd == "headroom" and not out["fits"]:
        return 1
    if a.cmd == "readback" and not out["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
