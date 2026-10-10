"""Public release mechanics for Decision 2.5 (run only on the lead's GO; the default is a dry run).

Steps, each refused unless its preconditions hold:

- ``move``: rename the private model repo (``HfApi.move_repo`` keeps history; the Hub redirects the old id).
  Pre: source exists and is private, target absent. Post: target private, same commits, old id redirects.
- ``flip-model``: make the model public. Pre: private, main == ``--revision`` (the revision that passed the
  fresh-download readback), README front matter parsed by the Hub. Post: public, main unchanged.
- ``flip-dataset``: make the results dataset public. Pre: exists, private, holds ``submissions/``.

The upload of the final package and its readback run in between with ``private_push.sh`` (REPO = the new id)
and ``hub.py readback --download`` from a fresh download (see STATUS.md, release checklist).

    python -m d25.vega.release.release_public move --source vllm-sr/Decision-2.5-Vega-27B \\
        --target vllm-sr/Decision-2.5-Omni-27B [--execute]
    python -m d25.vega.release.release_public flip-model --repo vllm-sr/Decision-2.5-Omni-27B --revision <sha> [--execute]
    python -m d25.vega.release.release_public flip-dataset --repo vllm-sr/decision-2.5-decision-index [--execute]
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def info(api, repo: str, repo_type: str = "model"):
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        return api.repo_info(repo, repo_type=repo_type)
    except RepositoryNotFoundError:
        return None


def redirects(old: str, new: str, token: str | None, repo_type: str = "model") -> bool:
    """The Hub answers the old id's API URL with a redirect to the new id (or serves the new repo)."""
    request = urllib.request.Request(f"https://huggingface.co/api/{repo_type}s/{old}")
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read()).get("id") == new
    except urllib.error.HTTPError:
        return False


def move(
    api,
    source: str,
    target: str,
    execute: bool,
    token: str | None,
    repo_type: str = "model",
) -> dict[str, Any]:
    src, dst = info(api, source, repo_type), info(api, target, repo_type)
    problems = [] if src else [f"{source} not found"]
    if src and not src.private:
        problems.append(f"{source} is public; this step moves a private repo only")
    if dst:
        problems.append(f"{target} already exists")
    report: dict[str, Any] = {
        "step": "move",
        "repo_type": repo_type,
        "source": source,
        "target": target,
        "problems": problems,
    }
    if src:
        report["source_sha"] = src.sha
        report["source_commits"] = len(
            api.list_repo_commits(source, repo_type=repo_type)
        )
    if problems or not execute:
        report["executed"] = False
        return report
    api.move_repo(from_id=source, to_id=target, repo_type=repo_type)
    moved = info(api, target, repo_type)
    report.update(
        executed=True,
        target_private=moved.private,
        target_sha=moved.sha,
        target_commits=len(api.list_repo_commits(target, repo_type=repo_type)),
        old_id_redirects=redirects(source, target, token, repo_type),
    )
    report["ok"] = (
        moved.private
        and moved.sha == report["source_sha"]
        and report["target_commits"] == report["source_commits"]
    )
    return report


def flip(
    api, repo: str, repo_type: str, revision: str | None, execute: bool
) -> dict[str, Any]:
    current = info(api, repo, repo_type)
    problems = [] if current else [f"{repo} not found"]
    if current and not current.private:
        problems.append(f"{repo} is already public")
    if current and revision and current.sha != revision:
        problems.append(f"main is {current.sha}, not the verified {revision}")
    if current and repo_type == "model" and not getattr(current, "card_data", None):
        problems.append("README front matter not parsed by the Hub")
    if current and repo_type == "dataset":
        files = api.list_repo_files(repo, repo_type="dataset")
        runs = [
            f for f in files if f.startswith("runs/") and f.endswith("/scores.json")
        ]
        if not any(f.startswith("submissions/") for f in files) and not runs:
            problems.append(
                "no submissions/ or runs/<run>/scores.json in the results dataset"
            )
        for path in runs:
            scores = json.loads(
                Path(api.hf_hub_download(repo, path, repo_type="dataset")).read_text()
            )
            if scores.get("complete") is not True:
                problems.append(f"{path} is not a complete run")
    report: dict[str, Any] = {
        "step": f"flip-{repo_type}",
        "repo": repo,
        "problems": problems,
        "sha": current.sha if current else None,
    }
    if problems or not execute:
        report["executed"] = False
        return report
    api.update_repo_settings(repo_id=repo, repo_type=repo_type, private=False)
    after = info(api, repo, repo_type)
    report.update(
        executed=True,
        private=after.private,
        sha_after=after.sha,
        ok=not after.private and after.sha == current.sha,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    import os

    from huggingface_hub import HfApi, get_token

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("move", "flip-model", "flip-dataset"))
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--repo")
    ap.add_argument(
        "--revision",
        help="flip-model: the revision that passed the fresh-download readback",
    )
    ap.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset"),
        help="move: kind of repository",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="without it: preconditions only (dry run)",
    )
    args = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    if args.step == "move":
        report = move(
            api, args.source, args.target, args.execute, token, args.repo_type
        )
    elif args.step == "flip-model":
        if args.execute and not args.revision:
            raise SystemExit("flip-model --execute needs --revision")
        report = flip(api, args.repo, "model", args.revision, args.execute)
    else:
        report = flip(api, args.repo, "dataset", None, args.execute)
    print(json.dumps(report, indent=1))
    return (
        1
        if report["problems"] or (report.get("executed") and not report.get("ok"))
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
