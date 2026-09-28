"""Fetch every link and image of an uploaded card from the Hub (run on a node).

``hub readback`` checks that relative card links name packaged files. This
follows them on the Hub instead: each relative link or image of README.md and
ATTRIBUTIONS.md is downloaded from ``resolve/<revision>`` and re-hashed against
MODEL_MANIFEST.json, each Hugging Face model or collection link is looked up
through the Hub API (existence and private flag), in-page anchors must match
a heading, and the rendered model page must load. The token stays in the
node's default HF token file; receipts hold URLs, status codes and digests.

    <hf-cli python> -m v2.release.hub_links --repo R --revision SHA --package PKG --output OUT
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from v2.release import layout

HUB = "https://huggingface.co"
LINK = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)\)")
MODEL_URL = re.compile(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/?\Z")
COLLECTION_URL = re.compile(
    r"https://huggingface\.co/collections/([\w.-]+/[\w.-]+)/?\Z"
)


def links(text: str) -> list[tuple[bool, str]]:
    """(is_image, target) for every markdown link, in order, without duplicates."""
    seen, out = set(), []
    for bang, target in LINK.findall(text):
        key = (bool(bang), target)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def anchors(text: str) -> set[str]:
    return {
        re.sub(r"[^\w\- ]", "", title.strip().lower()).replace(" ", "-")
        for title in re.findall(r"^#+ (.+)$", text, flags=re.M)
    }


def fetch(url: str, token: str | None) -> tuple[int, bytes]:
    headers = {"User-Agent": "dev2-release-readback"}
    if token and url.startswith(HUB):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def check(repo: str, revision: str, package: Path, token: str | None) -> dict[str, Any]:
    manifest = json.loads((package / layout.MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = {
        **manifest["files_sha256"],
        layout.MANIFEST_NAME: layout.sha_file(package / layout.MANIFEST_NAME),
    }
    results = []
    for source in ("README.md", "ATTRIBUTIONS.md"):
        text = (package / source).read_text(encoding="utf-8")
        heads = anchors(text)
        for image, target in links(text):
            entry: dict[str, Any] = {"source": source, "image": image, "target": target}
            if target.startswith("#"):
                entry.update(kind="anchor", ok=target[1:] in heads)
            elif target.startswith("http"):
                model, collection = MODEL_URL.match(target), COLLECTION_URL.match(
                    target
                )
                api = (
                    f"{HUB}/api/models/{model.group(1)}"
                    if model
                    else (
                        f"{HUB}/api/collections/{collection.group(1)}"
                        if collection
                        else None
                    )
                )
                status, body = fetch(api or target, token)
                entry.update(
                    kind="api" if api else "external", status=status, ok=status == 200
                )
                if api and status == 200:
                    entry["private"] = json.loads(body).get("private")
            else:
                name = target.split("#", 1)[0]
                status, body = fetch(f"{HUB}/{repo}/resolve/{revision}/{name}", token)
                digest = hashlib.sha256(body).hexdigest() if status == 200 else None
                entry.update(
                    kind="file",
                    status=status,
                    sha256=digest,
                    ok=status == 200 and digest == expected.get(name),
                )
            results.append(entry)
    status, body = fetch(f"{HUB}/{repo}", token)
    page = body.decode("utf-8", "replace")
    rendered = {
        "status": status,
        "bytes": len(body),
        "title_present": manifest["model_name"] in page,
        "banner_referenced": "owl-banner.png" in page,
    }
    return {
        "links": results,
        "checked": len(results),
        "failed": [r["target"] for r in results if not r["ok"]],
        "rendered_page": rendered,
        "passed": all(r["ok"] for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not layout.REVISION.fullmatch(args.revision):
        parser.error("Check an exact 40-hex revision")
    from huggingface_hub import get_token

    result = check(args.repo, args.revision, args.package, get_token())
    receipt = {
        "schema": "dev2-hub-links/1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": args.repo,
        "revision": args.revision,
        **result,
    }
    layout.write_json(args.output, receipt)
    print(json.dumps({k: receipt[k] for k in ("checked", "failed", "passed")}))
    sys.exit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
