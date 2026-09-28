"""Fetch a private release card's images and links from the Hub over HTTP (run on a node).

Complements ``hub readback`` (metadata, remote hashes, links against the package
file list) with what a signed-in reader receives: every relative image and link
of README.md is downloaded at the exact revision and hashed against
MODEL_MANIFEST.json, in-page anchors must name a heading, absolute links must
answer 200 (Hub links with the node's token), the model API at that revision
must report the repository private with the card's metadata, and anonymous
requests for the repository, its README and its model API must be refused. Hub
web pages refuse token auth, so a private collection link is checked through the
collections API and the rendered model page's status is only recorded. The token
stays in the HF CLI's default file and never enters the receipt.

    <hf-cli python> -m v2.release.tests.hub_card_http_check --repo R --revision SHA \
        --package PKG --output OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from v2.release import hub, layout

IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
REFUSED = (401, 403, 404)


def anchor(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


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
        raise ValueError("Check an exact 40-hex revision")
    from huggingface_hub import get_session, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    session = get_session()
    auth = build_hf_headers()

    def get(url: str, signed: bool):
        return session.get(
            url, headers=auth if signed else {}, follow_redirects=True, timeout=120
        )

    manifest_path = args.package / layout.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        **manifest["files_sha256"],
        layout.MANIFEST_NAME: layout.sha_file(manifest_path),
    }
    readme = (args.package / "README.md").read_text(encoding="utf-8")
    images = set(IMAGE.findall(readme))
    headings = {anchor(h) for h in re.findall(r"^#+\s+(.+)$", readme, re.M)}
    checks = []
    for target in sorted(images | set(LINK.findall(readme))):
        if target.startswith("#"):
            checks.append(
                {"target": target, "kind": "anchor", "passed": target[1:] in headings}
            )
        elif target.startswith("http"):
            # Hub web pages refuse token auth; a private collection is checked through its API.
            collection = re.fullmatch(
                r"https://huggingface\.co/collections/(.+)", target
            )
            url = (
                f"https://huggingface.co/api/collections/{collection.group(1)}"
                if collection
                else target
            )
            response = get(url, target.startswith("https://huggingface.co/"))
            checks.append(
                {
                    "target": target,
                    "kind": "absolute",
                    "via": "api" if collection else "page",
                    "status": response.status_code,
                    "passed": response.status_code == 200,
                }
            )
        else:
            path = target.split("#", 1)[0]
            response = get(hf_hub_url(args.repo, path, revision=args.revision), True)
            digest = hashlib.sha256(response.content).hexdigest()
            checks.append(
                {
                    "target": path,
                    "kind": "image" if target in images else "file",
                    "status": response.status_code,
                    "bytes": len(response.content),
                    "sha256_matches_manifest": digest == expected.get(path),
                    "passed": response.status_code == 200
                    and digest == expected.get(path),
                }
            )
    api_url = f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}"
    signed = get(api_url, True)
    info = signed.json() if signed.status_code == 200 else {}
    card = info.get("cardData") or {}
    model_api = {
        "status": signed.status_code,
        "sha": info.get("sha"),
        "private": info.get("private"),
        "license": card.get("license"),
        "base_model": card.get("base_model"),
        "base_model_relation": card.get("base_model_relation"),
        "files": len(info.get("siblings") or []),
    }
    anonymous = {
        name: get(url, False).status_code
        for name, url in (
            ("model_api", f"https://huggingface.co/api/models/{args.repo}"),
            ("readme", hf_hub_url(args.repo, "README.md", revision=args.revision)),
            ("model_page", f"https://huggingface.co/{args.repo}"),
        )
    }
    page = get(f"https://huggingface.co/{args.repo}", True)
    title = manifest["model_name"]
    rendered = {"status": page.status_code}
    if page.status_code == 200:
        rendered.update(
            contains_title=title in page.text,
            contains_banner=f"{title}-owl-banner.png" in page.text,
            contains_charts=all(Path(c).name in page.text for c in layout.CHART_FILES),
        )
    result = {
        "schema": "dev2-hub-card-http/1",
        "utc": hub.now(),
        "repo": args.repo,
        "revision": args.revision,
        "manifest_sha256": expected[layout.MANIFEST_NAME],
        "readme_sha256": layout.sha_file(args.package / "README.md"),
        "targets": checks,
        "model_api": model_api,
        "anonymous_status": anonymous,
        "rendered_page": rendered,
        "passed": all(c["passed"] for c in checks)
        and model_api["status"] == 200
        and model_api["sha"] == args.revision
        and model_api["private"] is True
        and all(status in REFUSED for status in anonymous.values()),
    }
    layout.write_json(args.output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "targets": len(checks),
                "anonymous": anonymous,
                "rendered": rendered,
            }
        )
    )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
