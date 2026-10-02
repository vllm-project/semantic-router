"""After a round-4 upload: main, privacy, redirects, the card-only diff, the reviewed files and the collection.

Run on a node with the HF CLI Python (token in the default HF token file):

    <hf-cli python> verify_card4.py --decisions DIR --revisions revisions.json \
        --expected expected.json --output verify-card4.json

revisions.json: {tier: revision} of the revised tiers; expected.json: {tier: {path: sha256}} of the reviewed
preview's changed card files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ORG = "llm-semantic-router"
COLLECTION = f"{ORG}/decision-20-6ab7cf7bdfb506bf8269cb00"
TIERS = {
    "0.6B": "Kai",
    "0.8B": "Eos",
    "2B": "Sol",
    "4B": "Nox",
    "9B": "Lux",
    "27B": "Vega",
}
BANNER = "assets/banner.png"
AUDIT = "Training data audited at row level against all Index test items.</sub>"
# The collection as the user left it (2026-10-01 17:01:30Z); a card revision must not change it.
COLLECTION_ITEMS = [
    f"{ORG}/Decision-2.0-{n}"
    for n in ("Vega-27B", "Lux-9B", "Nox-4B", "Sol-2B", "Eos-0.8B", "Kai-0.6B")
]


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(api: HfApi, tier: str, revision: str, previous: str, expected: dict) -> dict:
    name = f"Decision-2.0-{TIERS[tier]}-{tier}"
    repo, former = f"{ORG}/{name}", f"{ORG}/DEV2.0-{tier}"
    info = api.model_info(repo)
    redirected = api.model_info(former)

    def files(rev: str) -> dict:
        path = hf_hub_download(repo, "MODEL_MANIFEST.json", revision=rev)
        return json.loads(Path(path).read_text())["files_sha256"]

    old, new = files(previous), files(revision)
    changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    card_files = {BANNER}
    published = {
        n: sha(hf_hub_download(repo, n, revision=revision)) for n in card_files
    }
    readme = Path(hf_hub_download(repo, "README.md", revision=revision)).read_text(
        encoding="utf-8"
    )
    result = {
        "repo": repo,
        "revision": revision,
        "superseded": previous,
        "main_is_revision": info.sha == revision,
        "private": info.private,
        "former_redirects": redirected.id == repo and redirected.sha == revision,
        "changed_vs_superseded": changed,
        "only_card_files_changed": set(changed) == card_files and set(old) == set(new),
        "matches_reviewed_preview": published == expected,
        "footnote_discloses_audit": AUDIT in readme,
        "readme_chars": len(readme),
    }
    result["passed"] = all(
        result[k] is True
        for k in (
            "main_is_revision",
            "private",
            "former_redirects",
            "only_card_files_changed",
            "matches_reviewed_preview",
            "footnote_discloses_audit",
        )
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--revisions", type=Path, required=True, help="{tier: revision}")
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    api = HfApi()
    revisions = json.loads(args.revisions.read_text())
    expected = json.loads(args.expected.read_text())
    repos = []
    for tier, revision in revisions.items():
        name = f"Decision-2.0-{TIERS[tier]}-{tier}"
        decision = json.loads(
            (args.decisions / f"{name}.decision.card4.json").read_text()
        )
        previous = decision["supersedes"]["released_as"].split("@")[1]
        repos.append(check(api, tier, revision, previous, expected[tier]))
    collection = api.get_collection(COLLECTION)
    items = [i.item_id for i in collection.items]
    result = {
        "schema": "dev2-card4-verify/1",
        "repos": repos,
        "collection": {
            "slug": COLLECTION,
            "private": collection.private,
            "last_updated": str(collection.last_updated),
            "items": items,
            "unchanged": items == COLLECTION_ITEMS,
        },
    }
    result["passed"] = (
        all(r["passed"] for r in repos)
        and result["collection"]["unchanged"]
        and collection.private is True
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed": [r["repo"] for r in repos if not r["passed"]],
                "collection_unchanged": result["collection"]["unchanged"],
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
