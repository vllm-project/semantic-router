"""After the round-2 uploads: main, privacy, redirects, collection and the card-only diff of each repository.

Run on a node with the HF CLI Python (token in the default HF token file):

    <hf-cli python> verify_card2.py --decisions DIR --output verify-card2.json
"""

from __future__ import annotations

import argparse
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
GONE = ("NOTICE", "ATTRIBUTIONS.md")
GONE_PREFIXES = ("LICENSES/", "evaluation/")
CARD_ASSETS = {
    "assets/banner.png",
    "assets/jevarena.png",
    "assets/jevarena-types.png",
    "assets/index-pareto.png",
    "assets/index-areas.png",
}


def manifest(repo: str, revision: str) -> dict:
    return json.load(
        open(hf_hub_download(repo, "MODEL_MANIFEST.json", revision=revision))
    )


def config(repo: str, revision: str) -> dict:
    return json.load(open(hf_hub_download(repo, "config.json", revision=revision)))


def check(api: HfApi, tier: str, codename: str, revision: str, previous: str) -> dict:
    name = f"Decision-2.0-{codename}-{tier}"
    repo, former = f"{ORG}/{name}", f"{ORG}/DEV2.0-{tier}"
    info = api.model_info(repo, files_metadata=False)
    redirected = api.model_info(former)
    files = set(api.list_repo_files(repo, revision=revision))
    old, new = (
        manifest(repo, previous)["files_sha256"],
        manifest(repo, revision)["files_sha256"],
    )
    changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    allowed = {"README.md", *CARD_ASSETS}
    removed_ok = (
        lambda n: (n in GONE or n.startswith(GONE_PREFIXES) or n.endswith(".svg"))
        and n not in new
    )
    other = [
        n
        for n in changed
        if n not in allowed and not removed_ok(n) and n != "config.json"
    ]
    before, after = config(repo, previous), config(repo, revision)
    config_keys = sorted(
        k for k in set(before) | set(after) if before.get(k) != after.get(k)
    )
    readme = Path(hf_hub_download(repo, "README.md", revision=revision)).read_text(
        encoding="utf-8"
    )
    previous_readme = Path(
        hf_hub_download(repo, "README.md", revision=previous)
    ).read_text(encoding="utf-8")
    result = {
        "repo": repo,
        "revision": revision,
        "main_is_revision": info.sha == revision,
        "private": info.private,
        "former_id": former,
        "former_resolves_to": redirected.id,
        "former_redirects": redirected.id == repo and redirected.sha == revision,
        "superseded": previous,
        "changed_vs_superseded": changed,
        "non_card_changes": other,
        "config_changed_keys": config_keys,
        "config_model_name": after.get("model_name"),
        "removed_files_absent": not any(
            f in GONE or f.startswith(GONE_PREFIXES) or f.endswith(".svg")
            for f in files
        ),
        "card_assets_present": CARD_ASSETS <= files,
        "readme_chars": [len(previous_readme), len(readme)],
        "readme_title": f"# {name}\n" in readme,
    }
    result["passed"] = (
        result["main_is_revision"]
        and result["private"] is True
        and result["former_redirects"]
        and not other
        and config_keys in ([], ["model_name"])
        and result["config_model_name"] == name
        and result["removed_files_absent"]
        and result["card_assets_present"]
        and result["readme_title"]
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument(
        "--revisions", type=Path, required=True, help="JSON {tier: new revision}"
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    api = HfApi()
    revisions = json.loads(args.revisions.read_text())
    repos = []
    for tier, codename in TIERS.items():
        decision = json.loads(
            (
                args.decisions / f"Decision-2.0-{codename}-{tier}.decision.card2.json"
            ).read_text()
        )
        previous = decision["supersedes"]["released_as"].split("@")[1]
        repos.append(check(api, tier, codename, revisions[tier], previous))
    collection = api.get_collection(COLLECTION)
    items = [i.item_id for i in collection.items]
    expected = [f"{ORG}/Decision-2.0-{c}-{t}" for t, c in TIERS.items()]
    # The order and any non-family items are the user's to curate; the release needs all six present.
    result = {
        "schema": "dev2-card2-verify/1",
        "repos": repos,
        "collection": {
            "slug": COLLECTION,
            "private": collection.private,
            "last_updated": str(collection.last_updated),
            "items": items,
            "all_six_present": set(expected) <= set(items),
            "ascending_size_order": [i for i in items if i in expected] == expected,
        },
    }
    result["passed"] = (
        all(r["passed"] for r in repos)
        and result["collection"]["all_six_present"]
        and collection.private is True
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed": [r["repo"] for r in repos if not r["passed"]],
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
