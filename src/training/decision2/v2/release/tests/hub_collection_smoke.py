"""Exercise the collection calls of ``hub collect`` on a throwaway PRIVATE collection.

Never touches the "Decision 2.0" collection: creates a private scratch collection,
adds one private staging repository, reads the collection back, deletes it and
confirms the deletion. Run on a node with the HF CLI environment:

    <hf-cli python> -m v2.release.tests.hub_collection_smoke --repo llm-semantic-router/dev2-release-staging --output R.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v2.release import hub, layout

TITLE = "dev2-release-staging-collection-smoke"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not layout.STAGING_REPO.fullmatch(args.repo):
        raise ValueError("The smoke test only uses private staging repositories")
    api = hub._api()
    if api.model_info(args.repo).private is not True:
        raise RuntimeError("Staging repository must be private")
    created = api.create_collection(
        TITLE,
        namespace=layout.ORG,
        private=True,
        description="Temporary release-pipeline smoke test",
    )
    slug = created.slug
    result = {
        "schema": "dev2-hub-collection-smoke/1",
        "utc": hub.now(),
        "slug": slug,
        "created_private": created.private,
    }
    try:
        if created.private is not True or created.title == hub.COLLECTION_TITLE:
            raise RuntimeError("Scratch collection is not a private scratch collection")
        api.add_collection_item(
            slug, item_id=args.repo, item_type="model", exists_ok=True
        )
        collection, items = hub._collection_items(api, slug)
        result["readback"] = {"private": collection.private, "items": items}
        real, real_items = hub._collection_items(api, hub.COLLECTION)
        result["decision_2_0"] = {
            "private": real.private,
            "items": len(real_items),
            "contains_repo": args.repo in real_items,
        }
    finally:
        api.delete_collection(slug)
    try:
        api.get_collection(slug)
        result["deleted"] = False
    except Exception as exc:
        result["deleted"] = type(exc).__name__ in (
            "HfHubHTTPError",
            "RepositoryNotFoundError",
            "EntryNotFoundError",
        )
    result["passed"] = (
        result["created_private"] is True
        and result["readback"]["private"] is True
        and result["readback"]["items"] == [args.repo]
        and result["decision_2_0"]["private"] is True
        and not result["decision_2_0"]["contains_repo"]
        and result["deleted"]
    )
    layout.write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "deleted": result["deleted"]}))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
