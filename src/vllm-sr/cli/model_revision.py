"""Resolve serving shorthand to one immutable revision before replica fan-out."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")
RELEASE_FILENAME = "releases.generated.json"


def builtin_releases() -> dict[str, str]:
    """Read the runtime-owned release projection bundled with this CLI."""
    packaged = files("cli.config_schema").joinpath(RELEASE_FILENAME)
    if packaged.is_file():
        return json.loads(packaged.read_text(encoding="utf-8"))
    source = (
        Path(__file__).resolve().parents[3]
        / "src/model-runtime/vllm_srun/registry"
        / RELEASE_FILENAME
    )
    if source.is_file():
        return json.loads(source.read_text(encoding="utf-8"))
    raise RuntimeError("canonical model runtime releases are not bundled with this CLI")


def resolve_model_revision(artifact: str, revision: str | None) -> str | None:
    """Use a release pin or resolve a Hub branch/tag/default exactly once.

    Explicit immutable commits and built-in defaults work without Hub access.
    The canonical deployment receives only the resolved commit; workers never
    resolve a moving ref independently. Credentials remain Hub-client owned.
    """
    if Path(artifact).is_absolute():
        if revision is not None:
            raise ValueError("--revision applies only to Hub repositories")
        return None
    if revision is not None:
        if not revision or revision.strip() != revision:
            raise ValueError("--revision must be a non-empty branch, tag or commit")
        if COMMIT.fullmatch(revision):
            return revision.lower()
    releases = builtin_releases()
    known = next(
        (pin for name, pin in releases.items() if name.lower() == artifact.lower()),
        None,
    )
    if known and revision is None:
        return known
    if known and revision and len(revision) >= 7 and known.startswith(revision.lower()):
        return known

    from huggingface_hub import HfApi, constants, try_to_load_from_cache

    if constants.HF_HUB_OFFLINE:
        for filename in ("config.json", "MODEL_MANIFEST.json"):
            cached = try_to_load_from_cache(artifact, filename, revision=revision)
            if isinstance(cached, str) and COMMIT.fullmatch(Path(cached).parent.name):
                return Path(cached).parent.name.lower()
        raise ValueError(
            "Model revision is not cached; connect to the Hub or specify a cached commit"
        )
    try:
        commit = HfApi().model_info(artifact, revision=revision, timeout=15).sha
    except Exception as error:
        # Hub exceptions can include request/endpoint details. Do not echo
        # credentials, signed URLs or response bodies in the CLI error.
        raise ValueError(
            f"Cannot resolve model revision for {artifact} ({type(error).__name__})"
        ) from None
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError(
            f"Hub did not return an immutable model revision for {artifact}"
        )
    return commit.lower()
