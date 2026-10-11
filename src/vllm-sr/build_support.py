"""Stage canonical Router schema, runtime releases, and proposal assets."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

SCHEMA_FILENAME = "router-config-v0.3.schema.json"
PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_SCHEMA = PROJECT_ROOT / "cli" / "config_schema" / SCHEMA_FILENAME
REPOSITORY_SCHEMA = (
    PROJECT_ROOT.parent / "semantic-router" / "pkg" / "configschema" / SCHEMA_FILENAME
)
RELEASE_FILENAME = "releases.generated.json"
REPOSITORY_RELEASES = (
    PROJECT_ROOT.parent / "model-runtime/vllm_srun/registry" / RELEASE_FILENAME
)
PROPOSAL_ASSET_PATHS = ("fragments/algorithm/selection/latency-aware.yaml",)


def release_source() -> Path:
    if REPOSITORY_RELEASES.is_file():
        return REPOSITORY_RELEASES
    packaged = PROJECT_ROOT / "cli/config_schema" / RELEASE_FILENAME
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("canonical model runtime releases are unavailable")


def schema_source() -> Path:
    if REPOSITORY_SCHEMA.is_file():
        return REPOSITORY_SCHEMA
    if PACKAGE_SCHEMA.is_file():
        return PACKAGE_SCHEMA
    raise FileNotFoundError("canonical Router config schema is unavailable")


def proposal_asset_source(relative: str) -> Path:
    """Return one proposal asset from the repo, or from a staged sdist copy."""

    repository = PROJECT_ROOT.parents[1] / "config" / relative
    packaged = PROJECT_ROOT / "cli" / "proposal_assets" / relative
    if repository.is_file():
        return repository
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"proposal asset {relative} is unavailable")


def stage_proposal_assets(destination_root: Path) -> None:
    """Copy the latency-aware fragment into a CLI package tree."""

    for relative in PROPOSAL_ASSET_PATHS:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(proposal_asset_source(relative), destination)


class BuildPy(build_py):
    """Copy canonical generated artifacts into build output, never the worktree."""

    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "cli" / "config_schema" / SCHEMA_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(schema_source(), destination)
        copy2(release_source(), destination.with_name(RELEASE_FILENAME))
        stage_proposal_assets(Path(self.build_lib) / "cli" / "proposal_assets")

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        schema_output = str(
            Path(self.build_lib) / "cli" / "config_schema" / SCHEMA_FILENAME
        )
        release_output = str(Path(schema_output).with_name(RELEASE_FILENAME))
        extra = [schema_output, release_output]
        asset_root = Path(self.build_lib) / "cli" / "proposal_assets"
        for relative in PROPOSAL_ASSET_PATHS:
            extra.append(str(asset_root / relative))
        return [*outputs, *[path for path in extra if path not in outputs]]


class SDist(sdist):
    """Include generated resources in an sdist without tracking duplicate copies."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir) / "cli" / "config_schema" / SCHEMA_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(schema_source(), destination)
        copy2(release_source(), destination.with_name(RELEASE_FILENAME))
        stage_proposal_assets(Path(base_dir) / "cli" / "proposal_assets")
