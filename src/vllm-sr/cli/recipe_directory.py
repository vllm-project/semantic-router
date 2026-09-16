"""Resolve the managed Recipe directory associated with a source config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RECIPE_CONFIG_FILENAME = "config.yaml"


class RecipeDirectoryError(ValueError):
    """Raised when a directory looks like a managed Recipe but is incomplete."""


@dataclass(frozen=True)
class ActiveRecipeDirectory:
    """The fixed, validated files that make up one active managed Recipe."""

    root: Path
    config: Path
    metadata: Path
    probes: Path
    dsl: Path
    readme: Path

    @property
    def assets(self) -> tuple[tuple[str, Path], ...]:
        return (
            (RECIPE_CONFIG_FILENAME, self.config),
            ("metadata.yaml", self.metadata),
            ("probes.yaml", self.probes),
            ("recipe.dsl", self.dsl),
            ("README.md", self.readme),
        )


def resolve_active_recipe_directory(
    source_config_file: str | Path,
) -> ActiveRecipeDirectory | None:
    """Return the managed Recipe containing ``source_config_file``.

    A bare config remains supported. A directory becomes managed when it has a
    ``metadata.yaml`` file. A partial Recipe source (DSL or probes without
    metadata) is rejected so it cannot silently lose its identity in the
    Dashboard. Only the five fixed paths are inspected; adjacent directories
    and catalog roots are never scanned.
    """

    source_config = Path(source_config_file).expanduser().absolute()
    root = source_config.parent
    metadata = root / "metadata.yaml"
    probes = root / "probes.yaml"
    dsl = root / "recipe.dsl"
    readme = root / "README.md"

    if not metadata.exists():
        partial_assets = [path.name for path in (probes, dsl) if path.exists()]
        if partial_assets:
            joined = ", ".join(partial_assets)
            raise RecipeDirectoryError(
                f"Incomplete managed Recipe at {root}: found {joined} but "
                "metadata.yaml is missing"
            )
        return None

    expected_config = root / RECIPE_CONFIG_FILENAME
    if source_config.name != RECIPE_CONFIG_FILENAME:
        raise RecipeDirectoryError(
            "Managed Recipe configs must be named config.yaml; "
            f"received {source_config.name}"
        )

    paths = (expected_config, metadata, probes, dsl, readme)
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise RecipeDirectoryError(
            f"Incomplete managed Recipe at {root}: missing {', '.join(missing)}"
        )

    for path in paths:
        _validate_regular_recipe_file(root, path)

    return ActiveRecipeDirectory(
        root=root,
        config=expected_config,
        metadata=metadata,
        probes=probes,
        dsl=dsl,
        readme=readme,
    )


def _validate_regular_recipe_file(root: Path, path: Path) -> None:
    if path.is_symlink():
        raise RecipeDirectoryError(
            f"Managed Recipe asset must not be a symbolic link: {path.name}"
        )
    if not path.is_file():
        raise RecipeDirectoryError(f"Managed Recipe asset is not a file: {path.name}")

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_root:
        raise RecipeDirectoryError(
            f"Managed Recipe asset escapes the active directory: {path.name}"
        )
