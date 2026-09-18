"""Recipe source discovery and deterministic live-matrix planning."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

BUILT_IN_LATEST_SUBDIR = Path("built-in") / "latest"


class RecipeLike(Protocol):
    name: str
    variants: int
    entrypoints: tuple[str, ...]
    auto_entrypoints: tuple[str, ...]
    decisions: tuple[str, ...]
    signal_families: tuple[str, ...]
    algorithms: tuple[str, ...]
    plugins: tuple[str, ...]
    coverage: dict[str, Any]


RecipeType = TypeVar("RecipeType", bound=RecipeLike)


@dataclass(frozen=True)
class RecipeSource:
    name: str
    recipes_root: Path
    report_subdir: Path
    validate_readme: bool


@dataclass(frozen=True)
class RecipeSourceInventory(Generic[RecipeType]):
    source: RecipeSource
    recipes: tuple[RecipeType, ...]


def discover_recipe_sources(root: Path) -> list[RecipeSource]:
    """Return live conformance sources without including release snapshots."""

    sources = [
        RecipeSource(
            name="standalone",
            recipes_root=root,
            report_subdir=Path("."),
            validate_readme=True,
        )
    ]
    built_in_latest = root / BUILT_IN_LATEST_SUBDIR
    if built_in_latest.is_dir():
        sources.append(
            RecipeSource(
                name="built-in-latest",
                recipes_root=built_in_latest,
                report_subdir=BUILT_IN_LATEST_SUBDIR,
                validate_readme=False,
            )
        )
    return sources


def discover_source_inventories(
    root: Path, discover_inventory: Callable[[Path], list[RecipeType]]
) -> list[RecipeSourceInventory[RecipeType]]:
    return [
        RecipeSourceInventory(
            source=source,
            recipes=tuple(discover_inventory(source.recipes_root)),
        )
        for source in discover_recipe_sources(root)
    ]


def shard_inventory(
    inventory: list[RecipeType], shard_count: int
) -> list[list[RecipeType]]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    shards: list[list[RecipeType]] = [[] for _ in range(shard_count)]
    weights = [0] * shard_count
    for recipe in sorted(inventory, key=lambda item: (-item.variants, item.name)):
        index = min(range(shard_count), key=lambda item: (weights[item], item))
        shards[index].append(recipe)
        weights[index] += recipe.variants
    return [shard for shard in shards if shard]


def matrix_payload(inventory: list[RecipeType], shard_count: int) -> dict[str, Any]:
    return {
        "include": [
            {
                "shard": index,
                "recipes": ",".join(recipe.name for recipe in shard),
                "variants": sum(recipe.variants for recipe in shard),
            }
            for index, shard in enumerate(shard_inventory(inventory, shard_count))
        ]
    }


def source_matrix_payload(
    source_inventories: list[RecipeSourceInventory[RecipeType]],
    shard_count: int,
    repo_root: Path,
) -> dict[str, Any]:
    include: list[dict[str, Any]] = []
    for source_inventory in source_inventories:
        source = source_inventory.source
        for index, shard in enumerate(
            shard_inventory(list(source_inventory.recipes), shard_count)
        ):
            include.append(
                {
                    "source": source.name,
                    "shard": f"{source.name}-{index}",
                    "recipes": ",".join(recipe.name for recipe in shard),
                    "variants": sum(recipe.variants for recipe in shard),
                    "recipes_root": repo_relative_path(source.recipes_root, repo_root),
                    "report_dir": source.report_subdir.as_posix(),
                }
            )
    return {"include": include}


def render_source_rows(
    source_inventories: list[RecipeSourceInventory[RecipeType]],
    repo_root: Path,
    output_format: str,
) -> str:
    rows = [
        {
            "source": source_inventory.source.name,
            "recipes_root": repo_relative_path(
                source_inventory.source.recipes_root, repo_root
            ),
            "report_dir": source_inventory.source.report_subdir.as_posix(),
            "recipes": ",".join(recipe.name for recipe in source_inventory.recipes),
        }
        for source_inventory in source_inventories
    ]
    if output_format == "json":
        return json.dumps(rows, indent=2)
    return "\n".join(
        "|".join(
            (
                row["source"],
                row["recipes_root"],
                row["report_dir"],
                row["recipes"],
            )
        )
        for row in rows
    )


def repo_relative_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()
