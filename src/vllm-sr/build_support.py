"""Setuptools hooks for staging the canonical Router schema into artifacts."""

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


def schema_source() -> Path:
    if REPOSITORY_SCHEMA.is_file():
        return REPOSITORY_SCHEMA
    if PACKAGE_SCHEMA.is_file():
        return PACKAGE_SCHEMA
    raise FileNotFoundError("canonical Router config schema is unavailable")


class BuildPy(build_py):
    """Copy the single source artifact into build output, never the worktree."""

    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "cli" / "config_schema" / SCHEMA_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(schema_source(), destination)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        schema_output = str(
            Path(self.build_lib) / "cli" / "config_schema" / SCHEMA_FILENAME
        )
        return outputs if schema_output in outputs else [*outputs, schema_output]


class SDist(sdist):
    """Include the schema in an sdist release tree without tracking a copy."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir) / "cli" / "config_schema" / SCHEMA_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(schema_source(), destination)
