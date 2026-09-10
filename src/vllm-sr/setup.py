"""Setuptools entrypoint that stages the canonical built-in catalog."""

from __future__ import annotations

import runpy
from pathlib import Path

from setuptools import setup


def _stage_model_catalog() -> None:
    project_root = Path(__file__).resolve().parent
    repository_root = project_root.parents[1]
    stager = repository_root / "tools" / "release" / "stage_model_catalog_package.py"
    if stager.is_file():
        stage = runpy.run_path(str(stager))["stage"]
        if stage() != 0:
            raise RuntimeError("failed to stage the built-in model catalog")
    catalog = project_root / "cli" / "model_assets" / "latest" / "catalog.yaml"
    if not catalog.is_file():
        raise RuntimeError("the built-in model catalog was not staged for packaging")


_stage_model_catalog()
setup()
