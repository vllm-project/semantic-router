"""Every config the model-selection pages show passes `config validate`.

The pages document each example as a ```yaml title="..."``` block. Most are
fragments: like the Router's docs test, this one merges each onto
testdata/model_catalog_docs/base.yaml, which declares the request models and
the deployments the examples refer to without declaring. The Router's verdict
comes through the seam the other `config validate` tests use: a stand-in
Router binary that accepts every document, so the test checks what the CLI
itself refuses. A built Router named in VLLM_SR_ROUTER_BINARY when the tests
are collected judges the examples instead.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from cli.main import main
from cli.router_validation import ROUTER_BINARY_ENV
from click.testing import CliRunner

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "website" / "docs" / "installation"
BASE = REPO / "src/semantic-router/pkg/config/testdata/model_catalog_docs/base.yaml"
PAGES = (
    "choose-a-model-for-a-signal.md",
    "model-catalog.md",
    "model-resource-migration.md",
)
FENCE = re.compile(r"^([ \t]*)```(\w*)([^\n]*)\n(.*?)^\1```", re.M | re.S)
TITLE = re.compile(r'title="([^"]+)"')
# Read at collection: the suite's fixtures clear it before each test runs.
BUILT_ROUTER = os.environ.get(ROUTER_BINARY_ENV, "").strip()


def _titled_blocks(page: str) -> list[tuple[str, str]]:
    blocks = []
    text = (DOCS / page).read_text(encoding="utf-8")
    for _, language, info, fenced in FENCE.findall(text):
        title = TITLE.search(info)
        if language in ("yaml", "yml") and title:
            blocks.append((title.group(1), textwrap.dedent(fenced)))
    return blocks


def _merge(base: dict, overlay: dict) -> dict:
    """Maps merge key by key; anything else the overlay writes replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _router(tmp_path: Path) -> str:
    if BUILT_ROUTER:
        return BUILT_ROUTER
    stand_in = tmp_path / "router"
    stand_in.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "open(sys.argv[sys.argv.index('-config') + 1]).read()\n"
        'print(\'{"valid": true, "warnings": []}\')\n',
        encoding="utf-8",
    )
    stand_in.chmod(0o755)
    return str(stand_in)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_shows_titled_examples(page):
    assert _titled_blocks(page), f"{page} shows no titled YAML example"


@pytest.mark.parametrize(
    "page, title, text",
    [
        pytest.param(page, title, text, id=f"{page}:{title}")
        for page in PAGES
        for title, text in _titled_blocks(page)
    ],
)
def test_documented_configs_validate(monkeypatch, tmp_path, page, title, text):
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(_merge(base, yaml.safe_load(text)), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setenv(ROUTER_BINARY_ENV, _router(tmp_path))

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(config)])

    assert result.exit_code == 0, f'{page} "{title}":\n{result.output}'
    assert "Checked by the Router binary" in result.stdout, result.output
