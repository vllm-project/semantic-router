"""Guard: the recipe-lab assets speak the canonical config layout.

The smoke assets, the bootstrap templates and the real-component tests only
run against a live router, so CI never exercises them and a removed-layout
(pre-v0.5) document or read can sit undetected for a whole release. This
guard runs everywhere:

- every YAML under tools/recipe-lab that carries a ``version`` must declare
  ``v0.5``, and no YAML may carry the removed top-level ``routing`` block;
- production code may not read ``routing`` or ``modelCards``;
- the real-component tests may not build or read removed-layout documents.

Unit tests that deliberately feed legacy documents to rejection paths (the
``test_legacy_*`` family) are outside this guard's scope: building a legacy
document to assert it is rejected is the contract, not a violation.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# The removed top-level block and its card list, in the forms a reader uses.
FORBIDDEN_READS = (
    '["routing"]',
    "['routing']",
    '.get("routing"',
    ".get('routing'",
    '"modelCards"',
    "'modelCards'",
)
# The real-component tests additionally may not write removed-layout
# documents (version strings, card blocks) or read them from the fixtures.
FORBIDDEN_INTEGRATION = (*FORBIDDEN_READS, "v0.3")


def _py_files(base: Path) -> list[Path]:
    return sorted(base.rglob("*.py"))


def _yaml_violations() -> list[str]:
    problems: list[str] = []
    for path in sorted(ROOT.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if document is None:
            continue
        if not isinstance(document, dict):
            problems.append(f"{path.relative_to(ROOT)}: not a mapping")
            continue
        version = document.get("version")
        if version is not None and version != "v0.5":
            problems.append(
                f"{path.relative_to(ROOT)}: version {version!r} is not canonical"
            )
        if "routing" in document:
            problems.append(
                f"{path.relative_to(ROOT)}: top-level routing block is removed"
            )
    return problems


def _py_violations(base: Path, patterns: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for path in _py_files(base):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in text:
                problems.append(
                    f"{path.relative_to(ROOT)}: removed-layout read {pattern!r}"
                )
    return problems


def test_yaml_assets_are_canonical():
    assert _yaml_violations() == []


def test_production_code_reads_the_canonical_layout_only():
    assert _py_violations(ROOT / "recipe_lab", FORBIDDEN_READS) == []


def test_integration_tests_speak_the_canonical_layout():
    assert _py_violations(ROOT / "tests" / "integration", FORBIDDEN_INTEGRATION) == []
