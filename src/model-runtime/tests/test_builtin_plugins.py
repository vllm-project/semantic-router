"""A new built-in family, added the way every built-in is: its own package and its entry points.

The test writes a temporary distribution whose family names its own table of
pinned models and its own fixture writer, and whose engine declares an
``auto`` priority ahead of ``native``. It builds the wheel with the
distribution's build backend and installs it into a fresh directory. Nothing
in the runtime names the family, its table, its writer or its engine; the
runtime finds them through the entry points and the classes' declarations.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from vllm_sr_runtime.cli import main
from vllm_sr_runtime.config import ModelConfig, ServeConfig
from vllm_sr_runtime.families.decision2.family import Decision2Family
from vllm_sr_runtime.plugins import registry
from vllm_sr_runtime.plugins.base import DeviceInfo, PackageRef
from vllm_sr_runtime.registry import builtin
from vllm_sr_runtime.runtime import Runtime, choose_engine
from vllm_sr_runtime.testing.fixtures import write_fixture, write_package

BUILD_WHEEL = (
    "import setuptools.build_meta as backend, sys; backend.build_wheel(sys.argv[1])"
)
FAMILY, ENGINE = "tiny_decisions", "tiny_native"
REPO = "vllm-sr-fixtures/Tiny-Decisions"
SEED = 11
CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")

SOURCES = {
    "pyproject.toml": f"""
        [build-system]
        requires = ["setuptools>=69", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "vllm-sr-runtime-tiny-decisions"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = ["vllm-sr-runtime"]

        [project.entry-points."vllm_sr_runtime.families"]
        {FAMILY} = "tiny_decisions.family:TinyDecisionsFamily"

        [project.entry-points."vllm_sr_runtime.engines"]
        {ENGINE} = "tiny_decisions.engine:TinyNativeEngine"

        [tool.setuptools.packages.find]
        include = ["tiny_decisions*"]

        [tool.setuptools.package-data]
        tiny_decisions = ["pinned.json"]
        """,
    "tiny_decisions/__init__.py": "",
    "tiny_decisions/family.py": f'''
        """Decision 2.0 packages as a family of their own, with its own table and fixture writer."""

        from vllm_sr_runtime.families.decision2.family import Decision2Family


        class TinyDecisionsFamily(Decision2Family):
            name = "{FAMILY}"
            builtin_table = "tiny_decisions.table"
            fixture_writer = "tiny_decisions.fixtures"
        ''',
    "tiny_decisions/engine.py": f'''
        """The native engine under another name, which ``--engine auto`` tries first."""

        from vllm_sr_runtime.engines.native.engine import NativeEngine


        class TinyNativeEngine(NativeEngine):
            name = "{ENGINE}"
            auto_priority = -1
        ''',
    "tiny_decisions/fixtures.py": '''
        """The family's tiny packages: Decision 2.0 fixtures on a Qwen3 backbone."""

        from vllm_sr_runtime.testing.fixtures import write_package

        VARIANTS = ("qwen3",)


        def write_fixture(output, variant, seed):
            return write_package(output, backbone=variant or VARIANTS[0], seed=seed)
        ''',
    "tiny_decisions/table.py": '''
        """The family's pinned model and its recorded golden answers."""

        import json
        from pathlib import Path

        from vllm_sr_runtime.registry.tables.common import BuiltinModel

        PINNED = json.loads((Path(__file__).parent / "pinned.json").read_text())
        MODELS = (BuiltinModel(**PINNED),)
        ''',
}


def _identity(package: Path) -> str:
    manifest = json.loads((package / "MODEL_MANIFEST.json").read_text())
    return str(manifest["identity"]["model_sha256"])


def _cpu_answers(package: Path) -> dict:
    """The package's golden answers on this CPU, served by the built-in family."""
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device="cpu"),))
    )
    runtime.start(background=False)
    try:
        served = runtime.lookup(None)
        assert served.health.golden.status == "unverified"
        (golden,) = served.family.golden(served.package)
        return served.golden_surface(golden["surface"], golden["body"])
    finally:
        runtime.stop()


def _install(root: Path, pinned: dict) -> Path:
    """Build the distribution's wheel and unpack it into a fresh directory."""
    source = root / "source"
    for name, text in SOURCES.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip())
    (source / "tiny_decisions" / "pinned.json").write_text(json.dumps(pinned))
    subprocess.run(
        [sys.executable, "-c", BUILD_WHEEL, str(root)],
        cwd=source,
        check=True,
        capture_output=True,
    )
    (wheel,) = root.glob("*.whl")
    site = root / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    return site


@pytest.fixture(scope="module")
def tiny_family(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-decisions")
    recorded = write_package(root / "recorded", backbone="qwen3", seed=SEED)
    pinned = {
        "repo_id": REPO,
        "revision": "0" * 40,
        "family": FAMILY,
        "model_sha256": _identity(recorded),
        "manifest_sha256": "",
        "loaded_parameters": 0,
        "backbone": "qwen3",
        "min_device_memory_gib": 1,
        "golden_answers": {"cpu": _cpu_answers(recorded)},
    }
    site = _install(root, pinned)
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(site))
    registry.discover.cache_clear()
    yield root, site, pinned
    patch.undo()
    registry.discover.cache_clear()


def test_the_family_brings_its_table_and_its_fixture_writer(tiny_family, capsys):
    root, _, pinned = tiny_family
    assert FAMILY in registry.names("families") and ENGINE in registry.names("engines")
    known = builtin.lookup("Tiny-Decisions")
    assert known is not None and builtin.lookup(REPO.upper()) is known
    assert builtin.all_models(FAMILY) == (known,)
    assert builtin.by_identity(pinned["model_sha256"]) is known
    assert known.golden_answers == pinned["golden_answers"]
    assert builtin.lookup("Decision-2.0-Kai-0.6B").family == "decision2"
    written = write_fixture(root / "written", family=FAMILY, seed=SEED)
    assert _identity(written) == pinned["model_sha256"]
    with pytest.raises(ValueError, match="fixtures are qwen3"):
        write_fixture(root / "other", family=FAMILY, variant="qwen3_5")
    assert main(["fixture", str(root / "cli"), "--family", FAMILY]) == 0
    assert main(["models"]) == 0
    assert f"{REPO}@{'0' * 40}" in capsys.readouterr().out


def test_its_engine_priority_and_its_pinned_answers_serve_its_model(tiny_family):
    root, _, pinned = tiny_family
    package = write_fixture(root / "served", family=FAMILY, seed=SEED)
    family = Decision2Family()
    spec = family.describe(family.verify(PackageRef(package)))
    assert choose_engine("auto", spec, CPU)[0] == ENGINE
    assert choose_engine("auto", spec, CPU, preferred="native")[0] == "native"
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(package), device="cpu", family=FAMILY),)
        )
    )
    runtime.start(background=False)
    try:
        (card,) = runtime.model_cards()
        questions = len(pinned["golden_answers"]["cpu"])
        assert card["family"] == FAMILY and card["engine"] == ENGINE
        assert card["golden"] == {
            "status": "matched",
            "checked": questions,
            "matched": questions,
            "reference": "cpu",
        }
    finally:
        runtime.stop()


def test_uninstalled_its_models_are_no_longer_built_in(tiny_family, monkeypatch):
    _, site, _ = tiny_family
    assert builtin.lookup(REPO) is not None
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(site)])
    registry.discover.cache_clear()
    try:
        assert FAMILY not in registry.names("families")
        assert builtin.lookup(REPO) is None
        assert builtin.lookup("Decision-2.0-Kai-0.6B") is not None
    finally:
        monkeypatch.undo()
        registry.discover.cache_clear()
    assert builtin.lookup(REPO) is not None


class StrayFamily(Decision2Family):
    name = "stray"
    builtin_table = "vllm_sr_runtime.registry.tables.decision2"


class TwinFamily(Decision2Family):
    name = "twin"
    builtin_table = "twin_table"


def test_a_table_pins_only_its_family_and_a_repository_once(monkeypatch):
    def entry(name: str, target: str) -> registry.PluginEntry:
        return registry.PluginEntry("families", name, target, None, None)

    decision2 = registry.discover()["families"]["decision2"]
    stray = entry("stray", f"{__name__}:StrayFamily")
    with pytest.raises(registry.PluginError, match="pins models of decision2"):
        builtin._read({"decision2": decision2, "stray": stray})
    twins = types.ModuleType("twin_table")
    twins.MODELS = tuple(
        replace(model, family="twin") for model in builtin.all_models("decision2")
    )
    monkeypatch.setitem(sys.modules, "twin_table", twins)
    twin = entry("twin", f"{__name__}:TwinFamily")
    with pytest.raises(registry.PluginConflictError, match="decision2 and twin"):
        builtin._read({"decision2": decision2, "twin": twin})
