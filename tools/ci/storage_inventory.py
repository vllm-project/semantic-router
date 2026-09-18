"""Discover storage test ownership from test-source annotations and Ginkgo labels."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = re.compile(
    r"// StorageIntegration: ([a-z_]+)\nfunc (Test\w+)\(t \*testing.T\)"
)


def go_storage_tests(root=ROOT):
    """Return package -> backend -> test names; each case has one source owner."""
    inventory = {}
    source_root = root / "src/semantic-router"
    seen = set()
    for path in sorted((source_root / "pkg").rglob("*_test.go")):
        package = "./" + str(path.parent.relative_to(source_root))
        for backend, name in DECLARATION.findall(path.read_text(encoding="utf-8")):
            identity = package + "/" + name
            if identity in seen:
                raise ValueError(f"duplicate storage test {identity}")
            seen.add(identity)
            inventory.setdefault(package, {}).setdefault(backend, []).append(name)
    return inventory


def ginkgo_storage_suites():
    # Cases themselves remain discovered by Ginkgo, including nested It/DescribeTable nodes.
    return {"./pkg/cache": "TestCache", "./pkg/vectorstore": "TestVectorStore"}
