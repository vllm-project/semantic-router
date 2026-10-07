"""The dashboard image builds its backend from a subset of the Router's packages.

dashboard/backend/Dockerfile copies them one by one, so a Router package that
the backend starts to import, directly or through another Router package, must
gain a COPY line or the image fails to build.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_BACKEND = REPO_ROOT / "dashboard" / "backend"
ROUTER_ROOT = REPO_ROOT / "src" / "semantic-router"
ROUTER_MODULE = "github.com/vllm-project/semantic-router/src/semantic-router/"
GO_IMPORT_BLOCK = re.compile(r"^import\s*\((.*?)^\)", re.MULTILINE | re.DOTALL)
GO_IMPORT_LINE = re.compile(r'^import\s+(?:[\w.]+\s+)?"([^"]+)"', re.MULTILINE)
GO_QUOTED = re.compile(r'"([^"]+)"')
DOCKERFILE_COPY = re.compile(r"^COPY src/semantic-router/(pkg/\S+?)/ ", re.MULTILINE)


def _router_imports(sources) -> set[str]:
    """The Router packages the given non-test Go files import."""

    imports: set[str] = set()
    for source in sources:
        if source.name.endswith("_test.go"):
            continue
        text = source.read_text(encoding="utf-8")
        found = GO_IMPORT_LINE.findall(text)
        for block in GO_IMPORT_BLOCK.findall(text):
            found += GO_QUOTED.findall(block)
        imports |= {
            path.removeprefix(ROUTER_MODULE)
            for path in found
            if path.startswith(ROUTER_MODULE)
        }
    return imports


def _router_closure(packages: set[str]) -> set[str]:
    closure: set[str] = set()
    pending = set(packages)
    while pending:
        package = pending.pop()
        closure.add(package)
        pending |= _router_imports((ROUTER_ROOT / package).glob("*.go")) - closure
    return closure


def test_the_dockerfile_copies_every_router_package_the_backend_builds():
    copied = DOCKERFILE_COPY.findall(
        (DASHBOARD_BACKEND / "Dockerfile").read_text(encoding="utf-8")
    )
    needed = _router_closure(_router_imports(DASHBOARD_BACKEND.rglob("*.go")))

    missing = sorted(
        package
        for package in needed
        if not any(package == root or package.startswith(root + "/") for root in copied)
    )
    assert {"pkg/config", "pkg/extension"} <= needed
    assert missing == [], f"dashboard/backend/Dockerfile does not copy {missing}"
