"""Server-observed runner identity; never accepted from an evaluation manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .contracts import digest
from .setup import PACKAGES, harness_paths
from .snapshots import capture_recipes

ENVIRONMENT_PROBE = """
import importlib.metadata,json,platform
packages=sorted((d.metadata.get('Name',''),d.version) for d in importlib.metadata.distributions())
value={'python':platform.python_version(),'implementation':platform.python_implementation(),'system':platform.system(),'machine':platform.machine(),'packages':packages}
encoded=json.dumps(value,sort_keys=True)
if len(encoded)>131072: raise SystemExit('Dependency inventory exceeds limit')
print(encoded)
"""


def capture_runner(manifest):
    """Capture runner identity and optional observed recipes before dispatch.

    Recipe receipts have their own content hash; their observation timestamps
    are intentionally excluded from the code/environment identity fingerprint.
    """
    package = Path(__file__).parent
    sources = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": sorted(
            [dist.metadata.get("Name", ""), dist.version]
            for dist in importlib.metadata.distributions()
        ),
    }
    harnesses = {}
    if manifest["mode"] == "live":
        for benchmark in sorted({case["benchmark"] for case in manifest["cases"]}):
            if benchmark not in PACKAGES:
                continue
            root, interpreter = harness_paths(benchmark)
            try:
                observed = subprocess.run(
                    [str(interpreter), "-c", ENVIRONMENT_PROBE],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                dependencies = json.loads(observed.stdout)
                revision = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.strip()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise ValueError(
                    "Could not capture installed harness provenance"
                ) from exc
            harnesses[benchmark] = {
                "source_revision": revision,
                "environment": dependencies,
                "environment_sha256": digest(dependencies),
            }
    observed = {
        "version": 1,
        "source_files": sources,
        "source_sha256": digest(sources),
        "environment": environment,
        "environment_sha256": digest(environment),
        "harnesses": harnesses,
    }
    return {
        **observed,
        "fingerprint_sha256": digest(observed),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "recipe_snapshots": capture_recipes(manifest),
    }
