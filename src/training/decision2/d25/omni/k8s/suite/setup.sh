#!/usr/bin/env bash
# Usage: setup.sh <overlay dir>. Installs the suite's missing Python packages into the overlay
# (pip --target, never into the image) and records what the image already provides. Idempotent.
set -euo pipefail
OVERLAY=$1
mkdir -p "$OVERLAY"
python3 - "$OVERLAY" <<'PY'
import importlib, json, subprocess, sys
overlay = sys.argv[1]
wanted = {"pyarrow": "pyarrow>=15", "PIL": "pillow>=10", "numpy": "numpy>=1.26", "huggingface_hub": "huggingface_hub>=0.30"}
have, missing = {}, []
for module, spec in wanted.items():
    try:
        have[module] = getattr(importlib.import_module(module), "__version__", "?")
    except ImportError:
        missing.append(spec)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "--target", overlay, *missing])
print(json.dumps({"python": sys.version.split()[0], "image": have, "installed": missing}))
PY
