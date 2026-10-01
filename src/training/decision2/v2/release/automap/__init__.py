"""🤗 Transformers remote code (``trust_remote_code=True``) shipped at the root of every Decision 2.0 package.

The three modules here are copied byte for byte into the package root; ``config.json`` gains
``CONFIG_FIELDS``. The modules wrap the package's own ``decision2/`` runtime (see ``API.md``). Stdlib only.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parent
FILES = ("configuration_decision2.py", "modeling_decision2.py", "pipeline_decision2.py")
CONFIG_FIELDS: dict[str, Any] = {
    "model_type": "decision2",
    "architectures": ["Decision2Model"],
    "auto_map": {
        "AutoConfig": "configuration_decision2.Decision2Config",
        "AutoModel": "modeling_decision2.Decision2Model",
    },
    "custom_pipelines": {
        "decision": {
            "impl": "pipeline_decision2.Decision2Pipeline",
            "pt": ["AutoModel"],
            "type": "text",
        }
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_into(stage: Path, tree: Path | None = None) -> dict[str, dict[str, Any]]:
    """Copy the remote code from ``tree`` (a decision2 source tree; default this one) to ``stage``."""
    source = SOURCE if tree is None else Path(tree) / "v2/release/automap"
    records = {}
    for name in FILES:
        shutil.copyfile(source / name, Path(stage) / name)
        records[name] = {
            "source": f"v2/release/automap/{name}",
            "source_sha256": _sha(source / name),
            "sha256": _sha(Path(stage) / name),
        }
    return records
