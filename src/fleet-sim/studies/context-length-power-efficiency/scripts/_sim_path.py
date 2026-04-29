from __future__ import annotations

import os
import sys
from pathlib import Path


def add_sim_to_syspath() -> Path:
    """Locate the local `fleet-sim` repo root and prepend it to `sys.path`."""
    here = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get("FLEET_SIM_ROOT") or os.environ.get("INFERENCE_FLEET_SIMULATOR_DIR")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            here.parents[2],
            Path.cwd(),
            Path.cwd().parent,
            Path.cwd().parent.parent,
        ]
    )

    for candidate in candidates:
        if (candidate / "fleet_sim").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(
        "Could not locate the `fleet-sim` repository root.\n"
        "Run the study from inside `src/fleet-sim`, or set FLEET_SIM_ROOT.\n"
        f"Checked:\n{checked}"
    )
