"""Formal-runner entry for an untouched Decision 1.0 package via ``v2.dec.infer_1p0``.

Adds only the runner's ``--max-items`` smoke option (first N gold-free prompts
written to a private temporary input); without it the call is exactly
``v2.dec.infer_1p0``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if "--max-items" in argv:
        at = argv.index("--max-items")
        count = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
        source = Path(argv[argv.index("--input") + 1])
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)[:count]
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        handle.writelines(lines)
        handle.close()
        argv[argv.index("--input") + 1] = handle.name
    from v2.dec import infer_1p0

    sys.argv = [infer_1p0.__file__, *argv]
    infer_1p0.main()


if __name__ == "__main__":
    main()
