"""Run the kernel test suites on a node (``run.sh`` passes ``--out``; results go to RUN/selftest.txt)."""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ["v2.runtime_kernels.tests.test_cpu", "v2.runtime_kernels.tests.test_gpu"]
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.out / "selftest.txt").write_text(stream.getvalue())
    print(stream.getvalue())
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
