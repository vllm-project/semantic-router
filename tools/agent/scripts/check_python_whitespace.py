from __future__ import annotations

import argparse
import io
from pathlib import Path

from pre_commit_hooks import end_of_file_fixer, trailing_whitespace_fixer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hook", choices=("trailing-whitespace", "end-of-file-fixer"))
    parser.add_argument("files", nargs="+")
    args: argparse.Namespace = parser.parse_args()
    result = 0
    for filename in args.files:
        with Path(filename).open("rb") as source:
            if args.hook == "trailing-whitespace":
                changed: bool = any(
                    line
                    != trailing_whitespace_fixer._process_line(
                        line=line, is_markdown=False, chars=None
                    )
                    for line in source
                )
            else:
                changed = bool(end_of_file_fixer.fix_file(io.BytesIO(source.read())))
        if changed:
            print(f"{filename}: {args.hook} would make changes")
            result = 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
