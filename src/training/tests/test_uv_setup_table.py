"""The training README setup table matches the uv.lock files on disk."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
README = TRAINING_ROOT / "README.md"
ROW = re.compile(
    r"^\| \[`(?P<area>[^`]+)`\]\((?P<href>[^)]+)\) \| "
    r"(?P<locked>yes|no) \| (?P<setup>.+) \|$"
)
LOCKED_SETUP = "`uv sync --locked`"
UNLOCKED_SETUP = "no `uv.lock` yet"


def setup_rows(text: str) -> list[dict[str, str]]:
    start = text.index("## Environment setup")
    end = text.index("\n## ", start + 1)
    return [
        match.groupdict()
        for line in text[start:end].splitlines()
        if (match := ROW.match(line))
    ]


def lock_areas() -> set[str]:
    areas = set()
    for path in TRAINING_ROOT.rglob("uv.lock"):
        relative = path.relative_to(TRAINING_ROOT)
        if any(part.startswith(".") for part in relative.parts):
            continue
        areas.add(relative.parent.as_posix())
    return areas


class UvSetupTableTest(unittest.TestCase):
    def test_rows_match_lock_files(self) -> None:
        rows = setup_rows(README.read_text(encoding="utf-8"))
        areas = [row["area"] for row in rows]
        self.assertEqual(areas, sorted(areas))
        self.assertEqual(len(areas), len(set(areas)))

        locked = set()
        unlocked = set()
        for row in rows:
            self.assertEqual(row["href"], f"{row['area']}/")
            self.assertTrue((TRAINING_ROOT / row["area"]).is_dir(), row["area"])
            if row["locked"] == "yes":
                self.assertEqual(row["setup"], LOCKED_SETUP)
                locked.add(row["area"])
            else:
                self.assertEqual(row["setup"], UNLOCKED_SETUP)
                unlocked.add(row["area"])

        on_disk = lock_areas()
        self.assertEqual(locked, on_disk)
        self.assertTrue(unlocked.isdisjoint(on_disk))


if __name__ == "__main__":
    unittest.main()
