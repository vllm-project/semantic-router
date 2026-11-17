"""Jobs that fit in a hosted runner's free disk skip the two-minute cleanup."""

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CLEANUP = "./.github/actions/free-disk-space"


def cleanup_steps(name: str) -> list[dict]:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses") == CLEANUP
    ]


class FreeDiskSpaceCallerTests(unittest.TestCase):
    def test_jobs_with_room_skip_the_cleanup(self) -> None:
        for name in ("recipe-conformance.yml", "test-local.yml", "test-platform.yml"):
            with self.subTest(workflow=name):
                steps = cleanup_steps(name)
                self.assertEqual(len(steps), 1)
                self.assertEqual(steps[0]["with"]["min-free-gb"], "40")

    def test_only_accelerator_image_builds_always_clean(self) -> None:
        (step,) = cleanup_steps("build-artifacts.yml")
        self.assertEqual(
            step["with"]["min-free-gb"],
            '${{ contains(fromJSON(\'["vllm-sr-cuda", "vllm-sr-rocm"]\'), '
            "matrix.image) && '0' || '40' }}",
        )


if __name__ == "__main__":
    unittest.main()
