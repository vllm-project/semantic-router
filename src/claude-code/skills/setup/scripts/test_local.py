import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import local


class Install(unittest.TestCase):
    def test_installs_the_runtime_from_a_pinned_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = local.install_steps(Path(tmp) / "venv", "cpu", sys.executable)
        self.assertRegex(
            steps[-1][-1],
            r"^vllm-srun @ git\+https://github\.com/vllm-project/semantic-router\.git@[0-9a-f]{40}#subdirectory=src/model-runtime$",
        )

    def test_constrains_the_dependencies_to_the_same_commits_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = local.install_steps(Path(tmp) / "venv", "cpu", sys.executable)
        locked = sys.version_info >= local.LOCKED_PYTHON
        self.assertEqual("-c" in steps[-1], locked)
        if locked:
            lock = steps[-1][steps[-1].index("-c") + 1]
            self.assertIn(
                f"/{local.COMMIT}/src/model-runtime/requirements-lock.txt", lock
            )
            self.assertIn(f"@{local.COMMIT}#", steps[-1][-1])


if __name__ == "__main__":
    unittest.main()
