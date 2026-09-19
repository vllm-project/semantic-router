"""Exercise standalone experiment imports without importing training datasets."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = TRAINING_DIR / "model_experiment" / "mmlu_pro_solver_lora"

# Execute the actual entrypoint's path setup in an isolated interpreter. Import
# resolution needs no model dependencies; when torch is installed, also execute
# the real shared utilities module (as the training image does).
IMPORT_PROBE = """
import ast
import importlib
import importlib.util
import os
from pathlib import Path
import sys

entrypoint = Path(sys.argv[1])
shared_dir = Path(sys.argv[2])
tree = ast.parse(entrypoint.read_text())
for index, node in enumerate(tree.body):
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == '_parent_dir'
        for target in node.targets
    ):
        bootstrap = ast.Module(body=tree.body[index:index + 2], type_ignores=[])
        namespace = {'__file__': str(entrypoint), 'os': os, 'sys': sys, 'Path': Path}
        exec(compile(bootstrap, str(entrypoint), 'exec'), namespace)
        break
else:
    raise AssertionError('standalone entrypoint has no shared import bootstrap')

for name in ('common_lora_utils', 'training_args_compat'):
    spec = importlib.util.find_spec(name)
    assert spec is not None, name + ' cannot be resolved without PYTHONPATH'
    assert Path(spec.origin).resolve() == shared_dir / (name + '.py'), spec.origin
importlib.import_module('training_args_compat')
if importlib.util.find_spec('torch') is not None:
    importlib.import_module('common_lora_utils')
    print('real shared utility import passed')
else:
    print('shared utility import resolution passed; torch unavailable')
"""


class StandaloneExperimentImportsTest(unittest.TestCase):
    def test_experiment_entrypoints_resolve_shared_modules_without_pythonpath(self):
        for filename in (
            "ft_qwen3_mmlu_solver_lora.py",
            "ft_qwen3_mmlu_solver_lora_no_leakage.py",
        ):
            with self.subTest(entrypoint=filename), tempfile.TemporaryDirectory() as cwd:
                environment = dict(os.environ)
                environment.pop("PYTHONPATH", None)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        IMPORT_PROBE,
                        str(EXPERIMENT_DIR / filename),
                        str(TRAINING_DIR / "model_classifier"),
                    ],
                    cwd=cwd,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
