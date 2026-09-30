"""Exercise OpenVINO's report-directory contract through its real Make recipes."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_results import collection_errors
from runtime_evidence import host_platform, native_evidence

ROOT = Path(__file__).resolve().parents[3]
PROCESS_STUB = """
import json, os, pathlib, sys
args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
if '-c' in args:
    print(os.environ['OPENVINO_CONTRACT_ROOT'])
elif args[0].endswith('create_owned_fixture.py'):
    pathlib.Path(args[1]).mkdir(parents=True)
elif '--manifest' in args:
    path = pathlib.Path(option('--manifest'))
    path.write_text('{}')
    with open(os.environ['OPENVINO_CONTRACT_CALLS'], 'a') as stream:
        stream.write(json.dumps({'manifest': str(path)}) + '\\n')
elif args[0] == 'test':
    output = pathlib.Path(os.environ['OPENVINO_TEST_REPORT_DIR'])
    assert pathlib.Path(os.environ['OPENVINO_OWNED_FIXTURE_DIR']) == output / 'owned-fixtures'
    assert os.environ['SEMANTIC_ROUTER_OPENVINO_TEST_ARTIFACTS'] == os.environ['OPENVINO_OWNED_FIXTURE_DIR']
    package = 'github.com/vllm-project/semantic-router/openvino-binding'
    if '-tags' in args and option('-tags') == 'openvino':
        package = 'github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native'
        roots = ['TestOpenVINOQualifiedFixtureMetadata', 'TestOwnedOpenVINORuntimeIntegration']
        children = {'TestOwnedOpenVINORuntimeIntegration': ['embedding', 'classifier']}
    elif '-tags' in args and option('-tags') == 'published_model_tests':
        roots = ['TestPublishedOpenVINO']
        children = {'TestPublishedOpenVINO': ['domain', 'embedding']}
        assert pathlib.Path(os.environ['OPENVINO_TEST_MANIFEST']) == output / 'models.json'
        assert (output / 'sources.json').is_file() and (output / 'models.json').is_file()
        (output / 'inference.json').write_text(json.dumps({
            'passed': True, 'provider': 'openvino', 'device': 'CPU',
            'platform': os.environ['OPENVINO_CONTRACT_PLATFORM'], 'models': [],
        }))
    else:
        roots = ['TestOwnedHandlesRemainIndependent', 'TestOwnedBudgetsAndPadding',
            'TestOwnedCountsBeyondDeclaredModelLimit', 'TestOwnedRejectsNullText',
            'TestOwnedInitializationCanRetry', 'TestOwnedReopenOnSameThread',
            'TestOwnedConcurrentInferAndClose']
        children = {'TestOwnedBudgetsAndPadding': ['padding', 'exact',
            'truncate_preserves_sep', 'reject', 'envelope_cannot_fit']}
    def event(action, name=None, **extra):
        row = {'Package': package, 'Action': action, **extra}
        if name: row['Test'] = name
        print(json.dumps(row))
    event('start')
    if '-list' in args:
        for name in roots: event('output', Output=name + '\\n')
    else:
        repeats = 1 if roots == ['TestPublishedOpenVINO'] else 3
        assert '-count=' + str(repeats) in args
        if repeats == 3: assert '-race' in args
        for _ in range(repeats):
            for name in roots:
                event('run', name)
                for child in children.get(name, []):
                    event('run', name + '/' + child)
                    event('pass', name + '/' + child)
                event('pass', name)
    event('pass')
"""


class OpenVINOReportDirectoryTests(unittest.TestCase):
    def test_default_generic_destination_and_dedicated_override(self):
        for generic, dedicated in (
            (None, None),
            ("environment", None),
            ("command", None),
            ("environment", "environment"),
            ("command", "command"),
        ):
            with self.subTest(generic=generic, dedicated=dedicated):
                self.run_target(generic, dedicated)

    def run_target(self, generic, dedicated):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("bin", "openvino-binding", "src/semantic-router"):
                (root / name).mkdir(parents=True)
            for name in ("go", "python3"):
                executable = root / "bin" / name
                executable.write_text(f"#!{sys.executable}\n" + PROCESS_STUB)
                executable.chmod(0o755)
            generic_path = root / "reports/native.openvino-cpu"
            dedicated_path = root / "dedicated-openvino"
            expected = (
                dedicated_path
                if dedicated
                else (
                    generic_path
                    if generic
                    else root / ".agent-harness/model-tests/openvino-cpu"
                )
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "MAKEFLAGS",
                    "MFLAGS",
                    "MAKEOVERRIDES",
                    "MAKEFILES",
                    "MODEL_TEST_REPORT_DIR",
                    "OPENVINO_TEST_REPORT_DIR",
                }
            }
            environment.update(
                PATH=str(root / "bin") + os.pathsep + environment.get("PATH", ""),
                OPENVINO_CONTRACT_ROOT=str(root),
                OPENVINO_CONTRACT_CALLS=str(root / "calls.jsonl"),
                OPENVINO_CONTRACT_PLATFORM=host_platform(),
            )
            flags = []
            for kind, name, value in (
                (generic, "MODEL_TEST_REPORT_DIR", generic_path),
                (dedicated, "OPENVINO_TEST_REPORT_DIR", dedicated_path),
            ):
                if kind == "environment":
                    environment[name] = str(value)
                elif kind == "command":
                    flags.append(f"{name}={value}")
            result = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    "-f",
                    str(ROOT / "tools/make/common.mk"),
                    # Resolve models.mk's generic file default before OpenVINO;
                    # standalone runs must still keep their own destination.
                    "-f",
                    str(ROOT / "tools/make/models.mk"),
                    "-f",
                    str(ROOT / "tools/make/openvino.mk"),
                    "-o",
                    "build-openvino-binding",
                    "-o",
                    "rust-ci",
                    "-o",
                    "build-onnx-binding",
                    "verify-openvino-binding",
                    "SHELL=/bin/sh",
                    *flags,
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                calls,
                [
                    {"manifest": str(expected / "sources.json")},
                    {"manifest": str(expected / "models.json")},
                ],
            )
            evidence = native_evidence(expected)
            self.assertEqual(collection_errors(evidence, "test"), [])
            self.assertEqual(len(evidence["cases"]), 51)
            self.assertEqual(evidence["runtime"], "openvino")


if __name__ == "__main__":
    unittest.main()
