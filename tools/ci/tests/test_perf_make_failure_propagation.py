import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PERFORMANCE_MAKE = REPO_ROOT / "tools/make/performance.mk"


class PerfMakeFailurePropagationTests(unittest.TestCase):
    def test_failed_benchmark_stops_captured_targets(self) -> None:
        for target, suite, calls, report in (
            ("perf-check", "perf", 1, "bench-output.txt"),
            ("perf-check", "src/semantic-router", 2, "bench-output.txt"),
            ("perf-bench-looper", "src/semantic-router", 1, "bench-results-looper.txt"),
            ("perf-baseline-update", "perf", 1, "bench-results.txt"),
            ("perf-baseline-update", "src/semantic-router", 2, "bench-results.txt"),
        ):
            with self.subTest(target=target, suite=suite):
                result, invocations, output = self._run_target(target, suite, report)

                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("injected benchmark failure", result.stderr)
                self.assertEqual(len(invocations), calls, invocations)
                self.assertTrue(all(line.startswith("test ") for line in invocations))
                self.assertIn("BenchmarkCaptured", output)

    def test_successful_benchmarks_continue_to_consumers(self) -> None:
        for target, expected, report in (
            ("perf-check", ["test", "test", "run", "run"], "bench-output.txt"),
            ("perf-bench-looper", ["test"], "bench-results-looper.txt"),
            ("perf-baseline-update", ["test", "test", "baseline"], "bench-results.txt"),
        ):
            with self.subTest(target=target):
                result, invocations, output = self._run_target(target, "", report)

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual([line.split()[0] for line in invocations], expected)
                self.assertEqual(
                    output.count("BenchmarkCaptured"), expected.count("test")
                )

    def _run_target(
        self, target: str, failed_suite: str, report: str
    ) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("bin", "perf/scripts", "src/semantic-router"):
                (root / name).mkdir(parents=True)
            fixture = root / "fixture.mk"
            fixture.write_text(
                "LOG_TARGET = :\nNATIVE_ENV = PERF_TEST_NATIVE=1\nbuild-router:\n\t@:\n"
            )
            go = root / "bin/go"
            go.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" >> "$PERF_TEST_CALLS"\n'
                'if [ "$1" = test ]; then\n'
                '  echo "BenchmarkCaptured-1 1 100 ns/op"\n'
                '  if [ "$PWD" = "$PERF_TEST_FAIL_DIRECTORY" ]; then\n'
                '    echo "injected benchmark failure" >&2\n'
                "    exit 37\n"
                "  fi\n"
                "fi\n"
            )
            go.chmod(0o755)
            updater = root / "perf/scripts/update-baseline.sh"
            updater.write_text('#!/bin/sh\necho baseline >> "$PERF_TEST_CALLS"\n')
            updater.chmod(0o755)
            calls = root / "calls.txt"
            environment = dict(os.environ)
            environment.update(
                PATH=str(root / "bin") + os.pathsep + environment.get("PATH", ""),
                PERF_TEST_CALLS=str(calls),
                PERF_TEST_FAIL_DIRECTORY=(
                    str((root / failed_suite).resolve()) if failed_suite else ""
                ),
            )
            result = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    "-f",
                    str(PERFORMANCE_MAKE),
                    "-f",
                    str(fixture),
                    target,
                    "SHELL=/bin/sh",
                ],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            return (
                result,
                calls.read_text().splitlines(),
                (root / "reports" / report).read_text(),
            )


if __name__ == "__main__":
    unittest.main()
