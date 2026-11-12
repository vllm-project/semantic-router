"""Read-only localhost smoke test using the Claude plugin's exact synthetic probes."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8790/v1/decisions")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

assert urlsplit(args.url).hostname in {
    "localhost",
    "127.0.0.1",
    "::1",
}, "This experiment only sends fixtures to localhost"
repo = Path(__file__).resolve().parents[5]
helper = repo / "src/agent-harness/claude-code/skills/setup/scripts/runtime.py"
sys.path.insert(0, str(helper.parent))
spec = importlib.util.spec_from_file_location("claude_runtime_probe", helper)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
health = runtime.health(args.url, None)
assert health["ready"], health
rows = runtime.run_probes(args.url, None, runtime.settings.defaults(), None, True)
report = {
    "fixture_only": True,
    "health": health,
    "cases": rows,
    "median_ms": runtime.median_ms([r["got"] for r in rows]),
    "warning": runtime.slow_note(runtime.median_ms([r["got"] for r in rows])),
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
for row in rows:
    print(row["app"], row["status"], row["got"], flush=True)
assert {"router", "escalate", "guard", "shield", "check", "redact"} <= {
    r["app"] for r in rows
}
assert not any(
    r["status"] == "error" for r in rows
), "See report for runtime/protocol failures"
