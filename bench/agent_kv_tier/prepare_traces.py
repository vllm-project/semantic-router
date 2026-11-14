"""Download the SemiAnalysis Claude Code session traces at a pinned revision and split them into the one file per
trace that WEKA's kv-cache-tester reads (`--trace-directory`).

The dataset is semianalysisai/cc-traces-weka-with-subagents-060526 (Apache-2.0): 336 sessions with token counts
and 64-token KV block hashes per request and subagents marked, no text.
Usage: prepare_traces.py --out DIR [--revision SHA]
"""

import argparse
import json
from pathlib import Path

REPO = "semianalysisai/cc-traces-weka-with-subagents-060526"
REVISION = "8a5df339ba3c31a48dd79a2f5ceb81e08679266b"


def split_traces(lines, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in lines:
        if not line.strip():
            continue
        (out_dir / f"trace_{count:04d}.json").write_text(json.dumps(json.loads(line)))
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()
    # Lazy: split_traces and its tests need only the standard library.
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = hf_hub_download(
        REPO, "traces.jsonl", repo_type="dataset", revision=a.revision
    )
    with open(path) as f:
        print(split_traces(f, a.out), "traces in", a.out)


if __name__ == "__main__":
    main()
