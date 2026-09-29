"""Scan peer Hugging Face model cards (pinned revisions) for JevBench / training-data declarations.

Usage (stdin driver, node A): ssh NODE_A "cd /tmp && python3 - '<json>'" < card_scan.py
  json: {"out": DIR, "repos": [{"name", "repo", "revision"}]}
Downloads README.md (and any *.md / dataset list under the repo root) with the `hf` CLI into
DIR/cards/<name>/ (HF_HUB_CACHE=/data/dev2/hf-cache; the token is never printed), then prints
per repo: card `datasets:` metadata, and each line matching the keywords (trimmed to 240 chars).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

KEYS = re.compile(
    r"jevbench|jev[- ]?bench|jevarena|jev arena|public (items|split|set|decisions)|held[- ]?out|"
    r"sealed|contamin|fstandhartinger|datasets/public|easy\.jsonl|hard\.jsonl|standard\.jsonl|"
    r"benchmark (data|items)|train(ed|ing)? on",
    re.I,
)


def main():
    args = json.loads(sys.argv[1])
    out = Path(args["out"]) / "cards"
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o700)
    env = dict(os.environ, HF_HUB_CACHE="/data/dev2/hf-cache")
    report = {}
    for r in args["repos"]:
        d = out / r["name"]
        d.mkdir(exist_ok=True)
        entry = {"repo": r["repo"], "revision": r["revision"]}
        ls = subprocess.run(
            [
                "hf",
                "download",
                r["repo"],
                "--revision",
                r["revision"],
                "--include",
                "*.md",
                "--local-dir",
                str(d),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        entry["download_rc"] = ls.returncode
        if ls.returncode:
            entry["err"] = (
                ls.stderr.strip().splitlines()[-1][:200] if ls.stderr.strip() else ""
            )
        files = sorted(p for p in d.rglob("*.md") if ".cache" not in p.parts)
        entry["md_files"] = [str(p.relative_to(d)) for p in files]
        hits = []
        datasets = []
        for p in files:
            text = p.read_text(errors="replace")
            m = re.match(r"---\n(.*?)\n---", text, re.S)
            if m:
                block = m.group(1)
                dm = re.search(r"^datasets:\s*\n((?:\s*-\s*.+\n?)+)", block, re.M)
                if dm:
                    datasets += [
                        x.strip()[1:].strip()
                        for x in dm.group(1).splitlines()
                        if x.strip().startswith("-")
                    ]
            for i, line in enumerate(text.splitlines(), 1):
                if KEYS.search(line):
                    hits.append(f"{p.name}:{i}: {line.strip()[:240]}")
        entry["datasets_meta"] = datasets
        entry["keyword_lines"] = hits[:40]
        entry["keyword_line_count"] = len(hits)
        report[r["name"]] = entry
    (out / "card-scan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


main()
