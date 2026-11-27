"""Send the labeled cases through the router and print per-case status."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HTTP_OK = 200


def main() -> int:
    parser = argparse.ArgumentParser(description="send smoke cases through the router")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--router", default="http://127.0.0.1:8899")
    parser.add_argument("--model", default="vllm-sr/auto")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    cases_path = Path(args.cases)
    ok = failed = 0
    with cases_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            case = json.loads(line)
            body = json.dumps(
                {
                    "model": args.model,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": case["prompt"]}],
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                args.router.rstrip("/") + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"{case['case_id']}\tERROR\t{exc}", file=sys.stderr)
                failed += 1
                continue
            print(f"{case['case_id']}\t{status}")
            if status == _HTTP_OK:
                ok += 1
            else:
                failed += 1
    print(f"sent ok={ok} failed={failed}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
