"""Round-2 probe: verify the tool-loop binding vs admitted-set conflict.

- E: hard turn (-> frontier, loop binding candidate) then a tool-result turn
  whose content ALSO reads as hard code: admitted set should contain the
  bound candidate -> expected 200, sticky to the bound model.
- F: hard turn (-> frontier) then a tool-result turn whose content reads as
  simple: admitted set is local-only, binding is frontier -> predicted 503
  (selection_rejected: no_eligible_model).
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

ROUTER = "http://127.0.0.1:8899"
TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]
}
HARD = (
    "Redesign our distributed rate limiter: compare token bucket, sliding window and GCRA from first "
    "principles, analyse the failure modes under clock skew, and justify a recommendation."
)
HARD_TOOL_RESULT = "GCRA burst analysis: the per-region drift terms matter; see src/limiter/gcra.go line 40."
SIMPLE_TOOL_RESULT = "Done. The file says hello."


def post(body: dict, session: str, conversation: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        ROUTER + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-session-id": session,
            "x-conversation-id": conversation,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")[:300]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []

    def turn(tag: str, conversation: str, messages: list[dict], tools: bool) -> None:
        body = {"model": "vllm-sr/auto", "max_tokens": 64, "messages": messages}
        if tools:
            body.update(TOOLS)
        status, data = post(body, "probe-r2", conversation)
        entry = {
            "turn": tag,
            "status": status,
            "model": data.get("model"),
            "error": data.get("error"),
        }
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    # E: binding candidate inside the admitted set
    turn("E1-hard", "conv-E", [{"role": "user", "content": HARD}], False)
    turn(
        "E2-tool-result-hard",
        "conv-E",
        [
            {"role": "user", "content": HARD},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "src/limiter/gcra.go"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": HARD_TOOL_RESULT},
        ],
        True,
    )
    # F: binding candidate outside the admitted set
    turn("F1-hard", "conv-F", [{"role": "user", "content": HARD}], False)
    turn(
        "F2-tool-result-simple",
        "conv-F",
        [
            {"role": "user", "content": HARD},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "notes.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": SIMPLE_TOOL_RESULT},
        ],
        True,
    )
    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
