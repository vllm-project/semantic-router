"""Opt-in v0.5 Speed wiring test. Start fixtures, start the real SR, then probe.

Run `python live_speed_smoke.py --help` to see the fixture and probe modes. Use a private test
deployment: the two responders have distinct model identities but generate no
real answers. A probe checks the physical responders' request counters too.
"""

import argparse
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/count":
            self.reply({"id": self.server.model, "count": self.server.count})
        elif self.path == "/last-tool-choice":
            self.reply(self.server.last_tool_choice)
        elif self.path == "/v1/models":
            self.reply({"object": "list", "data": [{"id": self.server.model}]})
        else:
            self.reply({}, 404)

    def stream_text(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        event = {
            "id": "chatcmpl-" + secrets.token_hex(8),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.server.model,
        }
        for delta, finish in [
            ({"role": "assistant", "content": "served by "}, None),
            ({"content": self.server.model}, None),
            ({}, "stop"),
        ]:
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
            payload = json.dumps({**event, "choices": [choice]})
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self):
        # Count attempted dispatch even when the router sends an invalid payload.
        with self.server.lock:
            self.server.count += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.last_tool_choice = {
            "present": "tool_choice" in body,
            "value": body.get("tool_choice"),
            "offered_tools": [
                tool["function"]["name"] for tool in body.get("tools", [])
            ],
            "parallel_tool_calls_present": "parallel_tool_calls" in body,
        }
        if (
            self.path != "/v1/chat/completions"
            or body.get("model") != self.server.model
        ):
            self.reply({"error": "wrong path or physical model"}, 400)
            return
        if body.get("stream"):
            if body.get("tools"):
                self.reply({"error": "fixture only supports text streaming"}, 400)
            else:
                self.stream_text()
            return
        msg = {"role": "assistant", "content": "served by " + self.server.model}
        end = "stop"
        if body["messages"][-1]["role"] == "tool":
            msg["content"] = body["messages"][-1]["content"]
        elif body.get("tools") and wants_nonce_call(body):
            end = "tool_calls"
            msg.update(
                content=None,
                tool_calls=[
                    {
                        "id": "local-call",
                        "type": "function",
                        "function": {"name": "get_nonce", "arguments": "{}"},
                    }
                ],
            )
        self.reply(
            {
                "id": "fixture",
                "object": "chat.completion",
                "model": self.server.model,
                "choices": [{"index": 0, "finish_reason": end, "message": msg}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )


def wants_nonce_call(body):
    choice = body.get("tool_choice", "auto")
    offered = {tool.get("function", {}).get("name") for tool in body["tools"]}
    if "get_nonce" not in offered or choice == "none":
        return False
    if isinstance(choice, dict):
        return choice.get("function", {}).get("name") == "get_nonce"
    last_message = body["messages"][-1]
    return choice == "required" or (
        choice == "auto"
        and last_message.get("role") == "user"
        and "call get_nonce" in str(last_message.get("content", "")).lower()
    )


def fixtures(args):
    servers = []
    try:
        for model, port in [
            (args.strong, args.strong_port),
            (args.fast, args.fast_port),
        ]:
            server = ThreadingHTTPServer((args.bind, port), Fixture)
            server.model, server.count, server.lock = model, 0, threading.Lock()
            server.last_tool_choice = {}
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def read_fixture_stream(response, model):
    require(response.headers.get_content_type() == "text/event-stream", "expected SSE")
    # The fixture emits single-line data fields. Comments/keepalives carry no data.
    events = [line[5:].strip() for line in response if line.startswith(b"data:")]
    require(events and events[-1] == b"[DONE]", "missing terminal [DONE]")
    chunks = [json.loads(event) for event in events[:-1]]
    require(len(chunks) >= 3, "expected two text deltas followed by a stop chunk")
    chunk_ids = {chunk["id"] for chunk in chunks}
    require(
        len(chunk_ids) == 1 and all(chunk_ids), "missing or inconsistent stream IDs"
    )
    choices = []
    for chunk in chunks:
        require(
            chunk.get("object") == "chat.completion.chunk"
            and chunk.get("model") == model,
            "wrong physical model or stream object",
        )
        (choice,) = chunk["choices"]
        require(choice["index"] == 0, "unexpected streaming choice")
        choices.append(choice)
    require(
        all(c.get("finish_reason") is None for c in choices[:-1])
        and choices[-1].get("finish_reason") == "stop",
        "unexpected finish reason",
    )
    parts = [c["delta"]["content"] for c in choices if c["delta"].get("content")]
    text = "".join(parts)
    require(
        len(parts) >= 2 and text == "served by " + model, "incomplete streamed text"
    )
    return {
        "model": model,
        "finish_reason": "stop",
        "text": text,
        "content_deltas": len(parts),
        "id": chunks[0]["id"],
        "done": True,
    }


def probe(args):
    backends = {args.strong: args.strong_url, args.fast: args.fast_url}
    require(len(backends) == 2, "distinct physical backends required")

    def counts():
        out = {}
        for model, origin in backends.items():
            with urlopen(origin.rstrip("/") + "/count", timeout=10) as resp:
                state = json.load(resp)
            require(state["id"] == model, "counter belongs to the wrong fixture")
            out[model] = state["count"]
        return out

    def call(label, msgs, lane=None, model=None, finish="stop", **options):
        before = counts()
        headers = {"Content-Type": "application/json"}
        if args.key_env:
            headers["Authorization"] = "Bearer " + os.environ[args.key_env]
        body = {
            "model": args.public_model,
            "messages": msgs,
            "max_tokens": 256,
            **options,
        }
        request = Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        try:
            resp = urlopen(request, timeout=120)
        except HTTPError as error:
            resp = error
        stream = options.get("stream", False)
        with resp:
            data = read_fixture_stream(resp, model) if stream else json.load(resp)
        after = counts()
        delta = {m: after[m] - before[m] for m in before}
        if model is None:
            require(
                resp.status == 503
                and data.get("error", {}).get("code") == "no_eligible_model",
                f"{label}: expected rejection; got HTTP {resp.status}",
            )
            require(
                not any(delta.values()), f"{label}: request reached backend: {delta}"
            )
        else:
            require(
                resp.status == 200 and data.get("model") == model,
                f"{label}: wrong responder",
            )
            lanes = (lane,) if isinstance(lane, str) else lane
            selected = resp.headers.get("x-vsr-selected-decision")
            require(
                selected in lanes and resp.headers.get("x-vsr-selected-model") == model,
                f"{label}: expected {lanes} and {model}, got {selected} and "
                f"{resp.headers.get('x-vsr-selected-model')}",
            )
            require(
                all(n == int(m == model) for m, n in delta.items()), f"{label}: {delta}"
            )
            require(
                (
                    data["finish_reason"]
                    if stream
                    else data["choices"][0]["finish_reason"]
                )
                == finish,
                f"{label}: incomplete reply",
            )
        tool_choice_receipt = {}
        if args.mode == "probe-optional-tools":
            origin = backends[model].rstrip("/")
            with urlopen(origin + "/last-tool-choice", timeout=10) as receipt:
                actual = json.load(receipt)
            client_choice = {
                "present": "tool_choice" in options,
                "value": options.get("tool_choice"),
            }
            # OpenAI Chat backends receive no tool fields when calls are disabled.
            # With tools enabled, the codec makes an omitted choice explicit.
            enabled = options.get("tool_choice") != "none"
            expected = {
                "present": enabled,
                "value": options.get("tool_choice", "auto") if enabled else None,
                "offered_tools": ["get_nonce"] if enabled else [],
                "parallel_tool_calls_present": enabled
                and "parallel_tool_calls" in options,
            }
            require(
                actual == expected,
                f"{label}: physical backend received {actual}, expected {expected}",
            )
            message = data["choices"][0]["message"]
            if finish == "tool_calls":
                (tc,) = message["tool_calls"]
                require(
                    bool(tc["id"])
                    and tc["function"]["name"] == "get_nonce"
                    and json.loads(tc["function"]["arguments"]) == {},
                    f"{label}: wrong tool",
                )
            else:
                require(
                    not message.get("tool_calls")
                    and message.get("content") == "served by " + model,
                    f"{label}: expected complete text without tool calls",
                )
            tool_choice_receipt = {
                "client_tool_choice": client_choice,
                "upstream_tool_choice": actual,
                "finish_reason": finish,
                "tool_calls": [
                    t["function"]["name"] for t in message.get("tool_calls", [])
                ],
            }
        print(
            json.dumps(
                {
                    "case": label,
                    "status": resp.status,
                    "model": data.get("model"),
                    "decision": resp.headers.get("x-vsr-selected-decision"),
                    "backend_dispatch_delta": delta,
                    **tool_choice_receipt,
                    **({"stream": data} if stream else {}),
                }
            )
        )
        return data

    def user(text):
        return [{"role": "user", "content": text}]

    if args.mode == "probe-optional-tools":
        probe_optional_tools(call, args)
        return

    call(
        "easy", user("Hello! Say hello back in one short sentence."), "fast", args.fast
    )
    call(
        "hard",
        user(
            "Investigate a production queue that loses events only during network "
            "partitions. Compare multiple designs, reason through failure scenarios "
            "step by step, and prove which recovery protocol prevents duplicate delivery."
        ),
        "reasoning",
        args.strong,
    )
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO"
        "+jhYoAAAAASUVORK5CYII="
    )
    call(
        "unsupported-image",
        user(
            [
                {"type": "text", "text": "Describe the image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + png},
                },
            ]
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_nonce",
                "description": "Read a receipt code",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    msgs = user("Call get_nonce now and report the receipt code.")
    first = call(
        "tool-start",
        msgs,
        "tools",
        args.strong,
        "tool_calls",
        tools=tools,
        tool_choice="required",
    )
    assistant = first["choices"][0]["message"]
    (tc,) = assistant["tool_calls"]
    require(
        tc["function"]["name"] == "get_nonce"
        and json.loads(tc["function"]["arguments"]) == {},
        "unexpected tool call",
    )
    nonce = secrets.token_hex(8)
    msgs += [assistant, {"role": "tool", "tool_call_id": tc["id"], "content": nonce}]
    last = call("tool-return", msgs, "tools", args.strong, tools=tools)
    require(
        last["choices"][0]["message"]["content"] == nonce,
        "tool result was not delivered",
    )
    call(
        "stream-text",
        user("Hello! Say hello back in one short sentence."),
        "fast",
        args.fast,
        stream=True,
    )


def probe_optional_tools(call, args):
    """Speed fixture policy for a tools-capable strong model and text-only fast model."""
    greeting = [
        {"role": "user", "content": "Hello! Say hello back in one short sentence."}
    ]
    explicit = [
        {"role": "user", "content": "Call get_nonce now and report the receipt code."}
    ]
    schema = [
        {
            "type": "function",
            "function": {
                "name": "get_nonce",
                "description": "Read a receipt code",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    disabled = {"tool_choice": "none", "parallel_tool_calls": False}
    for label, messages, options, lanes, finish in [
        ("greeting-implicit", greeting, {}, "fast", "stop"),
        ("greeting-auto", greeting, {"tool_choice": "auto"}, "fast", "stop"),
        ("greeting-none", greeting, disabled, "fast", "stop"),
        (
            "explicit-auto",
            explicit,
            {"tool_choice": "auto"},
            ("tools", "fast"),
            "tool_calls",
        ),
        ("explicit-none", explicit, disabled, "fast", "stop"),
        (
            "named",
            explicit,
            {"tool_choice": {"type": "function", "function": {"name": "get_nonce"}}},
            "tools",
            "tool_calls",
        ),
    ]:
        model = args.fast if options.get("tool_choice") == "none" else args.strong
        call(label, messages, lanes, model, finish, tools=schema, **options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fixtures")
    p = sub.add_parser("probe")
    optional = sub.add_parser("probe-optional-tools")
    for opts in [f, p, optional]:
        opts.add_argument("--strong", required=True)
        opts.add_argument("--fast", required=True)
    f.add_argument("--strong-port", type=int, required=True)
    f.add_argument("--fast-port", type=int, required=True)
    f.add_argument("--bind", default="127.0.0.1")
    for opts in [p, optional]:
        opts.add_argument("--base-url", required=True, help="OpenAI /v1 URL")
        opts.add_argument("--public-model", required=True)
        opts.add_argument("--key-env")
        opts.add_argument(
            "--strong-url", required=True, help="test fixture origin, without /v1"
        )
        opts.add_argument(
            "--fast-url", required=True, help="test fixture origin, without /v1"
        )
    args = parser.parse_args()
    (fixtures if args.mode == "fixtures" else probe)(args)
