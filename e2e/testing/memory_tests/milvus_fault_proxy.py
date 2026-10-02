"""Test-only Milvus gRPC proxy with an independently controlled write gate.

Only Insert/Upsert can be held or rejected. Reads and initialization continue
to use the real backend, so the client response can finish while persistence
is blocked. No request payloads are decoded, logged, or retained after the RPC.
The control listener is loopback-only and the harness owns this process.
"""

import argparse
import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc

WRITE_METHODS = {
    "/milvus.proto.milvus.MilvusService/Insert",
    "/milvus.proto.milvus.MilvusService/Upsert",
}
MAX_FORWARD_TIMEOUT_SECONDS = 120
MAX_CONTROL_BODY_BYTES = 1024


class WriteGate:
    def __init__(self):
        self.condition = threading.Condition()
        self.mode = "pass"
        self.entered = self.active = self.cancelled = self.rejected = 0

    def snapshot(self):
        with self.condition:
            return {
                "mode": self.mode,
                "entered": self.entered,
                "active": self.active,
                "cancelled": self.cancelled,
                "rejected": self.rejected,
            }

    def set_mode(self, mode):
        if mode not in {"pass", "hold", "fail"}:
            raise ValueError("mode must be pass, hold, or fail")
        with self.condition:
            if mode != "pass":
                if self.active:
                    raise ValueError("previous writes have not drained")
                self.entered = self.cancelled = self.rejected = 0
            self.mode = mode
            self.condition.notify_all()

    def enter(self, context):
        with self.condition:
            self.entered += 1
            self.active += 1
            try:
                while self.mode == "hold" and context.is_active():
                    self.condition.wait(timeout=0.05)
                if not context.is_active():
                    self.cancelled += 1
                    context.abort(grpc.StatusCode.CANCELLED, "fixture write cancelled")
                if self.mode == "fail":
                    self.rejected += 1
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT, "fixture write rejected"
                    )
            finally:
                self.active -= 1


class MilvusFaultProxy(grpc.GenericRpcHandler):
    def __init__(self, upstream):
        self.channel = grpc.insecure_channel(upstream)
        self.gate = WriteGate()

    def service(self, details):
        method = details.method

        def forward(request, context):
            if method in WRITE_METHODS:
                self.gate.enter(context)
            remaining = context.time_remaining()
            if remaining is not None and remaining <= 0:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "proxy caller expired")
            if not context.is_active():
                context.abort(grpc.StatusCode.CANCELLED, "proxy caller cancelled")
            call = self.channel.unary_unary(method).future(
                request,
                timeout=(
                    MAX_FORWARD_TIMEOUT_SECONDS
                    if remaining is None
                    else min(remaining, MAX_FORWARD_TIMEOUT_SECONDS)
                ),
                metadata=context.invocation_metadata(),
            )
            if not context.add_callback(call.cancel):
                call.cancel()
                context.abort(grpc.StatusCode.CANCELLED, "proxy caller cancelled")
            try:
                result = call.result()
                context.send_initial_metadata(call.initial_metadata())
                context.set_trailing_metadata(call.trailing_metadata())
                return result
            except grpc.FutureCancelledError:
                context.abort(grpc.StatusCode.CANCELLED, "proxy caller cancelled")
            except grpc.RpcError as error:
                context.abort(error.code(), error.details())

        return grpc.unary_unary_rpc_method_handler(forward)


def control_handler(gate):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/state":
                self.reply(404, {})
                return
            self.reply(200, gate.snapshot())

        def do_POST(self):
            if self.path != "/mode":
                self.reply(404, {})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_CONTROL_BODY_BYTES:
                    raise ValueError("invalid control body length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("control body must be an object")
                gate.set_mode(payload.get("mode"))
            except (ValueError, TypeError) as error:
                self.reply(409, {"error": str(error)})
                return
            self.reply(200, gate.snapshot())

        def log_message(self, *_args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    proxy = MilvusFaultProxy(args.upstream)
    server = grpc.server(ThreadPoolExecutor(max_workers=32))
    server.add_generic_rpc_handlers((proxy,))
    port = server.add_insecure_port(f"{args.listen_host}:0")
    if not port:
        raise RuntimeError("cannot bind Milvus proxy")
    control = ThreadingHTTPServer(("127.0.0.1", 0), control_handler(proxy.gate))
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_args: stopped.set())
    server.start()
    thread = threading.Thread(target=control.serve_forever, daemon=True)
    thread.start()
    args.ready_file.write_text(
        json.dumps(
            {"port": port, "control_url": f"http://127.0.0.1:{control.server_port}"}
        )
    )
    try:
        stopped.wait()
    finally:
        server.stop(0).wait()
        control.shutdown()
        control.server_close()
        thread.join()
        proxy.channel.close()


if __name__ == "__main__":
    main()
