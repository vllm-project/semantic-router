"""Loopback RPC tests for the write fault fixture; no Milvus or model needed."""

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import grpc

from memory_tests.milvus_fault_proxy import (
    MAX_FORWARD_TIMEOUT_SECONDS,
    MilvusFaultProxy,
    control_handler,
)

SERVICE = "milvus.proto.milvus.MilvusService"


class MilvusFaultProxyTest(unittest.TestCase):
    def setUp(self):
        self.forwarded = []

        def backend(request, context):
            self.forwarded.append(request)
            if request == b"reject":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "backend rejected")
            return b"ok:" + request

        self.backend = grpc.server(ThreadPoolExecutor(max_workers=4))
        self.backend.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    SERVICE,
                    {
                        name: grpc.unary_unary_rpc_method_handler(backend)
                        for name in ("Insert", "Upsert", "Query")
                    },
                ),
            )
        )
        port = self.backend.add_insecure_port("127.0.0.1:0")
        self.backend.start()
        self.addCleanup(lambda: self.backend.stop(0).wait())
        self.proxy = MilvusFaultProxy(f"127.0.0.1:{port}")
        self.addCleanup(self.proxy.channel.close)
        self.server = grpc.server(ThreadPoolExecutor(max_workers=8))
        self.server.add_generic_rpc_handlers((self.proxy,))
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.addCleanup(lambda: self.server.stop(0).wait())
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.addCleanup(self.channel.close)
        grpc.channel_ready_future(self.channel).result(timeout=3)

    def rpc(self, method):
        return self.channel.unary_unary(f"/{SERVICE}/{method}")

    def wait_state(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self.proxy.gate.snapshot()
            if predicate(state):
                return state
            time.sleep(0.01)
        self.fail(f"fixture did not reach expected state: {state}")

    def test_passthrough_preserves_bytes_and_backend_errors(self):
        self.assertEqual(self.rpc("Query")(b"query", timeout=3), b"ok:query")
        self.assertEqual(self.rpc("Insert")(b"write", timeout=3), b"ok:write")
        with self.assertRaises(grpc.RpcError) as error:
            self.rpc("Query")(b"reject", timeout=3)
        self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(error.exception.details(), "backend rejected")

    def test_held_write_cannot_commit_but_reads_continue(self):
        self.proxy.gate.set_mode("hold")
        write = self.rpc("Insert").future(b"held", timeout=3)
        self.wait_state(lambda state: state["active"] == 1)
        self.assertFalse(write.done())
        self.assertEqual(self.rpc("Query")(b"read", timeout=1), b"ok:read")
        self.assertNotIn(b"held", self.forwarded)
        self.proxy.gate.set_mode("pass")
        self.assertEqual(write.result(timeout=3), b"ok:held")

    def test_deadline_cancels_a_held_rpc_without_forwarding_the_write(self):
        self.proxy.gate.set_mode("hold")
        write = self.rpc("Insert").future(b"expired", timeout=0.3)
        self.wait_state(lambda state: state["active"] == 1)
        with self.assertRaises(grpc.RpcError) as error:
            write.result(timeout=3)
        self.assertEqual(error.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        self.wait_state(lambda state: state["active"] == 0 and state["cancelled"] == 1)
        self.proxy.gate.set_mode("pass")
        self.assertNotIn(b"expired", self.forwarded)

    def test_client_cancellation_unwinds_held_write_without_releasing_gate(self):
        self.proxy.gate.set_mode("hold")
        write = self.rpc("Upsert").future(b"cancelled", timeout=3)
        self.wait_state(lambda state: state["active"] == 1)
        self.assertTrue(write.cancel())
        self.wait_state(lambda state: state["active"] == 0 and state["cancelled"] == 1)
        self.assertNotIn(b"cancelled", self.forwarded)
        self.assertEqual(self.proxy.gate.snapshot()["mode"], "hold")

    def test_rejected_writes_fail_promptly_without_disrupting_reads(self):
        self.proxy.gate.set_mode("fail")
        for name in ("Insert", "Upsert"):
            with self.assertRaises(grpc.RpcError) as error:
                self.rpc(name)(b"denied", timeout=3)
            self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(self.proxy.gate.snapshot()["rejected"], 2)
        self.assertEqual(self.rpc("Query")(b"read", timeout=1), b"ok:read")
        self.assertEqual(self.forwarded, [b"read"])

    def test_http_control_cannot_reset_an_active_gate(self):
        control = ThreadingHTTPServer(
            ("127.0.0.1", 0), control_handler(self.proxy.gate)
        )
        thread = threading.Thread(target=control.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(control.server_close)
        self.addCleanup(control.shutdown)
        base = f"http://127.0.0.1:{control.server_port}"

        def mode(value):
            request = Request(
                base + "/mode",
                data=json.dumps({"mode": value}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=3) as response:
                return json.load(response)

        self.assertEqual(mode("hold")["mode"], "hold")
        write = self.rpc("Insert").future(b"held", timeout=3)
        self.wait_state(lambda state: state["active"] == 1)
        with self.assertRaises(HTTPError) as error:
            mode("fail")
        self.assertEqual(error.exception.code, 409)
        error.exception.close()
        with urlopen(base + "/state", timeout=3) as response:
            self.assertEqual(json.load(response)["active"], 1)
        mode("pass")
        self.assertEqual(write.result(timeout=3), b"ok:held")


class MilvusFaultProxyDeadlineTest(unittest.TestCase):
    def setUp(self):
        self.proxy = MilvusFaultProxy("127.0.0.1:1")
        self.addCleanup(self.proxy.channel.close)
        self.proxy.channel = Mock()
        self.context = Mock()
        self.context.is_active.return_value = True
        self.context.add_callback.return_value = True
        self.context.abort.side_effect = grpc.RpcError
        self.context.invocation_metadata.return_value = ()
        self.handler = self.proxy.service(SimpleNamespace(method=f"/{SERVICE}/Query"))

    def test_expired_rpc_does_not_reach_upstream(self):
        self.context.time_remaining.return_value = 0.0
        with self.assertRaises(grpc.RpcError):
            self.handler.unary_unary(b"expired", self.context)
        self.context.abort.assert_called_once_with(
            grpc.StatusCode.DEADLINE_EXCEEDED, "proxy caller expired"
        )
        self.proxy.channel.unary_unary.assert_not_called()

    def test_cancelled_rpc_without_deadline_does_not_reach_upstream(self):
        self.context.time_remaining.return_value = None
        self.context.is_active.return_value = False
        with self.assertRaises(grpc.RpcError):
            self.handler.unary_unary(b"cancelled", self.context)
        self.context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED, "proxy caller cancelled"
        )
        self.proxy.channel.unary_unary.assert_not_called()

    def test_forwarding_preserves_remaining_budget_and_caps_unbounded_calls(self):
        for remaining, expected in (
            (None, MAX_FORWARD_TIMEOUT_SECONDS),
            (600, MAX_FORWARD_TIMEOUT_SECONDS),
            (0.05, 0.05),
        ):
            with self.subTest(remaining=remaining):
                self.context.time_remaining.return_value = remaining
                self.handler.unary_unary(b"query", self.context)
                future = self.proxy.channel.unary_unary.return_value.future
                self.assertEqual(future.call_args.kwargs["timeout"], expected)

    def test_cancellation_during_callback_registration_cancels_upstream(self):
        self.context.time_remaining.return_value = 1.0
        self.context.add_callback.return_value = False
        with self.assertRaises(grpc.RpcError):
            self.handler.unary_unary(b"cancelled", self.context)
        call = self.proxy.channel.unary_unary.return_value.future.return_value
        call.cancel.assert_called_once_with()
        call.result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
