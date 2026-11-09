"""Check that response status describes completed reuse and preserves the body."""

import json
import unittest

from src.kv_connector.http_adapter import KVTransferResponseMiddleware


class HTTPAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def run_response(
        self, body, content_type=b"application/json", rewritten=False
    ):
        sent = []

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", content_type)],
                }
            )
            await send(
                {"type": "http.response.body", "body": body[:5], "more_body": True}
            )
            await send({"type": "http.response.body", "body": body[5:]})

        async def send(message):
            sent.append(message)

        await KVTransferResponseMiddleware(app)(
            {"type": "http", "path": "/v1/chat/completions"}, None, send
        )
        actual = b"".join(m.get("body", b"") for m in sent)
        if rewritten:
            expected = json.loads(body)
            expected["kv_transfer_params"] = None
            self.assertEqual(json.loads(actual), expected)
            self.assertEqual(
                int(dict(sent[0]["headers"])[b"content-length"]), len(actual)
            )
        else:
            self.assertEqual(actual, body)
        return dict(sent[0]["headers"])

    async def test_completed_outcomes(self):
        for status in ("applied", "fallback_reprefill", "unsupported"):
            with self.subTest(status=status):
                body = json.dumps(
                    {
                        "kv_transfer_params": {
                            "vsr_kv_transfer": {"status": status, "reason": "loaded"}
                        }
                    }
                ).encode()
                headers = await self.run_response(body, rewritten=True)
                self.assertEqual(headers[b"x-vsr-kv-transfer-status"], status.encode())

    async def test_json_charset_reports_completed_outcome(self):
        body = json.dumps(
            {"kv_transfer_params": {"vsr_kv_transfer": {"status": "applied"}}}
        ).encode()
        headers = await self.run_response(
            body, b"application/json; charset=utf-8", rewritten=True
        )
        self.assertEqual(headers[b"x-vsr-kv-transfer-status"], b"applied")

    async def test_plain_streaming_invalid_and_large_responses_are_preserved(self):
        for body, content_type in (
            (b'{"choices": []}', b"application/json"),
            (b"data: [DONE]\n\n", b"text/event-stream"),
            (b'{"kv_transfer_params": "invalid"}', b"application/json"),
            (b"[" + b"0," * 600000 + b"0]", b"application/json"),
        ):
            with self.subTest(length=len(body), content_type=content_type):
                headers = await self.run_response(body, content_type)
                self.assertNotIn(b"x-vsr-kv-transfer-status", headers)
