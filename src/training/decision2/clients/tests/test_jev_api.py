"""Collector retry and resume checks without network access."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from clients.jev_api import run


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, *_args):
        return b'{"model":"jev-1.13.0","answers":{}}'


class CollectorTests(unittest.TestCase):
    def test_non_json_rate_limit_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": str(i),
                            "state": {},
                            "questions": {"q": {"type": "noul"}},
                        }
                    )
                    + "\n"
                    for i in range(2)
                ),
                encoding="utf-8",
            )
            receipts = root / "receipts.jsonl"
            rate_limit = HTTPError(
                "https://example.invalid",
                429,
                "rate limited",
                {},
                io.BytesIO(b"<html>busy</html>"),
            )
            with patch(
                "clients.jev_api.request.urlopen", side_effect=[rate_limit, Response()]
            ) as fetch, patch("clients.jev_api.time.sleep"):
                run(prompts, receipts, "jev-1.13.0", "test-token", 1)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(len(receipts.read_text().splitlines()), 1)
            with patch(
                "clients.jev_api.request.urlopen", return_value=Response()
            ) as fetch:
                run(prompts, receipts, "jev-1.13.0", "test-token", 0, resume=True)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(
                [json.loads(line)["id"] for line in receipts.read_text().splitlines()],
                ["0", "1"],
            )


if __name__ == "__main__":
    unittest.main()
