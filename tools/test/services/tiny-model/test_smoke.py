"""Failure-path coverage for the real-generation smoke's acceptance criteria."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run
from smoke import check_stop, read_stream


def frame(content=None, finish=None):
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
    )


class SmokeAssertionsTest(unittest.TestCase):
    def test_stream_requires_finish_and_terminator(self):
        with self.assertRaisesRegex(AssertionError, "finish reason"):
            read_stream([frame("hello"), "data: [DONE]"])
        with self.assertRaisesRegex(AssertionError, "without"):
            read_stream([frame("hello"), frame(finish="stop")])
        self.assertEqual(
            read_stream([frame("hello"), frame(finish="stop"), "data: [DONE]"]),
            "hello",
        )

    def test_cached_weights_require_the_pinned_checksum(self):
        payload = b"small test model fixture"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fixture.gguf"
            model.write_bytes(payload)
            with patch.dict(
                run.SPEC,
                {
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            ):
                self.assertTrue(run.verify_model(model))
                model.write_bytes(b"x" + payload[1:])
                self.assertFalse(run.verify_model(model))

    def test_stop_reason_alone_does_not_pass(self):
        with self.assertRaisesRegex(AssertionError, "leaked"):
            check_stop("hello world", "hello world", "world", "stop")
        with self.assertRaisesRegex(AssertionError, "truncate"):
            check_stop("hello world", "unrelated", "world", "stop")
        check_stop("hello world", "hello ", "world", "stop")


if __name__ == "__main__":
    unittest.main()
