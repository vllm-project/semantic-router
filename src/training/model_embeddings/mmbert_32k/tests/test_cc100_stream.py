"""Offline tests for the fixed CC-100 HTTP/XZ stream contract."""

from __future__ import annotations

import hashlib
import io
import lzma
import unittest

from src.training.model_embeddings.mmbert_32k.cc100_stream import (
    CC100SourceStats,
    iter_cc100_documents,
)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, etag: str, *, quote_etag: bool = True):
        super().__init__(payload)
        etag_header = etag if etag.startswith("W/") or not quote_etag else f'"{etag}"'
        self.headers = {
            "ETag": etag_header,
            "Last-Modified": "fixed",
            "Content-Length": str(len(payload)),
        }

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


class CC100StreamTest(unittest.TestCase):
    def test_stream_preserves_document_and_paragraph_order(self) -> None:
        source = "first paragraph\nsecond paragraph\n\nnext document\n\n"
        compressed = lzma.compress(source.encode("utf-8"))
        stats = CC100SourceStats()

        def opener(_request, *, timeout):
            self.assertEqual(timeout, 60)
            return FakeResponse(compressed, "fixed-etag")

        documents = list(
            iter_cc100_documents(
                "en",
                expected_etag="fixed-etag",
                expected_content_length=len(compressed),
                stats=stats,
                expected_prefix_bytes=len(compressed),
                expected_prefix_sha256=hashlib.sha256(compressed).hexdigest(),
                opener=opener,
            )
        )
        self.assertEqual(
            documents,
            ["first paragraph\nsecond paragraph", "next document"],
        )
        self.assertEqual(stats.etag, "fixed-etag")
        self.assertEqual(stats.compressed_bytes_read, len(compressed))
        self.assertEqual(
            stats.compressed_prefix_sha256, hashlib.sha256(compressed).hexdigest()
        )

    def test_changed_etag_fails_before_decompression(self) -> None:
        stats = CC100SourceStats()

        def opener(_request, *, timeout):
            return FakeResponse(lzma.compress(b"text\n"), "changed")

        with self.assertRaisesRegex(RuntimeError, "ETag changed"):
            list(
                iter_cc100_documents(
                    "en",
                    expected_etag="expected",
                    expected_content_length=len(lzma.compress(b"text\n")),
                    stats=stats,
                    opener=opener,
                )
            )
        self.assertEqual(stats.compressed_bytes_read, 0)

    def test_same_strong_etag_with_different_bytes_fails_prefix_hash(self) -> None:
        expected = lzma.compress(b"document A\n\n")
        changed = lzma.compress(b"document B\n\n")
        self.assertEqual(len(expected), len(changed))
        stats = CC100SourceStats()

        def opener(_request, *, timeout):
            return FakeResponse(changed, "same-etag")

        with self.assertRaisesRegex(RuntimeError, "prefix SHA-256 changed"):
            list(
                iter_cc100_documents(
                    "en",
                    expected_etag="same-etag",
                    expected_content_length=len(changed),
                    expected_prefix_bytes=len(changed),
                    expected_prefix_sha256=hashlib.sha256(expected).hexdigest(),
                    stats=stats,
                    opener=opener,
                )
            )

    def test_weak_etag_and_large_document_fail_closed(self) -> None:
        compressed = lzma.compress(b"one two three\n\n")

        def weak_opener(_request, *, timeout):
            return FakeResponse(compressed, 'W/"weak"')

        with self.assertRaisesRegex(RuntimeError, "weak ETag"):
            list(
                iter_cc100_documents(
                    "en",
                    expected_etag="strong",
                    expected_content_length=len(compressed),
                    stats=CC100SourceStats(),
                    opener=weak_opener,
                )
            )

        def unquoted_opener(_request, *, timeout):
            return FakeResponse(compressed, "strong", quote_etag=False)

        with self.assertRaisesRegex(RuntimeError, "non-strong ETag"):
            list(
                iter_cc100_documents(
                    "en",
                    expected_etag="strong",
                    expected_content_length=len(compressed),
                    stats=CC100SourceStats(),
                    opener=unquoted_opener,
                )
            )

        def strong_opener(_request, *, timeout):
            return FakeResponse(compressed, "strong")

        with self.assertRaisesRegex(ValueError, "document exceeds"):
            list(
                iter_cc100_documents(
                    "en",
                    expected_etag="strong",
                    expected_content_length=len(compressed),
                    max_document_bytes=3,
                    stats=CC100SourceStats(),
                    opener=strong_opener,
                )
            )


if __name__ == "__main__":
    unittest.main()
