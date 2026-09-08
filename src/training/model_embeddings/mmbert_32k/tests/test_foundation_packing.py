"""Standard-library tests for deterministic 32K packing primitives."""

from __future__ import annotations

import hashlib
import unittest

from src.training.model_embeddings.mmbert_32k.foundation_packing import (
    PackingStats,
    iter_packed_sequences,
    real_token_ids,
    stable_language_quotas,
    update_token_digest,
)


class FakeTokenizer:
    sep_token_id = 99
    eos_token_id = 98

    def __call__(self, text, *, add_special_tokens, truncation):
        if add_special_tokens or truncation:
            raise AssertionError("packing must tokenize complete raw documents")
        return {"input_ids": [int(piece) for piece in text.split()]}

    @staticmethod
    def num_special_tokens_to_add(*, pair):
        if pair:
            raise AssertionError("foundation packing is single-sequence")
        return 2

    @staticmethod
    def build_inputs_with_special_tokens(payload):
        return [101, *payload, 102]


class FoundationPackingTest(unittest.TestCase):
    def test_stable_30774_sequence_quotas(self) -> None:
        languages = [
            "en",
            "zh-Hans",
            "de",
            "fr",
            "es",
            "ru",
            "ar",
            "ja",
            "ko",
        ]
        quotas = stable_language_quotas(languages, 30_774)
        self.assertEqual(sum(quotas.values()), 30_774)
        self.assertEqual([quotas[language] for language in languages[:3]], [3420] * 3)
        self.assertEqual([quotas[language] for language in languages[3:]], [3419] * 6)

    def test_source_order_separator_and_exact_unpadded_output(self) -> None:
        stats = PackingStats()
        examples = list(
            iter_packed_sequences(
                ["1 2", "3 4 5", "6 7 8 9 10"],
                FakeTokenizer(),
                language="en",
                target_length=8,
                sequence_quota=2,
                stats=stats,
            )
        )
        self.assertEqual(
            [example["input_ids"] for example in examples],
            [
                [101, 1, 2, 99, 3, 4, 5, 102],
                [101, 99, 6, 7, 8, 9, 10, 102],
            ],
        )
        self.assertTrue(
            all(example["attention_mask"] == [1] * 8 for example in examples)
        )
        self.assertEqual(stats.documents_consumed, 3)
        self.assertEqual(stats.source_tokens_consumed, 10)
        self.assertEqual(stats.separators_inserted, 2)
        self.assertEqual(stats.sequences_emitted, 2)

    def test_exact_quota_does_not_read_an_extra_document(self) -> None:
        reads = []

        def documents():
            for text in ("1 2 3 4 5 6", "7 8 9"):
                reads.append(text)
                yield text

        examples = list(
            iter_packed_sequences(
                documents(),
                FakeTokenizer(),
                language="en",
                target_length=8,
                sequence_quota=1,
            )
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(reads, ["1 2 3 4 5 6"])

    def test_terminal_separator_closes_consumed_document_without_preread(self) -> None:
        reads = []

        def documents():
            for text in ("1 2 3 4 5", "6 7 8"):
                reads.append(text)
                yield text

        stats = PackingStats()
        examples = list(
            iter_packed_sequences(
                documents(),
                FakeTokenizer(),
                language="en",
                target_length=8,
                sequence_quota=1,
                stats=stats,
            )
        )
        self.assertEqual(examples[0]["input_ids"], [101, 1, 2, 3, 4, 5, 99, 102])
        self.assertEqual(reads, ["1 2 3 4 5"])
        self.assertEqual(stats.documents_consumed, 1)
        self.assertEqual(stats.separators_inserted, 1)

    def test_padding_cannot_masquerade_as_long_context(self) -> None:
        token_ids = [11, 12, 0, 0, 0, 0, 0, 0]
        attention_mask = [1, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(real_token_ids(token_ids, attention_mask), [11, 12])
        self.assertEqual(len(real_token_ids(token_ids, attention_mask)), 2)

    def test_short_source_fails_closed_instead_of_padding(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "0/1 complete"):
            list(
                iter_packed_sequences(
                    ["1 2"],
                    FakeTokenizer(),
                    language="en",
                    target_length=8,
                    sequence_quota=1,
                )
            )

    def test_document_byte_and_token_caps_fail_closed(self) -> None:
        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "byte count"):
            list(
                iter_packed_sequences(
                    ["1 2"],
                    tokenizer,
                    language="en",
                    target_length=8,
                    sequence_quota=1,
                    max_document_bytes=2,
                )
            )
        with self.assertRaisesRegex(ValueError, "token count"):
            list(
                iter_packed_sequences(
                    ["1 2 3"],
                    tokenizer,
                    language="en",
                    target_length=8,
                    sequence_quota=1,
                    max_document_tokens=2,
                )
            )

    def test_token_digest_uses_stable_little_endian_uint32_batches(self) -> None:
        digest = hashlib.sha256()
        update_token_digest(digest, [1, 256, 65_535])
        expected = hashlib.sha256(
            b"\x01\x00\x00\x00\x00\x01\x00\x00\xff\xff\x00\x00"
        ).hexdigest()
        self.assertEqual(digest.hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
