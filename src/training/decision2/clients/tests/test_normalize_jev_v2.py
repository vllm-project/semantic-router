"""The Jev v2 rebind must prove API-body and prediction provenance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.generate import digest
from clients.jev_api import canonical
from clients.normalize_jev_v2 import normalize


class JevV2NormalizationTests(unittest.TestCase):
    def test_verified_rebind_preserves_answer_and_original_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = {
                "id": "one",
                "state": "a state",
                "questions": {
                    "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
                },
            }
            answer = {"q": {"choice": "a", "probabilities": {"a": 0.99, "b": 0.01}}}
            body_hash = hashlib.sha256(
                canonical(
                    {
                        "state": prompt["state"],
                        "model": "jev-1.13.0",
                        "questions": prompt["questions"],
                    }
                )
            ).hexdigest()
            receipt = {
                "id": "one",
                "input_sha256": body_hash,
                "requested_model": "jev-1.13.0",
                "returned_model": "jev-1.13.0",
                "http_status": 200,
                "response": {"answers": answer, "usage": {"input_tokens": 12}},
            }
            old = {
                "id": "one",
                "source_input_sha256": body_hash,
                "answers": answer,
                "model": "jev-1.13.0",
                "http_status": 200,
                "usage": {"input_tokens": 12},
                "latency_ms": 23.0,
            }
            files = [
                root / name
                for name in ("prompts.jsonl", "receipts.jsonl", "legacy.jsonl")
            ]
            for path, row in zip(files, (prompt, receipt, old)):
                path.write_text(json.dumps(row) + "\n")
            originals = [path.read_bytes() for path in files]
            output = root / "bound.jsonl"
            stats = normalize(*files, output)
            rebound = json.loads(output.read_text())
            self.assertEqual(rebound["answers"], answer)
            self.assertEqual(
                rebound["source_input_sha256"],
                digest(
                    {
                        "state": prompt["state"],
                        "questions": prompt["questions"],
                    }
                ),
            )
            self.assertEqual(rebound["api_body_sha256"], body_hash)
            self.assertEqual(stats["items"], 1)
            self.assertEqual([path.read_bytes() for path in files], originals)
            with self.assertRaises(FileExistsError):
                normalize(*files, output)

            output.unlink()
            receipt["input_sha256"] = "0" * 64
            files[1].write_text(json.dumps(receipt) + "\n")
            with self.assertRaisesRegex(ValueError, "receipt does not bind"):
                normalize(*files, output)
            self.assertFalse(output.exists())

            receipt["input_sha256"] = body_hash
            files[1].write_text(json.dumps(receipt) + "\n")
            old["answers"] = {"q": {"choice": "b"}}
            files[2].write_text(json.dumps(old) + "\n")
            with self.assertRaisesRegex(ValueError, "legacy prediction differs"):
                normalize(*files, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
