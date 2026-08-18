import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.config import (
    ContractError,
    contract_sha256,
    distributed_batch_parameters,
    load_contract,
)


class ContractTest(unittest.TestCase):
    def test_checked_in_contract_is_valid_and_stable(self):
        contract = load_contract()
        self.assertEqual(contract["taxonomy_version"], "legacy-9-v1")
        self.assertEqual(len(contract_sha256(contract)), 64)
        self.assertEqual(
            contract["base_model"]["revision"],
            "72a23a6640489471eb4ff7ad3ec5bc80af8a27de",
        )

    def test_eight_gpu_batch_matches_historical_global_batch(self):
        contract = load_contract()
        self.assertEqual(distributed_batch_parameters(contract, 8), (8, 1))
        self.assertEqual(distributed_batch_parameters(contract, 1), (8, 8))

    def test_invalid_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text('{"contract_version": 2}', encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)


if __name__ == "__main__":
    unittest.main()
