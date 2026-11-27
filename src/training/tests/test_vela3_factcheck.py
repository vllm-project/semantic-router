import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.training.model_experiment.vela3_factcheck.data import overlaps
from src.training.model_experiment.vela3_factcheck.package import export_variant


class FactCheckDataTests(unittest.TestCase):
    def test_audit_detects_copied_prompts_and_source_groups(self):
        training = [
            {"id": "a", "text": "What time does the library close today?"},
            {"id": "b", "text": "Sum the values.", "group_sha256": "source-group"},
        ]
        held_out = [
            {"id": "c", "text": " WHAT time does the library close today？ "},
            {"id": "d", "text": "Suma los valores.", "group_sha256": "source-group"},
            {"id": "e", "text": "What time does the station close today?"},
        ]
        self.assertEqual(overlaps(training, held_out), {("a", "c"), ("b", "d")})

    def test_export_replaces_only_adapter_and_attests_output_bytes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_bytes(b"{}")
            (base / "factcheck_adapter.safetensors").write_bytes(b"base")
            adapter = root / "trained.safetensors"
            adapter.write_bytes(b"candidate")
            out = root / "variant"
            export_variant(
                base, ("config.json", "factcheck_adapter.safetensors"), adapter, out
            )
            self.assertEqual(
                (base / "factcheck_adapter.safetensors").read_bytes(), b"base"
            )
            self.assertEqual(
                (out / "factcheck_adapter.safetensors").read_bytes(), b"candidate"
            )
            self.assertEqual((out / "config.json").read_bytes(), b"{}")
            self.assertEqual(
                (out / "SHA256SUMS").read_text().splitlines()[0],
                "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a  config.json",
            )
