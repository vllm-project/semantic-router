"""The legacy 4B export remains reproducible but cannot publish third-party starts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from publication import export_hf_v3 as export


class SlimExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "private"
        self.source.mkdir()
        (self.source / "native").mkdir()
        (self.source / "native" / "config.json").write_text('{"model_type":"test"}\n')
        (self.source / "native" / "model.safetensors").write_bytes(b"model weights")
        (self.source / "native" / "LICENSE").write_text("Eikos MIT terms\n")
        (self.source / "native" / "LICENSE-Qwen").write_text("Qwen Apache terms\n")
        (self.source / "native" / "NOTICE").write_text("Original Eikos notice\n")
        (self.source / "native" / "decision2_provenance.json").write_text(
            '{"training_source_rights":"CC BY-SA 4.0"}\n'
        )
        native_files = export._inventory(self.source / "native")
        (self.source / "native" / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in native_files.items())
        )
        (self.source / "card-artifacts").mkdir()
        for name in export.CHARTS:
            (self.source / "card-artifacts" / name).write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"/>\n'
            )
        (self.source / "LICENSE").write_text("Apache License\nVersion 2.0\n")
        (self.source / "NOTICE").write_text("Decision 2.0 modification notice\n")
        source_card = (
            "---\nlicense: apache-2.0\n---\n\n"
            + export.CARD_FOX
            + "\n\n# llm-semantic-router/DEV2.0-4B\n\n"
            + "\n".join(f"![chart]({name})" for name in export.CHARTS)
            + "\n\nThe six figures, scorer versions, frozen panel digests and paired intervals are\n"
            + "bound in `card-artifacts/manifest.json`.\n\n"
            + "`PACKAGE_MANIFEST.json` binds this card, native model, provenance, package\n"
            + "parity and all three scored native runs. The packager checks reviewed release\n"
            + "receipts and exact bytes; it does not rerun GPU inference or independently\n"
            + "authenticate the declared reviewer and timestamp log.\n"
            + "\n## Training, rights and limitations\n\n"
            + "| Source | Terms | Attribution | Use | Redistribution |\n"
            + "| --- | --- | --- | --- | --- |\n"
            + "| Test source | CC BY-SA 4.0 | Test author | Train | Private |\n\n"
            + "Known training/evaluation overlap:\nNone.\n"
        )
        (self.source / "README.md").write_text(source_card)
        (self.source / "release-record.json").write_text(
            json.dumps(
                {
                    "model_id": export.MODEL_ID,
                    "license_id": "apache-2.0",
                    "base_model": {
                        "id": "caiovicentino1/Eikos-4B",
                        "revision": "582ffb13f19a4da3f455e3db198584190bd7755b",
                    },
                    "rights": {
                        "sources": [
                            {
                                "name": "Source A",
                                "license": "CC BY 4.0",
                                "attribution": "Author A",
                            }
                        ]
                    },
                }
            )
        )
        (self.source / "release-gate.json").write_text(
            json.dumps({"status": "passed_postkey_user_directed"})
        )
        files = export._inventory(self.source)
        (self.source / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(
                {
                    "model_id": export.MODEL_ID,
                    "model_revision": "checkpoint-0232",
                    "parameter_count": 4_205_751_296,
                    "native_model_sha256": export._sha(
                        self.source / "native" / "SHA256SUMS"
                    ),
                    "panel_sha256": {"typed": "a" * 64},
                    "files_sha256": files,
                }
            )
        )
        self.owl = self.root / "owl.png"
        self.owl.write_bytes(b"pinned owned Decision 1.0 owl bytes")
        self.output = self.root / "hf-export"

    def _export(self) -> dict:
        with patch.object(
            export, "OWL_SHA256", hashlib.sha256(self.owl.read_bytes()).hexdigest()
        ):
            return export.export(self.source, self.owl, self.output, research_only=True)

    def test_official_export_rejects_third_party_decision_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "own Decision 1.0 or unmodified"):
            export.export(self.source, self.owl, self.output)
        self.assertFalse(self.output.exists())

    def test_legacy_export_never_certifies_official_release(self) -> None:
        with patch.object(export, "_require_official_lineage"):
            with self.assertRaisesRegex(ValueError, "research-only"):
                export.export(self.source, self.owl, self.output)
        self.assertFalse(self.output.exists())

    def test_official_lineage_predicate_requires_pinned_own_or_qwen_source(
        self,
    ) -> None:
        for source_id in (
            "llm-semantic-router/Decision-1.0-Nox-4B",
            "Qwen/Qwen3.5-4B-Base",
            "Qwen/Qwen3.8-27B",
        ):
            export._require_official_lineage(
                {"base_model": {"id": source_id, "revision": "a" * 40}}
            )
        for source_id in (
            "caiovicentino1/Eikos-4B",
            "Mapika/decider-4b",
            "Qwen/ThirdPartyDecision-4B",
        ):
            with self.assertRaisesRegex(ValueError, "Official Decision 2.0"):
                export._require_official_lineage(
                    {"base_model": {"id": source_id, "revision": "a" * 40}}
                )

    def test_slim_tree_preserves_exact_native_and_figures(self) -> None:
        result = self._export()
        self.assertEqual(result["model_id"], export.MODEL_ID)
        self.assertEqual(result["publication_scope"], "research_only")
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {
                export.RESEARCH_MARKER,
                "README.md",
                "ATTRIBUTIONS.md",
                "LICENSE",
                "LICENSE-Eikos",
                "LICENSE-Qwen",
                "NOTICE",
                "assets",
                "model",
                "evaluation",
            },
        )
        self.assertEqual(
            (self.output / "model" / "model.safetensors").read_bytes(), b"model weights"
        )
        self.assertFalse((self.output / "model" / "decision2_provenance.json").exists())
        self.assertNotIn(
            "decision2_provenance.json",
            (self.output / "model" / "SHA256SUMS").read_text(),
        )
        self.assertNotEqual(
            result["scored_native_model_sha256"],
            result["public_runtime_manifest_sha256"],
        )
        self.assertEqual(
            (self.output / "assets" / export.CHARTS[0]).read_bytes(),
            (self.source / "card-artifacts" / export.CHARTS[0]).read_bytes(),
        )
        self.assertEqual(
            (self.output / "LICENSE-Eikos").read_bytes(),
            (self.source / "native" / "LICENSE").read_bytes(),
        )
        self.assertEqual(
            (self.output / "LICENSE-Qwen").read_bytes(),
            (self.source / "native" / "LICENSE-Qwen").read_bytes(),
        )
        self.assertEqual(
            (self.output / "NOTICE").read_bytes(), (self.source / "NOTICE").read_bytes()
        )
        self.assertIn(
            "[Eikos native notice](model/NOTICE)",
            (self.output / "ATTRIBUTIONS.md").read_text(),
        )
        card = (self.output / "README.md").read_text()
        self.assertIn("**Research only.**", card)
        self.assertIn("DEV2.0-4B mosaic owl", card)
        self.assertIn("assets/jevarena-rank.svg", card)
        self.assertNotIn("crossroads fox", card)
        self.assertNotIn("CC BY-SA", card)
        self.assertNotIn("PACKAGE_MANIFEST.json", card)
        self.assertNotIn("release-gate.json", export._inventory(self.output))
        with self.assertRaisesRegex(ValueError, "not an official release"):
            export.verify(self.output)
        self.assertEqual(
            export.verify(self.output, allow_research_only=True)["parameter_count"],
            4_205_751_296,
        )

    def test_source_tampering_is_rejected(self) -> None:
        (self.source / "native" / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Private package bytes differ"):
            self._export()
        self.assertFalse(self.output.exists())

    def test_public_tampering_is_rejected(self) -> None:
        self._export()
        (self.output / "model" / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            export.verify(self.output, allow_research_only=True)

    def test_broken_card_reference_is_rejected_even_if_rehashed(self) -> None:
        self._export()
        card_path = self.output / "README.md"
        card_path.write_text(
            card_path.read_text() + "\n![missing](assets/missing.svg)\n"
        )
        manifest_path = self.output / "evaluation" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["README.md"] = export._sha(card_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Broken local model-card reference"):
            export.verify(self.output, allow_research_only=True)

    def test_unpinned_owl_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned owned asset"):
            export.export(self.source, self.owl, self.output, research_only=True)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
