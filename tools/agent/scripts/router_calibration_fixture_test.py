import base64
import hashlib
import importlib
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Any

import yaml
from yaml.tokens import AliasToken, AnchorToken, TagToken

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

router_calibration_manifest = importlib.import_module("router_calibration_manifest")
router_calibration_support = importlib.import_module("router_calibration_support")
router_calibration_evaluation = importlib.import_module("router_calibration_evaluation")
router_calibration_schema = importlib.import_module("router_calibration_schema")

ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)
ONE_PIXEL_PNG_SHA256 = (
    "sha256:431ced6916a2a21a156e38701afe55bbd7f88969fbbfc56d7fe099d47f265460"
)


def _png_with_dimensions(width: int, height: int) -> tuple[str, str]:
    payload = bytearray(base64.b64decode(ONE_PIXEL_PNG_BASE64, validate=True))
    payload[16:24] = struct.pack(">II", width, height)
    payload[29:33] = struct.pack(">I", zlib.crc32(payload[12:29]))
    encoded = base64.b64encode(payload).decode("ascii")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return encoded, digest


def _write_probe_manifest(
    path: Path,
    decisions: str,
    fixtures: str = "",
) -> None:
    path.write_text(
        f"""\
schema_version: v1
name: compact-fixture-test
routing_assets:
  yaml: test.yaml
  dsl: test.dsl
{fixtures}coverage:
  min_signal_assertion_percent: 0
  min_projection_assertion_percent: 0
  min_algorithm_assertion_percent: 0
  min_plugin_assertion_percent: 0
  required_request_shapes: []
  min_tag_counts: {{}}
  min_tag_pass_rate: {{}}
decisions:
{decisions}""",
        encoding="utf-8",
    )


def _normalize_adjacent_message_text_parts(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ignore content-part framing while retaining every request-bearing value."""
    normalized: list[dict[str, Any]] = []
    for original in messages:
        message = dict(original)
        content = original.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        merged: list[Any] = []
        for raw_item in content:
            item = dict(raw_item) if isinstance(raw_item, dict) else raw_item
            kind = (
                str(item.get("type") or "").strip().lower()
                if isinstance(item, dict)
                else None
            )
            text_item = (
                isinstance(item, dict)
                and kind in ("", "text", "input_text", "output_text")
                and isinstance(item.get("text"), str)
            )
            if text_item and merged and isinstance(merged[-1], dict):
                previous = merged[-1]
                previous_kind = str(previous.get("type") or "").strip().lower()
                previous_shape = {
                    key: value for key, value in previous.items() if key != "text"
                }
                item_shape = {
                    key: value for key, value in item.items() if key != "text"
                }
                if (
                    previous_kind == kind
                    and previous_shape == item_shape
                    and isinstance(previous.get("text"), str)
                ):
                    previous["text"] += item["text"]
                    continue
            merged.append(item)
        message["content"] = merged
        normalized.append(message)
    return normalized


def _materialized_image_url(item: dict[str, Any]) -> str | None:
    if str(item.get("type") or "").strip().lower() != "image_url":
        return None
    image_url = item.get("image_url")
    if not isinstance(image_url, dict):
        return None
    url = image_url.get("url")
    return url if isinstance(url, str) else None


def _message_payload_receipt(
    messages: list[dict[str, Any]],
) -> tuple[bytes, list[str], int]:
    parts: list[bytes] = []
    image_urls: list[str] = []
    image_parts = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content.encode("utf-8"))
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            url = _materialized_image_url(item)
            if url is not None:
                image_parts += 1
                image_urls.append(url)
            kind = str(item.get("type") or "").strip().lower()
            if kind in ("", "text", "input_text", "output_text") and isinstance(
                item.get("text"), str
            ):
                parts.append(item["text"].encode("utf-8"))
    return b"".join(parts), image_urls, image_parts


def _update_probe_digest(digest: Any, probe_id: str, payload: bytes) -> None:
    digest.update(probe_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _mom_materialization_receipt(probes: list[Any]) -> dict[str, Any]:
    text_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    receipt: dict[str, Any] = {
        "message_probes": 0,
        "generated_probes": 0,
        "text_bytes": 0,
        "image_parts": 0,
        "image_urls": set(),
    }
    for probe in probes:
        if not probe.messages:
            continue
        receipt["message_probes"] += 1
        receipt["generated_probes"] += probe.generated_text is not None
        messages = router_calibration_support.materialize_probe_messages(probe)
        payload, image_urls, image_parts = _message_payload_receipt(messages)
        _update_probe_digest(text_digest, probe.probe_id, payload)
        semantic_request = json.dumps(
            {
                "messages": _normalize_adjacent_message_text_parts(messages),
                "tools": list(probe.tools),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _update_probe_digest(semantic_digest, probe.probe_id, semantic_request)
        receipt["text_bytes"] += len(payload)
        receipt["image_parts"] += image_parts
        receipt["image_urls"].update(image_urls)
    receipt["text_sha256"] = text_digest.hexdigest()
    receipt["semantic_sha256"] = semantic_digest.hexdigest()
    return receipt


class CompactProbeFixtureTest(unittest.TestCase):
    def test_manifest_rejects_raw_image_sources_without_echoing_them(self) -> None:
        raw_sources = (
            "http://169.254.169.254/latest/meta-data",
            "http://127.0.0.1/private",
            "file:///etc/passwd",
            "data:image/png;base64,AA==",
        )
        for source in raw_sources:
            with self.subTest(source=source):
                decisions = f"""\
  - id: vision
    expected_decision: vision
    variants:
      - id: raw
        messages:
          - role: user
            content:
              - type: text
                text: inspect
              - type: image_url
                image_url:
                  url: {source}
"""
                with tempfile.TemporaryDirectory() as tempdir:
                    manifest_path = Path(tempdir) / "probes.yaml"
                    _write_probe_manifest(manifest_path, decisions)
                    with self.assertRaisesRegex(
                        ValueError, "must use a declared image_fixture"
                    ) as caught:
                        router_calibration_manifest.load_probe_manifest(manifest_path)
                self.assertNotIn(source, str(caught.exception))

    def test_generated_text_counts_every_router_text_part_and_stays_compact(
        self,
    ) -> None:
        decisions = """\
  - id: long_context
    expected_decision: long_context
    variants:
      - id: response_shape
        messages:
          - role: user
            content:
              - type: text
                text: 'ask '
              - type: output_text
                text: ok
              - type: metadata
                value: ignored-non-text-part
        generated_text:
          message_index: 0
          content_index: 2
          target_text_bytes: 11
"""
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            _write_probe_manifest(manifest_path, decisions)
            _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)

        probe = probes[0]
        messages = router_calibration_support.materialize_probe_messages(probe)
        content = messages[0]["content"]
        self.assertEqual(len(content), 4)
        self.assertEqual(content[2], {"type": "text", "text": "xxxxx"})
        receipt = router_calibration_support.probe_materialized_messages_metadata(probe)
        self.assertEqual(receipt["text_bytes"], 11)
        report_json = json.dumps(
            router_calibration_support.failed_probe_result(
                probe, RuntimeError("fixture failure")
            )
        )
        self.assertNotIn('"text": "xxxxx"', report_json)
        self.assertIn('"target_text_bytes": 11', report_json)
        self.assertEqual(len(probe.messages[0]["content"]), 3)

    def test_image_fixture_type_requires_canonical_lowercase(self) -> None:
        fixtures = f"""\
fixtures:
  images:
    pixel:
      description: One-pixel fixture for loader tests.
      media_type: image/png
      data_base64: {ONE_PIXEL_PNG_BASE64}
      sha256: {ONE_PIXEL_PNG_SHA256}
"""
        decisions_template = """\
  - id: vision
    expected_decision: vision
    variants:
      - id: fixture
        messages:
          - role: user
            content:
              - type: {item_type}
                fixture: pixel
"""
        for item_type in ("IMAGE_FIXTURE", "' image_fixture '"):
            with (
                self.subTest(item_type=item_type),
                tempfile.TemporaryDirectory() as tempdir,
            ):
                manifest_path = Path(tempdir) / "probes.yaml"
                _write_probe_manifest(
                    manifest_path,
                    decisions_template.format(item_type=item_type),
                    fixtures,
                )
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                self.assertTrue(
                    list(
                        router_calibration_schema.probe_manifest_validator().iter_errors(
                            manifest
                        )
                    )
                )
                with self.assertRaisesRegex(
                    ValueError, "type must be exactly 'image_fixture'"
                ):
                    router_calibration_manifest.load_probe_manifest(manifest_path)

        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            _write_probe_manifest(
                manifest_path,
                decisions_template.format(item_type="image_fixture"),
                fixtures,
            )
            _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)
        probes[0].messages[0]["content"][0]["type"] = "IMAGE_FIXTURE"
        with self.assertRaisesRegex(ValueError, "type must be exactly 'image_fixture'"):
            router_calibration_support.materialize_probe_messages(probes[0])

    def test_generated_text_rejects_target_below_explicit_text(self) -> None:
        decisions = """\
  - id: long_context
    expected_decision: long_context
    variants:
      - id: invalid
        messages:
          - role: user
            content:
              - type: text
                text: explicit
        generated_text:
          message_index: 0
          content_index: 1
          target_text_bytes: 8
"""
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            _write_probe_manifest(manifest_path, decisions)
            with self.assertRaisesRegex(ValueError, "must exceed.*explicit text"):
                router_calibration_manifest.load_probe_manifest(manifest_path)

    def test_image_fixture_materializes_metadata_without_report_payload(self) -> None:
        fixtures = f"""\
fixtures:
  images:
    pixel:
      description: One-pixel fixture for loader tests.
      media_type: image/png
      data_base64: {ONE_PIXEL_PNG_BASE64}
      sha256: {ONE_PIXEL_PNG_SHA256}
"""
        decisions = """\
  - id: vision
    expected_decision: vision
    variants:
      - id: fixture
        messages:
          - role: user
            content:
              - type: text
                text: inspect
              - type: image_fixture
                fixture: pixel
"""
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            _write_probe_manifest(manifest_path, decisions, fixtures)
            manifest, probes = router_calibration_manifest.load_probe_manifest(
                manifest_path
            )

        messages = router_calibration_support.materialize_probe_messages(probes[0])
        self.assertEqual(
            messages[0]["content"][1],
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ONE_PIXEL_PNG_BASE64}"},
            },
        )
        safe_manifest = router_calibration_manifest.report_safe_probe_manifest(manifest)
        image_metadata = safe_manifest["fixtures"]["images"]["pixel"]
        self.assertEqual(
            image_metadata["description"], "One-pixel fixture for loader tests."
        )
        self.assertNotIn("data_base64", image_metadata)
        self.assertEqual(image_metadata["bytes"], 68)
        report_json = json.dumps(
            router_calibration_support.failed_probe_result(
                probes[0], RuntimeError("fixture failure")
            )
        )
        self.assertNotIn("data:image/png;base64", report_json)
        self.assertNotIn('"data_base64"', report_json)
        self.assertIn('"fixture": "pixel"', report_json)
        self.assertIn('"materialized_messages"', report_json)

    def test_image_fixture_rejects_unsafe_payload_integrity_and_references(
        self,
    ) -> None:
        fixture_template = """\
fixtures:
  images:
    pixel:
      description: One-pixel fixture for loader tests.
      media_type: {media_type}
      data_base64: {data_base64}
      sha256: {sha256}
"""
        decisions_template = """\
  - id: vision
    expected_decision: vision
    variants:
      - id: fixture
        messages:
          - role: user
            content:
              - type: image_fixture
                fixture: {reference}
"""
        fake_payload = b"not an encoded image"
        fake_base64 = base64.b64encode(fake_payload).decode("ascii")
        fake_sha256 = f"sha256:{hashlib.sha256(fake_payload).hexdigest()}"
        static_gif = base64.b64decode(
            "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
            validate=True,
        )
        animated_gif = static_gif[:-1] + static_gif[19:-1] + static_gif[-1:]
        animated_gif_base64 = base64.b64encode(animated_gif).decode("ascii")
        animated_gif_sha256 = f"sha256:{hashlib.sha256(animated_gif).hexdigest()}"
        bomb_base64, bomb_sha256 = _png_with_dimensions(4097, 4096)
        cases = (
            (
                ONE_PIXEL_PNG_BASE64,
                ONE_PIXEL_PNG_SHA256,
                "image/png",
                "missing",
                "unknown image fixture",
            ),
            (
                ONE_PIXEL_PNG_BASE64,
                "sha256:" + "0" * 64,
                "image/png",
                "pixel",
                "sha256 mismatch",
            ),
            (
                "AB==",
                ONE_PIXEL_PNG_SHA256,
                "image/png",
                "pixel",
                "canonical padding",
            ),
            (
                fake_base64,
                fake_sha256,
                "image/png",
                "pixel",
                "valid supported image",
            ),
            (
                ONE_PIXEL_PNG_BASE64,
                ONE_PIXEL_PNG_SHA256,
                "image/jpeg",
                "pixel",
                "does not match detected",
            ),
            (
                bomb_base64,
                bomb_sha256,
                "image/png",
                "pixel",
                "pixel canvas limit",
            ),
            (
                animated_gif_base64,
                animated_gif_sha256,
                "image/gif",
                "pixel",
                "valid supported image",
            ),
        )
        for data_base64, sha256, media_type, reference, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tempdir:
                manifest_path = Path(tempdir) / "probes.yaml"
                _write_probe_manifest(
                    manifest_path,
                    decisions_template.format(reference=reference),
                    fixture_template.format(
                        data_base64=data_base64,
                        sha256=sha256,
                        media_type=media_type,
                    ),
                )
                with self.assertRaisesRegex(ValueError, error):
                    router_calibration_manifest.load_probe_manifest(manifest_path)

    def test_image_fixture_reference_requires_string(self) -> None:
        fixtures = f"""\
fixtures:
  images:
    pixel:
      description: One-pixel fixture for loader tests.
      media_type: image/png
      data_base64: {ONE_PIXEL_PNG_BASE64}
      sha256: {ONE_PIXEL_PNG_SHA256}
"""
        decisions = """\
  - id: vision
    expected_decision: vision
    variants:
      - id: fixture
        messages:
          - role: user
            content:
              - type: image_fixture
                fixture: 1
"""
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            _write_probe_manifest(manifest_path, decisions, fixtures)
            with self.assertRaisesRegex(
                ValueError, "does not satisfy recipe probe schema"
            ):
                router_calibration_manifest.load_probe_manifest(manifest_path)

    def test_image_fixture_expansion_respects_complete_request_budget(self) -> None:
        fixtures = f"""\
fixtures:
  images:
    pixel:
      description: One-pixel fixture for loader tests.
      media_type: image/png
      data_base64: {ONE_PIXEL_PNG_BASE64}
      sha256: {ONE_PIXEL_PNG_SHA256}
"""
        decisions = """\
  - id: vision
    model: test/model
    expected_decision: vision
    variants:
      - id: fixture
        messages:
          - role: user
            content:
              - type: image_fixture
                fixture: pixel
              - type: image_fixture
                fixture: pixel
              - type: image_fixture
                fixture: pixel
"""
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "probes.yaml"
            _write_probe_manifest(manifest_path, decisions, fixtures)
            _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)
        probe = probes[0]
        compact_payload = {"messages": list(probe.messages), "model": probe.model}
        compact_bytes = len(json.dumps(compact_payload).encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "materialized messages exceed"):
            router_calibration_evaluation._build_request_payload(
                probe, max_request_bytes=compact_bytes + 1
            )

    def test_probe_loader_rejects_yaml_indirection(self) -> None:
        cases = (
            ("value: &hidden 1\n", "anchors"),
            ("value: *hidden\n", "aliases"),
            ("value: !!str hidden\n", "tags"),
            ("value:\n  <<: {}\n", "merge keys"),
        )
        for document, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                router_calibration_manifest.reject_yaml_indirection(
                    document, Path("untrusted/probes.yaml")
                )

    def test_mom_materialization_preserves_legacy_semantic_receipt(self) -> None:
        manifest_path = (
            SCRIPT_DIR.parents[2]
            / "config"
            / "recipes"
            / "built-in"
            / "latest"
            / "mom-v1"
            / "probes.yaml"
        )
        document = manifest_path.read_text(encoding="utf-8")
        self.assertLess(len(document.encode("utf-8")), 300_000)
        self.assertLess(len(document.splitlines()), 8_000)
        tokens = tuple(yaml.scan(document))
        self.assertFalse(
            any(
                isinstance(token, (AnchorToken, AliasToken, TagToken))
                for token in tokens
            )
        )
        self.assertFalse(
            router_calibration_manifest._contains_yaml_merge(yaml.compose(document))
        )

        _, probes = router_calibration_manifest.load_probe_manifest(manifest_path)
        receipt = _mom_materialization_receipt(probes)
        self.assertEqual(len(probes), 226)
        self.assertEqual(receipt["message_probes"], 85)
        self.assertEqual(receipt["generated_probes"], 50)
        self.assertEqual(receipt["image_parts"], 57)
        self.assertEqual(receipt["text_bytes"], 26_229_731)
        self.assertEqual(len(receipt["image_urls"]), 1)
        image_url = next(iter(receipt["image_urls"]))
        self.assertEqual(
            hashlib.sha256(image_url.encode("utf-8")).hexdigest(),
            "c5506a9f23e55bafd166bd07c682566c21363a25c1433cd6e92bf22d3e28a1b3",
        )
        fixture = probes[0].image_fixtures["mom_v1_architecture"]
        self.assertEqual(fixture.bytes, 651)
        self.assertEqual(
            fixture.description,
            "Compact Client-to-Router-to-Backend diagram used only to exercise "
            "image payload shape and vision-routing branches.",
        )
        self.assertEqual(
            fixture.sha256,
            "sha256:76f8c378648b27757ac72a7ea00f32cc76d4a4240c6e84002a182a0c2de5fde3",
        )
        self.assertEqual(
            receipt["text_sha256"],
            "17e56c0457f3ffd9419a25ad8b8e0de60c21fbcf62a88d31fced47c2d1a189b7",
        )
        self.assertEqual(
            receipt["semantic_sha256"],
            "83e3b91d616567bd6f0bc0825c0523108d85a05c802dd6d1139cf3b9c265fac3",
        )


if __name__ == "__main__":
    unittest.main()
